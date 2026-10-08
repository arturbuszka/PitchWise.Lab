using Microsoft.Extensions.Logging;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;

namespace PitchWise.GameState;

/// <summary>Available ONNX Runtime execution providers.</summary>
public enum OnnxProvider
{
    Cpu,
    /// <summary>DirectML — any DX12-capable GPU on Windows, no extra installation.</summary>
    DirectML,
    /// <summary>CUDA — NVIDIA GPU on Linux; needs the NVIDIA driver, CUDA 12 and cuDNN 9 installed.</summary>
    Cuda,
}

/// <summary>
/// Central factory for inference sessions so the execution provider (CPU / DirectML GPU)
/// is configured in one place — the "device" config equivalent.
/// </summary>
public static class OnnxSessions
{
    public static OnnxProvider Provider { get; set; } = OnnxProvider.Cpu;
    public static int DeviceId { get; set; }

    /// <summary>Receives which provider each session really ended up on (set by PipelineConfig.BuildPipeline).</summary>
    public static ILogger? Log { get; set; }

    public static InferenceSession Create(string modelPath)
    {
        var name = Path.GetFileName(modelPath);

        // each GPU provider only exists on its own platform's package; anything else runs on CPU
        if (Provider == OnnxProvider.Cpu)
        {
            Log?.LogInformation("ONNX {Model}: running on CPU", name);
            return new InferenceSession(modelPath);
        }
        if ((Provider == OnnxProvider.DirectML && !OperatingSystem.IsWindows())
            || (Provider == OnnxProvider.Cuda && !OperatingSystem.IsLinux()))
        {
            Log?.LogWarning("ONNX {Model}: provider {Provider} is not available on this OS, running on CPU", name, Provider);
            return new InferenceSession(modelPath);
        }

        if (Provider == OnnxProvider.Cuda)
        {
            try
            {
                var cuda = new SessionOptions();
                cuda.AppendExecutionProvider_CUDA(DeviceId);
                var session = new InferenceSession(modelPath, cuda);
                Log?.LogInformation("ONNX {Model}: running on CUDA (device {Device})", name, DeviceId);
                return session;
            }
            catch (Exception e) when (e is OnnxRuntimeException or DllNotFoundException or EntryPointNotFoundException)
            {
                // CUDA libraries missing or incompatible — still usable, just slow
                var msg = $"ONNX CUDA provider unavailable for {name}, falling back to CPU: {e.Message} [{CudaDiagnostics()}]";
                if (Log is null) Console.Error.WriteLine(msg); else Log.LogWarning("{Message}", msg);
                return new InferenceSession(modelPath);
            }
        }

        var options = new SessionOptions();
        // DML requires disabling memory-pattern optimization and works best sequential
        options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        options.EnableMemoryPattern = false;
        options.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
        // the DML graph-fusion pass hung the device (DXGI_ERROR_DEVICE_HUNG) on a
        // GTX 1660 Ti — run the graph unfused, kernels still execute on the GPU
        options.AddSessionConfigEntry("ep.dml.disable_graph_fusion", "1");
        options.AppendExecutionProvider_DML(DeviceId);
        var dml = new InferenceSession(modelPath, options);
        Log?.LogInformation("ONNX {Model}: running on DirectML (device {Device})", name, DeviceId);
        return dml;
    }

    [DllImport("libcudart.so.12", EntryPoint = "cudaGetDeviceCount")]
    private static extern int CudaGetDeviceCount(out int count);

    /// <summary>What the CUDA runtime sees from inside this process, to tell "no GPU visible" from "GPU present, EP broken".</summary>
    private static string CudaDiagnostics()
    {
        var visible = Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES") ?? "<unset>";
        try
        {
            var rc = CudaGetDeviceCount(out var count);
            return $"cudaGetDeviceCount rc={rc} count={count}, CUDA_VISIBLE_DEVICES={visible}";
        }
        catch (Exception e) when (e is DllNotFoundException or EntryPointNotFoundException)
        {
            return $"libcudart.so.12 not loadable ({e.GetType().Name}), CUDA_VISIBLE_DEVICES={visible}";
        }
    }
}
