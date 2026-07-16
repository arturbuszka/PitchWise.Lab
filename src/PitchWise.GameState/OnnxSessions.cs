using Microsoft.ML.OnnxRuntime;

namespace PitchWise.GameState;

/// <summary>Available ONNX Runtime execution providers.</summary>
public enum OnnxProvider
{
    Cpu,
    /// <summary>DirectML — any DX12-capable GPU on Windows, no extra installation.</summary>
    DirectML,
}

/// <summary>
/// Central factory for inference sessions so the execution provider (CPU / DirectML GPU)
/// is configured in one place — the "device" config equivalent.
/// </summary>
public static class OnnxSessions
{
    public static OnnxProvider Provider { get; set; } = OnnxProvider.Cpu;
    public static int DeviceId { get; set; }

    public static InferenceSession Create(string modelPath)
    {
        if (Provider == OnnxProvider.Cpu)
            return new InferenceSession(modelPath);

        var options = new SessionOptions();
        // DML requires disabling memory-pattern optimization and works best sequential
        options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        options.EnableMemoryPattern = false;
        options.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
        // the DML graph-fusion pass hung the device (DXGI_ERROR_DEVICE_HUNG) on a
        // GTX 1660 Ti — run the graph unfused, kernels still execute on the GPU
        options.AddSessionConfigEntry("ep.dml.disable_graph_fusion", "1");
        options.AppendExecutionProvider_DML(DeviceId);
        return new InferenceSession(modelPath, options);
    }
}
