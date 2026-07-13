using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using TrackLab.Datastruct;
using TrackLab.Pipeline;

namespace SnGameState.Calibration;

/// <summary>
/// Port of sn_gamestate.calibration.nbjw_calib.NBJW_Calib_Keypoints running ONNX exports
/// of the two "No Bells, Just Whistles" HRNet models (SV_kp keypoints + SV_lines lines,
/// see tools/export/export_nbjw.py). Produces per-image pitch "keypoints" (ids 1..72,
/// normalized to [0,1] of the 960x540 input) and the derived "lines".
/// </summary>
public sealed class NbjwCalibKeypoints : ImageLevelModule, IDisposable
{
    private const int InputWidth = 960;
    private const int InputHeight = 540;

    private readonly InferenceSession _kpSession;
    private readonly InferenceSession _linesSession;

    public override IReadOnlyDictionary<string, string[]> InputColumns { get; } = new Dictionary<string, string[]>();

    public override IReadOnlyDictionary<string, string[]> OutputColumns { get; } = new Dictionary<string, string[]>
    {
        ["image"] = new[] { "keypoints", "lines" },
    };

    public NbjwCalibKeypoints(string kpModelPath, string linesModelPath, int batchSize = 1)
        : base(batchSize)
    {
        _kpSession = new InferenceSession(kpModelPath);
        _linesSession = new InferenceSession(linesModelPath);
    }

    public override object Preprocess(Mat image, DataFrame detections, Series metadata)
    {
        // T.Compose([T.Resize((540, 960)), T.ToTensor()]) — RGB in [0, 1], no normalization
        using var resized = image.Resize(new Size(InputWidth, InputHeight));
        var tensor = new float[3 * InputHeight * InputWidth];
        resized.GetArray(out Vec3b[] pixels);
        for (var y = 0; y < InputHeight; y++)
            for (var x = 0; x < InputWidth; x++)
            {
                var px = pixels[y * InputWidth + x];
                tensor[0 * InputHeight * InputWidth + y * InputWidth + x] = px.Item2 / 255f;
                tensor[1 * InputHeight * InputWidth + y * InputWidth + x] = px.Item1 / 255f;
                tensor[2 * InputHeight * InputWidth + y * InputWidth + x] = px.Item0 / 255f;
            }
        return tensor;
    }

    public override ModuleOutput Process(IReadOnlyList<object> batch, DataFrame detections, IReadOnlyList<Series> metadatas)
    {
        var samples = batch.Cast<float[]>().ToList();
        var input = new DenseTensor<float>(new[] { samples.Count, 3, InputHeight, InputWidth });
        for (var n = 0; n < samples.Count; n++)
            samples[n].CopyTo(input.Buffer.Span.Slice(n * samples[n].Length));

        var kpHeatmaps = Run(_kpSession, input);
        var lineHeatmaps = Run(_linesSession, input);

        var kpDicts = HeatmapDecoder.DecodeKeypoints(kpHeatmaps);
        var lineDicts = HeatmapDecoder.DecodeLines(lineHeatmaps);

        var imageUpdates = new List<Series>(samples.Count);
        for (var n = 0; n < samples.Count; n++)
        {
            var complete = KeypointCompleter.Complete(
                kpDicts[n], lineDicts[n], InputWidth, InputHeight, normalize: true);
            var row = new Series(metadatas[n].Index);
            row["keypoints"] = complete;
            row["lines"] = PitchGeometry.KeypointsToLines(complete);
            imageUpdates.Add(row);
        }
        return new ModuleOutput(Array.Empty<Series>(), imageUpdates);
    }

    private static Tensor<float> Run(InferenceSession session, DenseTensor<float> input)
    {
        var inputName = session.InputMetadata.Keys.First();
        using var results = session.Run(new[] { NamedOnnxValue.CreateFromTensor(inputName, input) });
        // clone out of the disposed result buffer
        return results.First().AsTensor<float>().ToDenseTensor();
    }

    public void Dispose()
    {
        _kpSession.Dispose();
        _linesSession.Dispose();
    }
}
