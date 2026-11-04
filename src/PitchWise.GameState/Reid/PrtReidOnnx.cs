using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using PitchWise.Tracking.Datastruct;
using PitchWise.Tracking.Pipeline;
using PitchWise.Tracking.Utils;

namespace PitchWise.GameState.Reid;

/// <summary>
/// Re-identification running an ONNX export of the
/// prtreid-soccernet-baseline model (see tools/export/export_prtreid.py).
/// For every detection crop it produces the global ReID embedding, part visibility
/// scores and the role classification (ball/goalkeeper/other/player/referee).
/// </summary>
public sealed class PrtReidOnnx : DetectionLevelModule, IDisposable
{
    /// <summary>role_mapping of the Python module: class index -> role name.</summary>
    private static readonly string[] Roles = { "ball", "goalkeeper", "other", "player", "referee" };

    private static readonly float[] ImageNetMean = { 0.485f, 0.456f, 0.406f };
    private static readonly float[] ImageNetStd = { 0.229f, 0.224f, 0.225f };

    private readonly InferenceSession _session;
    private readonly int _height;
    private readonly int _width;

    public override IReadOnlyDictionary<string, string[]> InputColumns { get; } = new Dictionary<string, string[]>
    {
        ["detection"] = new[] { "bbox_ltwh" },
    };

    public override IReadOnlyDictionary<string, string[]> OutputColumns { get; } = new Dictionary<string, string[]>
    {
        ["detection"] = new[] { "embeddings", "visibility_scores", "role_detection", "role_confidence" },
    };

    public PrtReidOnnx(string modelPath, int batchSize = 8, int height = 256, int width = 128)
        : base(batchSize)
    {
        _height = height;
        _width = width;
        _session = OnnxSessions.Create(modelPath);
    }

    public override object Preprocess(Mat image, Series detection, Series metadata)
    {
        var bbox = Coordinates.ClipLtwhToImageDim(detection.Get<float[]>("bbox_ltwh"), image.Width, image.Height);
        var rect = new Rect((int)bbox[0], (int)bbox[1], Math.Max((int)bbox[2], 1), Math.Max((int)bbox[3], 1));
        using var crop = new Mat(image, rect);
        using var resized = crop.Resize(new Size(_width, _height));

        // BGR -> RGB + ImageNet normalization (torchreid transforms), vectorized
        var tensor = new float[3 * _height * _width];
        TensorOps.CopyNormalizedChw(resized, tensor, ImageNetMean, ImageNetStd);
        return new Sample(tensor, detection.Index);
    }

    public override IReadOnlyList<Series> Process(IReadOnlyList<object> batch, DataFrame detections, IReadOnlyList<Series> metadatas)
    {
        var samples = batch.Cast<Sample>().ToList();
        var input = new DenseTensor<float>(new[] { samples.Count, 3, _height, _width });
        for (var n = 0; n < samples.Count; n++)
            samples[n].Tensor.CopyTo(input.Buffer.Span.Slice(n * samples[n].Tensor.Length));

        var inputName = _session.InputMetadata.Keys.First();
        using var results = OnnxSessions.Run(_session, new[] { NamedOnnxValue.CreateFromTensor(inputName, input) });
        var outputs = results.ToDictionary(r => r.Name, r => r.AsTensor<float>());
        var embeddings = outputs["embeddings"];           // [N, D]
        var visibility = outputs["visibility_scores"];    // [N, K]
        var roleScores = outputs["role_scores"];          // [N, 5]

        var updated = new List<Series>(samples.Count);
        for (var n = 0; n < samples.Count; n++)
        {
            var row = new Series(samples[n].DetectionIndex);
            row["embeddings"] = SliceRow(embeddings, n);
            row["visibility_scores"] = SliceRow(visibility, n);

            var scores = Softmax(SliceRow(roleScores, n));
            var best = Array.IndexOf(scores, scores.Max());
            row["role_detection"] = Roles[best];
            row["role_confidence"] = (double)scores[best];
            updated.Add(row);
        }
        return updated;
    }

    private static float[] SliceRow(Tensor<float> tensor, int n)
    {
        var size = tensor.Dimensions[1];
        var row = new float[size];
        for (var i = 0; i < size; i++) row[i] = tensor[n, i];
        return row;
    }

    private static float[] Softmax(float[] logits)
    {
        var max = logits.Max();
        var exp = logits.Select(v => MathF.Exp(v - max)).ToArray();
        var sum = exp.Sum();
        return exp.Select(v => v / sum).ToArray();
    }

    private sealed record Sample(float[] Tensor, long DetectionIndex);

    public void Dispose() => _session.Dispose();
}
