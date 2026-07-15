using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using TrackLab.Datastruct;
using TrackLab.Pipeline;

namespace SnGameState.Detection;

/// <summary>
/// Port of tracklab.wrappers.bbox_detector.yolo_ultralytics_api.YOLOv11 (the sn-gamestate
/// baseline detector), running an Ultralytics-exported ONNX model (yolo export format=onnx).
/// Keeps the "person" class like the baseline (cfg.cls = [0]) and, additionally, the single
/// most-confident "sports ball" detection per frame — the baseline is person-only, but the
/// world-state / statistics layer needs the ball. Ball rows carry category_id = 4 (see
/// <see cref="BallCategoryId"/>) and are skipped by the person tracker; everything else
/// (calibration → bbox_pitch) treats them uniformly.
/// </summary>
public sealed class YoloOnnxDetector : ImageLevelModule, IDisposable
{
    /// <summary>category_id written on person detection rows (matches the SoccerNet GT convention).</summary>
    public const long PersonCategoryId = 1L;

    /// <summary>category_id written on the ball detection row. Matches the SoccerNet GT
    /// convention (Labels-GameState.json categories: 1 player, 2 goalkeeper, 3 referee, 4 ball),
    /// so a predicted ball lines up with a ground-truth ball if ever compared by category.</summary>
    public const long BallCategoryId = 4L;

    /// <summary>COCO class index of the "sports ball" class in the stock yolo11 model.</summary>
    private const int CocoSportsBallClass = 32;

    private readonly InferenceSession _session;
    private readonly float _minConfidence;
    private readonly float _ballMinConfidence;
    private readonly int _inputSize;

    public override IReadOnlyDictionary<string, string[]> InputColumns { get; } = new Dictionary<string, string[]>();

    public override IReadOnlyDictionary<string, string[]> OutputColumns { get; } = new Dictionary<string, string[]>
    {
        ["detection"] = new[] { "image_id", "video_id", "category_id", "bbox_ltwh", "bbox_conf" },
    };

    public YoloOnnxDetector(string modelPath, float minConfidence = 0.4f, int batchSize = 1, int inputSize = 640,
        float ballMinConfidence = 0.15f)
        : base(batchSize)
    {
        _minConfidence = minConfidence;
        // The ball is small and often motion-blurred, so it scores far lower than a person.
        // A separate, looser threshold keeps it detectable without loosening person detection.
        _ballMinConfidence = ballMinConfidence;
        _inputSize = inputSize;
        _session = OnnxSessions.Create(modelPath);
    }

    public override object Preprocess(Mat image, DataFrame detections, Series metadata)
    {
        var (tensor, scale, padX, padY) = Letterbox(image, _inputSize);
        return new PreprocessedImage(tensor, scale, padX, padY, metadata);
    }

    public override ModuleOutput Process(IReadOnlyList<object> batch, DataFrame detections, IReadOnlyList<Series> metadatas)
    {
        var newRows = new List<Series>();
        foreach (var sample in batch.Cast<PreprocessedImage>())
        {
            var inputName = _session.InputMetadata.Keys.First();
            using var results = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(inputName, sample.Tensor) });
            var output = results.First().AsTensor<float>();

            foreach (var (bbox, conf) in DecodeAndNms(output, sample.Scale, sample.PadX, sample.PadY))
            {
                var row = new Series(detections.NextIndex++);
                row["image_id"] = sample.Metadata.Index;
                row["video_id"] = sample.Metadata.Get<long>("video_id");
                row["category_id"] = PersonCategoryId;
                row["bbox_ltwh"] = bbox;
                row["bbox_conf"] = conf;
                newRows.Add(row);
            }

            // One ball per frame: the single most-confident "sports ball" detection, if any.
            if (DecodeBestBall(output, sample.Scale, sample.PadX, sample.PadY) is var (ballBbox, ballConf))
            {
                var row = new Series(detections.NextIndex++);
                row["image_id"] = sample.Metadata.Index;
                row["video_id"] = sample.Metadata.Get<long>("video_id");
                row["category_id"] = BallCategoryId;
                row["bbox_ltwh"] = ballBbox;
                row["bbox_conf"] = ballConf;
                newRows.Add(row);
            }
        }
        return new ModuleOutput(newRows);
    }

    /// <summary>Decodes Ultralytics output [1, 4+nc, N] (cx cy w h + class scores),
    /// keeps class 0 (person) above the confidence threshold, then applies NMS.</summary>
    private IEnumerable<(float[] BboxLtwh, float Conf)> DecodeAndNms(
        Tensor<float> output, float scale, float padX, float padY)
    {
        var channels = output.Dimensions[1];
        var anchors = output.Dimensions[2];
        var boxes = new List<Rect2d>();
        var scores = new List<float>();
        for (var i = 0; i < anchors; i++)
        {
            var personScore = output[0, 4, i];
            for (var c = 5; c < channels; c++)
                if (output[0, c, i] > personScore)
                    goto notPerson; // baseline keeps detections whose best class is person
            if (personScore < _minConfidence) continue;

            var cx = (output[0, 0, i] - padX) / scale;
            var cy = (output[0, 1, i] - padY) / scale;
            var w = output[0, 2, i] / scale;
            var h = output[0, 3, i] / scale;
            boxes.Add(new Rect2d(cx - w / 2, cy - h / 2, w, h));
            scores.Add(personScore);
        notPerson:;
        }

        CvDnn.NMSBoxes(boxes, scores, _minConfidence, nmsThreshold: 0.45f, out var keep);
        foreach (var i in keep)
            yield return (new[] { (float)boxes[i].X, (float)boxes[i].Y, (float)boxes[i].Width, (float)boxes[i].Height }, scores[i]);
    }

    /// <summary>Returns the single highest-scoring "sports ball" detection above the ball
    /// threshold, or null. There is at most one ball on the pitch, so we skip NMS and just take
    /// the argmax — this avoids duplicate ball rows and keeps the downstream ball signal a clean
    /// one-per-frame measurement.</summary>
    private (float[] BboxLtwh, float Conf)? DecodeBestBall(
        Tensor<float> output, float scale, float padX, float padY)
    {
        var channels = output.Dimensions[1];
        var anchors = output.Dimensions[2];
        var ballChannel = 4 + CocoSportsBallClass;
        if (ballChannel >= channels) return null; // model without the COCO ball class

        var bestScore = _ballMinConfidence;
        float[]? bestBox = null;
        for (var i = 0; i < anchors; i++)
        {
            var score = output[0, ballChannel, i];
            if (score <= bestScore) continue;
            // Only accept anchors whose best class actually is the ball, mirroring the
            // person path (best-class wins), so a person anchor with weak ball score can't leak.
            var isBest = true;
            for (var c = 4; c < channels; c++)
                if (output[0, c, i] > score) { isBest = false; break; }
            if (!isBest) continue;

            var cx = (output[0, 0, i] - padX) / scale;
            var cy = (output[0, 1, i] - padY) / scale;
            var w = output[0, 2, i] / scale;
            var h = output[0, 3, i] / scale;
            bestScore = score;
            bestBox = new[] { cx - w / 2, cy - h / 2, w, h };
        }

        return bestBox is null ? null : (bestBox, bestScore);
    }

    private static (DenseTensor<float> tensor, float scale, float padX, float padY) Letterbox(Mat image, int size)
    {
        var scale = Math.Min((float)size / image.Width, (float)size / image.Height);
        var newW = (int)Math.Round(image.Width * scale);
        var newH = (int)Math.Round(image.Height * scale);
        var padX = (size - newW) / 2f;
        var padY = (size - newH) / 2f;

        using var resized = image.Resize(new Size(newW, newH));
        using var padded = new Mat();
        Cv2.CopyMakeBorder(resized, padded,
            (int)Math.Floor(padY), (int)Math.Ceiling(padY),
            (int)Math.Floor(padX), (int)Math.Ceiling(padX),
            BorderTypes.Constant, new Scalar(114, 114, 114));

        var tensor = new DenseTensor<float>(new[] { 1, 3, size, size });
        TensorOps.CopyNormalizedChw(padded, tensor.Buffer.Span);
        return (tensor, scale, padX, padY);
    }

    private sealed record PreprocessedImage(DenseTensor<float> Tensor, float Scale, float PadX, float PadY, Series Metadata);

    public void Dispose() => _session.Dispose();
}
