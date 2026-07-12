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
/// Keeps only the "person" class, like the baseline (cfg.cls = [0]).
/// </summary>
public sealed class YoloOnnxDetector : ImageLevelModule, IDisposable
{
    private readonly InferenceSession _session;
    private readonly float _minConfidence;
    private readonly int _inputSize;

    public override IReadOnlyDictionary<string, string[]> InputColumns { get; } = new Dictionary<string, string[]>();

    public override IReadOnlyDictionary<string, string[]> OutputColumns { get; } = new Dictionary<string, string[]>
    {
        ["detection"] = new[] { "image_id", "video_id", "category_id", "bbox_ltwh", "bbox_conf" },
    };

    public YoloOnnxDetector(string modelPath, float minConfidence = 0.4f, int batchSize = 1, int inputSize = 640)
        : base(batchSize)
    {
        _minConfidence = minConfidence;
        _inputSize = inputSize;
        _session = new InferenceSession(modelPath);
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
                row["category_id"] = 1L; // person
                row["bbox_ltwh"] = bbox;
                row["bbox_conf"] = conf;
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
        padded.GetArray(out Vec3b[] pixels);
        for (var y = 0; y < size; y++)
            for (var x = 0; x < size; x++)
            {
                var px = pixels[y * size + x];
                tensor[0, 0, y, x] = px.Item2 / 255f; // BGR -> RGB
                tensor[0, 1, y, x] = px.Item1 / 255f;
                tensor[0, 2, y, x] = px.Item0 / 255f;
            }
        return (tensor, scale, padX, padY);
    }

    private sealed record PreprocessedImage(DenseTensor<float> Tensor, float Scale, float PadX, float PadY, Series Metadata);

    public void Dispose() => _session.Dispose();
}
