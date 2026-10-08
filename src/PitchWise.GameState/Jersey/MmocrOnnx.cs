using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using PitchWise.Tracking.Datastruct;
using PitchWise.Tracking.Pipeline;
using PitchWise.Tracking.Utils;

namespace PitchWise.GameState.Jersey;

/// <summary>How to decode the recognizer output logits into characters.</summary>
public enum RecDecodeMode
{
    /// <summary>CTC head (PP-OCR/CRNN/SVTR): blank at class 0, charset shifted by one;
    /// collapse repeats and skip blanks.</summary>
    Ctc,
    /// <summary>Attention head (SAR): read until the end-of-sequence class.</summary>
    Attention,
}

/// <summary>
/// Jersey-number OCR running ONNX exports of the text
/// detector (DBNet) and recognizer (see tools/export/export_mmocr.py). For each person
/// crop, detects text regions, recognizes them, keeps the digits and outputs the
/// highest-confidence jersey number (max two digits), exactly like the Python module.
/// </summary>
public sealed class MmocrOnnx : DetectionLevelModule, IDisposable
{
    private readonly InferenceSession _det;
    private readonly InferenceSession _rec;
    private readonly string[] _charset;
    private readonly RecDecodeMode _decodeMode;
    private readonly int _recHeight;
    private readonly int _recWidth;

    private static readonly float[] ImageNetMean = { 0.485f, 0.456f, 0.406f };
    private static readonly float[] ImageNetStd = { 0.229f, 0.224f, 0.225f };

    public override IReadOnlyDictionary<string, string[]> InputColumns { get; } = new Dictionary<string, string[]>
    {
        ["detection"] = new[] { "bbox_ltwh" },
    };

    public override IReadOnlyDictionary<string, string[]> OutputColumns { get; } = new Dictionary<string, string[]>
    {
        ["detection"] = new[] { "jersey_number_detection", "jersey_number_confidence" },
    };

    /// <param name="charsetPath">Text file with one recognizer dictionary character per line.</param>
    public MmocrOnnx(string detModelPath, string recModelPath, string charsetPath,
        int batchSize = 8, RecDecodeMode decodeMode = RecDecodeMode.Ctc,
        int recHeight = 48, int recWidth = 320)
        : base(batchSize)
    {
        _det = OnnxSessions.Create(detModelPath);
        _rec = OnnxSessions.Create(recModelPath);
        _charset = File.ReadAllLines(charsetPath).ToArray();
        _decodeMode = decodeMode;
        _recHeight = recHeight;
        _recWidth = recWidth;
    }

    public override object Preprocess(Mat image, Series detection, Series metadata)
    {
        var bbox = Coordinates.ClipLtwhToImageDim(detection.Get<float[]>("bbox_ltwh"), image.Width, image.Height);
        var rect = new Rect((int)bbox[0], (int)bbox[1], Math.Max((int)bbox[2], 1), Math.Max((int)bbox[3], 1));
        var crop = new Mat(image, rect).Clone();
        if (crop.Width < 10 || crop.Height < 10)
        {
            crop.Dispose();
            crop = new Mat(new Size(10, 10), MatType.CV_8UC3, Scalar.Black);
        }
        return new Sample(crop, detection.Index);
    }

    public override IReadOnlyList<Series> Process(IReadOnlyList<object> batch, DataFrame detections, IReadOnlyList<Series> metadatas)
    {
        var updated = new List<Series>(batch.Count);
        foreach (var sample in batch.Cast<Sample>())
        {
            using var crop = sample.Crop;
            var texts = RunOcr(crop);
            var (jerseyNumber, confidence) = ExtractJerseyNumbersFromOcr(texts);

            var row = new Series(sample.DetectionIndex);
            row["jersey_number_detection"] = jerseyNumber;
            row["jersey_number_confidence"] = confidence;
            updated.Add(row);
        }
        return updated;
    }

    /// <summary>Detects text regions in the crop and recognizes each one.</summary>
    private List<(string Text, double Score)> RunOcr(Mat crop)
    {
        var results = new List<(string, double)>();
        foreach (var region in DetectTextRegions(crop))
        {
            using var textCrop = new Mat(crop, region);
            var (text, score) = RecognizeText(textCrop);
            if (text.Length > 0)
                results.Add((text, score));
        }
        return results;
    }

    private List<Rect> DetectTextRegions(Mat crop)
    {
        // keep-ratio resize so both sides are multiples of 32 (DBNet requirement)
        var scale = Math.Min(736.0 / Math.Max(crop.Width, crop.Height), 4.0);
        var w = Math.Max(32, (int)Math.Round(crop.Width * scale / 32) * 32);
        var h = Math.Max(32, (int)Math.Round(crop.Height * scale / 32) * 32);
        using var resized = crop.Resize(new Size(w, h));

        var input = ToNormalizedTensor(resized, ImageNetMean, ImageNetStd);
        using var outputs = OnnxSessions.Run(_det, new[] { NamedOnnxValue.CreateFromTensor(_det.InputMetadata.Keys.First(), input) });
        var probMap = outputs.First().AsTensor<float>();

        // output can be [1, 1, H, W] or [1, H, W]
        var dims = probMap.Dimensions.ToArray();
        var (mh, mw) = (dims[^2], dims[^1]);
        var map = new float[mh, mw];
        for (var y = 0; y < mh; y++)
            for (var x = 0; x < mw; x++)
                map[y, x] = dims.Length == 4 ? probMap[0, 0, y, x] : probMap[0, y, x];

        var scaleX = (double)crop.Width / mw;
        var scaleY = (double)crop.Height / mh;
        return DbPostProcessor.ExtractTextBoxes(map)
            .Select(r => new Rect(
                (int)(r.Box.X * scaleX), (int)(r.Box.Y * scaleY),
                (int)(r.Box.Width * scaleX), (int)(r.Box.Height * scaleY)))
            .Select(r => r.Intersect(new Rect(0, 0, crop.Width, crop.Height)))
            .Where(r => r.Width > 0 && r.Height > 0)
            .ToList();
    }

    private (string Text, double Score) RecognizeText(Mat textCrop)
    {
        // keep-ratio resize to the recognizer height, pad to the right
        var scale = (double)_recHeight / textCrop.Height;
        var w = Math.Min(_recWidth, Math.Max(1, (int)Math.Round(textCrop.Width * scale)));
        using var resized = textCrop.Resize(new Size(w, _recHeight));
        using var padded = new Mat();
        Cv2.CopyMakeBorder(resized, padded, 0, 0, 0, _recWidth - w, BorderTypes.Constant, Scalar.Black);

        // mmocr recognizers normalize with mean/std 127.5
        var input = ToNormalizedTensor(padded, new[] { 0.5f, 0.5f, 0.5f }, new[] { 0.5f, 0.5f, 0.5f });
        using var outputs = OnnxSessions.Run(_rec, new[] { NamedOnnxValue.CreateFromTensor(_rec.InputMetadata.Keys.First(), input) });
        var logits = outputs.First().AsTensor<float>(); // [1, T, C]

        int steps = logits.Dimensions[1], classes = logits.Dimensions[2];
        var text = new System.Text.StringBuilder();
        var scores = new List<double>();
        var previous = -1;
        for (var t = 0; t < steps; t++)
        {
            var best = 0;
            var bestLogit = float.MinValue;
            for (var c = 0; c < classes; c++)
                if (logits[0, t, c] > bestLogit) { bestLogit = logits[0, t, c]; best = c; }

            var probability = Probability(logits, t, best, classes);
            if (_decodeMode == RecDecodeMode.Ctc)
            {
                // blank = class 0, characters shifted by one (PP-OCR convention)
                if (best != 0 && best != previous && best - 1 < _charset.Length)
                {
                    text.Append(_charset[best - 1]);
                    scores.Add(probability);
                }
                previous = best;
            }
            else
            {
                // attention: classes beyond the charset are <EOS>/<PAD> — stop there
                if (best >= _charset.Length) break;
                text.Append(_charset[best]);
                scores.Add(probability);
            }
        }
        return (text.ToString(), scores.Count > 0 ? scores.Average() : 0);
    }

    /// <summary>Confidence of the chosen class; applies softmax only when the model
    /// outputs raw logits (PP-OCR models already emit probabilities).</summary>
    private static double Probability(Tensor<float> logits, int t, int target, int classes)
    {
        var sum = 0.0;
        for (var c = 0; c < classes; c++) sum += logits[0, t, c];
        if (Math.Abs(sum - 1.0) < 0.01)
            return logits[0, t, target];

        var max = float.MinValue;
        for (var c = 0; c < classes; c++) max = Math.Max(max, logits[0, t, c]);
        var expSum = 0.0;
        for (var c = 0; c < classes; c++) expSum += Math.Exp(logits[0, t, c] - max);
        return Math.Exp(logits[0, t, target] - max) / expSum;
    }

    private static DenseTensor<float> ToNormalizedTensor(Mat image, float[] mean, float[] std)
    {
        var tensor = new DenseTensor<float>(new[] { 1, 3, image.Height, image.Width });
        TensorOps.CopyNormalizedChw(image, tensor.Buffer.Span, mean, std);
        return tensor;
    }

    /// <summary>extract_numbers: keeps only the digit characters of a recognized text.</summary>
    internal static string? ExtractNumbers(string text)
    {
        var digits = new string(text.Where(char.IsDigit).ToArray());
        return digits.Length > 0 ? digits : null;
    }

    /// <summary>extract_jersey_numbers_from_ocr + choose_best_jersey_number: digit-only
    /// candidates, highest confidence wins, truncated to two digits.</summary>
    internal static (string? JerseyNumber, double Confidence) ExtractJerseyNumbersFromOcr(
        IReadOnlyList<(string Text, double Score)> predictions)
    {
        var candidates = predictions
            .Select(p => (Number: ExtractNumbers(p.Text), p.Score))
            .Where(p => p.Number is not null)
            .ToList();
        if (candidates.Count == 0)
            return (null, 0);
        var (number, score) = candidates.MaxBy(c => c.Score);
        var digits = number!;
        return (digits[..Math.Min(2, digits.Length)], score);
    }

    private sealed record Sample(Mat Crop, long DetectionIndex);

    public void Dispose()
    {
        _det.Dispose();
        _rec.Dispose();
    }
}
