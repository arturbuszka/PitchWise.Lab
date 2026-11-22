using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Dnn;

namespace PitchWise.Foosball.Vision.Detection;

/// <summary>
/// Runs the Ultralytics-exported foosball model (classes 0 ball, 1 figure) in ONNX Runtime.
/// Follows <c>PitchWise.GameState</c>'s person detector, which Foosball may not reference.
/// </summary>
public sealed class YoloOnnxDetector : IFoosballDetector
{
    private const int BallClass = 0;
    private const int FigureClass = 1;
    private const int Classes = 2;

    private readonly DetectorOptions _options;
    private readonly InferenceSession _session;
    private readonly string _inputName;

    public YoloOnnxDetector(DetectorOptions options)
    {
        if (!File.Exists(options.ModelPath))
            throw new FileNotFoundException("Foosball model not found", options.ModelPath);
        _options = options;
        _session = CreateSession(options);
        _inputName = _session.InputMetadata.Keys.Single();
    }

    /// <summary>Which execution provider the session ended up on, for logs.</summary>
    public string Provider { get; private set; } = "CPU";

    public FrameDetections Detect(Mat image)
    {
        var (input, scale, padX, padY) = Letterbox(image, _options.InputSize);
        using var results = _session.Run([NamedOnnxValue.CreateFromTensor(_inputName, input)]);
        var output = (DenseTensor<float>)results[0].AsTensor<float>();
        return Decode(output, scale, padX, padY);
    }

    /// <summary>
    /// Output is [1, 4 + classes, N]: per candidate cx, cy, w, h in input pixels, then one score
    /// per class. A candidate belongs to its highest-scoring class, so a yellow figure that also
    /// scores a little as ball does not count as a ball.
    /// </summary>
    private FrameDetections Decode(DenseTensor<float> output, float scale, float padX, float padY)
    {
        var n = output.Dimensions[2];
        if (output.Dimensions[1] != 4 + Classes)
            throw new InvalidDataException($"Expected {4 + Classes} output channels, got {output.Dimensions[1]}");
        var data = output.Buffer.Span;

        Detection? ball = null;
        var figureBoxes = new List<Rect2d>();
        var figureScores = new List<float>();
        for (var i = 0; i < n; i++)
        {
            var ballScore = data[(4 + BallClass) * n + i];
            var figureScore = data[(4 + FigureClass) * n + i];
            var isBall = ballScore >= figureScore;
            if (isBall ? ballScore < _options.BallMinConfidence : figureScore < _options.FigureMinConfidence)
                continue;

            // Undo the letterbox: remove the padding, then the resize.
            var w = data[2 * n + i] / scale;
            var h = data[3 * n + i] / scale;
            var x = (data[i] - padX) / scale - w / 2;
            var y = (data[n + i] - padY) / scale - h / 2;

            if (isBall)
            {
                if (ball is null || ballScore > ball.Confidence)
                    ball = new Detection(new Rect2f(x, y, w, h), ballScore);
            }
            else
            {
                figureBoxes.Add(new Rect2d(x, y, w, h));
                figureScores.Add(figureScore);
            }
        }

        CvDnn.NMSBoxes(figureBoxes, figureScores, _options.FigureMinConfidence, _options.NmsIou, out var keep);
        var figures = keep
            .Select(k => new Detection(new Rect2f((float)figureBoxes[k].X, (float)figureBoxes[k].Y,
                (float)figureBoxes[k].Width, (float)figureBoxes[k].Height), figureScores[k]))
            .ToList();
        return new FrameDetections(ball, figures);
    }

    /// <summary>
    /// Scales the frame to fit the square input and pads the rest with grey 114, as Ultralytics
    /// does in training. Stretching instead would squash a 16:9 frame and the round ball with it.
    /// </summary>
    private static (DenseTensor<float> Input, float Scale, float PadX, float PadY) Letterbox(Mat image, int size)
    {
        var scale = Math.Min((float)size / image.Width, (float)size / image.Height);
        var newW = (int)Math.Round(image.Width * scale);
        var newH = (int)Math.Round(image.Height * scale);
        var padX = (size - newW) / 2f;
        var padY = (size - newH) / 2f;

        using var resized = image.Resize(new Size(newW, newH));
        using var padded = new Mat();
        Cv2.CopyMakeBorder(resized, padded,
            (int)Math.Floor(padY), (int)Math.Ceiling(padY), (int)Math.Floor(padX), (int)Math.Ceiling(padX),
            BorderTypes.Constant, new Scalar(114, 114, 114));

        // BGR to RGB, 0..1, HWC to CHW in one call.
        using var blob = CvDnn.BlobFromImage(padded, 1.0 / 255, swapRB: true, crop: false);
        var values = new float[3 * size * size];
        System.Runtime.InteropServices.Marshal.Copy(blob.Data, values, 0, values.Length);
        return (new DenseTensor<float>(values, [1, 3, size, size]), scale, padX, padY);
    }

    private InferenceSession CreateSession(DetectorOptions options)
    {
        if (options.UseGpu && OperatingSystem.IsLinux())
        {
            try
            {
                var cuda = new SessionOptions();
                cuda.AppendExecutionProvider_CUDA();
                var session = new InferenceSession(options.ModelPath, cuda);
                Provider = "CUDA";
                return session;
            }
            catch (Exception e) when (e is OnnxRuntimeException or DllNotFoundException or EntryPointNotFoundException)
            {
                // CUDA libraries missing: still correct, only slower.
                Console.Error.WriteLine($"Foosball detector: CUDA unavailable, running on CPU ({e.Message})");
            }
        }
        return new InferenceSession(options.ModelPath);
    }

    public void Dispose() => _session.Dispose();
}
