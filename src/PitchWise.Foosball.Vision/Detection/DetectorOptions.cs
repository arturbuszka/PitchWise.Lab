using System.Text.Json;

namespace PitchWise.Foosball.Vision.Detection;

/// <summary>Detector settings, read from <c>configs/foosball/detector.json</c>.</summary>
/// <param name="ModelPath">ONNX model; relative paths resolve against the working directory.</param>
/// <param name="InputSize">Square input the model was exported with.</param>
/// <param name="BallMinConfidence">Below this no ball is reported for the frame.</param>
/// <param name="FigureMinConfidence">Figures below this are dropped before NMS.</param>
/// <param name="NmsIou">Overlap above which two figure boxes count as the same figure.</param>
/// <param name="UseGpu">Try CUDA first; falls back to CPU when it is not available.</param>
public sealed record DetectorOptions(
    string ModelPath,
    int InputSize = 1280,
    float BallMinConfidence = 0.25f,
    float FigureMinConfidence = 0.4f,
    float NmsIou = 0.45f,
    bool UseGpu = true)
{
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web)
    {
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true,
    };

    public static DetectorOptions Load(string path) =>
        JsonSerializer.Deserialize<DetectorOptions>(File.ReadAllText(path), JsonOptions)
        ?? throw new InvalidDataException($"Empty detector config {path}");
}
