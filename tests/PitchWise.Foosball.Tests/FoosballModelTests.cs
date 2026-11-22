using Microsoft.ML.OnnxRuntime;

namespace PitchWise.Foosball.Tests;

/// <summary>
/// Checks the exported foosball detector loads in ONNX Runtime with the shapes the detector (step 3.2)
/// will rely on. The model lives outside git, so the test skips when it is absent.
/// </summary>
public class FoosballModelTests
{
    private const string ModelFile = "pretrained_models/foosball-yolo11n-v4.onnx";
    private const int ImageSize = 1280;
    private const int Classes = 2; // ball, figure

    private static readonly string? RepoRoot = FindRepoRoot();

    /// <summary>Walks up from the test binary to the repo, which holds the models.</summary>
    private static string? FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "PitchWise.slnx")))
            dir = dir.Parent;
        return dir?.FullName;
    }

    private sealed class RequiresModelFactAttribute : FactAttribute
    {
        public RequiresModelFactAttribute()
        {
            if (RepoRoot is null)
                Skip = "could not locate the repo root from the test binary";
            else if (!File.Exists(Path.Combine(RepoRoot, ModelFile)))
                Skip = $"missing model: {ModelFile}";
        }
    }

    [RequiresModelFact]
    public void ModelLoadsWithExpectedShapes()
    {
        using var session = new InferenceSession(Path.Combine(RepoRoot!, ModelFile));

        var input = Assert.Single(session.InputMetadata).Value;
        Assert.Equal([1, 3, ImageSize, ImageSize], input.Dimensions);

        // YOLO11 head: per candidate box 4 coordinates + one score per class, over the
        // 160², 80², 40² grids at strides 8, 16, 32 of a 1280 input.
        var anchors = (ImageSize / 8) * (ImageSize / 8) + (ImageSize / 16) * (ImageSize / 16)
                      + (ImageSize / 32) * (ImageSize / 32);
        var output = Assert.Single(session.OutputMetadata).Value;
        Assert.Equal([1, 4 + Classes, anchors], output.Dimensions);
    }
}
