namespace PitchWise.Foosball.Tests;

/// <summary>Where the foosball model lives. It is kept outside git, like the other models.</summary>
internal static class FoosballModel
{
    public const string RelativePath = "pretrained_models/foosball-yolo11n-v4.onnx";

    public static readonly string? RepoRoot = FindRepoRoot();

    public static string Path => System.IO.Path.Combine(RepoRoot!, RelativePath);

    /// <summary>Walks up from the test binary to the repo, which holds the models.</summary>
    private static string? FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(System.IO.Path.Combine(dir.FullName, "PitchWise.slnx")))
            dir = dir.Parent;
        return dir?.FullName;
    }
}

/// <summary>A fact that skips when the model file is absent.</summary>
internal sealed class RequiresModelFactAttribute : FactAttribute
{
    public RequiresModelFactAttribute()
    {
        if (FoosballModel.RepoRoot is null)
            Skip = "could not locate the repo root from the test binary";
        else if (!File.Exists(FoosballModel.Path))
            Skip = $"missing model: {FoosballModel.RelativePath}";
    }
}
