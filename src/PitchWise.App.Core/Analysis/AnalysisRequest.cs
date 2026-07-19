using PitchWise.App.Core.Workspace;

namespace PitchWise.App.Core.Analysis;

/// <summary>What the user asked to analyse, and how.</summary>
/// <remarks>
/// Paths are made absolute on the way in. A job outlives the process that queued it and its
/// artifacts are read back later, so a path that depends on some working directory is a path that
/// stops meaning anything — and the app's working directory is its own bin folder, not the repo
/// root the CLI is run from.
/// </remarks>
public sealed record AnalysisRequest
{
    public required SourceKind SourceKind { get; init; }

    /// <summary>Video file, frames directory or SoccerNet video folder, per <see cref="SourceKind"/>.</summary>
    public required string SourcePath
    {
        get => _sourcePath;
        init => _sourcePath = Absolute(value) ?? value;
    }

    private readonly string _sourcePath = "";

    /// <summary>Name shown in the queue; defaults to the source's file or folder name.</summary>
    public string? DisplayName { get; init; }

    public string? ConfigPath
    {
        get => _configPath;
        init => _configPath = Absolute(value);
    }

    private readonly string? _configPath;

    /// <summary>Where the ONNX models live. Null means "wherever the config says".</summary>
    public string? ModelsDir
    {
        get => _modelsDir;
        init => _modelsDir = Absolute(value);
    }

    private readonly string? _modelsDir;

    private static string? Absolute(string? path) =>
        string.IsNullOrWhiteSpace(path) ? null : Path.GetFullPath(path.Trim());

    /// <summary>
    /// Keep every n-th frame. Pinned to 1 for now: the observation timestamps downstream are
    /// derived from the kept frame index, so a stride greater than 1 makes the world-state
    /// timeline run at the wrong speed. Fixing that means passing fps/stride into
    /// ObservationAdapter.ToObservations, which is a change to make deliberately and test,
    /// not to smuggle in here.
    /// </summary>
    public int FrameStride { get; init; } = 1;

    /// <summary>Stop after this many kept frames (0 = whole source).</summary>
    public int MaxFrames { get; init; }

    /// <summary>Skip into a video before extracting (video sources only).</summary>
    public double StartSeconds { get; init; }

    /// <summary>Frame rate to assume for loose image sequences, which record none.</summary>
    public double SourceFps { get; init; } = 25;

    /// <summary>Also write an overlay mp4 next to the other artifacts.</summary>
    public bool ExportVideo { get; init; }

    /// <summary>
    /// Names a frames directory whose own name says nothing about the material after its parent:
    /// SoccerNet lays frames out as SNGS-021/img1, so the useful name is one level up.
    /// </summary>
    private static readonly string[] GenericFrameFolderNames = ["img1", "img", "frames", "images"];

    /// <summary>The queue label: the display name, or the source's own name.</summary>
    public string ResolveDisplayName()
    {
        if (!string.IsNullOrWhiteSpace(DisplayName)) return DisplayName;

        var full = Path.TrimEndingDirectorySeparator(Path.GetFullPath(SourcePath));
        if (SourceKind == SourceKind.VideoFile)
            return Path.GetFileNameWithoutExtension(full);

        var name = Path.GetFileName(full);
        if (SourceKind == SourceKind.FrameSequence
            && GenericFrameFolderNames.Contains(name, StringComparer.OrdinalIgnoreCase)
            && Path.GetDirectoryName(full) is { } parent
            && Path.GetFileName(parent) is { Length: > 0 } parentName)
            return parentName;

        return name;
    }
}
