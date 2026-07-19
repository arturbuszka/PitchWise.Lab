namespace PitchWise.App.Core.Workspace;

/// <summary>
/// The file layout of one job folder. Names are fixed so a folder is readable without its
/// manifest, and the manifest records which of them a given run actually produced.
/// </summary>
public sealed class WorkspacePaths
{
    public const string ManifestFileName = "manifest.json";
    public const string FramesDirName = "frames";
    public const string DetectionsFileName = "detections.json";
    public const string ImagePredsFileName = "image_preds.json";
    public const string StatsFileName = "stats.json";
    public const string RenderFileName = "render.mp4";
    public const string LogFileName = "job.log";

    /// <summary>
    /// WorldStateJsonl.TryCreate is handed Path.ChangeExtension(stats.json, ".world_state.jsonl"),
    /// exactly as the CLI stats verb does it, which yields this name.
    /// </summary>
    public const string WorldStateJsonlFileName = "stats.world_state.jsonl";

    public WorkspacePaths(string jobDirectory, string jobId)
    {
        JobDirectory = jobDirectory;
        JobId = jobId;
    }

    public string JobId { get; }
    public string JobDirectory { get; }

    public string Manifest => Path.Combine(JobDirectory, ManifestFileName);
    public string FramesDir => Path.Combine(JobDirectory, FramesDirName);
    public string Detections => Path.Combine(JobDirectory, DetectionsFileName);
    public string ImagePreds => Path.Combine(JobDirectory, ImagePredsFileName);
    public string Stats => Path.Combine(JobDirectory, StatsFileName);
    public string WorldStateJsonl => Path.Combine(JobDirectory, WorldStateJsonlFileName);
    public string Render => Path.Combine(JobDirectory, RenderFileName);
    public string Log => Path.Combine(JobDirectory, LogFileName);
}
