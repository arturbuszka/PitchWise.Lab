namespace PitchWise.App.Core.Workspace;

/// <summary>Where the frames of an analysis come from.</summary>
public enum SourceKind
{
    /// <summary>A video file; frames are extracted to the job's own frames directory.</summary>
    VideoFile,

    /// <summary>A directory of numbered images, used in place without copying.</summary>
    FrameSequence,

    /// <summary>A SoccerNet game-state video folder, which also carries ground truth.</summary>
    SoccerNetVideo,
}

public enum JobStatus
{
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// <summary>Artifacts a completed job leaves behind, as paths relative to the job folder.</summary>
public sealed record ArtifactSet
{
    public string? Detections { get; init; }
    public string? ImagePreds { get; init; }
    public string? Stats { get; init; }
    public string? WorldStateJsonl { get; init; }
    public string? Render { get; init; }
    public string? Log { get; init; }
}

/// <summary>
/// Everything needed to reopen one analysis, stored beside its artifacts.
/// </summary>
/// <remarks>
/// Self-describing job folders make discovery a directory scan: no database, no index to fall
/// out of sync, and a job survives being copied to another machine. The manifest is written when
/// the job is queued — so a crashed run leaves a Running tombstone rather than a silent gap —
/// and rewritten on every status change.
/// </remarks>
public sealed record AnalysisManifest
{
    /// <summary>Bumped when the on-disk shape changes; a reader rejects what it cannot read.</summary>
    public const int CurrentSchemaVersion = 1;

    public int SchemaVersion { get; init; } = CurrentSchemaVersion;

    public required string JobId { get; init; }
    public required string DisplayName { get; init; }
    public required SourceKind SourceKind { get; init; }
    public required string SourcePath { get; init; }

    public required DateTimeOffset CreatedUtc { get; init; }
    public DateTimeOffset? CompletedUtc { get; init; }
    public required JobStatus Status { get; init; }
    public string? FailureMessage { get; init; }

    public int FrameCount { get; init; }
    public double Fps { get; init; }
    public int ImWidth { get; init; }
    public int ImHeight { get; init; }

    public int FrameStride { get; init; } = 1;
    public int MaxFrames { get; init; }
    public double StartSeconds { get; init; }

    public string? ConfigPath { get; init; }
    public string? ModelsDir { get; init; }

    /// <summary>Pipeline.ToString() of the run — an audit trail of which modules actually ran,
    /// since `enabled: auto` means the config alone does not determine that.</summary>
    public string? PipelineDescription { get; init; }

    public ArtifactSet Artifacts { get; init; } = new();
}
