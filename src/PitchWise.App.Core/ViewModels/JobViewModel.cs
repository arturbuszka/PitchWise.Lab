using CommunityToolkit.Mvvm.ComponentModel;
using PitchWise.App.Core.Analysis;
using PitchWise.App.Core.Workspace;

namespace PitchWise.App.Core.ViewModels;

/// <summary>One row of the job queue: a job's status, progress and what can be done to it.</summary>
public sealed partial class JobViewModel : ViewModelBase
{
    private readonly IWorkspaceStore _store;

    public JobViewModel(IWorkspaceStore store, WorkspacePaths paths, AnalysisManifest manifest)
    {
        _store = store;
        Paths = paths;
        _manifest = manifest;
        _status = manifest.Status;
        _detail = DescribeIdleStatus(manifest);
        _fraction = manifest.Status == JobStatus.Completed ? 1 : 0;
        _hasFrames = Directory.Exists(paths.FramesDir);
    }

    public WorkspacePaths Paths { get; }

    public string JobId => Paths.JobId;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(DisplayName))]
    [NotifyPropertyChangedFor(nameof(Subtitle))]
    private AnalysisManifest _manifest;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsRunning))]
    [NotifyPropertyChangedFor(nameof(IsBrowsable))]
    [NotifyPropertyChangedFor(nameof(CanCancel))]
    [NotifyPropertyChangedFor(nameof(CanDeleteFrames))]
    private JobStatus _status;

    [ObservableProperty] private string _detail;
    [ObservableProperty] private double _fraction;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(EtaText))]
    private TimeSpan? _eta;

    /// <summary>Remaining time, rounded to something worth reading. Empty when unknown.</summary>
    public string EtaText => Eta switch
    {
        null => "",
        { TotalSeconds: < 60 } eta => $"~{eta.TotalSeconds:F0}s left",
        { TotalMinutes: < 60 } eta => $"~{eta.TotalMinutes:F0} min left",
        var eta => $"~{eta.Value.TotalHours:F1} h left",
    };

    /// <summary>
    /// Whether this job still has its extracted frames. Frame-sequence jobs never had their own —
    /// they read the originals in place — so there is nothing to reclaim there.
    /// </summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanDeleteFrames))]
    private bool _hasFrames;

    public bool CanDeleteFrames => HasFrames && !IsRunning;

    /// <summary>Re-checks the frames directory after it has been deleted.</summary>
    public void RefreshFrames() => HasFrames = Directory.Exists(Paths.FramesDir);

    public string DisplayName => Manifest.DisplayName;

    /// <summary>Secondary line: what was analysed and how much of it.</summary>
    public string Subtitle => Manifest.FrameCount > 0
        ? $"{Manifest.SourceKind} · {Manifest.FrameCount} frames"
        : $"{Manifest.SourceKind}";

    public bool IsRunning => Status is JobStatus.Running or JobStatus.Queued;

    /// <summary>Only a completed job has results worth opening.</summary>
    public bool IsBrowsable => Status == JobStatus.Completed;

    public bool CanCancel => IsRunning;

    /// <summary>Folds a progress report from the runner into the bound properties.</summary>
    public void Apply(AnalysisProgress progress)
    {
        Status = JobStatus.Running;
        Detail = progress.Detail;
        Fraction = progress.Fraction;
        Eta = progress.Eta;
    }

    /// <summary>Records a terminal state, re-reading the manifest the runner just wrote.</summary>
    public void Finish(JobStatus status)
    {
        Status = status;
        Eta = null;
        Fraction = status == JobStatus.Completed ? 1 : Fraction;
        try
        {
            Manifest = _store.ReadManifest(Paths.JobDirectory);
        }
        catch (Exception e) when (e is IOException or InvalidDataException)
        {
            // the in-memory status is still worth showing even if the manifest is unreadable
        }
        Detail = DescribeIdleStatus(Manifest);
        RefreshFrames();   // a video job only grows a frames directory once it has run
    }

    private static string DescribeIdleStatus(AnalysisManifest manifest) => manifest.Status switch
    {
        JobStatus.Completed => manifest.CompletedUtc is { } done
            ? $"Completed {done.ToLocalTime():g}"
            : "Completed",
        JobStatus.Failed => manifest.FailureMessage ?? "Failed",
        JobStatus.Cancelled => "Cancelled",
        JobStatus.Running => "Interrupted",
        _ => "Queued",
    };
}
