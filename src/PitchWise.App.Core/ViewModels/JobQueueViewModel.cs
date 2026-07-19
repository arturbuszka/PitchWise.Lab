using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using PitchWise.App.Core.Analysis;
using PitchWise.App.Core.Services;
using PitchWise.App.Core.Workspace;

namespace PitchWise.App.Core.ViewModels;

/// <summary>
/// The queue: past jobs discovered from disk, plus the ones this session adds, pumped one at a
/// time.
/// </summary>
/// <remarks>
/// Strictly serial, and not only to be kind to the GPU: BuildPipeline sets the static
/// OnnxSessions.Provider, so two concurrent jobs with different configs would race over it.
/// </remarks>
public sealed partial class JobQueueViewModel : ViewModelBase
{
    private readonly IWorkspaceStore _store;
    private readonly IAnalysisJobRunner _runner;
    private readonly IUiDispatcher _ui;
    private readonly Queue<(JobViewModel Job, AnalysisRequest Request)> _pending = new();

    private CancellationTokenSource? _currentCts;

    public JobQueueViewModel(IWorkspaceStore store, IAnalysisJobRunner runner, IUiDispatcher ui)
    {
        _store = store;
        _runner = runner;
        _ui = ui;
    }

    public ObservableCollection<JobViewModel> Jobs { get; } = [];

    [ObservableProperty] private JobViewModel? _selectedJob;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsIdle))]
    private JobViewModel? _runningJob;

    public bool IsIdle => RunningJob is null;

    /// <summary>Raised when a job finishes successfully, so the shell can offer to open it.</summary>
    public event Action<JobViewModel>? JobCompleted;

    /// <summary>
    /// Loads past jobs. A manifest still marked Running is a crash tombstone — nothing is
    /// running at startup — so it is recorded as failed rather than left to look alive.
    /// </summary>
    public void LoadExisting()
    {
        Jobs.Clear();
        foreach (var (paths, manifest) in _store.DiscoverAll())
        {
            var recovered = manifest.Status == JobStatus.Running
                ? manifest with { Status = JobStatus.Failed, FailureMessage = "interrupted" }
                : manifest;
            if (!ReferenceEquals(recovered, manifest))
                TrySaveManifest(paths, recovered);
            Jobs.Add(new JobViewModel(_store, paths, recovered));
        }
    }

    /// <summary>Adds a request to the queue and starts it when nothing else is running.</summary>
    public JobViewModel Enqueue(AnalysisRequest request)
    {
        var (paths, manifest) = _store.CreateJob(request);
        var job = new JobViewModel(_store, paths, manifest);
        Jobs.Insert(0, job);
        _pending.Enqueue((job, request));
        SelectedJob ??= job;
        PumpAsync();
        return job;
    }

    [RelayCommand]
    private void CancelJob(JobViewModel? job)
    {
        if (job is null) return;
        if (ReferenceEquals(job, RunningJob))
        {
            _currentCts?.Cancel();
            return;
        }

        // not started yet: drop it from the queue without troubling the runner
        if (job.Status == JobStatus.Queued)
        {
            RemoveFromPending(job);
            job.Finish(JobStatus.Cancelled);
            TrySaveManifest(job.Paths, job.Manifest with
            {
                Status = JobStatus.Cancelled,
                CompletedUtc = DateTimeOffset.UtcNow,
            });
        }
    }

    [RelayCommand]
    private void DeleteJob(JobViewModel? job)
    {
        if (job is null || ReferenceEquals(job, RunningJob)) return;
        RemoveFromPending(job);
        _store.DeleteJob(job.Paths);
        Jobs.Remove(job);
        if (ReferenceEquals(SelectedJob, job)) SelectedJob = Jobs.FirstOrDefault();
    }

    /// <summary>
    /// Drops a job's extracted frames, keeping its analysis.
    /// </summary>
    /// <remarks>
    /// Frames are the bulk of a job on disk — a full match at 25 fps is tens of gigabytes — and
    /// they are only needed for playback. Afterwards playback falls back to the source video,
    /// whose seeking is approximate, which the player says so on screen.
    /// </remarks>
    [RelayCommand]
    private void DeleteFrames(JobViewModel? job)
    {
        if (job is null || ReferenceEquals(job, RunningJob) || !job.HasFrames) return;
        try
        {
            _store.DeleteFrames(job.Paths);
            job.RefreshFrames();
        }
        catch (IOException)
        {
            // usually the frames are open in a player; leaving them is the harmless outcome
        }
    }

    /// <summary>
    /// Runs queued jobs one after another. Fire-and-forget by design: the queue drives itself and
    /// nothing awaits it, so every failure is handled inside.
    /// </summary>
    private async void PumpAsync()
    {
        if (RunningJob is not null) return;

        while (_pending.Count > 0)
        {
            var (job, request) = _pending.Dequeue();
            RunningJob = job;
            using var cts = new CancellationTokenSource();
            _currentCts = cts;

            try
            {
                // Dispatched explicitly rather than via Progress<T>: that posts asynchronously,
                // so the last report can still be in flight when the job is already finished,
                // leaving a stale label under a completed job.
                var progress = new DispatchedProgress(_ui, job.Apply);
                await _runner.RunAsync(job.Paths, request, progress, cts.Token);
                job.Finish(JobStatus.Completed);
                JobCompleted?.Invoke(job);
            }
            catch (OperationCanceledException)
            {
                job.Finish(JobStatus.Cancelled);
            }
            catch (Exception)
            {
                // the runner already recorded why in the manifest; a failed job must never take
                // the app down with it
                job.Finish(JobStatus.Failed);
            }
            finally
            {
                _currentCts = null;
                RunningJob = null;
            }
        }
    }

    private void RemoveFromPending(JobViewModel job)
    {
        var remaining = _pending.Where(p => !ReferenceEquals(p.Job, job)).ToList();
        _pending.Clear();
        foreach (var entry in remaining) _pending.Enqueue(entry);
    }

    private void TrySaveManifest(WorkspacePaths paths, AnalysisManifest manifest)
    {
        try
        {
            _store.SaveManifest(paths, manifest);
        }
        catch (IOException)
        {
            // best-effort: a manifest we cannot rewrite is not worth losing the queue over
        }
    }

    /// <summary>
    /// Applies progress on the UI thread, synchronously when already there.
    /// </summary>
    /// <remarks>
    /// The runner throttles to ~10 Hz, so there is no flood to coalesce; what matters is that a
    /// report never overtakes the completion that follows it.
    /// </remarks>
    private sealed class DispatchedProgress(IUiDispatcher ui, Action<AnalysisProgress> apply)
        : IProgress<AnalysisProgress>
    {
        public void Report(AnalysisProgress value)
        {
            if (ui.IsOnUiThread) apply(value);
            else ui.Post(() => apply(value));
        }
    }
}
