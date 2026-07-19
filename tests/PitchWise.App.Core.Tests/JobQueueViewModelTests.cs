using PitchWise.App.Core.Analysis;
using PitchWise.App.Core.Services;
using PitchWise.App.Core.ViewModels;
using PitchWise.App.Core.Workspace;

namespace PitchWise.App.Core.Tests;

/// <summary>
/// The queue's state machine, driven by a fake runner. No window, no pipeline, no GPU — which is
/// the whole reason the view-models live in a project that cannot reference Avalonia.
/// </summary>
public class JobQueueViewModelTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), $"queue-test-{Guid.NewGuid():N}");
    private readonly WorkspaceStore _store;

    public JobQueueViewModelTests() => _store = new WorkspaceStore(_root);

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    private static AnalysisRequest Request(string name = "clip") => new()
    {
        SourceKind = SourceKind.FrameSequence,
        SourcePath = Path.Combine(Path.GetTempPath(), "pw-tests", name, "img1"),
    };

    /// <summary>A runner the test drives by hand: nothing finishes until it is told to.</summary>
    private sealed class FakeRunner : IAnalysisJobRunner
    {
        private readonly Queue<TaskCompletionSource<AnalysisResult>> _pending = new();

        public List<AnalysisRequest> Started { get; } = [];
        public List<WorkspacePaths> StartedPaths { get; } = [];
        public IProgress<AnalysisProgress>? LastProgress { get; private set; }
        public CancellationToken LastToken { get; private set; }
        public int InFlight { get; private set; }
        public int MaxConcurrent { get; private set; }

        public Task<AnalysisResult> RunAsync(WorkspacePaths paths, AnalysisRequest request,
            IProgress<AnalysisProgress> progress, CancellationToken ct)
        {
            Started.Add(request);
            StartedPaths.Add(paths);
            LastProgress = progress;
            LastToken = ct;
            InFlight++;
            MaxConcurrent = Math.Max(MaxConcurrent, InFlight);

            var tcs = new TaskCompletionSource<AnalysisResult>();
            _pending.Enqueue(tcs);
            ct.Register(() =>
            {
                InFlight--;
                tcs.TrySetCanceled(ct);
            });
            return tcs.Task;
        }

        public void Complete(string jobId, string directory)
        {
            InFlight--;
            _pending.Dequeue().SetResult(new AnalysisResult(jobId, directory));
        }

        public void Fail(Exception e)
        {
            InFlight--;
            _pending.Dequeue().SetException(e);
        }
    }

    /// <summary>Runs everything inline: the test thread stands in for the UI thread.</summary>
    private sealed class InlineDispatcher : IUiDispatcher
    {
        public bool IsOnUiThread => true;
        public void Post(Action action) => action();
        public Task InvokeAsync(Action action)
        {
            action();
            return Task.CompletedTask;
        }
    }

    private (JobQueueViewModel Queue, FakeRunner Runner) Create()
    {
        var runner = new FakeRunner();
        return (new JobQueueViewModel(_store, runner, new InlineDispatcher()), runner);
    }

    [Fact]
    public async Task Enqueue_StartsTheJobAndTracksItsProgress()
    {
        var (queue, runner) = Create();

        var job = queue.Enqueue(Request());
        await Task.Yield();

        Assert.Same(job, queue.RunningJob);
        Assert.False(queue.IsIdle);
        Assert.True(job.IsRunning);
        Assert.False(job.IsBrowsable);
        Assert.Single(runner.Started);

        runner.LastProgress!.Report(new AnalysisProgress(AnalysisStage.Tracking,
            "YoloOnnxDetector (1/9)", 0.25, 10, 40, TimeSpan.FromSeconds(5), TimeSpan.FromSeconds(15)));

        Assert.Equal("YoloOnnxDetector (1/9)", job.Detail);
        Assert.Equal(0.25, job.Fraction);
        Assert.Equal(TimeSpan.FromSeconds(15), job.Eta);

        runner.Complete(job.JobId, job.Paths.JobDirectory);
        await WaitUntil(() => queue.IsIdle);

        Assert.True(job.IsBrowsable);
        Assert.Equal(1, job.Fraction);
        Assert.Null(job.Eta);
    }

    [Fact]
    public async Task Enqueue_RunsTheJobItShowsRatherThanCreatingASecondOne()
    {
        var (queue, runner) = Create();

        var job = queue.Enqueue(Request());
        await Task.Yield();

        // The runner is handed the queued job's own folder. Creating its own would leave the
        // queue watching a folder that stays Queued for ever while the results land elsewhere,
        // and Open would then refuse a job that had in fact finished.
        Assert.Same(job.Paths, Assert.Single(runner.StartedPaths));
        Assert.Single(Directory.GetDirectories(_root));

        runner.Complete(job.JobId, job.Paths.JobDirectory);
        await WaitUntil(() => queue.IsIdle);

        Assert.Single(Directory.GetDirectories(_root));
        Assert.True(job.IsBrowsable);
    }

    [Fact]
    public async Task Enqueue_RunsJobsOneAtATimeInOrder()
    {
        var (queue, runner) = Create();

        var first = queue.Enqueue(Request("first"));
        var second = queue.Enqueue(Request("second"));
        await Task.Yield();

        // the second must wait: BuildPipeline sets static ONNX provider state, so concurrent
        // jobs with different configs would race
        Assert.Same(first, queue.RunningJob);
        Assert.Equal(JobStatus.Queued, second.Status);
        Assert.Single(runner.Started);

        runner.Complete(first.JobId, first.Paths.JobDirectory);
        await WaitUntil(() => ReferenceEquals(queue.RunningJob, second));

        Assert.Equal(2, runner.Started.Count);
        Assert.Equal(1, runner.MaxConcurrent);

        runner.Complete(second.JobId, second.Paths.JobDirectory);
        await WaitUntil(() => queue.IsIdle);
    }

    [Fact]
    public async Task FailedJob_IsRecordedAndTheQueueCarriesOn()
    {
        var (queue, runner) = Create();
        var failing = queue.Enqueue(Request("bad"));
        var next = queue.Enqueue(Request("good"));
        await Task.Yield();

        runner.Fail(new InvalidOperationException("model missing"));
        await WaitUntil(() => ReferenceEquals(queue.RunningJob, next));

        // a failed job must never take the app down, nor stall everything behind it
        Assert.Equal(JobStatus.Failed, failing.Status);
        Assert.False(failing.IsBrowsable);
        Assert.Equal(2, runner.Started.Count);

        runner.Complete(next.JobId, next.Paths.JobDirectory);
        await WaitUntil(() => queue.IsIdle);
        Assert.Equal(JobStatus.Completed, next.Status);
    }

    [Fact]
    public async Task Cancel_StopsTheRunningJobThroughItsToken()
    {
        var (queue, runner) = Create();
        var job = queue.Enqueue(Request());
        await Task.Yield();

        queue.CancelJobCommand.Execute(job);
        await WaitUntil(() => queue.IsIdle);

        Assert.True(runner.LastToken.IsCancellationRequested);
        Assert.Equal(JobStatus.Cancelled, job.Status);
        Assert.False(job.IsBrowsable);
    }

    [Fact]
    public async Task Cancel_OnAQueuedJobDropsItWithoutEverStartingIt()
    {
        var (queue, runner) = Create();
        var running = queue.Enqueue(Request("running"));
        var waiting = queue.Enqueue(Request("waiting"));
        await Task.Yield();

        queue.CancelJobCommand.Execute(waiting);
        runner.Complete(running.JobId, running.Paths.JobDirectory);
        await WaitUntil(() => queue.IsIdle);

        Assert.Equal(JobStatus.Cancelled, waiting.Status);
        Assert.Single(runner.Started);   // never handed to the runner at all
        Assert.Equal(JobStatus.Cancelled, _store.ReadManifest(waiting.Paths.JobDirectory).Status);
    }

    [Fact]
    public async Task JobCompleted_FiresOnlyForJobsThatActuallySucceeded()
    {
        var (queue, runner) = Create();
        var completed = new List<JobViewModel>();
        queue.JobCompleted += completed.Add;

        var ok = queue.Enqueue(Request("ok"));
        var bad = queue.Enqueue(Request("bad"));
        await Task.Yield();

        runner.Complete(ok.JobId, ok.Paths.JobDirectory);
        await WaitUntil(() => runner.Started.Count == 2);
        runner.Fail(new IOException("disk gone"));
        await WaitUntil(() => queue.IsIdle);

        Assert.Same(ok, Assert.Single(completed));
        Assert.Equal(JobStatus.Failed, bad.Status);
    }

    [Fact]
    public void LoadExisting_TurnsACrashTombstoneIntoAFailedJob()
    {
        // a manifest still marked Running at startup can only mean the app died mid-run
        var (paths, manifest) = _store.CreateJob(Request("interrupted"));
        _store.SaveManifest(paths, manifest with { Status = JobStatus.Running });

        var (queue, _) = Create();
        queue.LoadExisting();

        var job = Assert.Single(queue.Jobs);
        Assert.Equal(JobStatus.Failed, job.Status);
        Assert.Equal("interrupted", job.Detail);
        Assert.False(job.IsRunning);
        // and the recovery is persisted, so it does not have to be rediscovered every start
        Assert.Equal(JobStatus.Failed, _store.ReadManifest(paths.JobDirectory).Status);
    }

    [Fact]
    public void LoadExisting_ShowsPastJobsNewestFirst()
    {
        var older = _store.CreateJob(Request("older"));
        _store.SaveManifest(older.Paths, older.Manifest with
        {
            Status = JobStatus.Completed,
            CreatedUtc = DateTimeOffset.UtcNow.AddDays(-1),
            CompletedUtc = DateTimeOffset.UtcNow.AddDays(-1),
        });
        var newer = _store.CreateJob(Request("newer"));
        _store.SaveManifest(newer.Paths, newer.Manifest with { Status = JobStatus.Completed });

        var (queue, _) = Create();
        queue.LoadExisting();

        Assert.Equal(["newer", "older"], queue.Jobs.Select(j => j.DisplayName));
        Assert.All(queue.Jobs, j => Assert.True(j.IsBrowsable));
    }

    [Fact]
    public async Task DeleteJob_RemovesItFromTheQueueAndFromDisk()
    {
        var (queue, runner) = Create();
        var job = queue.Enqueue(Request());
        await Task.Yield();
        runner.Complete(job.JobId, job.Paths.JobDirectory);
        await WaitUntil(() => queue.IsIdle);

        queue.DeleteJobCommand.Execute(job);

        Assert.Empty(queue.Jobs);
        Assert.False(Directory.Exists(job.Paths.JobDirectory));
    }

    [Fact]
    public async Task DeleteJob_RefusesToPullTheRugFromUnderARunningJob()
    {
        var (queue, runner) = Create();
        var job = queue.Enqueue(Request());
        await Task.Yield();

        queue.DeleteJobCommand.Execute(job);

        Assert.Single(queue.Jobs);
        Assert.True(Directory.Exists(job.Paths.JobDirectory));

        runner.Complete(job.JobId, job.Paths.JobDirectory);
        await WaitUntil(() => queue.IsIdle);
    }

    [Fact]
    public void DeleteFrames_ReclaimsTheImagesButKeepsTheAnalysis()
    {
        var (paths, manifest) = _store.CreateJob(Request());
        _store.SaveManifest(paths, manifest with { Status = JobStatus.Completed });
        Directory.CreateDirectory(paths.FramesDir);
        File.WriteAllText(Path.Combine(paths.FramesDir, "000001.jpg"), "pretend frame");
        File.WriteAllText(paths.Stats, "{}");

        var (queue, _) = Create();
        queue.LoadExisting();
        var job = Assert.Single(queue.Jobs);
        Assert.True(job.CanDeleteFrames);

        queue.DeleteFramesCommand.Execute(job);

        Assert.False(Directory.Exists(paths.FramesDir));
        Assert.False(job.CanDeleteFrames);   // nothing left to reclaim
        // the analysis itself survives; only playback is degraded
        Assert.True(File.Exists(paths.Stats));
        Assert.True(job.IsBrowsable);
    }

    [Fact]
    public void DeleteFrames_IsOfferedOnlyWhenThereAreFramesToDelete()
    {
        // a frame-sequence job reads the originals in place, so it never has its own copy
        var (paths, manifest) = _store.CreateJob(Request());
        _store.SaveManifest(paths, manifest with { Status = JobStatus.Completed });

        var (queue, _) = Create();
        queue.LoadExisting();

        Assert.False(Assert.Single(queue.Jobs).CanDeleteFrames);
    }

    /// <summary>Pumps until a condition holds, so tests do not depend on continuation timing.</summary>
    private static async Task WaitUntil(Func<bool> condition)
    {
        for (var i = 0; i < 100 && !condition(); i++)
            await Task.Delay(5);
        Assert.True(condition(), "condition never became true");
    }
}
