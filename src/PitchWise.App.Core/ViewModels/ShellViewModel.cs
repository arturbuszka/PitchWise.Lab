using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using PitchWise.App.Core.Analysis;
using PitchWise.App.Core.Playback;
using PitchWise.App.Core.Services;
using PitchWise.App.Core.Tuning;
using PitchWise.App.Core.Workspace;

namespace PitchWise.App.Core.ViewModels;

/// <summary>
/// The main window: the job queue down one side, and either the new-analysis form or an open
/// analysis on the other.
/// </summary>
public sealed partial class ShellViewModel : ViewModelBase, IDisposable
{
    private readonly IWorkspaceStore _store;
    private readonly IFrameSourceFactory _frameSources;
    private readonly IUiDispatcher _ui;
    private readonly Func<IFrameClock> _clockFactory;
    private readonly ITuningService _tuning;

    /// <param name="clockFactory">
    /// A factory, not a clock: each playback loop paces from its own last-frame timestamp, so
    /// every analysis opened needs a fresh one.
    /// </param>
    public ShellViewModel(JobQueueViewModel queue, NewAnalysisViewModel newAnalysis,
        IWorkspaceStore store, IFrameSourceFactory frameSources, IUiDispatcher ui,
        Func<IFrameClock> clockFactory, ITuningService tuning)
    {
        Queue = queue;
        NewAnalysis = newAnalysis;
        _store = store;
        _frameSources = frameSources;
        _ui = ui;
        _clockFactory = clockFactory;
        _tuning = tuning;

        Queue.LoadExisting();
    }

    public JobQueueViewModel Queue { get; }

    public NewAnalysisViewModel NewAnalysis { get; }

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasOpenAnalysis))]
    private AnalysisViewModel? _analysis;

    [ObservableProperty] private string _statusMessage = "Ready";

    [ObservableProperty] private bool _isOpening;

    public bool HasOpenAnalysis => Analysis is not null;

    /// <summary>Loads a finished job's results and shows them.</summary>
    [RelayCommand]
    public async Task OpenJobAsync(JobViewModel? job)
    {
        if (job is null || !job.IsBrowsable || IsOpening) return;

        IsOpening = true;
        StatusMessage = $"Opening {job.DisplayName}…";
        try
        {
            // reading a run's detections is hundreds of milliseconds of json; off the UI thread
            var session = await Task.Run(() =>
                AnalysisSession.Open(_store, job.Paths, _frameSources));

            Analysis?.Dispose();
            var analysis = new AnalysisViewModel(session, _ui, _clockFactory(), _tuning);
            Analysis = analysis;
            StatusMessage = session.Frames.SeekIsExact
                ? $"{job.DisplayName} — {session.Frames.FrameCount} frames"
                : $"{job.DisplayName} — frames deleted, seeking from the video is approximate";

            // milliseconds, but it reads the dump, so it does not belong in the constructor
            await analysis.LoadTimelineAsync();
        }
        catch (Exception e)
        {
            // an unopenable job is worth a message, not a crash
            StatusMessage = $"Could not open {job.DisplayName}: {e.Message}";
        }
        finally
        {
            IsOpening = false;
        }
    }

    [RelayCommand]
    private void CloseAnalysis()
    {
        Analysis?.Dispose();
        Analysis = null;
        StatusMessage = "Ready";
    }

    public void Dispose() => Analysis?.Dispose();
}
