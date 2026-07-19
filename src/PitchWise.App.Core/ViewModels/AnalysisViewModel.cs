using CommunityToolkit.Mvvm.ComponentModel;
using PitchWise.App.Core.Analysis;
using PitchWise.App.Core.Playback;
using PitchWise.App.Core.Services;
using PitchWise.App.Core.Tuning;
using PitchWise.Engine;

namespace PitchWise.App.Core.ViewModels;

/// <summary>
/// One open analysis, and the panels looking at it.
/// </summary>
/// <remarks>
/// Owns the session and every disposable under it. Panels never reach into each other: a click on
/// a timeline marker or a tracklet comes back through <see cref="SeekTo"/> here, and a re-tune
/// fans out from here to the panels it affects. That keeps the coupling one-way and in one place.
/// </remarks>
public sealed partial class AnalysisViewModel : ViewModelBase, IDisposable
{
    private readonly AnalysisSession _session;
    private readonly ITuningService _tuning;

    public AnalysisViewModel(AnalysisSession session, IUiDispatcher ui, IFrameClock clock,
        ITuningService tuning)
    {
        _session = session;
        _tuning = tuning;

        Player = new PlayerViewModel(session, ui, clock);
        Stats = new StatsViewModel(session.Stats);
        Timeline = new TimelineViewModel(SeekTo);
        Debug = new DebugViewModel(session, tuning, ApplyRetune, SeekTo);

        // the debug grid and the timeline's playhead both follow whatever frame is on screen
        Player.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName != nameof(PlayerViewModel.CurrentFrame)) return;
            Debug.ShowFrame(Player.CurrentFrame);
            Timeline.CurrentFrame = Player.CurrentFrame;
        };
    }

    public PlayerViewModel Player { get; }

    public StatsViewModel Stats { get; }

    public TimelineViewModel Timeline { get; }

    public DebugViewModel Debug { get; }

    public string Title => _session.Manifest.DisplayName;

    public AnalysisSession Session => _session;

    [ObservableProperty] private string? _timelineError;

    /// <summary>
    /// Derives the timeline from the recorded observations. Separate from the constructor because
    /// it is a replay, however fast, and the caller decides when to await it.
    /// </summary>
    public async Task LoadTimelineAsync(CancellationToken ct = default)
    {
        try
        {
            var result = await _tuning.ReplayAsync(_session.WorldStateJsonlPath, new EngineConfig(), ct);
            Timeline.Update(result.Markers, _session.Frames.FrameCount);
        }
        catch (OperationCanceledException)
        {
        }
        catch (Exception e)
        {
            // an unreadable dump costs the timeline, not the analysis: video and stats still work
            TimelineError = $"No timeline: {e.Message}";
        }
    }

    /// <summary>The one way any panel moves the video.</summary>
    public void SeekTo(int frameIndex) => Player.SeekTo(frameIndex);

    /// <summary>
    /// Pushes a re-tuned replay into the panels that show derived numbers. The video and the
    /// detections are untouched — those are vision output, and no engine threshold changes them.
    /// </summary>
    private void ApplyRetune(TuningResult result)
    {
        Timeline.Update(result.Markers, _session.Frames.FrameCount);
        Stats.Report = result.Stats;
    }

    public void Dispose()
    {
        Debug.Dispose();
        Player.Dispose();
        _session.Dispose();
    }
}
