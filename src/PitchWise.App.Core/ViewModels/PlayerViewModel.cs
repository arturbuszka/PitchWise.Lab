using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using PitchWise.App.Core.Analysis;
using PitchWise.App.Core.Playback;
using PitchWise.App.Core.Services;

namespace PitchWise.App.Core.ViewModels;

/// <summary>The video panel: transport controls, scrub bar and overlay toggles.</summary>
public sealed partial class PlayerViewModel : ViewModelBase, IDisposable
{
    private readonly PlaybackLoop _loop;
    private readonly IUiDispatcher _ui;
    private readonly double _fps;

    /// <summary>Guards the scrub bar from seeking to the frame the loop just reported.</summary>
    private bool _updatingFromLoop;

    public PlayerViewModel(AnalysisSession session, IUiDispatcher ui, IFrameClock clock)
    {
        _ui = ui;
        _fps = session.Frames.Fps;
        FrameCount = session.Frames.FrameCount;
        SeekIsExact = session.Frames.SeekIsExact;
        HasGroundTruth = session.Overlay.HasGroundTruth;

        _loop = new PlaybackLoop(session.Frames, session.Overlay, clock, frame =>
        {
            FrameReady?.Invoke(frame);
            _ui.Post(() =>
            {
                _updatingFromLoop = true;
                CurrentFrame = frame.Index;
                _updatingFromLoop = false;
            });
        });
    }

    /// <summary>
    /// Raised on the playback thread once a frame is drawn; the view blits it.
    /// </summary>
    /// <remarks>
    /// An event rather than a bindable property because a Mat is not something to bind to, and
    /// the handler must copy the pixels out synchronously — the loop reuses that Mat for a later
    /// frame as soon as the call returns.
    /// </remarks>
    public event Action<RenderedFrame>? FrameReady;

    /// <summary>
    /// Draws the first frame. Called once the view is listening — doing it in the constructor
    /// would render into a void, since nothing has subscribed yet.
    /// </summary>
    public void ShowFirstFrame() => _loop.SeekTo(0);

    public int FrameCount { get; }

    public bool HasGroundTruth { get; }

    /// <summary>False when playing from a video file, whose seeking lands near — not on — the
    /// requested frame, so the overlay can belong to a neighbouring one.</summary>
    public bool SeekIsExact { get; }

    public bool ShowSeekWarning => !SeekIsExact;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(PositionText))]
    private int _currentFrame;

    [ObservableProperty] private bool _isPlaying;

    [ObservableProperty] private double _rate = 1;

    public IReadOnlyList<double> Rates { get; } = [0.25, 0.5, 1, 2, 4];

    public string PositionText
    {
        get
        {
            var seconds = _fps > 0 ? CurrentFrame / _fps : 0;
            return $"{TimeSpan.FromSeconds(seconds):mm\\:ss\\.ff}  ·  frame {CurrentFrame + 1}/{FrameCount}";
        }
    }

    // Overlay toggles. Each rewrites the options and redraws, so a click is visible at once.
    [ObservableProperty] private bool _showPitch = true;
    [ObservableProperty] private bool _showEllipses = true;
    [ObservableProperty] private bool _showRadar = true;
    [ObservableProperty] private bool _showGroundTruthRadar;
    [ObservableProperty] private bool _showTrackIds = true;
    [ObservableProperty] private bool _showJerseyNumbers = true;
    [ObservableProperty] private bool _showRoles = true;

    partial void OnCurrentFrameChanged(int value)
    {
        if (!_updatingFromLoop) _loop.SeekTo(value);
    }

    partial void OnRateChanged(double value) => _loop.SetRate(value);

    partial void OnShowPitchChanged(bool value) => ApplyOverlay();
    partial void OnShowEllipsesChanged(bool value) => ApplyOverlay();
    partial void OnShowRadarChanged(bool value) => ApplyOverlay();
    partial void OnShowGroundTruthRadarChanged(bool value) => ApplyOverlay();
    partial void OnShowTrackIdsChanged(bool value) => ApplyOverlay();
    partial void OnShowJerseyNumbersChanged(bool value) => ApplyOverlay();
    partial void OnShowRolesChanged(bool value) => ApplyOverlay();

    private void ApplyOverlay() => _loop.SetOverlay(new OverlayOptions(
        ShowPitch, ShowEllipses, ShowRadar, ShowGroundTruthRadar,
        ShowTrackIds, ShowJerseyNumbers, ShowRoles));

    [RelayCommand]
    private void TogglePlay()
    {
        if (_loop.IsPlaying)
        {
            _loop.Pause();
            IsPlaying = false;
        }
        else
        {
            // restarting at the end would otherwise sit there doing nothing
            if (CurrentFrame >= FrameCount - 1) _loop.SeekTo(0);
            _loop.Play();
            IsPlaying = true;
        }
    }

    [RelayCommand]
    private void StepBack() => Step(-1);

    [RelayCommand]
    private void StepForward() => Step(1);

    private void Step(int delta)
    {
        _loop.Step(delta);
        IsPlaying = false;
    }

    /// <summary>Jumps to a frame, used by the timeline and the tracklet inspector.</summary>
    public void SeekTo(int index)
    {
        _loop.Pause();
        IsPlaying = false;
        _loop.SeekTo(index);
    }

    public void Dispose() => _loop.Dispose();
}
