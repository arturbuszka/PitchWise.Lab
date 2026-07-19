using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using PitchWise.App.Core.Analysis;
using PitchWise.App.Core.Tuning;
using PitchWise.Tracking.Datastruct;

namespace PitchWise.App.Core.ViewModels;

/// <summary>One raw detection, flattened for a grid.</summary>
public sealed record DetectionRow(
    long TrackId, string Role, string Team, string Jersey,
    string Bbox, string Pitch, string Confidence);

/// <summary>
/// The debug and tuning panel: what the pipeline produced, and what the engine makes of it under
/// thresholds the user can move.
/// </summary>
public sealed partial class DebugViewModel : ViewModelBase, IDisposable
{
    /// <summary>
    /// Slider settle time before re-running. A replay is milliseconds, so this is only here to
    /// stop a drag from queueing a run per pixel — not because the work is expensive.
    /// </summary>
    private static readonly TimeSpan Debounce = TimeSpan.FromMilliseconds(150);

    private readonly AnalysisSession _session;
    private readonly ITuningService _tuning;
    private readonly Action<TuningResult> _onRetuned;
    private readonly Action<int> _seek;

    /// <summary>image_id to frame index — built once, since every tracklet needs the lookup.</summary>
    private readonly Dictionary<long, int> _frameOfImage;

    private CancellationTokenSource? _pending;

    public DebugViewModel(AnalysisSession session, ITuningService tuning,
        Action<TuningResult> onRetuned, Action<int> seek)
    {
        _session = session;
        _tuning = tuning;
        _onRetuned = onRetuned;
        _seek = seek;

        _frameOfImage = BuildFrameIndex(session);
        Config = new EngineConfigViewModel(OnKnobChanged);
        Tracklets = TrackletSummary.From(session.Detections,
            imageId => _frameOfImage.GetValueOrDefault(imageId, -1));
        PipelineDescription = session.Manifest.PipelineDescription ?? "(not recorded)";
        LoadLog();
    }

    private static Dictionary<long, int> BuildFrameIndex(AnalysisSession session)
    {
        var index = new Dictionary<long, int>(session.Frames.FrameCount);
        for (var i = 0; i < session.Frames.FrameCount; i++)
            if (session.Frames.Metadata(i) is { } metadata)
                index[metadata.Index] = i;
        return index;
    }

    public EngineConfigViewModel Config { get; }

    public IReadOnlyList<TrackletSummary> Tracklets { get; }

    /// <summary>Which modules actually ran — `enabled: auto` means the config alone does not say.</summary>
    public string PipelineDescription { get; }

    public ObservableCollection<DetectionRow> FrameDetections { get; } = [];

    [ObservableProperty] private string _log = "";

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(FrameSummary))]
    private int _currentFrame;

    [ObservableProperty] private string _tuningStatus = "";

    [ObservableProperty] private bool _isRetuning;

    public string FrameSummary => $"Frame {CurrentFrame + 1}: {FrameDetections.Count} detections";

    /// <summary>Shows the detections of the frame currently on screen.</summary>
    public void ShowFrame(int frameIndex)
    {
        CurrentFrame = frameIndex;
        FrameDetections.Clear();

        if (_session.Frames.Metadata(frameIndex) is not { } metadata) return;

        foreach (var d in _session.Overlay.DetectionsOf(metadata.Index))
            FrameDetections.Add(new DetectionRow(
                d.GetOrDefault<long>("track_id"),
                d.GetOrDefault<string>("role") ?? "",
                d.GetOrDefault<string>("team") ?? "",
                d.GetOrDefault<string>("jersey_number") ?? "",
                Format(d.GetOrDefault<float[]>("bbox_ltwh")),
                FormatPitch(d),
                $"{d.GetOrDefault<float>("bbox_conf"):F2}"));

        OnPropertyChanged(nameof(FrameSummary));
    }

    [RelayCommand]
    private void GoToTracklet(TrackletSummary? tracklet)
    {
        if (tracklet is not null) _seek(tracklet.FirstFrame);
    }

    [RelayCommand]
    private void ResetConfig() => Config.ResetAll();

    /// <summary>
    /// Re-runs the whole replay on every change, debounced.
    /// </summary>
    /// <remarks>
    /// No incremental machinery and no caching: the dump is one line per frame, and replaying it
    /// is a Kalman update plus a few rule checks — milliseconds even for a full match. Re-running
    /// everything is both simpler and fast enough to feel instant.
    /// </remarks>
    private async void OnKnobChanged()
    {
        _pending?.Cancel();
        _pending?.Dispose();
        var cts = new CancellationTokenSource();
        _pending = cts;

        try
        {
            await Task.Delay(Debounce, cts.Token);

            IsRetuning = true;
            var result = await _tuning.ReplayAsync(_session.WorldStateJsonlPath, Config.ToConfig(),
                cts.Token);
            if (cts.IsCancellationRequested) return;

            _onRetuned(result);
            TuningStatus = $"{result.FramesReplayed} frames replayed in " +
                           $"{result.Duration.TotalMilliseconds:F0} ms";
        }
        catch (OperationCanceledException)
        {
            // superseded by a newer change; the newer one reports
        }
        catch (Exception e)
        {
            TuningStatus = $"Re-tune failed: {e.Message}";
        }
        finally
        {
            if (ReferenceEquals(_pending, cts)) IsRetuning = false;
        }
    }

    private void LoadLog()
    {
        try
        {
            if (File.Exists(_session.Paths.Log))
                Log = File.ReadAllText(_session.Paths.Log);
        }
        catch (IOException e)
        {
            Log = $"(could not read the job log: {e.Message})";
        }
    }

    private static string Format(float[]? bbox) => bbox is { Length: 4 }
        ? $"{bbox[0]:F0},{bbox[1]:F0} {bbox[2]:F0}×{bbox[3]:F0}"
        : "";

    private static string FormatPitch(Series detection)
        => detection["bbox_pitch"] is GameState.Calibration.BboxPitch p
            ? $"{p.XBottomMiddle:F1}, {p.YBottomMiddle:F1} m"
            : "";

    public void Dispose()
    {
        _pending?.Cancel();
        _pending?.Dispose();
    }
}
