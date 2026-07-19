using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using PitchWise.App.Core.Stats;
using PitchWise.Engine;

namespace PitchWise.App.Core.ViewModels;

/// <summary>
/// A marker placed on the timeline: the event plus where along the track it belongs.
/// </summary>
/// <param name="Position">0..1 along the video, so the view can lay markers out proportionally.</param>
public sealed record PlacedMarker(TimelineMarker Marker, double Position)
{
    public string Label => Marker.Label;
    public double TimestampSeconds => Marker.TimestampSeconds;

    /// <summary>Team colour, matching the ellipses the overlay draws: A red, B blue.</summary>
    public string Color => Marker.Type == GameEventType.WaywardPass
        ? "#e67e22"                       // turnovers stand out from the team colours
        : Marker.Team switch
        {
            TeamId.A => "#c0392b",
            TeamId.B => "#2980b9",
            _ => "#7f8c8d",
        };
}

/// <summary>The event timeline: passes, turnovers and possession changes, clickable to seek.</summary>
public sealed partial class TimelineViewModel : ViewModelBase
{
    private readonly Action<int> _seek;
    private IReadOnlyList<TimelineMarker> _all = [];

    /// <param name="seek">Called with a frame index when a marker is clicked.</param>
    public TimelineViewModel(Action<int> seek) => _seek = seek;

    public ObservableCollection<PlacedMarker> Markers { get; } = [];

    [ObservableProperty] private PlacedMarker? _selectedMarker;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(SummaryText))]
    [NotifyPropertyChangedFor(nameof(Playhead))]
    private int _totalFrames;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(Playhead))]
    private int _currentFrame;

    /// <summary>Where the video is, 0..1 along the track.</summary>
    public double Playhead =>
        TotalFrames > 1 ? Math.Clamp((double)CurrentFrame / (TotalFrames - 1), 0, 1) : 0;

    // Filters. Possession changes are the noisiest, so they start off.
    [ObservableProperty] private bool _showPasses = true;
    [ObservableProperty] private bool _showTurnovers = true;
    [ObservableProperty] private bool _showPossessionChanges;

    public string SummaryText => $"{Markers.Count} of {_all.Count} events";

    partial void OnShowPassesChanged(bool value) => ApplyFilter();
    partial void OnShowTurnoversChanged(bool value) => ApplyFilter();
    partial void OnShowPossessionChangesChanged(bool value) => ApplyFilter();

    /// <summary>Replaces the events, e.g. after a re-tune.</summary>
    public void Update(IReadOnlyList<TimelineMarker> markers, int totalFrames)
    {
        _all = markers;
        TotalFrames = totalFrames;
        ApplyFilter();
    }

    [RelayCommand]
    private void GoTo(PlacedMarker? marker)
    {
        if (marker is null) return;
        SelectedMarker = marker;
        _seek(marker.Marker.FrameIndex);
    }

    private void ApplyFilter()
    {
        Markers.Clear();
        foreach (var marker in _all.Where(Passes))
            Markers.Add(new PlacedMarker(marker, Position(marker)));
        OnPropertyChanged(nameof(SummaryText));
    }

    /// <summary>Where along the track a marker sits, 0..1.</summary>
    private double Position(TimelineMarker marker) =>
        TotalFrames > 1 ? Math.Clamp((double)marker.FrameIndex / (TotalFrames - 1), 0, 1) : 0;

    private bool Passes(TimelineMarker marker) => marker.Type switch
    {
        GameEventType.Pass => ShowPasses,
        GameEventType.WaywardPass => ShowTurnovers,
        GameEventType.PossessionChange => ShowPossessionChanges,
        _ => true,   // anything the engine grows later shows up rather than vanishing
    };
}
