using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Media;
using PitchWise.App.Core.ViewModels;

namespace PitchWise.App.Controls;

/// <summary>
/// Lays events out along the video's duration and lets one be picked.
/// </summary>
/// <remarks>
/// Drawn rather than composed from controls: a match can carry thousands of events, and a
/// Canvas of that many Buttons would cost more to lay out than the video costs to play. Hit
/// testing is a linear scan of the same list, which at these counts is nothing.
/// </remarks>
public sealed class TimelineBar : Control
{
    private const double MarkerWidth = 3;
    private const double HitSlop = 4;

    public static readonly StyledProperty<IEnumerable<PlacedMarker>?> MarkersProperty =
        AvaloniaProperty.Register<TimelineBar, IEnumerable<PlacedMarker>?>(nameof(Markers));

    public static readonly StyledProperty<PlacedMarker?> SelectedProperty =
        AvaloniaProperty.Register<TimelineBar, PlacedMarker?>(nameof(Selected),
            defaultBindingMode: Avalonia.Data.BindingMode.TwoWay);

    /// <summary>Playhead position, 0..1, so the bar can show where the video is.</summary>
    public static readonly StyledProperty<double> PlayheadProperty =
        AvaloniaProperty.Register<TimelineBar, double>(nameof(Playhead));

    static TimelineBar()
    {
        AffectsRender<TimelineBar>(MarkersProperty, SelectedProperty, PlayheadProperty);
    }

    public TimelineBar() => Cursor = new Cursor(StandardCursorType.Hand);

    public IEnumerable<PlacedMarker>? Markers
    {
        get => GetValue(MarkersProperty);
        set => SetValue(MarkersProperty, value);
    }

    public PlacedMarker? Selected
    {
        get => GetValue(SelectedProperty);
        set => SetValue(SelectedProperty, value);
    }

    public double Playhead
    {
        get => GetValue(PlayheadProperty);
        set => SetValue(PlayheadProperty, value);
    }

    /// <summary>Raised when a marker is clicked, so the view can seek to it.</summary>
    public event Action<PlacedMarker>? MarkerPicked;

    public override void Render(DrawingContext context)
    {
        var track = new Rect(0, Bounds.Height / 2 - 2, Bounds.Width, 4);
        context.FillRectangle(new SolidColorBrush(Color.Parse("#2a2a2a")), track);

        if (Markers is not null)
            foreach (var marker in Markers)
            {
                var x = marker.Position * Bounds.Width;
                var selected = ReferenceEquals(marker, Selected);
                var brush = new SolidColorBrush(Color.Parse(marker.Color));
                var height = selected ? Bounds.Height : Bounds.Height * 0.6;
                var y = (Bounds.Height - height) / 2;
                context.FillRectangle(brush, new Rect(x - MarkerWidth / 2, y, MarkerWidth, height));
            }

        var playheadX = Math.Clamp(Playhead, 0, 1) * Bounds.Width;
        context.DrawLine(new Pen(Brushes.White, 1), new Point(playheadX, 0),
            new Point(playheadX, Bounds.Height));
    }

    protected override void OnPointerPressed(PointerPressedEventArgs e)
    {
        base.OnPointerPressed(e);
        if (Markers is null || Bounds.Width <= 0) return;

        var x = e.GetPosition(this).X;
        var nearest = Markers
            .Select(m => (Marker: m, Distance: Math.Abs(m.Position * Bounds.Width - x)))
            .OrderBy(t => t.Distance)
            .FirstOrDefault();

        if (nearest.Marker is not null && nearest.Distance <= MarkerWidth / 2 + HitSlop)
        {
            Selected = nearest.Marker;
            MarkerPicked?.Invoke(nearest.Marker);
        }
    }
}
