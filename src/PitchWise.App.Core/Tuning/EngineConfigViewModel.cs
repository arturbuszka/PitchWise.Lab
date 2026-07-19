using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using PitchWise.Engine;

namespace PitchWise.App.Core.Tuning;

/// <summary>One tunable engine number, as a slider.</summary>
public sealed partial class TuningKnob : ObservableObject
{
    private readonly Action _onChanged;

    public TuningKnob(string group, string name, string unit, double min, double max, double step,
        double value, string description, Action onChanged)
    {
        Group = group;
        Name = name;
        Unit = unit;
        Minimum = min;
        Maximum = max;
        Step = step;
        Default = value;
        _value = value;
        Description = description;
        _onChanged = onChanged;
    }

    public string Group { get; }
    public string Name { get; }
    public string Unit { get; }
    public double Minimum { get; }
    public double Maximum { get; }
    public double Step { get; }
    public double Default { get; }

    /// <summary>Why this number is what it is — surfaced as the slider's tooltip.</summary>
    public string Description { get; }

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(ValueText))]
    [NotifyPropertyChangedFor(nameof(IsModified))]
    private double _value;

    public string ValueText => $"{Value:0.##} {Unit}".TrimEnd();

    /// <summary>True once moved off the engine's own default, so changes are easy to spot.</summary>
    public bool IsModified => Math.Abs(Value - Default) > 1e-9;

    partial void OnValueChanged(double value) => _onChanged();

    public void Reset() => Value = Default;
}

/// <summary>
/// The tunable half of <see cref="EngineConfig"/>, as sliders.
/// </summary>
/// <remarks>
/// Not every field is here: pitch dimensions are facts about the world rather than parameters to
/// tune, and changing them would invalidate the recorded positions. The descriptions are lifted
/// from EngineConfig's own doc comments, which are unusually good at saying WHY a number is what
/// it is — that reasoning is exactly what someone dragging the slider needs.
/// </remarks>
public sealed class EngineConfigViewModel
{
    private readonly EngineConfig _defaults = new();

    public EngineConfigViewModel(Action onChanged)
    {
        TuningKnob Knob(string group, string name, string unit, double min, double max, double step,
            double value, string description) =>
            new(group, name, unit, min, max, step, value, description, onChanged);

        Knobs =
        [
            Knob("Ball filter", "Min confidence", "", 0, 1, 0.05, _defaults.MinBallConfidence,
                "Ball detections below this are ignored outright."),
            Knob("Ball filter", "Max speed", "m/s", 5, 80, 1, _defaults.MaxBallSpeed,
                "A measurement further than this from the prediction is rejected as spurious and " +
                "the filter coasts. Scaled by dt, so it is really a speed cap: 40 m/s is far above " +
                "any struck ball's ground speed."),
            Knob("Ball filter", "Max coast", "s", 0.2, 6, 0.1, _defaults.MaxCoastSeconds,
                "After this long with no accepted measurement the ball is declared lost and the " +
                "possession and pass rules go quiet. On broadcast footage the gap between ball " +
                "detections has a median of ~0.7 s, so under a second declares it lost half the time."),
            Knob("Ball filter", "Process noise", "m/s²", 1, 40, 1, _defaults.BallProcessNoise,
                "How wrong the constant-velocity assumption is allowed to be — a ball gets kicked, " +
                "so acceleration is not zero."),
            Knob("Ball filter", "Measurement noise", "m", 0.1, 4, 0.1, _defaults.BallMeasurementNoise,
                "Expected error of a projected ball position. Dominated by homography error and the " +
                "Z=0 assumption, not by the detector's box jitter."),

            Knob("Ball carry", "Stick radius", "m", 0, 6, 0.25, _defaults.BallStickRadius,
                "When the ball goes unmeasured while its estimate sits this close to an eligible " +
                "player, assume they are carrying it. A controlled ball is hidden by the carrier's " +
                "own legs, so it goes undetected in exactly the frames where somebody has it; " +
                "without this, coasting drifts away from the carrier and possession never accumulates."),
            Knob("Ball carry", "Stick noise", "m", 0.2, 4, 0.1, _defaults.BallStickNoise,
                "Pseudo-measurement noise while stuck. Loose on purpose: the carrier's feet are a " +
                "guess, and a real detection must be able to pull the filter away."),
            Knob("Ball carry", "Max stick", "s", 0.5, 10, 0.5, _defaults.MaxStickSeconds,
                "How long a ball may stay stuck to a carrier with no real measurement. Bounds the " +
                "damage of sticking to the wrong player."),
            Knob("Ball carry", "Stick confidence", "", 0, 1, 0.05, _defaults.BallStickConfidence,
                "Confidence reported while stuck. Low: rules may trust where the ball is, but " +
                "anything velocity-based should treat a stuck ball as unknown."),

            Knob("Possession", "Control radius", "m", 0.5, 5, 0.1, _defaults.ControlRadius,
                "A player this close to the ball is close enough to control it."),
            Knob("Possession", "Capture dwell", "s", 0.05, 2, 0.05, _defaults.CaptureDwell,
                "How long a challenger must stay nearest before possession transfers. Shorter than " +
                "release grace — that asymmetry is the hysteresis."),
            Knob("Possession", "Release grace", "s", 0.05, 3, 0.05, _defaults.ReleaseGrace,
                "How long the owner keeps possession after the ball leaves their radius. Longer than " +
                "capture dwell, so ownership is sticky and does not flicker."),
            Knob("Possession", "Contested radius", "m", 0.5, 6, 0.25, _defaults.ContestedRadius,
                "Two opponents inside this radius may be duelling for the ball."),
            Knob("Possession", "Contest margin", "m", 0.1, 3, 0.1, _defaults.ContestMargin,
                "…and if their distances to the ball differ by less than this, neither owns it."),

            Knob("Rules", "Pass max flight", "s", 0.5, 10, 0.5, _defaults.PassMaxFlightSeconds,
                "Max time from release to reception for two touches to count as one pass. Longer " +
                "than this and the ball was loose, not passed."),
            Knob("Rules", "Pass min distance", "m", 0.5, 20, 0.5, _defaults.PassMinDistance,
                "The ball must travel this far for a release and receive to be a pass rather than a " +
                "scramble or a failed touch."),
            Knob("Rules", "Event cooldown", "s", 0, 5, 0.25, _defaults.EventCooldownSeconds,
                "Per-type cooldown, so one incident does not emit the same event repeatedly."),
        ];

        Groups = new ObservableCollection<KnobGroup>(
            Knobs.GroupBy(k => k.Group).Select(g => new KnobGroup(g.Key, [.. g])));
    }

    public IReadOnlyList<TuningKnob> Knobs { get; }

    public IReadOnlyList<KnobGroup> Groups { get; }

    public bool IsModified => Knobs.Any(k => k.IsModified);

    /// <summary>Builds the config the sliders currently describe.</summary>
    public EngineConfig ToConfig() => new()
    {
        MinBallConfidence = Value("Min confidence"),
        MaxBallSpeed = Value("Max speed"),
        MaxCoastSeconds = Value("Max coast"),
        BallProcessNoise = Value("Process noise"),
        BallMeasurementNoise = Value("Measurement noise"),
        BallStickRadius = Value("Stick radius"),
        BallStickNoise = Value("Stick noise"),
        MaxStickSeconds = Value("Max stick"),
        BallStickConfidence = Value("Stick confidence"),
        ControlRadius = Value("Control radius"),
        CaptureDwell = Value("Capture dwell"),
        ReleaseGrace = Value("Release grace"),
        ContestedRadius = Value("Contested radius"),
        ContestMargin = Value("Contest margin"),
        PassMaxFlightSeconds = Value("Pass max flight"),
        PassMinDistance = Value("Pass min distance"),
        EventCooldownSeconds = Value("Event cooldown"),
    };

    public void ResetAll()
    {
        foreach (var knob in Knobs) knob.Reset();
    }

    private double Value(string name) => Knobs.First(k => k.Name == name).Value;
}

/// <summary>Knobs that belong together, for a collapsible section.</summary>
public sealed record KnobGroup(string Name, IReadOnlyList<TuningKnob> Knobs);
