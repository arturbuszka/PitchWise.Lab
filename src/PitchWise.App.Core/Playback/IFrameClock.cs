namespace PitchWise.App.Core.Playback;

/// <summary>
/// Paces playback. Abstracted so the loop's timing can be driven instantly in tests instead of
/// waiting in real time.
/// </summary>
public interface IFrameClock
{
    /// <summary>Waits out the remainder of one frame's budget at the given rate.</summary>
    /// <param name="fps">Source frame rate.</param>
    /// <param name="rate">Playback speed multiplier; 1 is real time.</param>
    void WaitNextFrame(double fps, double rate, CancellationToken ct);
}

/// <summary>
/// Real-time pacing that measures from the previous frame, so decode and draw time counts
/// against the budget instead of adding to it.
/// </summary>
/// <remarks>
/// When a frame overruns its budget the wait is simply skipped: playback drops behind rather
/// than drifting further and further from the source's timing.
/// </remarks>
public sealed class FrameClock : IFrameClock
{
    private long _lastTicks;

    public void WaitNextFrame(double fps, double rate, CancellationToken ct)
    {
        var budget = TimeSpan.FromSeconds(1.0 / (fps * Math.Max(rate, 0.01)));
        var now = DateTime.UtcNow.Ticks;
        var elapsed = TimeSpan.FromTicks(now - _lastTicks);

        if (_lastTicks != 0 && elapsed < budget)
            ct.WaitHandle.WaitOne(budget - elapsed);

        _lastTicks = DateTime.UtcNow.Ticks;
    }
}
