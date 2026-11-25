using OpenCvSharp;
using PitchWise.Foosball.Vision.Calibration;
using PitchWise.Foosball.Vision.Detection;

namespace PitchWise.Foosball.Vision.Tracking;

/// <summary>Where one rod is on this frame.</summary>
/// <param name="Offset">How far the rod is slid along y from its centred position, metres.</param>
/// <param name="FigureYs">Table y of every figure on the rod, seen or not: they are fixed to it.</param>
/// <param name="Measured">False when the position is held from an earlier frame.</param>
/// <param name="Confidence">0 when the position is unknown; then <paramref name="Offset"/> means nothing.</param>
public sealed record RodState(RodSpec Rod, double Offset, IReadOnlyList<double> FigureYs, bool Measured, double Confidence);

/// <summary>
/// Follows how far each rod is slid, from the figure boxes. Figures on a rod move as one, so a single
/// offset per rod is fitted to whatever figures are visible. Only y is used for it: kicking swings a
/// figure along x and leaves y alone. Stateful: feed frames in order.
/// </summary>
public sealed class RodTracker
{
    private sealed class Memory
    {
        public double Offset;       // smoothed, reported
        public double LastMeasured; // raw, what the speed gate compares with
        public double MeasuredAt;
        public double Confidence;
        public bool Known;
    }

    private readonly record struct Fit(double Offset, int Count, double Confidence);

    private readonly HeightProjection _projection;
    private readonly TrackerOptions _options;
    private readonly Memory[] _memory;

    public RodTracker(TableCalibration calibration, TrackerOptions options)
    {
        _projection = new HeightProjection(calibration, options);
        _options = options;
        _memory = calibration.Table.Rods.Select(_ => new Memory()).ToArray();
    }

    /// <summary>One state per rod, in the order of <see cref="TableModel.Rods"/>.</summary>
    public IReadOnlyList<RodState> Step(FrameDetections detections, TimeSpan timestamp)
    {
        var rods = _projection.Table.Rods;
        var seen = rods.Select(_ => new List<(double Y, double Confidence)>()).ToArray();
        foreach (var figure in detections.Figures)
        {
            var p = _projection.ToTable(new Point2d(figure.Center.X, figure.Center.Y), _options.RodHeight);
            var nearest = Enumerable.Range(0, rods.Count).MinBy(i => Math.Abs(rods[i].X - p.X));
            if (Math.Abs(rods[nearest].X - p.X) <= _options.RodAssignDistance)
                seen[nearest].Add((p.Y, figure.Confidence));
        }
        return rods.Select((rod, i) => Update(rod, _memory[i], seen[i], timestamp.TotalSeconds)).ToList();
    }

    private RodState Update(RodSpec rod, Memory m, List<(double Y, double Confidence)> seen, double now)
    {
        var age = now - m.MeasuredAt;
        if (m.Known && age >= _options.RodMaxHoldSeconds) m.Known = false;

        var fit = Choose(Fits(rod, seen), m, age);
        if (fit is { } f)
        {
            // Smoothing by elapsed time rather than per frame, so it does not depend on fps or stride.
            var gain = m.Known ? 1 - Math.Exp(-Math.Max(age, 0) / _options.RodSmoothingSeconds) : 1;
            m.Offset += gain * (f.Offset - m.Offset);
            m.LastMeasured = f.Offset;
            m.MeasuredAt = now;
            m.Confidence = f.Confidence;
            m.Known = true;
        }

        var confidence = !m.Known ? 0 : fit is not null ? m.Confidence : m.Confidence * (1 - age / _options.RodMaxHoldSeconds);
        var ys = Enumerable.Range(0, rod.FigureCount).Select(i => Rest(rod, i) + m.Offset).ToList();
        return new RodState(rod, m.Offset, ys, fit is not null, confidence);
    }

    /// <summary>Every offset that explains at least one seen figure, with how many it explains.</summary>
    private List<Fit> Fits(RodSpec rod, List<(double Y, double Confidence)> seen)
    {
        var travel = (_projection.Table.Width - (rod.FigureCount - 1) * rod.FigureSpacing) / 2;
        var fits = new List<Fit>();
        foreach (var (y, _) in seen)
        for (var index = 0; index < rod.FigureCount; index++)
        {
            var candidate = y - Rest(rod, index);
            if (Math.Abs(candidate) > travel) continue; // would push a figure through the wall

            double sum = 0, confidence = 0;
            var count = 0;
            for (var i = 0; i < rod.FigureCount; i++)
            {
                var expected = Rest(rod, i) + candidate;
                var match = seen.MinBy(s => Math.Abs(s.Y - expected));
                if (Math.Abs(match.Y - expected) > _options.RodFitTolerance) continue;
                sum += match.Y - Rest(rod, i);
                confidence += match.Confidence;
                count++;
            }
            fits.Add(new Fit(sum / count, count, confidence / rod.FigureCount));
        }
        return fits;
    }

    private Fit? Choose(List<Fit> fits, Memory m, double age)
    {
        if (m.Known)
        {
            // A rod cannot jump; mistaking one figure for its neighbour would be a jump of a whole spacing.
            var reach = _options.RodMaxSpeed * Math.Max(age, 0) + _options.RodFitTolerance;
            fits = fits.Where(f => Math.Abs(f.Offset - m.LastMeasured) <= reach).ToList();
        }
        if (fits.Count == 0) return null;

        var most = fits.Max(f => f.Count);
        var best = fits.Where(f => f.Count == most).ToList();
        if (m.Known) return best.MinBy(f => Math.Abs(f.Offset - m.LastMeasured));

        // Nothing to compare with: a lone figure could be any of the three, so wait for a clear view.
        return best.Max(f => f.Offset) - best.Min(f => f.Offset) <= _options.RodFitTolerance ? best[0] : null;
    }

    /// <summary>Table y of a figure when its rod is centred.</summary>
    private double Rest(RodSpec rod, int index) =>
        _projection.Table.Width / 2 + (index - (rod.FigureCount - 1) / 2.0) * rod.FigureSpacing;
}
