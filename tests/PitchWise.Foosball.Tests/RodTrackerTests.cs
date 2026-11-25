using OpenCvSharp;
using PitchWise.Foosball.Vision.Calibration;
using PitchWise.Foosball.Vision.Detection;
using PitchWise.Foosball.Vision.Tracking;

namespace PitchWise.Foosball.Tests;

public class RodTrackerTests
{
    private const double FrameSeconds = 1 / 240.0;
    private const double Tolerance = 0.001;
    private const int Rod = 2; // any rod, figures at indices 0..2

    private static readonly TableModel Table = TableModel.Load(Path.Combine(AppContext.BaseDirectory, "table.json"));
    private static readonly TrackerOptions Options = TrackerOptions.Load(Path.Combine(AppContext.BaseDirectory, "tracker.json"));
    private static readonly TableCalibration Calibration = TableCalibration.FromCorners(
        [new(100, 100), new(1600, 100), new(1600, 977), new(100, 977)], new Size(1920, 1080), Table);
    private static readonly HeightProjection Projection = new(Calibration, Options);

    private readonly RodTracker _tracker = new(Calibration, Options);
    private int _frame;

    [Fact]
    public void HeightMovesOuterRodInwardsAndBack()
    {
        // 0.2925 m from the camera axis at rod height 50 mm and camera 615 mm: 23.8 mm.
        var edge = Calibration.ToImage(new Point2d(0, Table.Width / 2));
        var p = Projection.ToTable(edge, 0.050);
        Assert.Equal(0.0238, p.X, 2e-4);

        var back = Projection.ToImage(p, 0.050);
        Assert.Equal(edge.X, back.X, 0.01);
        Assert.Equal(edge.Y, back.Y, 0.01);
    }

    [Fact]
    public void FindsEveryRodFromAllFigures()
    {
        var figures = Table.Rods.SelectMany((r, i) => Figures(i, 0.01 * i - 0.02, 0, 1, 2)).ToArray();

        var states = Step(figures);

        for (var i = 0; i < Table.Rods.Count; i++)
        {
            Assert.True(states[i].Measured);
            Assert.Equal(0.01 * i - 0.02, states[i].Offset, Tolerance);
            Assert.Equal(Table.Rods[i].FigureCount, states[i].FigureYs.Count);
        }
    }

    [Fact]
    public void HiddenFigureKeepsItsPlace()
    {
        Step(Figures(Rod, 0.03, 0, 1, 2));

        var state = Step(Figures(Rod, 0.03, 0, 2))[Rod];

        Assert.True(state.Measured);
        Assert.Equal(0.03, state.Offset, Tolerance);
        Assert.Equal(Table.Width / 2 + 0.03, state.FigureYs[1], Tolerance);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void OneVisibleFigureIsNotMistakenForItsNeighbour(int visible)
    {
        Step(Figures(Rod, 0.02, 0, 1, 2));

        var state = Step(Figures(Rod, 0.022, visible))[Rod];

        Assert.True(state.Measured);
        Assert.InRange(state.Offset, 0.0195, 0.0225);
    }

    [Fact]
    public void LoneFigureAtTheStartIsAmbiguous()
    {
        // 40 mm off centre fits both the middle figure and, with the rod slid the other way, an end one.
        var state = Step(Figures(Rod, 0.04, 1))[Rod];

        Assert.False(state.Measured);
        Assert.Equal(0, state.Confidence);
    }

    [Fact]
    public void ExtraBoxOnTheRodIsIgnored()
    {
        Step(Figures(Rod, 0.01, 0, 1, 2));

        var state = Step([.. Figures(Rod, 0.01, 0, 1, 2), At(Table.Rods[Rod].X, 0.2)])[Rod];

        Assert.Equal(0.01, state.Offset, Tolerance);
    }

    [Fact]
    public void KickDoesNotMoveTheRod()
    {
        Step(Figures(Rod, 0.01, 0, 1, 2));

        var kicked = Enumerable.Range(0, 3).Select(i => At(Table.Rods[Rod].X + 0.02, RestY(Rod, i) + 0.01)).ToArray();
        var state = Step(kicked)[Rod];

        Assert.True(state.Measured);
        Assert.Equal(0.01, state.Offset, Tolerance);
    }

    [Fact]
    public void HoldsWithFallingConfidenceThenGivesUp()
    {
        var first = Step(Figures(Rod, 0.02, 0, 1, 2))[Rod];
        _frame += 5;
        var held = Step()[Rod];
        Assert.False(held.Measured);
        Assert.Equal(0.02, held.Offset, Tolerance);
        Assert.InRange(held.Confidence, 0.001, first.Confidence - 0.001);

        _frame += (int)(Options.RodMaxHoldSeconds / FrameSeconds) + 1;
        Assert.Equal(0, Step()[Rod].Confidence);
    }

    [Fact]
    public void FollowsAFastSlide()
    {
        RodState state = null!;
        for (var i = 0; i <= 24; i++) // 120 mm in 100 ms
            state = Step(Figures(Rod, -0.06 + 0.12 * i / 24, 0, 1, 2))[Rod];
        Assert.InRange(state.Offset, 0.06 - 0.008, 0.06 + Tolerance);
    }

    [Fact]
    public void LoneFigureAtTheWallCanOnlyBeTheEndOne()
    {
        // Any other figure there would push its neighbour through the wall.
        var state = Step(At(Table.Rods[Rod].X, 0.005))[Rod];

        Assert.True(state.Measured);
        Assert.Equal(0.005, state.FigureYs[0], Tolerance);
    }

    private IReadOnlyList<RodState> Step(params Detection[] figures) =>
        _tracker.Step(new FrameDetections(null, figures), TimeSpan.FromSeconds(_frame++ * FrameSeconds));

    private static Detection[] Figures(int rod, double offset, params int[] indices) =>
        indices.Select(i => At(Table.Rods[rod].X, RestY(rod, i) + offset)).ToArray();

    private static double RestY(int rod, int index) =>
        Table.Width / 2 + (index - (Table.Rods[rod].FigureCount - 1) / 2.0) * Table.Rods[rod].FigureSpacing;

    private static Detection At(double x, double y)
    {
        var p = Projection.ToImage(new Point2d(x, y), Options.RodHeight);
        return new Detection(new Rect2f((float)p.X - 20, (float)p.Y - 30, 40, 60), 0.9f);
    }
}
