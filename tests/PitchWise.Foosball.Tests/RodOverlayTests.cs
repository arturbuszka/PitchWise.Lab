using OpenCvSharp;
using PitchWise.Foosball.Vision.Calibration;
using PitchWise.Foosball.Vision.Tracking;

namespace PitchWise.Foosball.Tests;

public class RodOverlayTests
{
    private static readonly TableModel Table = TableModel.Load(Path.Combine(AppContext.BaseDirectory, "table.json"));
    private static readonly HeightProjection Projection = new(
        TableCalibration.FromCorners([new(100, 100), new(700, 100), new(700, 450), new(100, 450)], new Size(800, 600), Table),
        new TrackerOptions(0.017, 0.615, 0.03));

    [Fact]
    public void DrawsFiguresWhereTheyAreSeenAndMarksHeldOnes()
    {
        using var measured = Draw(measured: true, confidence: 0.9);
        using var held = Draw(measured: false, confidence: 0.5);

        var edge = FigureEdge();
        Assert.NotEqual(default, measured.At<Vec3b>(edge.Y, edge.X));
        Assert.NotEqual(default, held.At<Vec3b>(edge.Y, edge.X));
        Assert.NotEqual(measured.At<Vec3b>(edge.Y, edge.X), held.At<Vec3b>(edge.Y, edge.X));
    }

    [Fact]
    public void UnknownRodHasNoFigures()
    {
        using var image = Draw(measured: false, confidence: 0);

        var edge = FigureEdge();
        Assert.Equal(default, image.At<Vec3b>(edge.Y, edge.X));
    }

    /// <summary>A pixel on the circle of the middle figure, away from the rod line.</summary>
    private static Point FigureEdge()
    {
        var p = Projection.ToImage(new Point2d(Table.Rods[0].X, Table.Width / 2), 0.050);
        return new Point((int)Math.Round(p.X) + 9, (int)Math.Round(p.Y));
    }

    private static Mat Draw(bool measured, double confidence)
    {
        var image = new Mat(600, 800, MatType.CV_8UC3, Scalar.Black);
        new RodOverlay(Projection, 0.050).Draw(image,
            [new RodState(Table.Rods[0], 0, [Table.Width / 2], measured, confidence)]);
        return image;
    }
}
