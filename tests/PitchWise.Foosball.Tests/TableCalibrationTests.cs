using OpenCvSharp;
using PitchWise.Foosball.Vision.Calibration;

namespace PitchWise.Foosball.Tests;

public class TableCalibrationTests
{
    private const double MaxErrorMetres = 0.002; // 2 mm, the step's acceptance bar

    private static readonly TableModel Table = TableModel.Load(Path.Combine(AppContext.BaseDirectory, "table.json"));
    private static readonly Size Image = new(1920, 1080);

    /// <summary>A trapezoid like a slightly tilted camera: the far side (y = 0) looks shorter.</summary>
    private static readonly Point2f[] Trapezoid =
    [
        new(420, 180), new(1500, 200), new(1640, 940), new(300, 900),
    ];

    [Fact]
    public void CornersMapOntoTableCorners()
    {
        var calibration = TableCalibration.FromCorners(Trapezoid, Image, Table);
        Point2d[] expected = [new(0, 0), new(Table.Length, 0), new(Table.Length, Table.Width), new(0, Table.Width)];

        for (var i = 0; i < 4; i++)
        {
            var mapped = calibration.ToTable(new Point2d(Trapezoid[i].X, Trapezoid[i].Y));
            Assert.InRange(Distance(mapped, expected[i]), 0, MaxErrorMetres);
        }
    }

    [Fact]
    public void DiagonalCrossingMapsToTableCentre()
    {
        // Under perspective the centre is where the diagonals cross, not the mean of the corners.
        var calibration = TableCalibration.FromCorners(Trapezoid, Image, Table);

        var centre = calibration.ToTable(DiagonalCrossing(Trapezoid));

        Assert.InRange(Distance(centre, new Point2d(Table.Length / 2, Table.Width / 2)), 0, MaxErrorMetres);
    }

    [Fact]
    public void ImageToTableAndBackReturnsThePixel()
    {
        var calibration = TableCalibration.FromCorners(Trapezoid, Image, Table);
        var pixel = new Point2d(911.5, 377.25);

        var back = calibration.ToImage(calibration.ToTable(pixel));

        Assert.InRange(Distance(back, pixel), 0, 0.01);
    }

    [Fact]
    public void SaveThenLoadGivesTheSameMapping()
    {
        var path = Path.Combine(Path.GetTempPath(), $"calib-{Guid.NewGuid():N}.json");
        try
        {
            var original = TableCalibration.FromCorners(Trapezoid, Image, Table);
            original.Save(path);

            var loaded = TableCalibration.Load(path, Image, Table);

            Assert.Equal(original.Corners, loaded.Corners);
            Assert.Equal(original.ImageSize, loaded.ImageSize);
            var pixel = new Point2d(1000, 600);
            Assert.Equal(original.ToTable(pixel).X, loaded.ToTable(pixel).X, 1e-9);
            Assert.Equal(original.ToTable(pixel).Y, loaded.ToTable(pixel).Y, 1e-9);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void LoadRejectsAnotherImageSize()
    {
        var path = Path.Combine(Path.GetTempPath(), $"calib-{Guid.NewGuid():N}.json");
        try
        {
            TableCalibration.FromCorners(Trapezoid, Image, Table).Save(path);

            Assert.Throws<InvalidDataException>(() => TableCalibration.Load(path, new Size(3840, 2160), Table));
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static double Distance(Point2d a, Point2d b) => Math.Sqrt((a.X - b.X) * (a.X - b.X) + (a.Y - b.Y) * (a.Y - b.Y));

    private static Point2d DiagonalCrossing(Point2f[] c)
    {
        // Lines c0→c2 and c1→c3; solve c0 + t (c2 - c0) = c1 + s (c3 - c1).
        double ax = c[2].X - c[0].X, ay = c[2].Y - c[0].Y;
        double bx = c[3].X - c[1].X, by = c[3].Y - c[1].Y;
        double dx = c[1].X - c[0].X, dy = c[1].Y - c[0].Y;
        var t = (dx * by - dy * bx) / (ax * by - ay * bx);
        return new Point2d(c[0].X + t * ax, c[0].Y + t * ay);
    }
}
