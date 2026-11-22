using System.Text.Json;
using OpenCvSharp;

namespace PitchWise.Foosball.Vision.Calibration;

/// <summary>
/// Maps between image pixels and table metres (convention of <see cref="TableModel"/>) through a
/// homography of the playfield plane.
///
/// The homography holds only for points on the playfield surface. The ball centre sits about
/// 17 mm above it and the camera about 760 mm, so a ball near the edge lands a few millimetres
/// too far out; that parallax correction belongs to ball tracking (3.6/3.7), not here.
/// </summary>
public sealed class TableCalibration
{
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web) { WriteIndented = true };

    private readonly double[,] _toTable;
    private readonly double[,] _toImage;

    /// <summary>Playfield corners in pixels, in the order (0,0), (L,0), (L,W), (0,W).</summary>
    public IReadOnlyList<Point2f> Corners { get; }

    /// <summary>Frame size the corners were picked on; another resolution would need other corners.</summary>
    public Size ImageSize { get; }

    public TableModel Table { get; }

    private TableCalibration(Point2f[] corners, Size imageSize, TableModel table)
    {
        Corners = corners;
        ImageSize = imageSize;
        Table = table;

        var tableCorners = new[]
        {
            new Point2f(0, 0),
            new Point2f((float)table.Length, 0),
            new Point2f((float)table.Length, (float)table.Width),
            new Point2f(0, (float)table.Width),
        };
        using var toTable = Cv2.GetPerspectiveTransform(corners, tableCorners);
        using var toImage = Cv2.GetPerspectiveTransform(tableCorners, corners);
        _toTable = ToArray(toTable);
        _toImage = ToArray(toImage);
    }

    /// <param name="corners">Pixels of the playfield corners (0,0), (L,0), (L,W), (0,W) of the table model.</param>
    public static TableCalibration FromCorners(Point2f[] corners, Size imageSize, TableModel table)
    {
        if (corners.Length != 4)
            throw new ArgumentException($"Need 4 corners, got {corners.Length}", nameof(corners));
        return new TableCalibration((Point2f[])corners.Clone(), imageSize, table);
    }

    /// <summary>Pixel → table metres (x along the length, y across).</summary>
    public Point2d ToTable(Point2d pixel) => Apply(_toTable, pixel);

    /// <summary>Table metres → pixel.</summary>
    public Point2d ToImage(Point2d table) => Apply(_toImage, table);

    /// <summary>
    /// Writes only the corners and the image size: they are what a person picked, and the
    /// homography follows from them and the table model, so it cannot drift out of sync.
    /// </summary>
    public void Save(string path)
    {
        var file = new CalibrationFile(
            ImageSize.Width, ImageSize.Height,
            Corners.Select(c => new[] { c.X, c.Y }).ToArray());
        File.WriteAllText(path, JsonSerializer.Serialize(file, JsonOptions));
    }

    /// <summary>
    /// Reads a calibration and checks it was made for frames of <paramref name="imageSize"/>,
    /// so a 4K recording is not silently mapped with corners picked on 1080p.
    /// </summary>
    public static TableCalibration Load(string path, Size imageSize, TableModel table)
    {
        var file = JsonSerializer.Deserialize<CalibrationFile>(File.ReadAllText(path), JsonOptions)
            ?? throw new InvalidDataException($"Empty calibration {path}");
        if (file.Corners is not { Length: 4 } || file.Corners.Any(c => c is not { Length: 2 }))
            throw new InvalidDataException($"Calibration {path} needs 4 corners as [x, y]");
        var saved = new Size(file.ImageWidth, file.ImageHeight);
        if (saved != imageSize)
            throw new InvalidDataException(
                $"Calibration {path} was made for {saved.Width}x{saved.Height}, video is {imageSize.Width}x{imageSize.Height}");
        return FromCorners(file.Corners.Select(c => new Point2f(c[0], c[1])).ToArray(), saved, table);
    }

    private static Point2d Apply(double[,] h, Point2d p)
    {
        var w = h[2, 0] * p.X + h[2, 1] * p.Y + h[2, 2];
        return new Point2d(
            (h[0, 0] * p.X + h[0, 1] * p.Y + h[0, 2]) / w,
            (h[1, 0] * p.X + h[1, 1] * p.Y + h[1, 2]) / w);
    }

    private static double[,] ToArray(Mat m)
    {
        var a = new double[3, 3];
        for (var r = 0; r < 3; r++)
            for (var c = 0; c < 3; c++)
                a[r, c] = m.At<double>(r, c);
        return a;
    }

    private sealed record CalibrationFile(int ImageWidth, int ImageHeight, float[][] Corners);
}
