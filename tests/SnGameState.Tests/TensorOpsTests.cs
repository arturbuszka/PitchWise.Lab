using OpenCvSharp;

namespace SnGameState.Tests;

public class TensorOpsTests
{
    private static Mat RandomImage(int height, int width)
    {
        var image = new Mat(new Size(width, height), MatType.CV_8UC3);
        Cv2.Randu(image, Scalar.All(0), Scalar.All(256));
        return image;
    }

    /// <summary>The per-pixel reference implementation the modules used before vectorization.</summary>
    private static float[] NaiveReference(Mat image, float[]? mean, float[]? std)
    {
        int h = image.Rows, w = image.Cols;
        var tensor = new float[3 * h * w];
        image.GetArray(out Vec3b[] pixels);
        for (var y = 0; y < h; y++)
            for (var x = 0; x < w; x++)
            {
                var px = pixels[y * w + x];
                tensor[0 * h * w + y * w + x] = (px.Item2 / 255f - (mean?[0] ?? 0)) / (std?[0] ?? 1);
                tensor[1 * h * w + y * w + x] = (px.Item1 / 255f - (mean?[1] ?? 0)) / (std?[1] ?? 1);
                tensor[2 * h * w + y * w + x] = (px.Item0 / 255f - (mean?[2] ?? 0)) / (std?[2] ?? 1);
            }
        return tensor;
    }

    [Theory]
    [InlineData(null, null)]                     // plain /255 (YOLO, NBJW)
    [InlineData(new[] { 0.485f, 0.456f, 0.406f }, new[] { 0.229f, 0.224f, 0.225f })] // ImageNet (PRTReid, MMOCR det)
    [InlineData(new[] { 0.5f, 0.5f, 0.5f }, new[] { 0.5f, 0.5f, 0.5f })]             // MMOCR rec
    public void VectorizedConversion_MatchesPerPixelReference(float[]? mean, float[]? std)
    {
        using var image = RandomImage(37, 53); // odd sizes to catch stride bugs
        var expected = NaiveReference(image, mean, std);

        var actual = new float[3 * 37 * 53];
        SnGameState.TensorOps.CopyNormalizedChw(image, actual, mean, std);

        for (var i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-5f,
                $"mismatch at {i}: {expected[i]} vs {actual[i]}");
    }

    [Fact]
    public void TooSmallDestination_Throws()
    {
        using var image = RandomImage(10, 10);
        Assert.Throws<ArgumentException>(() =>
            SnGameState.TensorOps.CopyNormalizedChw(image, new float[10]));
    }
}
