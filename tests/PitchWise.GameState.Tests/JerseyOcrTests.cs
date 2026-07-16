using OpenCvSharp;
using PitchWise.GameState.Jersey;

namespace PitchWise.GameState.Tests;

public class JerseyOcrTests
{
    [Fact]
    public void ExtractNumbers_KeepsDigitsOnly()
    {
        Assert.Equal("10", MmocrOnnx.ExtractNumbers("N10!"));
        Assert.Equal("7", MmocrOnnx.ExtractNumbers("7"));
        Assert.Null(MmocrOnnx.ExtractNumbers("ADIDAS"));
        Assert.Null(MmocrOnnx.ExtractNumbers(""));
    }

    [Fact]
    public void ExtractJerseyNumbers_PicksHighestConfidenceAndTruncatesToTwoDigits()
    {
        var (number, confidence) = MmocrOnnx.ExtractJerseyNumbersFromOcr(new[]
        {
            ("fly", 0.99),      // no digits — ignored
            ("23", 0.61),
            ("235", 0.87),      // best digits candidate, truncated to "23"
            ("8", 0.42),
        });
        Assert.Equal("23", number);
        Assert.Equal(0.87, confidence);
    }

    [Fact]
    public void ExtractJerseyNumbers_NoDigits_ReturnsNullWithZeroConfidence()
    {
        var (number, confidence) = MmocrOnnx.ExtractJerseyNumbersFromOcr(new[] { ("nike", 0.9) });
        Assert.Null(number);
        Assert.Equal(0, confidence);
    }

    [Fact]
    public void DbPostProcessor_FindsTextRegionInProbabilityMap()
    {
        // synthetic 96x96 prob map with a bright 20x10 text blob at (30, 40)
        var map = new float[96, 96];
        for (var y = 40; y < 50; y++)
            for (var x = 30; x < 50; x++)
                map[y, x] = 0.9f;

        var boxes = DbPostProcessor.ExtractTextBoxes(map);

        var (box, score) = Assert.Single(boxes);
        Assert.True(score > 0.8, $"score {score}");
        // the unclipped box must contain the original blob
        Assert.True(box.Left <= 30 && box.Right >= 49 && box.Top <= 40 && box.Bottom >= 49,
            $"box {box} does not cover the blob");
    }

    [Fact]
    public void DbPostProcessor_EmptyMap_ReturnsNothing()
    {
        Assert.Empty(DbPostProcessor.ExtractTextBoxes(new float[64, 64]));
    }
}
