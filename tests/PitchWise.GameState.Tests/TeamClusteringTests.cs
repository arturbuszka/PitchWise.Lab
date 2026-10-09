using OpenCvSharp;
using PitchWise.GameState.Team;
using PitchWise.Tracking.Datastruct;

namespace PitchWise.GameState.Tests;

public class TeamClusteringTests
{
    private static readonly Scalar Grass = new(40, 140, 40);   // BGR

    [Fact]
    public void TorsoColor_IgnoresGrassAroundTheShirt()
    {
        using var image = new Mat(200, 200, MatType.CV_8UC3, Grass);
        // red shirt covering the left half of the torso region, grass in the right half
        Cv2.Rectangle(image, new Rect(100, 0, 25, 200), new Scalar(0, 0, 200), -1);

        var color = TorsoColor.Measure(image, new[] { 90f, 10f, 40f, 100f });

        using var red = new Mat(1, 1, MatType.CV_8UC3, new Scalar(0, 0, 200));
        using var redLab = new Mat();
        Cv2.CvtColor(red, redLab, ColorConversionCodes.BGR2Lab);
        var expected = redLab.Get<Vec3b>(0, 0);
        Assert.NotNull(color);
        Assert.Equal(new float[] { expected.Item0, expected.Item1, expected.Item2 }, color);
    }

    [Fact]
    public void TorsoColor_GivesNothingForAllGrassOrTinyBoxes()
    {
        using var image = new Mat(100, 100, MatType.CV_8UC3, Grass);

        Assert.Null(TorsoColor.Measure(image, new[] { 10f, 10f, 40f, 80f }));
        Assert.Null(TorsoColor.Measure(image, new[] { 10f, 10f, 3f, 4f }));
        Assert.Null(TorsoColor.Measure(image, new[] { 99f, 10f, 40f, 80f }));   // cut off at the edge
    }

    [Fact]
    public void Median_MatchesNumpyForEvenCounts()
    {
        Assert.Equal(2.5f, TorsoColor.Median(new List<float> { 4, 1, 3, 2 }));
        Assert.Equal(3f, TorsoColor.Median(new List<float> { 5, 3, 1 }));
    }

    [Fact]
    public void ColorFeature_SplitsTrackletsByShirtColourIgnoringNonPlayers()
    {
        var detections = new DataFrame();
        void Add(long trackId, string role, float[]? color)
        {
            var row = detections.NewRow();
            row["track_id"] = trackId;
            row["role"] = role;
            row["torso_color"] = color;
        }
        // team "white": tracks 1–4, team "red": tracks 5–8; a few detections each with noise
        for (long t = 1; t <= 8; t++)
            for (var k = 0; k < 5; k++)
                Add(t, "player", t <= 4 ? new[] { 240f + k, 128f, 128f - k } : new[] { 130f - k, 190f, 170f + k });
        Add(9, "referee", new[] { 20f, 128f, 128f });
        Add(10, "player", null);

        new TrackletTeamClustering(TeamFeature.Color).Process(detections, new DataFrame());

        int? Cluster(long trackId) => detections.Rows.First(r => r.Get<long>("track_id") == trackId)
            .GetOrDefault<int?>("team_cluster");
        var white = Cluster(1);
        Assert.NotNull(white);
        for (long t = 2; t <= 4; t++) Assert.Equal(white, Cluster(t));
        for (long t = 5; t <= 8; t++) Assert.Equal(1 - white, Cluster(t));
        Assert.Null(Cluster(9));    // referees are not clustered
        Assert.Null(Cluster(10));   // no colour, no cluster
    }
}
