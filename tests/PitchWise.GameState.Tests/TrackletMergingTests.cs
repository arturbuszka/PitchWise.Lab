using PitchWise.GameState.Calibration;
using PitchWise.GameState.Track;
using PitchWise.Tracking.Datastruct;

namespace PitchWise.GameState.Tests;

public class TrackletMergingTests
{
    private static readonly float[] White = { 240, 128, 128 };
    private static readonly float[] Red = { 130, 190, 170 };

    /// <summary>One video at 25 fps; image ids equal frame numbers.</summary>
    private sealed class Video
    {
        public DataFrame Detections { get; } = new();
        public DataFrame Images { get; } = new();

        public Video()
        {
            for (long f = 0; f < 1000; f++)
                Images.Upsert(new Series(f) { ["frame"] = f });
        }

        /// <summary>Detections of <paramref name="trackId"/> in frames [from, to), standing at (x, y).</summary>
        public void Track(long trackId, long from, long to, double x, double y, float[] embedding,
            float[]? color = null, string role = "player")
        {
            for (var f = from; f < to; f++)
                Detections.Upsert(new Series(Detections.NextIndex)
                {
                    ["image_id"] = f,
                    ["track_id"] = trackId,
                    ["bbox_pitch"] = new BboxPitch(x, y, x, y, x, y),
                    ["embeddings"] = embedding,
                    ["torso_color"] = color ?? White,
                    ["role_detection"] = role,
                });
        }

        public Dictionary<long, long> Run(TrackletMergingOptions? options = null)
        {
            var before = Detections.Rows.ToDictionary(r => r.Index, r => r.Get<long>("track_id"));
            new TrackletMerging(options).Process(Detections, Images);
            // original track id -> merged track id
            return Detections.Rows.GroupBy(r => before[r.Index])
                .ToDictionary(g => g.Key, g => g.Select(r => r.Get<long>("track_id")).Distinct().Single());
        }
    }

    private static float[] Embedding(params float[] v) => v;

    [Fact]
    public void JoinsAFragmentToTheTrackletItContinues()
    {
        var video = new Video();
        video.Track(1, 0, 50, 0, 0, Embedding(1, 0, 0));
        video.Track(2, 100, 150, 5, 0, Embedding(0.98f, 0.2f, 0));   // 2 s later, 5 m away
        video.Track(3, 0, 150, 20, 20, Embedding(0, 1, 0));           // someone else, all along

        var ids = video.Run();

        Assert.Equal(1, ids[1]);
        Assert.Equal(1, ids[2]);
        Assert.Equal(3, ids[3]);
    }

    [Fact]
    public void ChainsFragmentsUnderTheFirstId()
    {
        var video = new Video();
        video.Track(4, 0, 30, 0, 0, Embedding(1, 0, 0));
        video.Track(7, 40, 70, 1, 0, Embedding(1, 0, 0));
        video.Track(9, 80, 110, 2, 0, Embedding(1, 0, 0));

        var ids = video.Run();

        Assert.All(ids.Values, id => Assert.Equal(4, id));
    }

    [Fact]
    public void NeverJoinsTrackletsOverlappingInTime()
    {
        var video = new Video();
        video.Track(1, 0, 50, 0, 0, Embedding(1, 0, 0));
        video.Track(2, 49, 100, 0, 0, Embedding(1, 0, 0));

        Assert.Equal(2, video.Run()[2]);
    }

    [Theory]
    [InlineData(25, 30.0, false)]   // 1 s gap: reach 3 + 8 m
    [InlineData(25, 10.0, true)]
    [InlineData(300, 5.0, false)]   // 12 s: beyond the 10 s gap limit
    public void RespectsGapAndPitchReach(int gapFrames, double distance, bool joined)
    {
        var video = new Video();
        video.Track(1, 0, 50, 0, 0, Embedding(1, 0, 0));
        video.Track(2, 49 + gapFrames + 1, 49 + gapFrames + 51, distance, 0, Embedding(1, 0, 0));

        Assert.Equal(joined ? 1 : 2, video.Run(new TrackletMergingOptions { NoLinkCost = 1 })[2]);
    }

    [Fact]
    public void UnknownPitchPositionDoesNotBlockALink()
    {
        var video = new Video();
        video.Track(1, 0, 50, 0, 0, Embedding(1, 0, 0));
        video.Track(2, 60, 100, 80, 0, Embedding(1, 0, 0));
        foreach (var row in video.Detections.Rows.Where(r => r.Get<long>("track_id") == 2))
            row["bbox_pitch"] = null;

        Assert.Equal(1, video.Run()[2]);
    }

    [Fact]
    public void RefusesDissimilarAppearance()
    {
        var video = new Video();
        video.Track(1, 0, 50, 0, 0, Embedding(1, 0, 0));
        video.Track(2, 60, 100, 0, 0, Embedding(0.6f, 0.8f, 0));   // cosine distance 0.4

        Assert.Equal(2, video.Run(new TrackletMergingOptions { NoLinkCost = 1 })[2]);
    }

    [Fact]
    public void TeamConstraintAppliesOnlyToTrackletsWithReliableColour()
    {
        var video = new Video();
        // two clear teams so the colour clustering has something to split
        video.Track(10, 0, 200, 30, 30, Embedding(0, 0, 1), White);
        video.Track(11, 0, 200, -30, 30, Embedding(0, 0, 1), Red);
        video.Track(1, 0, 50, 0, 0, Embedding(1, 0, 0), White);
        video.Track(2, 60, 100, 0, 0, Embedding(1, 0, 0), Red);       // 40 detections: other team
        video.Track(3, 300, 350, 0, -20, Embedding(0, 1, 0), White);
        video.Track(4, 360, 365, 0, -20, Embedding(0, 1, 0), Red);    // 5 detections: colour not trusted

        var ids = video.Run();

        Assert.Equal(2, ids[2]);
        Assert.Equal(3, ids[4]);
    }

    [Fact]
    public void GoalkeeperIsNotJoinedToAPlayer()
    {
        var video = new Video();
        video.Track(1, 0, 50, 0, 0, Embedding(1, 0, 0), role: "goalkeeper");
        video.Track(2, 60, 100, 0, 0, Embedding(1, 0, 0), role: "player");

        Assert.Equal(2, video.Run()[2]);
    }

    [Fact]
    public void EachEndTakesItsCheapestStartOneToOne()
    {
        var video = new Video();
        video.Track(1, 0, 50, 0, 0, Embedding(1, 0, 0));
        video.Track(2, 0, 50, 4, 0, Embedding(0, 1, 0));
        video.Track(3, 60, 100, 2, 0, Embedding(0.1f, 1, 0));   // looks like 2
        video.Track(4, 60, 100, 2, 0, Embedding(1, 0.1f, 0));   // looks like 1

        var ids = video.Run();

        Assert.Equal(2, ids[3]);
        Assert.Equal(1, ids[4]);
    }

    [Fact]
    public void LinksCostingMoreThanNoLinkAreNotMade()
    {
        var video = new Video();
        video.Track(1, 0, 50, 0, 0, Embedding(1, 0, 0));
        video.Track(2, 60, 100, 0, 0, Embedding(0.8f, 0.6f, 0));   // cosine distance 0.2

        Assert.Equal(2, video.Run(new TrackletMergingOptions { NoLinkCost = 0.1 })[2]);
        Assert.Equal(1, video.Run(new TrackletMergingOptions { NoLinkCost = 0.3 })[2]);
    }
}
