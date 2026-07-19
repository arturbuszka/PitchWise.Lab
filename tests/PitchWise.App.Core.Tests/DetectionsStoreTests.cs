using PitchWise.App.Core.Workspace;
using PitchWise.GameState.Calibration;
using PitchWise.Tracking.Datastruct;

namespace PitchWise.App.Core.Tests;

/// <summary>
/// The column/type union covered here is what a real pipeline run actually produces, dumped
/// from the engine rather than guessed — including the two TrackerState.Save silently drops
/// (bbox_pitch, lines) and the rank-2 arrays System.Text.Json cannot handle on its own.
/// </summary>
public class DetectionsStoreTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), $"detstore-test-{Guid.NewGuid():N}");

    public DetectionsStoreTests() => Directory.CreateDirectory(_root);

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    private string Path_(string name) => System.IO.Path.Combine(_root, name);

    [Fact]
    public void Detections_RoundTripPreserveTheExactClrTypesTheVisualizerCastsTo()
    {
        var frame = new DataFrame();
        var row = frame.NewRow();
        row["image_id"] = 42L;
        row["video_id"] = 1L;
        row["category_id"] = 1L;
        row["track_id"] = 7L;
        row["bbox_ltwh"] = new[] { 1.5f, 2.5f, 30f, 60f };
        row["track_bbox_ltwh"] = new[] { 1.6f, 2.6f, 30f, 60f };
        row["bbox_conf"] = 0.87f;
        row["track_bbox_conf"] = 0.9f;
        row["visibility_scores"] = new[] { 0.5f, 0.25f };
        row["role"] = "player";
        row["role_detection"] = "player";
        row["role_confidence"] = 0.75;
        row["team"] = "left";
        row["team_cluster"] = 3;
        row["bbox_pitch"] = new BboxPitch(-10.5, 5.0, -9.5, 5.0, -10.0, 5.0);

        DetectionsStore.WriteDetections(Path_("d.json"), frame);
        var back = DetectionsStore.ReadDetections(Path_("d.json"));

        var r = back.Rows.Single();
        Assert.Equal(row.Index, r.Index);

        // the visualizer casts rather than converts, so the CLR type is the contract
        Assert.Equal(42L, r.Get<long>("image_id"));
        Assert.Equal(7L, r.Get<long>("track_id"));
        Assert.Equal([1.5f, 2.5f, 30f, 60f], r.Get<float[]>("bbox_ltwh"));
        Assert.Equal([1.6f, 2.6f, 30f, 60f], r.Get<float[]>("track_bbox_ltwh"));
        Assert.Equal(0.87f, r.Get<float>("bbox_conf"));
        Assert.Equal([0.5f, 0.25f], r.Get<float[]>("visibility_scores"));
        Assert.Equal("player", r.Get<string>("role"));
        Assert.Equal(0.75, r.Get<double>("role_confidence"));
        Assert.Equal(3, r.Get<int>("team_cluster"));

        // bbox_pitch is one of the two TrackerState.Save drops: no radar without it
        Assert.Equal(new BboxPitch(-10.5, 5.0, -9.5, 5.0, -10.0, 5.0), r.Get<BboxPitch>("bbox_pitch"));
    }

    [Fact]
    public void ImagePreds_RoundTripPreserveLinesKeypointsAndTheRank2Arrays()
    {
        var frame = new DataFrame();
        var row = frame.NewRow();
        row["id"] = row.Index;
        row["video_id"] = 1L;
        row["frame"] = 5L;
        row["nframes"] = 750L;
        row["file_path"] = @"C:\frames\000001.jpg";
        row["lines"] = new Dictionary<string, List<Keypoint>>
        {
            ["Big rect. left top"] = [new Keypoint(10, 20, 0.9), new Keypoint(30, 40, 0.8)],
            ["Circle central"] = [new Keypoint(1, 2, 1.0)],
        };
        row["keypoints"] = new Dictionary<int, Keypoint> { [1] = new Keypoint(5, 6, 0.7) };
        row["homography"] = new[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 }, { 7.0, 8.0, 9.0 } };
        row["parameters"] = new CameraParameters
        {
            PanDegrees = 1, TiltDegrees = 2, RollDegrees = 3,
            XFocalLength = 100, YFocalLength = 200,
            PrincipalPoint = [960, 540],
            PositionMeters = [1, 2, 3],
            RotationMatrix = new[,] { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1.0 } },
        };

        DetectionsStore.WriteImagePreds(Path_("i.json"), frame);
        var back = DetectionsStore.ReadImagePreds(Path_("i.json"));

        var r = back.Rows.Single();
        Assert.Equal(@"C:\frames\000001.jpg", r.Get<string>("file_path"));
        Assert.Equal(5L, r.Get<long>("frame"));

        // lines is the other TrackerState.Save drop: no pitch overlay without it.
        // GameStateVideoRenderer tests it as IReadOnlyDictionary<string, List<Keypoint>>.
        var lines = Assert.IsAssignableFrom<IReadOnlyDictionary<string, List<Keypoint>>>(r["lines"]);
        Assert.Equal(2, lines.Count);
        Assert.Equal([new Keypoint(10, 20, 0.9), new Keypoint(30, 40, 0.8)], lines["Big rect. left top"]);
        Assert.Equal(new Keypoint(1, 2, 1.0), lines["Circle central"].Single());

        var keypoints = r.Get<Dictionary<int, Keypoint>>("keypoints");
        Assert.Equal(new Keypoint(5, 6, 0.7), keypoints[1]);

        // rank-2 arrays: System.Text.Json has no built-in support, hence the custom converter
        var homography = r.Get<double[,]>("homography");
        Assert.Equal(3, homography.GetLength(0));
        Assert.Equal(3, homography.GetLength(1));
        Assert.Equal(6.0, homography[1, 2]);   // row-major order survives
        Assert.Equal(7.0, homography[2, 0]);

        var parameters = r.Get<CameraParameters>("parameters");
        Assert.Equal(100, parameters.XFocalLength);
        Assert.Equal([960, 540], parameters.PrincipalPoint);
        Assert.Equal(1.0, parameters.RotationMatrix[0, 0]);
        Assert.Equal(0.0, parameters.RotationMatrix[0, 1]);
    }

    [Fact]
    public void Write_DropsEmbeddingsBecauseNothingReadsThemBack()
    {
        var frame = new DataFrame();
        var row = frame.NewRow();
        row["track_id"] = 1L;
        row["embeddings"] = Enumerable.Range(0, 512).Select(i => (float)i).ToArray();

        DetectionsStore.WriteDetections(Path_("e.json"), frame);
        var back = DetectionsStore.ReadDetections(Path_("e.json"));

        Assert.False(back.Rows.Single().Has("embeddings"));
        Assert.Equal(1L, back.Rows.Single().Get<long>("track_id"));
    }

    [Fact]
    public void Write_FailsLoudlyOnAnUnknownTypeRatherThanDroppingIt()
    {
        var frame = new DataFrame();
        frame.NewRow()["mystery"] = new Uri("https://example.com");

        var ex = Assert.Throws<NotSupportedException>(
            () => DetectionsStore.WriteDetections(Path_("x.json"), frame));
        Assert.Contains("mystery", ex.Message);
    }

    [Fact]
    public void RoundTrip_PreservesRowIndicesTheAllocationCursorAndOrder()
    {
        var frame = new DataFrame();
        // a non-contiguous index set, as produced by filtering a larger frame
        foreach (var index in new long[] { 5, 2, 9 })
        {
            var row = new Series(index);
            row["track_id"] = index;
            frame.Upsert(row);
        }

        DetectionsStore.WriteDetections(Path_("o.json"), frame);
        var back = DetectionsStore.ReadDetections(Path_("o.json"));

        Assert.Equal([5L, 2L, 9L], back.Index);           // insertion order, not sorted
        Assert.Equal(frame.NextIndex, back.NextIndex);    // appending cannot collide
        Assert.Equal(2L, back[2L].Get<long>("track_id"));
    }

    [Fact]
    public void RoundTrip_HandlesAnEmptyFrameAndNullValues()
    {
        DetectionsStore.WriteDetections(Path_("empty.json"), new DataFrame());
        Assert.True(DetectionsStore.ReadDetections(Path_("empty.json")).IsEmpty);

        var frame = new DataFrame();
        var row = frame.NewRow();
        row["track_id"] = 1L;
        row["team_cluster"] = null;   // real runs leave this null before clustering

        DetectionsStore.WriteDetections(Path_("null.json"), frame);
        var back = DetectionsStore.ReadDetections(Path_("null.json"));

        Assert.False(back.Rows.Single().Has("team_cluster"));
    }
}
