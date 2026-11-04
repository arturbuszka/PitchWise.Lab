using PitchWise.GameState.Calibration;
using PitchWise.Tracking.Datastruct;

namespace PitchWise.GameState.Tests;

public class StateCacheTests
{
    [Fact]
    public void SaveLoad_RoundTripsEveryColumnTypeThePipelineWrites()
    {
        var detections = new DataFrame();
        var det = detections.NewRow();
        det["image_id"] = 1022000001L;
        det["bbox_ltwh"] = new[] { 1.5f, 2f, 30f, 60f };
        det["bbox_conf"] = 0.9f;
        det["role_confidence"] = 0.75;
        det["role_detection"] = "player";
        det["jersey_number_detection"] = null;
        det["track_id"] = 7;
        det["bbox_pitch"] = new BboxPitch(-1, 2, 1, 2, 0, 2);
        detections.NewRow()["bbox_ltwh"] = new[] { 0f, 0f, 1f, 1f };

        var imagePreds = new DataFrame();
        var image = new Series(1022000001L);
        image["file_path"] = "img1/000001.jpg";
        image["has_labeled_camera"] = false;
        image["homography"] = new[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 }, { 7.0, 8.0, 9.0 } };
        image["keypoints"] = new Dictionary<int, Keypoint> { [3] = new(10, 20, 0.9) };
        image["lines"] = new Dictionary<string, List<Keypoint>> { ["Middle line"] = new() { new(1, 2, 1) } };
        image["parameters"] = new CameraParameters
        {
            PanDegrees = 1, TiltDegrees = 2, RollDegrees = 3, XFocalLength = 4, YFocalLength = 5,
            PrincipalPoint = new[] { 960.0, 540.0 }, PositionMeters = new[] { 0.0, 50.0, -15.0 },
            RotationMatrix = new[,] { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1.0 } },
        };
        imagePreds.Upsert(image);

        var path = Path.Combine(Path.GetTempPath(), $"statecache-{Guid.NewGuid():N}.json.gz");
        try
        {
            StateCache.Save(path, detections, imagePreds, new Dictionary<string, string> { ["StrongSortModule"] = "fp" });
            var loaded = StateCache.Load(path);

            Assert.Equal("fp", loaded.Meta["StrongSortModule"]);
            Assert.Equal(detections.NextIndex, loaded.Detections.NextIndex);
            Assert.Equal(detections.Index, loaded.Detections.Index);
            var d = loaded.Detections[det.Index];
            Assert.Equal(1022000001L, d["image_id"]);
            Assert.Equal(new[] { 1.5f, 2f, 30f, 60f }, d["bbox_ltwh"]);
            Assert.Equal(0.9f, d["bbox_conf"]);
            Assert.Equal(0.75, d["role_confidence"]);
            Assert.Equal(7, d["track_id"]);
            Assert.False(d.Has("jersey_number_detection"));
            Assert.Equal(new BboxPitch(-1, 2, 1, 2, 0, 2), d["bbox_pitch"]);

            var i = loaded.ImagePreds[image.Index];
            Assert.Equal(false, i["has_labeled_camera"]);
            Assert.Equal(image["homography"], i["homography"]);
            Assert.Equal(new Keypoint(10, 20, 0.9), i.Get<Dictionary<int, Keypoint>>("keypoints")[3]);
            Assert.Equal(new Keypoint(1, 2, 1), i.Get<Dictionary<string, List<Keypoint>>>("lines")["Middle line"][0]);
            var parameters = i.Get<CameraParameters>("parameters");
            Assert.Equal(new[] { 0.0, 50.0, -15.0 }, parameters.PositionMeters);
            Assert.Equal(1.0, parameters.RotationMatrix[2, 2]);
        }
        finally
        {
            File.Delete(path);
        }
    }
}
