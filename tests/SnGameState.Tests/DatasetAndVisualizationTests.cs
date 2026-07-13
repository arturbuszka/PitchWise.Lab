using OpenCvSharp;
using SnGameState.Calibration;
using SnGameState.Dataset;
using SnGameState.Visualization;
using TrackLab.Datastruct;

namespace SnGameState.Tests;

public class DatasetAndVisualizationTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), $"sngs-test-{Guid.NewGuid():N}");

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    private string CreateSyntheticDataset()
    {
        var videoDir = Path.Combine(_root, "valid", "SNGS-021");
        var imgDir = Path.Combine(videoDir, "img1");
        Directory.CreateDirectory(imgDir);
        using (var frame = new Mat(new Size(64, 36), MatType.CV_8UC3, new Scalar(20, 90, 20)))
        {
            Cv2.ImWrite(Path.Combine(imgDir, "000001.jpg"), frame);
            Cv2.ImWrite(Path.Combine(imgDir, "000002.jpg"), frame);
        }

        const string labels = """
        {
          "info": {"id": "021", "name": "SNGS-021", "seq_length": "2", "frame_rate": "25", "im_dir": "img1"},
          "images": [
            {"image_id": "3021000001", "file_name": "000001.jpg", "width": "1920", "height": "1080", "is_labeled": true},
            {"image_id": "3021000002", "file_name": "000002.jpg", "width": "1920", "height": "1080", "is_labeled": true}
          ],
          "annotations": [
            {"id": "a1", "image_id": "3021000001", "supercategory": "object", "track_id": 7, "category_id": 1,
             "bbox_image": {"x_center": 100.0, "y_center": 200.0, "w": 40.0, "h": 100.0},
             "bbox_pitch": {"x_bottom_left": -10.5, "y_bottom_left": 5.0, "x_bottom_right": -9.5,
                            "y_bottom_right": 5.0, "x_bottom_middle": -10.0, "y_bottom_middle": 5.0},
             "attributes": {"role": "player", "team": "left", "jersey": "10"}},
            {"id": "a2", "image_id": "3021000001", "supercategory": "object", "track_id": 8, "category_id": 2,
             "bbox_image": {"x_center": 300.0, "y_center": 220.0, "w": 42.0, "h": 104.0},
             "bbox_pitch": null,
             "attributes": {"role": "referee", "team": null, "jersey": null}},
            {"id": "p1", "image_id": "3021000001", "supercategory": "pitch", "category_id": 5, "lines": {}}
          ],
          "categories": []
        }
        """;
        File.WriteAllText(Path.Combine(videoDir, "Labels-GameState.json"), labels);
        return _root;
    }

    [Fact]
    public void Loader_ParsesLabelsGameStateJson()
    {
        var dataset = new SoccerNetGameState(CreateSyntheticDataset());

        var set = dataset.Sets["valid"];
        var video = Assert.Single(set.VideoMetadatas.Rows);
        Assert.Equal(21L, video["id"]);
        Assert.Equal("SNGS-021", video["name"]);
        Assert.Equal(2, set.ImageMetadatas.Count);

        // pitch/camera supercategory rows are not loaded as detections
        Assert.Equal(2, set.DetectionsGt.Count);

        var player = set.DetectionsGt.Rows.First(r => r.Get<long>("track_id") == 7);
        Assert.Equal("player", player["role"]);
        Assert.Equal("left", player["team"]);
        Assert.Equal("10", player["jersey_number"]);
        Assert.Equal("player_left_10", player["category"]);
        // xywh (center) -> ltwh
        Assert.Equal(new[] { 80f, 150f, 40f, 100f }, player.Get<float[]>("bbox_ltwh"));
        Assert.Equal(-10.0, player.Get<BboxPitch>("bbox_pitch").XBottomMiddle);

        var referee = set.DetectionsGt.Rows.First(r => r.Get<long>("track_id") == 8);
        Assert.Equal("referee", referee["category"]);
        Assert.False(referee.Has("bbox_pitch"));

        // images of the video are ordered by frame and linked to files on disk
        var images = set.ImagesOfVideo(21).ToList();
        Assert.True(File.Exists(images[0].Get<string>("file_path")));
    }

    [Fact]
    public void Visualizer_DrawsEllipseRadarAndLines()
    {
        using var image = new Mat(new Size(1920, 1080), MatType.CV_8UC3, Scalar.Black);

        var detections = new DataFrame();
        var det = detections.NewRow();
        det["image_id"] = 1L;
        det["track_id"] = 7L;
        det["bbox_ltwh"] = new[] { 900f, 400f, 60f, 160f };
        det["role"] = "player";
        det["team"] = "left";
        det["jersey_number"] = "10";
        det["bbox_pitch"] = new BboxPitch(-10.5, 5, -9.5, 5, -10, 5);

        var metadata = new Series(1);
        metadata["frame"] = 0L;
        metadata["lines"] = new Dictionary<string, List<Keypoint>>
        {
            ["Middle line"] = new() { new Keypoint(0.5, 0.1, 1), new Keypoint(0.5, 0.9, 1) },
        };

        using var renderer = new GameStateVideoRenderer();
        renderer.RenderFrame(image, metadata, detections);

        Assert.True(Cv2.CountNonZero(image.CvtColor(ColorConversionCodes.BGR2GRAY)) > 5000,
            "expected ellipse, radar and line pixels to be drawn");
    }

    [Fact]
    public void Renderer_WritesPlayableVideo()
    {
        var datasetPath = CreateSyntheticDataset();
        var dataset = new SoccerNetGameState(datasetPath);
        var state = new TrackerState(dataset.Sets["valid"]);

        // pretend the pipeline ran: predictions = ground truth
        var (detections, imagePreds) = state.Load(21);
        foreach (var gt in dataset.Sets["valid"].DetectionsGt.Rows)
            detections.Upsert(gt);
        state.Update(21, detections, imagePreds);

        var output = Path.Combine(_root, "out.mp4");
        using (var renderer = new GameStateVideoRenderer())
            renderer.RenderVideo(state, 21, output);

        Assert.True(File.Exists(output) && new FileInfo(output).Length > 500,
            $"video file missing or too small: {output}");
    }
}
