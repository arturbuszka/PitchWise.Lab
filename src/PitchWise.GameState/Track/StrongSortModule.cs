using OpenCvSharp;
using PitchWise.GameState.Detection;
using StrongSort;
using PitchWise.Tracking.Datastruct;
using PitchWise.Tracking.Pipeline;

namespace PitchWise.GameState.Track;

/// <summary>
/// StrongSORT tracking adapter: feeds each frame's detections
/// (with ReID embeddings when available) into <see cref="StrongSortTracker"/> and writes
/// track_id / track_bbox_ltwh / track_bbox_conf back onto the detection rows.
/// </summary>
public sealed class StrongSortModule : ImageLevelModule
{
    private readonly StrongSortOptions _options;
    private StrongSortTracker _tracker;
    private readonly CameraMotion _cameraMotion;

    public override IReadOnlyDictionary<string, string[]> InputColumns { get; } = new Dictionary<string, string[]>
    {
        ["detection"] = new[] { "bbox_ltwh", "bbox_conf", "category_id" },
    };

    public override IReadOnlyDictionary<string, string[]> OutputColumns { get; } = new Dictionary<string, string[]>
    {
        ["detection"] = new[] { "track_id", "track_bbox_ltwh", "track_bbox_conf" },
    };

    public StrongSortModule(StrongSortOptions? options = null, CameraMotionMode cameraMotion = CameraMotionMode.None)
        : base(batchSize: 1)
    {
        _options = options ?? new StrongSortOptions();
        _tracker = new StrongSortTracker(_options);
        _cameraMotion = new CameraMotion(cameraMotion);
    }

    public override void Reset()
    {
        _tracker = new StrongSortTracker(_options);
        _cameraMotion.Reset();
    }

    public override object Preprocess(Mat image, DataFrame detections, Series metadata)
    {
        // Person-only tracker: the ball (category_id != person) rides in the same DataFrame for
        // calibration but must never enter the appearance tracker, which is built for people.
        var observations = detections.Rows
            .Where(row => Convert.ToInt64(row["category_id"]) == YoloOnnxDetector.PersonCategoryId)
            .Select(row =>
            {
                var bbox = row.Get<float[]>("bbox_ltwh");
                return new Observation(bbox[0], bbox[1], bbox[2], bbox[3],
                    Convert.ToSingle(row["bbox_conf"]),
                    Feature: row.GetOrDefault<float[]>("embeddings"),
                    Tag: row.Index);
            }).ToList();
        // Frames come in order one at a time (batch size 1), so the motion from the previous frame is known here.
        var motion = _cameraMotion.Next(image, observations.Select(o => new[] { o.Left, o.Top, o.Width, o.Height }).ToList());
        return (observations, motion);
    }

    public override ModuleOutput Process(IReadOnlyList<object> batch, DataFrame detections, IReadOnlyList<Series> metadatas)
    {
        var updated = new List<Series>();
        foreach (var (observations, motion) in batch.Cast<(List<Observation>, double[,]?)>())
        {
            foreach (var track in _tracker.Update(observations, motion))
            {
                if (track.Tag is not long detectionIndex) continue; // coasting track, no detection this frame
                var row = new Series(detectionIndex);
                row["track_id"] = track.TrackId;
                row["track_bbox_ltwh"] = track.BboxLtwh;
                row["track_bbox_conf"] = track.Confidence;
                updated.Add(row);
            }
        }
        return new ModuleOutput(updated);
    }
}
