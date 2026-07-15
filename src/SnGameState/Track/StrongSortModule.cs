using OpenCvSharp;
using SnGameState.Detection;
using StrongSort;
using TrackLab.Datastruct;
using TrackLab.Pipeline;

namespace SnGameState.Track;

/// <summary>
/// Port of tracklab.wrappers.track (bpbreid_)strong_sort_api: feeds each frame's detections
/// (with ReID embeddings when available) into <see cref="StrongSortTracker"/> and writes
/// track_id / track_bbox_ltwh / track_bbox_conf back onto the detection rows.
/// </summary>
public sealed class StrongSortModule : ImageLevelModule
{
    private readonly StrongSortOptions _options;
    private StrongSortTracker _tracker;

    public override IReadOnlyDictionary<string, string[]> InputColumns { get; } = new Dictionary<string, string[]>
    {
        ["detection"] = new[] { "bbox_ltwh", "bbox_conf", "category_id" },
    };

    public override IReadOnlyDictionary<string, string[]> OutputColumns { get; } = new Dictionary<string, string[]>
    {
        ["detection"] = new[] { "track_id", "track_bbox_ltwh", "track_bbox_conf" },
    };

    public StrongSortModule(StrongSortOptions? options = null) : base(batchSize: 1)
    {
        _options = options ?? new StrongSortOptions();
        _tracker = new StrongSortTracker(_options);
    }

    public override void Reset() => _tracker = new StrongSortTracker(_options);

    public override object Preprocess(Mat image, DataFrame detections, Series metadata)
    {
        // Person-only tracker: the ball (category_id != person) rides in the same DataFrame for
        // calibration but must never enter the appearance tracker, which is built for people.
        return detections.Rows
            .Where(row => Convert.ToInt64(row["category_id"]) == YoloOnnxDetector.PersonCategoryId)
            .Select(row =>
            {
                var bbox = row.Get<float[]>("bbox_ltwh");
                return new Observation(bbox[0], bbox[1], bbox[2], bbox[3],
                    Convert.ToSingle(row["bbox_conf"]),
                    Feature: row.GetOrDefault<float[]>("embeddings"),
                    Tag: row.Index);
            }).ToList();
    }

    public override ModuleOutput Process(IReadOnlyList<object> batch, DataFrame detections, IReadOnlyList<Series> metadatas)
    {
        var updated = new List<Series>();
        foreach (var observations in batch.Cast<List<Observation>>())
        {
            foreach (var track in _tracker.Update(observations))
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
