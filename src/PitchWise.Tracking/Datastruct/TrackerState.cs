using System.Text.Json;

namespace PitchWise.Tracking.Datastruct;

/// <summary>
/// Tracker state (in-memory, with JSON persistence).
/// Holds per-video predicted detections and image predictions produced by the pipeline.
/// The Python original persists to a zipped pickle/parquet; here we persist JSON-serializable
/// columns per video under a state directory.
/// </summary>
public sealed class TrackerState
{
    public TrackingSet Set { get; }
    private readonly Dictionary<long, (DataFrame Detections, DataFrame ImagePreds)> _perVideo = new();

    public TrackerState(TrackingSet set) => Set = set;

    /// <summary>Returns current detections + image predictions for the active video,
    /// seeding image predictions from the dataset's image metadata.</summary>
    public (DataFrame detections, DataFrame imagePreds) Load(long videoId)
    {
        if (!_perVideo.TryGetValue(videoId, out var entry))
        {
            var imagePreds = new DataFrame();
            foreach (var img in Set.ImagesOfVideo(videoId))
                imagePreds.Upsert(img);
            entry = (new DataFrame(), imagePreds);
            _perVideo[videoId] = entry;
        }
        return entry;
    }

    public void Update(long videoId, DataFrame detections, DataFrame imagePreds)
        => _perVideo[videoId] = (detections, imagePreds);

    public void Save(string directory, long videoId)
    {
        Directory.CreateDirectory(directory);
        var (detections, _) = _perVideo[videoId];
        var rows = detections.Rows.Select(r =>
            new Dictionary<string, object?>(r.Values.Where(kv => IsJsonSerializable(kv.Value))
                                                     .Append(new("index", r.Index))));
        var json = JsonSerializer.Serialize(rows, new JsonSerializerOptions { WriteIndented = false });
        File.WriteAllText(Path.Combine(directory, $"{videoId}.json"), json);
    }

    private static bool IsJsonSerializable(object? value)
        => value is null or string or bool or int or long or float or double
           or float[] or double[] or int[] or string[];
}
