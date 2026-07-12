namespace TrackLab.Datastruct;

/// <summary>Port of tracklab.datastruct.tracking_dataset — metadata tables for one split.</summary>
public sealed class TrackingSet
{
    /// <summary>One row per video: "id", "name", ...</summary>
    public required DataFrame VideoMetadatas { get; init; }

    /// <summary>One row per image: "id", "video_id", "frame", "file_path", ...</summary>
    public required DataFrame ImageMetadatas { get; init; }

    /// <summary>Ground-truth detections (empty for unlabeled data).</summary>
    public DataFrame DetectionsGt { get; init; } = DataFrame.Empty();

    public IEnumerable<Series> ImagesOfVideo(long videoId)
        => ImageMetadatas.Rows.Where(r => r.Get<long>("video_id") == videoId)
                              .OrderBy(r => r.GetOrDefault<long>("frame"));
}

public abstract class TrackingDataset
{
    public string DatasetPath { get; }
    public Dictionary<string, TrackingSet> Sets { get; } = new();

    protected TrackingDataset(string datasetPath) => DatasetPath = datasetPath;

    public abstract string Name { get; }
}
