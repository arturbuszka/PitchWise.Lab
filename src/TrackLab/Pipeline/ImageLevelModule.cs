using OpenCvSharp;
using TrackLab.Datastruct;

namespace TrackLab.Pipeline;

/// <summary>
/// Port of tracklab.pipeline.imagelevel_module.ImageLevelModule — a module that
/// operates directly on images (e.g. a bounding box detector).
/// </summary>
public abstract class ImageLevelModule : Module
{
    public int BatchSize { get; }

    protected ImageLevelModule(int batchSize) => BatchSize = batchSize;

    public override string Level => "image";

    /// <summary>Adapts the default input (one image + its current detections) to the
    /// model-specific sample fed to <see cref="Process"/>.</summary>
    public abstract object Preprocess(Mat image, DataFrame detections, Series metadata);

    /// <summary>Runs on a batch of preprocessed samples; returns new/updated detection
    /// rows (and optionally updated image metadata rows).</summary>
    public abstract ModuleOutput Process(IReadOnlyList<object> batch, DataFrame detections, IReadOnlyList<Series> metadatas);
}

/// <summary>Rows returned by a module: detections to merge, plus optional image metadata updates.</summary>
public sealed record ModuleOutput(IReadOnlyList<Series> Detections, IReadOnlyList<Series>? ImageMetadatas = null)
{
    public static readonly ModuleOutput Empty = new(Array.Empty<Series>());
}
