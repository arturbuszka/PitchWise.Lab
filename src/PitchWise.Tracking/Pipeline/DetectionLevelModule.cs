using OpenCvSharp;
using PitchWise.Tracking.Datastruct;

namespace PitchWise.Tracking.Pipeline;

/// <summary>
/// A module
/// that operates on individual detections (e.g. a re-identifier or top-down pose estimator).
/// </summary>
public abstract class DetectionLevelModule : Module
{
    public int BatchSize { get; }

    protected DetectionLevelModule(int batchSize) => BatchSize = batchSize;

    public override string Level => "detection";

    /// <summary>Adapts one detection (with its source image) to the sample fed to <see cref="Process"/>.</summary>
    public abstract object Preprocess(Mat image, Series detection, Series metadata);

    /// <summary>Runs on a batch of preprocessed samples; returns updated detection rows
    /// (must keep the original row indices so the merge overrides the right rows).</summary>
    public abstract IReadOnlyList<Series> Process(IReadOnlyList<object> batch, DataFrame detections, IReadOnlyList<Series> metadatas);
}
