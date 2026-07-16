using OpenCvSharp;
using PitchWise.Tracking.Datastruct;
using PitchWise.Tracking.Pipeline;
using PitchWise.Tracking.Utils;

namespace PitchWise.GameState.Calibration;

/// <summary>
/// Per frame, estimates the
/// image -> pitch homography from the detected keypoints (RANSAC threshold 50, inverse)
/// and projects the bottom of every detection bbox onto the pitch ("bbox_pitch").
/// With use_prev_homography, frames where the estimate fails reuse the last good one.
/// </summary>
public sealed class NbjwCalib : ImageLevelModule
{
    private readonly FrameByFrameCalib _cam;
    private readonly bool _usePrevHomography;
    private double[,]? _lastH;
    private CameraParameters? _lastParams;

    public override IReadOnlyDictionary<string, string[]> InputColumns { get; } = new Dictionary<string, string[]>
    {
        ["image"] = new[] { "keypoints" },
        ["detection"] = new[] { "bbox_ltwh" },
    };

    public override IReadOnlyDictionary<string, string[]> OutputColumns { get; } = new Dictionary<string, string[]>
    {
        ["image"] = new[] { "homography", "parameters" },
        ["detection"] = new[] { "bbox_pitch" },
    };

    public NbjwCalib(int imageWidth, int imageHeight, bool usePrevHomography = true)
        : base(batchSize: 1)
    {
        _cam = new FrameByFrameCalib(imageWidth, imageHeight, denormalize: true);
        _usePrevHomography = usePrevHomography;
    }

    public override void Reset()
    {
        _lastH = null;
        _lastParams = null;
    }

    public override object Preprocess(Mat image, DataFrame detections, Series metadata) => false;

    public override ModuleOutput Process(IReadOnlyList<object> batch, DataFrame detections, IReadOnlyList<Series> metadatas)
    {
        var metadata = metadatas[0];
        var keypoints = metadata.GetOrDefault<Dictionary<int, Keypoint>>("keypoints")
                        ?? new Dictionary<int, Keypoint>();

        _cam.Update(keypoints);
        var h = _cam.GetHomographyFromGroundPlane(useRansac: 50, inverse: true);
        CameraParameters? parameters;
        if (h is not null)
        {
            parameters = _cam.HeuristicVoting();
            _lastH = h;
            _lastParams = parameters;
        }
        else if (_usePrevHomography)
        {
            h = _lastH;
            parameters = _lastParams;
        }
        else
        {
            parameters = null;
        }

        var detectionUpdates = new List<Series>();
        foreach (var detection in detections.Rows.Where(d => Equals(d["image_id"], metadata.Index)))
        {
            var row = new Series(detection.Index);
            row["bbox_pitch"] = h is null
                ? null
                : GetBboxPitch(h, Coordinates.LtwhToLtrb(detection.Get<float[]>("bbox_ltwh")));
            detectionUpdates.Add(row);
        }

        var imageRow = new Series(metadata.Index);
        imageRow["homography"] = h;
        imageRow["parameters"] = parameters;
        return new ModuleOutput(detectionUpdates, new[] { imageRow });
    }

    /// <summary>get_bbox_pitch: unprojects the bbox bottom-left/right/middle onto the pitch plane.</summary>
    public static BboxPitch? GetBboxPitch(double[,] h, float[] bboxLtrb)
    {
        var (l, t, r, b) = (bboxLtrb[0], bboxLtrb[1], bboxLtrb[2], bboxLtrb[3]);
        var (blX, blY) = FrameByFrameCalib.Transform(h, l, b);
        var (brX, brY) = FrameByFrameCalib.Transform(h, r, b);
        var (bmX, bmY) = FrameByFrameCalib.Transform(h, l + (r - l) / 2, b);
        if (new[] { blX, blY, brX, brY, bmX, bmY }.Any(double.IsNaN))
            return null;
        return new BboxPitch(blX, blY, brX, brY, bmX, bmY);
    }
}
