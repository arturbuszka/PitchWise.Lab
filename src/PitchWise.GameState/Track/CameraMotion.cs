using OpenCvSharp;

namespace PitchWise.GameState.Track;

/// <summary>How the tracker compensates camera motion between frames.</summary>
public enum CameraMotionMode
{
    /// <summary>No compensation (sn-gamestate default, <c>ecc: False</c>).</summary>
    None,

    /// <summary>tracklab <c>bpbreid_strong_sort/ecc.py</c>: ECC, Euclidean, on grayscale frames scaled to 0.1.</summary>
    Ecc,

    /// <summary>Sparse optical flow (as BoT-SORT GMC): corners outside the person boxes of the previous frame,
    /// pyramidal Lucas-Kanade, rotation + uniform scale + translation fitted with RANSAC, on frames scaled to 0.5.</summary>
    Flow,
}

/// <summary>
/// Estimates the camera motion between consecutive frames as a 3x3 matrix that maps pixels of the previous
/// frame to the current one. Feed frames in order with <see cref="Next"/>; <see cref="Reset"/> between videos.
/// </summary>
public sealed class CameraMotion : IDisposable
{
    private const double EccScale = 0.1;
    private const double FlowScale = 0.5;

    private readonly CameraMotionMode _mode;
    private Mat? _previous;
    private IReadOnlyList<float[]> _previousBoxes = Array.Empty<float[]>();

    public CameraMotion(CameraMotionMode mode) => _mode = mode;

    /// <summary>The camera motion from the previous frame to <paramref name="image"/> (BGR), or null for the
    /// first frame, when disabled, or when the estimate fails (e.g. a cut to a replay).</summary>
    /// <param name="personBoxes">Person boxes (ltwh) of this frame; masked out of the next flow estimate.</param>
    public double[,]? Next(Mat image, IReadOnlyList<float[]> personBoxes)
    {
        if (_mode == CameraMotionMode.None) return null;
        var current = Downscale(image, _mode == CameraMotionMode.Ecc ? EccScale : FlowScale);
        double[,]? motion = null;
        if (_previous is not null && _previous.Size() == current.Size())
            motion = _mode == CameraMotionMode.Ecc ? Ecc(_previous, current) : Flow(_previous, current, _previousBoxes);
        _previous?.Dispose();
        _previous = current;
        _previousBoxes = personBoxes;
        return motion;
    }

    public void Reset()
    {
        _previous?.Dispose();
        _previous = null;
        _previousBoxes = Array.Empty<float[]>();
    }

    public void Dispose() => Reset();

    private static Mat Downscale(Mat image, double scale)
    {
        using var gray = new Mat();
        Cv2.CvtColor(image, gray, ColorConversionCodes.BGR2GRAY);
        var small = new Mat();
        Cv2.Resize(gray, small, new Size(0, 0), scale, scale, InterpolationFlags.Linear);
        return small;
    }

    // tracklab ecc.py: MOTION_EUCLIDEAN, 100 iterations, eps 1e-5, no Gaussian filtering; translation rescaled.
    private static double[,]? Ecc(Mat previous, Mat current)
    {
        using var warp = Mat.Eye(2, 3, MatType.CV_32FC1).ToMat();
        try
        {
            Cv2.FindTransformECC(previous, current, warp, MotionTypes.Euclidean,
                new TermCriteria(CriteriaTypes.Eps | CriteriaTypes.Count, 100, 1e-5), null, 1);
        }
        catch (OpenCVException)
        {
            return null;
        }
        var m = ToMatrix(warp, EccScale);
        // tracklab get_matrix: a warp too far from identity (Frobenius norm of I - M >= 100) is replaced by identity
        var dist = 0.0;
        for (var i = 0; i < 3; i++)
            for (var j = 0; j < 3; j++)
                dist += Math.Pow((i == j ? 1 : 0) - m[i, j], 2);
        return Math.Sqrt(dist) < 100 ? m : null;
    }

    private static double[,]? Flow(Mat previous, Mat current, IReadOnlyList<float[]> previousBoxes)
    {
        using var mask = new Mat(previous.Size(), MatType.CV_8UC1, Scalar.All(0));
        int h = previous.Rows, w = previous.Cols;
        mask[new Rect((int)(0.02 * w), (int)(0.02 * h), (int)(0.98 * w) - (int)(0.02 * w), (int)(0.98 * h) - (int)(0.02 * h))]
            .SetTo(Scalar.All(255));
        foreach (var box in previousBoxes)
        {
            var x1 = Math.Clamp((int)(box[0] * FlowScale), 0, w);
            var y1 = Math.Clamp((int)(box[1] * FlowScale), 0, h);
            var x2 = Math.Clamp((int)((box[0] + box[2]) * FlowScale), 0, w);
            var y2 = Math.Clamp((int)((box[1] + box[3]) * FlowScale), 0, h);
            if (x2 > x1 && y2 > y1)
                mask[new Rect(x1, y1, x2 - x1, y2 - y1)].SetTo(Scalar.All(0));
        }

        var corners = Cv2.GoodFeaturesToTrack(previous, 1000, 0.01, 1, mask, 3, false, 0.04);
        if (corners.Length < 10) return null;
        var moved = Array.Empty<Point2f>();
        Cv2.CalcOpticalFlowPyrLK(previous, current, corners, ref moved, out var status, out _);
        var from = new List<Point2f>();
        var to = new List<Point2f>();
        for (var i = 0; i < corners.Length; i++)
        {
            if (status[i] != 1) continue;
            from.Add(corners[i]);
            to.Add(moved[i]);
        }
        if (from.Count < 10) return null;

        using var fromMat = Mat.FromArray(from.ToArray());
        using var toMat = Mat.FromArray(to.ToArray());
        using var affine = Cv2.EstimateAffinePartial2D(fromMat, toMat, null, RobustEstimationAlgorithms.RANSAC);
        if (affine is null || affine.Empty()) return null;
        return ToMatrix(affine, FlowScale);
    }

    /// <summary>A 2x3 warp estimated on downscaled frames as a full-resolution 3x3 matrix.</summary>
    private static double[,] ToMatrix(Mat warp, double scale)
    {
        using var m = new Mat();
        warp.ConvertTo(m, MatType.CV_64FC1);
        return new double[,]
        {
            { m.At<double>(0, 0), m.At<double>(0, 1), m.At<double>(0, 2) / scale },
            { m.At<double>(1, 0), m.At<double>(1, 1), m.At<double>(1, 2) / scale },
            { 0, 0, 1 },
        };
    }
}
