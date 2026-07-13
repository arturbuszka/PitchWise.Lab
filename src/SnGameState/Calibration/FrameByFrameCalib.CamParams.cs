using OpenCvSharp;

namespace SnGameState.Calibration;

/// <summary>
/// Camera-parameter estimation part of the FramebyFrameCalib port: get_cam_params
/// (multi-plane cv2.calibrateCamera with all intrinsics except focal length fixed)
/// and heuristic_voting (best result over 3 modes x 6 RANSAC thresholds).
/// </summary>
public sealed partial class FrameByFrameCalib
{
    /// <summary>Message of the last OpenCV calibration failure (diagnostics only).</summary>
    public string? LastCalibrationError { get; private set; }

    private static readonly int[] LeftGoalPlaneKeypoints = { 1, 4, 8, 13, 17, 20, 24, 28 };
    private static readonly int[] RightGoalPlaneKeypoints = { 3, 7, 11, 14, 18, 23, 27, 30 };

    /// <summary>heuristic_voting: tries all mode/RANSAC combinations and returns the
    /// calibration with the lowest reprojection error, preferring a clean "full" fit.</summary>
    public CameraParameters? HeuristicVoting()
    {
        var results = new List<(string Mode, int UseRansac, double RepErr, CameraParameters Params)>();
        foreach (var mode in new[] { "full", "ground_plane", "main" })
            foreach (var useRansac in new[] { 0, 5, 10, 15, 25, 50 })
            {
                var result = GetCamParams(mode, useRansac);
                if (result is not null)
                    results.Add((mode, useRansac, result.Value.RepErr, result.Value.Params));
            }

        if (results.Count == 0)
            return null;

        results = results.OrderBy(r => r.RepErr).ThenBy(r => r.Mode, StringComparer.Ordinal).ToList();
        foreach (var res in results)
            if (res.Mode == "full" && res.UseRansac == 0 && res.RepErr <= 10.0)
                return res.Params;
        return results[0].Params;
    }

    /// <summary>get_cam_params: full camera calibration from the per-plane correspondences.
    /// Returns null when no usable plane exists or OpenCV fails to calibrate.</summary>
    public (CameraParameters Params, double RepErr)? GetCamParams(string mode = "full", double useRansac = 0)
    {
        var (objPlanes, imgPlanes, ordPlanes) = GetPerPlaneCorrespondences(mode, useRansac);
        if (objPlanes.Count == 0)
            return null;

        var cameraMatrix = new double[3, 3]
        {
            { 1, 0, _imageWidth / 2.0 },
            { 0, 1, _imageHeight / 2.0 },
            { 0, 0, 1 },
        };
        var distCoeffs = new double[5];
        // CALIB_FIX_TANGENT_DIST (0x200000) and CALIB_FIX_TAUX_TAUY (0x80000) are missing
        // from the OpenCvSharp enum — raw OpenCV values used instead.
        const CalibrationFlags flags =
            CalibrationFlags.FixPrincipalPoint | CalibrationFlags.FixAspectRatio |
            (CalibrationFlags)0x200000 | CalibrationFlags.FixS1S2S3S4 |
            (CalibrationFlags)0x80000 |
            CalibrationFlags.FixK1 | CalibrationFlags.FixK2 | CalibrationFlags.FixK3 |
            CalibrationFlags.FixK4 | CalibrationFlags.FixK5 | CalibrationFlags.FixK6;

        double repErr;
        Vec3d[] rvecs, tvecs;
        try
        {
            repErr = Cv2.CalibrateCamera(
                objPlanes, imgPlanes, new Size(_imageWidth, _imageHeight),
                cameraMatrix, distCoeffs, out rvecs, out tvecs, flags);
        }
        catch (OpenCVException e)
        {
            LastCalibrationError = e.Message;
            return null;
        }
        if (repErr <= 0)
            return null;

        Cv2.Rodrigues(new[] { rvecs[0].Item0, rvecs[0].Item1, rvecs[0].Item2 }, out var rotation, out _);
        var t = new[] { tvecs[0].Item0, tvecs[0].Item1, tvecs[0].Item2 };
        var position = MultiplyTransposed(rotation, t).Select(v => -v).ToArray();

        if (ordPlanes[0] != 0)
            (rotation, position) = ChangePlaneCoords(rotation, position, ordPlanes[0]);

        var (pan, tilt, roll) = RotationMatrixToPanTiltRoll(rotation);
        return (new CameraParameters
        {
            PanDegrees = pan * 180.0 / Math.PI,
            TiltDegrees = tilt * 180.0 / Math.PI,
            RollDegrees = roll * 180.0 / Math.PI,
            XFocalLength = cameraMatrix[0, 0],
            YFocalLength = cameraMatrix[1, 1],
            PrincipalPoint = new[] { _imageWidth / 2.0, _imageHeight / 2.0 },
            PositionMeters = position,
            RotationMatrix = rotation,
        }, repErr);
    }

    /// <summary>
    /// get_per_plane_correspondences: splits keypoints over the pitch ground plane (0),
    /// the left (1) and right (2) goal planes (world x replaced by -z there), applies the
    /// RANSAC outlier pre-filter to the ground plane, and drops planes with fewer than 4
    /// or degenerate correspondences.
    /// </summary>
    private (List<List<Point3f>> Obj, List<List<Point2f>> Img, List<int> Ord)
        GetPerPlaneCorrespondences(string mode, double useRansac)
    {
        var subset = GetSubset(mode);
        var objPlanes = new List<List<Point3f>> { new(), new(), new() };
        var imgPlanes = new List<List<Point2f>> { new(), new(), new() };

        foreach (var (kp, v) in subset)
        {
            var img = new Point2f((float)v.Xi, (float)v.Yi);
            if (kp is 12 or 16)
            {
                objPlanes[1].Add(new Point3f((float)-v.Zw, (float)v.Yw, 0));
                imgPlanes[1].Add(img);
            }
            else if (LeftGoalPlaneKeypoints.Contains(kp))
            {
                objPlanes[0].Add(new Point3f((float)v.Xw, (float)v.Yw, (float)v.Zw));
                objPlanes[1].Add(new Point3f((float)-v.Zw, (float)v.Yw, 0));
                imgPlanes[0].Add(img);
                imgPlanes[1].Add(img);
            }
            else if (RightGoalPlaneKeypoints.Contains(kp))
            {
                objPlanes[0].Add(new Point3f((float)v.Xw, (float)v.Yw, (float)v.Zw));
                objPlanes[2].Add(new Point3f((float)-v.Zw, (float)v.Yw, 0));
                imgPlanes[0].Add(img);
                imgPlanes[2].Add(img);
            }
            else if (kp is 15 or 19)
            {
                objPlanes[2].Add(new Point3f((float)-v.Zw, (float)v.Yw, 0));
                imgPlanes[2].Add(img);
            }
            else
            {
                objPlanes[0].Add(new Point3f((float)v.Xw, (float)v.Yw, (float)v.Zw));
                imgPlanes[0].Add(img);
            }
        }

        var planeCount = mode == "ground_plane" ? 1 : 3;

        // RANSAC pre-filter, applied to the ground plane only (i == 0 in the Python code)
        if (useRansac > 0 && IsUsablePlane(objPlanes[0]))
        {
            using var mask = new Mat();
            using var pre = Cv2.FindHomography(
                objPlanes[0].Select(p => new Point2d(p.X, p.Y)),
                imgPlanes[0].Select(p => new Point2d(p.X, p.Y)),
                HomographyMethods.Ransac, useRansac, mask);
            if (!pre.Empty())
            {
                mask.GetArray(out byte[] status);
                objPlanes[0] = objPlanes[0].Where((_, i) => status[i] == 1).ToList();
                imgPlanes[0] = imgPlanes[0].Where((_, i) => status[i] == 1).ToList();
            }
        }

        var obj = new List<List<Point3f>>();
        var img2 = new List<List<Point2f>>();
        var ord = new List<int>();
        for (var i = 0; i < planeCount; i++)
        {
            if (!IsUsablePlane(objPlanes[i])) continue;
            obj.Add(objPlanes[i]);
            img2.Add(imgPlanes[i]);
            ord.Add(i);
        }
        return (obj, img2, ord);
    }

    private Dictionary<int, (double Xi, double Yi, double Xw, double Yw, double Zw)> GetSubset(string mode)
    {
        var subset = new Dictionary<int, (double, double, double, double, double)>();
        foreach (var (kp, point) in _keypoints)
        {
            var include = mode switch
            {
                "full" => true,
                "main" => kp <= 30,
                "ground_plane" => !PitchGeometry.CrossbarKeypoints.Contains(kp),
                _ => throw new ArgumentException($"Wrong mode: {mode}"),
            };
            if (!include) continue;
            var (xw, yw) = PitchGeometry.WorldCoords(kp);
            var zw = PitchGeometry.CrossbarKeypoints.Contains(kp) ? -2.44 : 0.0;
            subset[kp] = (point.X, point.Y, xw, yw, zw);
        }
        return subset;
    }

    private static bool IsUsablePlane(List<Point3f> points)
        => points.Count >= 4
           && !points.All(p => p.X == points[0].X)
           && !points.All(p => p.Y == points[0].Y);

    /// <summary>change_plane_coords: maps a goal-plane calibration back to pitch coordinates.</summary>
    private static (double[,] Rotation, double[] Position) ChangePlaneCoords(
        double[,] rotation, double[] position, int ord, double pitchWidth = 105)
    {
        var rc = new double[3, 3] { { 0, 0, -1 }, { 0, 1, 0 }, { 1, 0, 0 } };
        var rcInv = new double[3, 3] { { 0, 0, 1 }, { 0, 1, 0 }, { -1, 0, 0 } }; // R^T = R^-1

        var newRotation = Multiply(rotation, rc);
        var newPosition = Multiply(rcInv, position);
        newPosition[0] += ord == 1 ? -pitchWidth / 2 : pitchWidth / 2;
        return (newRotation, newPosition);
    }

    /// <summary>rotation_matrix_to_pan_tilt_roll: of the two Euler decompositions, picks
    /// the one with the smallest roll (broadcast cameras minimize roll). Radians.</summary>
    public static (double Pan, double Tilt, double Roll) RotationMatrixToPanTiltRoll(double[,] rotation)
    {
        var o = Transpose(rotation);
        var firstTilt = Math.Acos(o[2, 2]);
        var secondTilt = -firstTilt;

        var signFirst = Math.Sin(firstTilt) > 0 ? 1.0 : -1.0;
        var signSecond = Math.Sin(secondTilt) > 0 ? 1.0 : -1.0;

        var firstPan = Math.Atan2(signFirst * o[0, 2], signFirst * -o[1, 2]);
        var secondPan = Math.Atan2(signSecond * o[0, 2], signSecond * -o[1, 2]);
        var firstRoll = Math.Atan2(signFirst * o[2, 0], signFirst * o[2, 1]);
        var secondRoll = Math.Atan2(signSecond * o[2, 0], signSecond * o[2, 1]);

        return Math.Abs(firstRoll) < Math.Abs(secondRoll)
            ? (firstPan, firstTilt, firstRoll)
            : (secondPan, secondTilt, secondRoll);
    }

    /// <summary>pan_tilt_roll_to_orientation: Rpan @ Rtilt @ Rroll (radians).</summary>
    public static double[,] PanTiltRollToOrientation(double pan, double tilt, double roll)
    {
        var rPan = new double[3, 3]
        {
            { Math.Cos(pan), -Math.Sin(pan), 0 }, { Math.Sin(pan), Math.Cos(pan), 0 }, { 0, 0, 1 },
        };
        var rRoll = new double[3, 3]
        {
            { Math.Cos(roll), -Math.Sin(roll), 0 }, { Math.Sin(roll), Math.Cos(roll), 0 }, { 0, 0, 1 },
        };
        var rTilt = new double[3, 3]
        {
            { 1, 0, 0 }, { 0, Math.Cos(tilt), -Math.Sin(tilt) }, { 0, Math.Sin(tilt), Math.Cos(tilt) },
        };
        return Multiply(rPan, Multiply(rTilt, rRoll));
    }

    private static double[,] Multiply(double[,] a, double[,] b)
    {
        var result = new double[3, 3];
        for (var i = 0; i < 3; i++)
            for (var j = 0; j < 3; j++)
                for (var k = 0; k < 3; k++)
                    result[i, j] += a[i, k] * b[k, j];
        return result;
    }

    private static double[] Multiply(double[,] a, double[] v)
    {
        var result = new double[3];
        for (var i = 0; i < 3; i++)
            for (var k = 0; k < 3; k++)
                result[i] += a[i, k] * v[k];
        return result;
    }

    private static double[] MultiplyTransposed(double[,] a, double[] v)
    {
        var result = new double[3];
        for (var i = 0; i < 3; i++)
            for (var k = 0; k < 3; k++)
                result[i] += a[k, i] * v[k];
        return result;
    }

    private static double[,] Transpose(double[,] a)
    {
        var result = new double[3, 3];
        for (var i = 0; i < 3; i++)
            for (var j = 0; j < 3; j++)
                result[i, j] = a[j, i];
        return result;
    }
}
