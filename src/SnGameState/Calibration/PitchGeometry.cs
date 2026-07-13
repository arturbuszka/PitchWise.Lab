namespace SnGameState.Calibration;

/// <summary>A detected pitch keypoint in image coordinates (or normalized when noted).</summary>
public readonly record struct Keypoint(double X, double Y, double Score);

/// <summary>A detected pitch line as two extremity points.</summary>
public readonly record struct LineSegment(Keypoint P1, Keypoint P2);

/// <summary>
/// Port of the pitch template of nbjw_calib.utils.utils_calib / nbjw_calib.py:
/// world coordinates (meters, origin at pitch center) of the 57 main + 16 auxiliary
/// keypoints, and the mapping between line names and the keypoints lying on them.
/// </summary>
public static class PitchGeometry
{
    /// <summary>Goal crossbar keypoints (z = -2.44 m) — excluded from the ground plane.</summary>
    public static readonly int[] CrossbarKeypoints = { 12, 15, 16, 19 };

    /// <summary>World coordinates of main keypoints 1..57, already centered (x-52.5, y-34).</summary>
    public static readonly double[][] KeypointWorldCoords2D = Center(new double[][]
    {
        new[] { 0.0, 0.0 }, new[] { 52.5, 0.0 }, new[] { 105.0, 0.0 }, new[] { 0.0, 13.84 },
        new[] { 16.5, 13.84 }, new[] { 88.5, 13.84 }, new[] { 105.0, 13.84 }, new[] { 0.0, 24.84 },
        new[] { 5.5, 24.84 }, new[] { 99.5, 24.84 }, new[] { 105.0, 24.84 }, new[] { 0.0, 30.34 },
        new[] { 0.0, 30.34 }, new[] { 105.0, 30.34 }, new[] { 105.0, 30.34 }, new[] { 0.0, 37.66 },
        new[] { 0.0, 37.66 }, new[] { 105.0, 37.66 }, new[] { 105.0, 37.66 }, new[] { 0.0, 43.16 },
        new[] { 5.5, 43.16 }, new[] { 99.5, 43.16 }, new[] { 105.0, 43.16 }, new[] { 0.0, 54.16 },
        new[] { 16.5, 54.16 }, new[] { 88.5, 54.16 }, new[] { 105.0, 54.16 }, new[] { 0.0, 68.0 },
        new[] { 52.5, 68.0 }, new[] { 105.0, 68.0 }, new[] { 16.5, 26.68 }, new[] { 52.5, 24.85 },
        new[] { 88.5, 26.68 }, new[] { 16.5, 41.31 }, new[] { 52.5, 43.15 }, new[] { 88.5, 41.31 },
        new[] { 19.99, 32.29 }, new[] { 43.68, 31.53 }, new[] { 61.31, 31.53 }, new[] { 85.0, 32.29 },
        new[] { 19.99, 35.7 }, new[] { 43.68, 36.46 }, new[] { 61.31, 36.46 }, new[] { 85.0, 35.7 },
        new[] { 11.0, 34.0 }, new[] { 16.5, 34.0 }, new[] { 20.15, 34.0 }, new[] { 46.03, 27.53 },
        new[] { 58.97, 27.53 }, new[] { 43.35, 34.0 }, new[] { 52.5, 34.0 }, new[] { 61.5, 34.0 },
        new[] { 46.03, 40.47 }, new[] { 58.97, 40.47 }, new[] { 84.85, 34.0 }, new[] { 88.5, 34.0 },
        new[] { 94.0, 34.0 },
    });

    /// <summary>World coordinates of auxiliary keypoints 58..73 (line-intersection constructs).</summary>
    public static readonly double[][] KeypointAuxWorldCoords2D = Center(new double[][]
    {
        new[] { 5.5, 0.0 }, new[] { 16.5, 0.0 }, new[] { 88.5, 0.0 }, new[] { 99.5, 0.0 },
        new[] { 5.5, 13.84 }, new[] { 99.5, 13.84 }, new[] { 16.5, 24.84 }, new[] { 88.5, 24.84 },
        new[] { 16.5, 43.16 }, new[] { 88.5, 43.16 }, new[] { 5.5, 54.16 }, new[] { 99.5, 54.16 },
        new[] { 5.5, 68.0 }, new[] { 16.5, 68.0 }, new[] { 88.5, 68.0 }, new[] { 99.5, 68.0 },
    });

    /// <summary>World (x, y) of keypoint id 1..73.</summary>
    public static (double Xw, double Yw) WorldCoords(int keypointId)
    {
        var wp = keypointId <= 57
            ? KeypointWorldCoords2D[keypointId - 1]
            : KeypointAuxWorldCoords2D[keypointId - 1 - 57];
        return (wp[0], wp[1]);
    }

    /// <summary>Line names in the order of the line-model channels (1-based ids = index + 1).</summary>
    public static readonly string[] LinesList =
    {
        "Big rect. left bottom", "Big rect. left main", "Big rect. left top", "Big rect. right bottom",
        "Big rect. right main", "Big rect. right top", "Goal left crossbar", "Goal left post left ",
        "Goal left post right", "Goal right crossbar", "Goal right post left", "Goal right post right",
        "Middle line", "Side line bottom", "Side line left", "Side line right", "Side line top",
        "Small rect. left bottom", "Small rect. left main", "Small rect. left top", "Small rect. right bottom",
        "Small rect. right main", "Small rect. right top",
    };

    /// <summary>For main keypoint id 1..30: the two lines whose intersection defines it.</summary>
    public static readonly string[][] KeypointsLineList =
    {
        new[] { "Side line top", "Side line left" }, new[] { "Side line top", "Middle line" },
        new[] { "Side line right", "Side line top" }, new[] { "Side line left", "Big rect. left top" },
        new[] { "Big rect. left top", "Big rect. left main" }, new[] { "Big rect. right top", "Big rect. right main" },
        new[] { "Side line right", "Big rect. right top" }, new[] { "Side line left", "Small rect. left top" },
        new[] { "Small rect. left top", "Small rect. left main" }, new[] { "Small rect. right top", "Small rect. right main" },
        new[] { "Side line right", "Small rect. right top" }, new[] { "Goal left crossbar", "Goal left post right" },
        new[] { "Side line left", "Goal left post right" }, new[] { "Side line right", "Goal right post left" },
        new[] { "Goal right crossbar", "Goal right post left" }, new[] { "Goal left crossbar", "Goal left post left " },
        new[] { "Side line left", "Goal left post left " }, new[] { "Side line right", "Goal right post right" },
        new[] { "Goal right crossbar", "Goal right post right" }, new[] { "Side line left", "Small rect. left bottom" },
        new[] { "Small rect. left bottom", "Small rect. left main" }, new[] { "Small rect. right bottom", "Small rect. right main" },
        new[] { "Side line right", "Small rect. right bottom" }, new[] { "Side line left", "Big rect. left bottom" },
        new[] { "Big rect. left bottom", "Big rect. left main" }, new[] { "Big rect. right main", "Big rect. right bottom" },
        new[] { "Side line right", "Big rect. right bottom" }, new[] { "Side line left", "Side line bottom" },
        new[] { "Side line bottom", "Middle line" }, new[] { "Side line bottom", "Side line right" },
    };

    /// <summary>For auxiliary keypoint (id - 57): the two lines whose intersection defines it.</summary>
    public static readonly string[][] KeypointAuxPairList =
    {
        new[] { "Small rect. left main", "Side line top" }, new[] { "Big rect. left main", "Side line top" },
        new[] { "Big rect. right main", "Side line top" }, new[] { "Small rect. right main", "Side line top" },
        new[] { "Small rect. left main", "Big rect. left top" }, new[] { "Big rect. right top", "Small rect. right main" },
        new[] { "Small rect. left top", "Big rect. left main" }, new[] { "Small rect. right top", "Big rect. right main" },
        new[] { "Small rect. left bottom", "Big rect. left main" }, new[] { "Small rect. right bottom", "Big rect. right main" },
        new[] { "Small rect. left main", "Big rect. left bottom" }, new[] { "Small rect. right main", "Big rect. right bottom" },
        new[] { "Small rect. left main", "Side line bottom" }, new[] { "Big rect. left main", "Side line bottom" },
        new[] { "Big rect. right main", "Side line bottom" }, new[] { "Small rect. right main", "Side line bottom" },
    };

    /// <summary>Keypoint ids lying on each named line (kp_to_line of nbjw_calib.py).</summary>
    public static readonly IReadOnlyDictionary<string, int[]> LineKeypointsMatch = new Dictionary<string, int[]>
    {
        ["Big rect. left bottom"] = new[] { 24, 68, 25 },
        ["Big rect. left main"] = new[] { 5, 64, 31, 46, 34, 66, 25 },
        ["Big rect. left top"] = new[] { 4, 62, 5 },
        ["Big rect. right bottom"] = new[] { 26, 69, 27 },
        ["Big rect. right main"] = new[] { 6, 65, 33, 56, 36, 67, 26 },
        ["Big rect. right top"] = new[] { 6, 63, 7 },
        ["Circle central"] = new[] { 32, 48, 38, 50, 42, 53, 35, 54, 43, 52, 39, 49 },
        ["Circle left"] = new[] { 31, 37, 47, 41, 34 },
        ["Circle right"] = new[] { 33, 40, 55, 44, 36 },
        ["Goal left crossbar"] = new[] { 16, 12 },
        ["Goal left post left"] = new[] { 16, 17 },
        ["Goal left post right"] = new[] { 12, 13 },
        ["Goal right crossbar"] = new[] { 15, 19 },
        ["Goal right post left"] = new[] { 15, 14 },
        ["Goal right post right"] = new[] { 19, 18 },
        ["Middle line"] = new[] { 2, 32, 51, 35, 29 },
        ["Side line bottom"] = new[] { 28, 70, 71, 29, 72, 73, 30 },
        ["Side line left"] = new[] { 1, 4, 8, 13, 17, 20, 24, 28 },
        ["Side line right"] = new[] { 3, 7, 11, 14, 18, 23, 27, 30 },
        ["Side line top"] = new[] { 1, 58, 59, 2, 60, 61, 3 },
        ["Small rect. left bottom"] = new[] { 20, 21 },
        ["Small rect. left main"] = new[] { 9, 21 },
        ["Small rect. left top"] = new[] { 8, 9 },
        ["Small rect. right bottom"] = new[] { 22, 23 },
        ["Small rect. right main"] = new[] { 10, 22 },
        ["Small rect. right top"] = new[] { 10, 11 },
    };

    /// <summary>Groups detected keypoints into the lines they lie on (kp_to_line).</summary>
    public static Dictionary<string, List<Keypoint>> KeypointsToLines(IReadOnlyDictionary<int, Keypoint> keypoints)
    {
        var lines = new Dictionary<string, List<Keypoint>>();
        foreach (var (lineName, ids) in LineKeypointsMatch)
        {
            var line = ids.Where(keypoints.ContainsKey).Select(id => keypoints[id]).ToList();
            if (line.Count > 0)
                lines[lineName] = line;
        }
        return lines;
    }

    private static double[][] Center(double[][] coords)
        => coords.Select(c => new[] { c[0] - 52.5, c[1] - 34.0 }).ToArray();
}
