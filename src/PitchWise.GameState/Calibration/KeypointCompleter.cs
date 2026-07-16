namespace PitchWise.GameState.Calibration;

/// <summary>
/// Fills in main keypoints
/// (1..30) missing from the keypoint model by intersecting the two pitch lines that
/// define them, and constructs the auxiliary keypoints (58..) from line-pair
/// intersections. Optionally normalizes coordinates to [0, 1] by the input size.
/// </summary>
public static class KeypointCompleter
{
    public static Dictionary<int, Keypoint> Complete(
        IReadOnlyDictionary<int, Keypoint> keypoints,
        IReadOnlyDictionary<int, LineSegment> lines,
        double width, double height, bool normalize = false)
    {
        var complete = keypoints.ToDictionary(kv => kv.Key, kv => kv.Value);
        var wExtra = 0.5 * width;
        var hExtra = 0.5 * height;

        for (var key = 1; key <= 30; key++)
        {
            if (complete.ContainsKey(key)) continue;
            if (TryIntersect(PitchGeometry.KeypointsLineList[key - 1], lines, out var p)
                && -wExtra < p.X && p.X < wExtra + width && -hExtra < p.Y && p.Y < hExtra + height)
                complete[key] = new Keypoint(Math.Round(p.X), Math.Round(p.Y), 1.0);
        }

        // Faithful to the Python loop `range(1, len(keypoint_aux_pair_list))`: the last
        // auxiliary keypoint (73) is never constructed.
        for (var key = 1; key < PitchGeometry.KeypointAuxPairList.Length; key++)
        {
            if (TryIntersect(PitchGeometry.KeypointAuxPairList[key - 1], lines, out var p)
                && -wExtra < p.X && p.X < wExtra + width && -hExtra < p.Y && p.Y < hExtra + height)
                complete[key + 57] = new Keypoint(Math.Round(p.X), Math.Round(p.Y), 1.0);
        }

        if (normalize)
            foreach (var key in complete.Keys.ToList())
                complete[key] = complete[key] with { X = complete[key].X / width, Y = complete[key].Y / height };

        return complete.OrderBy(kv => kv.Key).ToDictionary(kv => kv.Key, kv => kv.Value);
    }

    private static bool TryIntersect(
        string[] lineNames, IReadOnlyDictionary<int, LineSegment> lines, out (double X, double Y) point)
    {
        point = default;
        var id1 = Array.IndexOf(PitchGeometry.LinesList, lineNames[0]) + 1;
        var id2 = Array.IndexOf(PitchGeometry.LinesList, lineNames[1]) + 1;
        if (!lines.TryGetValue(id1, out var l1) || !lines.TryGetValue(id2, out var l2))
            return false;

        // Same computation as the Python line_intersection (linregress over the two
        // extremities is just the line through them, with the same 1e-7 nudges).
        var slope1 = (l1.P2.Y - l1.P1.Y) / (l1.P2.X + 1e-7 - l1.P1.X);
        var intercept1 = l1.P1.Y - slope1 * l1.P1.X;
        var slope2 = (l2.P2.Y - l2.P1.Y) / (l2.P2.X + 1e-7 - l2.P1.X);
        var intercept2 = l2.P1.Y - slope2 * l2.P1.X;

        var x = (intercept2 - intercept1) / (slope1 - slope2 + 1e-7);
        var y = slope1 * x + intercept1;
        if (double.IsNaN(x) || double.IsNaN(y) || double.IsInfinity(x) || double.IsInfinity(y))
            return false;
        point = (x, y);
        return true;
    }
}
