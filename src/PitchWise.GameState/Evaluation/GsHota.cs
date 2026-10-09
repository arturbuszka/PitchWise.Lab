using PitchWise.Tracking.Eval;

namespace PitchWise.GameState.Evaluation;

/// <summary>One evaluated detection: identity, pitch position and GS attributes.</summary>
public sealed record GsDetection(long TrackId, double X, double Y,
    string? Role = null, string? Team = null, string? Jersey = null);

public sealed record GsHotaOptions
{
    /// <summary>Sigma of the gaussian similarity kernel on pitch distance (meters). The official
    /// metric derives it from a 5 m distance tolerance (EVAL_DIST_TOL), i.e. ~2.04 m.</summary>
    public double Sigma { get; init; } = SigmaForTolerance(5.0);
    /// <summary>Role must match for a valid GT/pred pair.</summary>
    public bool UseRoles { get; init; } = true;
    /// <summary>Team must match — only enforced for players and goalkeepers.</summary>
    public bool UseTeams { get; init; } = true;
    /// <summary>Jersey number must match — only enforced for players (not goalkeepers).</summary>
    public bool UseJerseyNumbers { get; init; } = true;

    /// <summary>Sigma whose kernel drops to 0.05 at <paramref name="tolerance"/> meters
    /// (sn-trackeval calculate_sigma).</summary>
    public static double SigmaForTolerance(double tolerance)
        => Math.Sqrt(tolerance * tolerance / (-2 * Math.Log(0.05)));
}

/// <summary>
/// GS-HOTA — the SoccerNet Game State Reconstruction metric: HOTA evaluated on pitch
/// coordinates with a gaussian similarity kernel (5 m tolerance, sigma ~2.04 m) instead of IoU, where
/// pairs disagreeing on role/team/jersey get zero similarity and cannot be matched.
/// </summary>
public static class GsHota
{
    /// <param name="gtFrames">Ground truth detections per frame.</param>
    /// <param name="predFrames">Predicted detections per frame (same frame count).</param>
    public static HotaResult Evaluate(
        IReadOnlyList<IReadOnlyList<GsDetection>> gtFrames,
        IReadOnlyList<IReadOnlyList<GsDetection>> predFrames,
        GsHotaOptions? options = null)
    {
        options ??= new GsHotaOptions();
        if (gtFrames.Count != predFrames.Count)
            throw new ArgumentException("GT and prediction frame counts differ");

        // dense 0-based ids as required by the HOTA core
        var gtIdMap = BuildIdMap(gtFrames);
        var predIdMap = BuildIdMap(predFrames);

        var frames = new List<HotaFrame>(gtFrames.Count);
        for (var t = 0; t < gtFrames.Count; t++)
        {
            var gt = gtFrames[t];
            var pred = predFrames[t];
            var similarity = new double[gt.Count, pred.Count];
            for (var i = 0; i < gt.Count; i++)
                for (var j = 0; j < pred.Count; j++)
                    similarity[i, j] = Similarity(gt[i], pred[j], options);
            frames.Add(new HotaFrame(
                gt.Select(d => gtIdMap[d.TrackId]).ToArray(),
                pred.Select(d => predIdMap[d.TrackId]).ToArray(),
                similarity));
        }
        return Hota.Evaluate(frames, gtIdMap.Count, predIdMap.Count);
    }

    /// <summary>Gaussian kernel on pitch distance, gated by the enabled attributes.</summary>
    public static double Similarity(GsDetection gt, GsDetection pred, GsHotaOptions options)
    {
        if (Attributes(gt, options) != Attributes(pred, options))
            return 0;
        var dx = gt.X - pred.X;
        var dy = gt.Y - pred.Y;
        return Math.Exp(-(dx * dx + dy * dy) / (2 * options.Sigma * options.Sigma));
    }

    /// <summary>
    /// The attributes that must agree, as the official SoccerNetGS dataset class prepares them per
    /// detection: team only counts for players/goalkeepers and jersey only for players (when roles
    /// are disabled, team and jersey count for everyone); disabled attributes are dropped.
    /// </summary>
    private static (string? Role, string? Team, string? Jersey) Attributes(GsDetection d, GsHotaOptions options)
    {
        var role = options.UseRoles ? Normalize(d.Role) : null;
        var team = options.UseTeams && (!options.UseRoles || role is "player" or "goalkeeper") ? Normalize(d.Team) : null;
        var jersey = options.UseJerseyNumbers && (!options.UseRoles || role is "player") ? Normalize(d.Jersey) : null;
        return (role, team, jersey);
    }

    private static string? Normalize(string? value) => string.IsNullOrEmpty(value) ? null : value;

    private static Dictionary<long, int> BuildIdMap(IReadOnlyList<IReadOnlyList<GsDetection>> frames)
    {
        var map = new Dictionary<long, int>();
        foreach (var frame in frames)
            foreach (var detection in frame)
                if (!map.ContainsKey(detection.TrackId))
                    map[detection.TrackId] = map.Count;
        return map;
    }
}
