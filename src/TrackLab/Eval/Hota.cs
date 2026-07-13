namespace TrackLab.Eval;

/// <summary>One frame of evaluation input: which GT/pred identities appear and their
/// pairwise similarity in [0, 1] (rows = GT order, cols = pred order).</summary>
public sealed record HotaFrame(int[] GtIds, int[] PredIds, double[,] Similarity);

public sealed record HotaResult
{
    /// <summary>Final score: mean over the 19 alpha thresholds of sqrt(DetA * AssA).</summary>
    public required double Hota { get; init; }
    public required double DetA { get; init; }
    public required double AssA { get; init; }
    /// <summary>Mean similarity of true positives (localization accuracy).</summary>
    public required double LocA { get; init; }
    public required double[] HotaPerAlpha { get; init; }
}

/// <summary>
/// Port of the HOTA metric (TrackEval hota.py): detection accuracy and association
/// accuracy combined over 19 similarity thresholds. Identity-agnostic — the caller
/// supplies per-frame similarity matrices (IoU in image space, or the GS kernel).
/// </summary>
public static class Hota
{
    private const double Eps = 1e-10;
    public static readonly double[] Alphas =
        Enumerable.Range(1, 19).Select(i => i * 0.05).ToArray();

    /// <param name="numGtIds">Total distinct GT identities (ids are 0..numGtIds-1).</param>
    /// <param name="numPredIds">Total distinct predicted identities.</param>
    public static HotaResult Evaluate(IReadOnlyList<HotaFrame> frames, int numGtIds, int numPredIds)
    {
        if (numGtIds == 0 || numPredIds == 0)
        {
            var zero = numGtIds == 0 && numPredIds == 0 ? 1.0 : 0.0;
            return new HotaResult
            {
                Hota = zero, DetA = zero, AssA = zero, LocA = zero,
                HotaPerAlpha = Alphas.Select(_ => zero).ToArray(),
            };
        }

        // First pass: global alignment score (Jaccard of potential matches per id pair).
        var potentialMatches = new double[numGtIds, numPredIds];
        var gtIdCount = new double[numGtIds];
        var predIdCount = new double[numPredIds];
        foreach (var frame in frames)
        {
            var (gt, pred, sim) = (frame.GtIds, frame.PredIds, frame.Similarity);
            var rowSum = new double[gt.Length];
            var colSum = new double[pred.Length];
            for (var i = 0; i < gt.Length; i++)
                for (var j = 0; j < pred.Length; j++)
                {
                    rowSum[i] += sim[i, j];
                    colSum[j] += sim[i, j];
                }
            for (var i = 0; i < gt.Length; i++)
                for (var j = 0; j < pred.Length; j++)
                {
                    var denom = rowSum[i] + colSum[j] - sim[i, j];
                    if (denom > Eps)
                        potentialMatches[gt[i], pred[j]] += sim[i, j] / denom;
                }
            foreach (var g in gt) gtIdCount[g]++;
            foreach (var p in pred) predIdCount[p]++;
        }
        var globalAlignment = new double[numGtIds, numPredIds];
        for (var g = 0; g < numGtIds; g++)
            for (var p = 0; p < numPredIds; p++)
                globalAlignment[g, p] = potentialMatches[g, p]
                    / (gtIdCount[g] + predIdCount[p] - potentialMatches[g, p]);

        // Second pass: per-alpha Hungarian matching weighted by the alignment score.
        var alphaCount = Alphas.Length;
        var tp = new double[alphaCount];
        var fn = new double[alphaCount];
        var fp = new double[alphaCount];
        var locSum = new double[alphaCount];
        var matchesCount = new double[alphaCount][,];
        for (var a = 0; a < alphaCount; a++)
            matchesCount[a] = new double[numGtIds, numPredIds];

        foreach (var frame in frames)
        {
            var (gt, pred, sim) = (frame.GtIds, frame.PredIds, frame.Similarity);
            if (gt.Length == 0 || pred.Length == 0)
            {
                for (var a = 0; a < alphaCount; a++)
                {
                    fn[a] += gt.Length;
                    fp[a] += pred.Length;
                }
                continue;
            }

            var score = new float[gt.Length, pred.Length];
            for (var i = 0; i < gt.Length; i++)
                for (var j = 0; j < pred.Length; j++)
                    score[i, j] = (float)(globalAlignment[gt[i], pred[j]] * sim[i, j]);
            var assignment = SolveMaximize(score);

            for (var a = 0; a < alphaCount; a++)
            {
                var matched = 0;
                for (var i = 0; i < gt.Length; i++)
                {
                    var j = assignment[i];
                    if (j >= 0 && sim[i, j] >= Alphas[a] - Eps)
                    {
                        matched++;
                        matchesCount[a][gt[i], pred[j]]++;
                        locSum[a] += sim[i, j];
                    }
                }
                tp[a] += matched;
                fn[a] += gt.Length - matched;
                fp[a] += pred.Length - matched;
            }
        }

        var hotaPerAlpha = new double[alphaCount];
        double detASum = 0, assASum = 0, locASum = 0;
        for (var a = 0; a < alphaCount; a++)
        {
            var detA = tp[a] / Math.Max(Eps, tp[a] + fn[a] + fp[a]);
            var assA = 0.0;
            if (tp[a] > 0)
            {
                for (var g = 0; g < numGtIds; g++)
                    for (var p = 0; p < numPredIds; p++)
                        if (matchesCount[a][g, p] > 0)
                            assA += matchesCount[a][g, p] * (matchesCount[a][g, p]
                                / (gtIdCount[g] + predIdCount[p] - matchesCount[a][g, p]));
                assA /= tp[a];
            }
            hotaPerAlpha[a] = Math.Sqrt(detA * assA);
            detASum += detA;
            assASum += assA;
            locASum += tp[a] > 0 ? locSum[a] / tp[a] : 0;
        }

        return new HotaResult
        {
            Hota = hotaPerAlpha.Average(),
            DetA = detASum / alphaCount,
            AssA = assASum / alphaCount,
            LocA = locASum / alphaCount,
            HotaPerAlpha = hotaPerAlpha,
        };
    }

    /// <summary>Maximum-score assignment; returns per-row column or -1 (zero-score pairs stay unmatched
    /// only via the alpha gate, like TrackEval).</summary>
    private static int[] SolveMaximize(float[,] score)
    {
        var rows = score.GetLength(0);
        var cols = score.GetLength(1);
        var max = 0f;
        foreach (var v in score) max = Math.Max(max, v);
        var cost = new float[rows, cols];
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < cols; j++)
                cost[i, j] = max - score[i, j];
        return Hungarian.Solve(cost);
    }
}
