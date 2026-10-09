using PitchWise.GameState.Evaluation;

namespace PitchWise.GameState.Tests;

public class GsHotaTests
{
    private static List<IReadOnlyList<GsDetection>> Frames(params IReadOnlyList<GsDetection>[] frames)
        => frames.ToList();

    private static GsDetection Player(long id, double x, double y, string team = "left", string jersey = "10")
        => new(id, x, y, "player", team, jersey);

    [Fact]
    public void PerfectTracking_ScoresOne()
    {
        var frames = Enumerable.Range(0, 10)
            .Select(t => (IReadOnlyList<GsDetection>)new[]
            {
                Player(1, t, 0), Player(2, t, 10, "right", "7"),
            })
            .ToList();

        var result = GsHota.Evaluate(frames, frames);
        Assert.Equal(1.0, result.Hota, 6);
        Assert.Equal(1.0, result.DetA, 6);
        Assert.Equal(1.0, result.AssA, 6);
    }

    [Fact]
    public void WrongTeam_PreventsMatching()
    {
        var gt = Frames(new[] { Player(1, 0, 0, team: "left") });
        var pred = Frames(new[] { Player(1, 0, 0, team: "right") });
        var result = GsHota.Evaluate(gt, pred);
        Assert.Equal(0.0, result.Hota, 6);
    }

    [Fact]
    public void WrongJersey_PreventsMatching_ButOnlyForPlayers()
    {
        var gtReferee = Frames(new[] { new GsDetection(1, 0, 0, "referee", null, "99") });
        var predReferee = Frames(new[] { new GsDetection(1, 0, 0, "referee", null, null) });
        // jersey mismatch irrelevant for referees
        Assert.Equal(1.0, GsHota.Evaluate(gtReferee, predReferee).Hota, 6);

        var gtPlayer = Frames(new[] { Player(1, 0, 0, jersey: "10") });
        var predPlayer = Frames(new[] { Player(1, 0, 0, jersey: "9") });
        Assert.Equal(0.0, GsHota.Evaluate(gtPlayer, predPlayer).Hota, 6);
    }

    [Fact]
    public void Sigma_FollowsTheOfficialFiveMeterTolerance()
    {
        // sn-trackeval: EVAL_DIST_TOL = 5 -> similarity drops to 0.05 at 5 m
        var sigma = new GsHotaOptions().Sigma;
        Assert.Equal(2.0427, sigma, 4);
        Assert.Equal(0.05, GsHota.Similarity(Player(1, 0, 0), Player(1, 5, 0), new GsHotaOptions()), 6);
    }

    [Fact]
    public void PositionError_DegradesSimilarityWithGaussianKernel()
    {
        // 2.5 m error -> sim = exp(-0.5 * (2.5 / sigma)^2) ~= 0.473:
        // matched for alphas up to that value, unmatched above.
        var gt = Frames(new[] { Player(1, 0, 0) });
        var pred = Frames(new[] { Player(1, 2.5, 0) });
        var result = GsHota.Evaluate(gt, pred);

        var sigma = new GsHotaOptions().Sigma;
        var similarity = Math.Exp(-0.5 * Math.Pow(2.5 / sigma, 2));
        var expectedMatchedAlphas = PitchWise.Tracking.Eval.Hota.Alphas.Count(a => similarity >= a - 1e-10);
        var expectedHota = (double)expectedMatchedAlphas / PitchWise.Tracking.Eval.Hota.Alphas.Length;
        Assert.Equal(expectedHota, result.Hota, 3);
    }

    [Fact]
    public void GoalkeeperJersey_IsIgnored_ButTeamStillCounts()
    {
        // official metric: jersey only for role == player; team for players and goalkeepers
        var gt = Frames(new[] { new GsDetection(1, 0, 0, "goalkeeper", "left", "1") });
        var wrongJersey = Frames(new[] { new GsDetection(1, 0, 0, "goalkeeper", "left", "12") });
        var wrongTeam = Frames(new[] { new GsDetection(1, 0, 0, "goalkeeper", "right", "1") });
        Assert.Equal(1.0, GsHota.Evaluate(gt, wrongJersey).Hota, 6);
        Assert.Equal(0.0, GsHota.Evaluate(gt, wrongTeam).Hota, 6);
    }

    [Fact]
    public void IdentitySwitch_LowersAssociationNotDetection()
    {
        // two GT tracks; prediction swaps ids halfway through
        var gt = new List<IReadOnlyList<GsDetection>>();
        var pred = new List<IReadOnlyList<GsDetection>>();
        for (var t = 0; t < 20; t++)
        {
            gt.Add(new[] { Player(1, 0, 0), Player(2, 20, 0, "right", "7") });
            var (idA, idB) = t < 10 ? (1L, 2L) : (2L, 1L);
            pred.Add(new[]
            {
                Player(idA, 0, 0),
                new GsDetection(idB, 20, 0, "player", "right", "7"),
            });
        }
        var result = GsHota.Evaluate(gt, pred);
        Assert.Equal(1.0, result.DetA, 6);       // every detection is still correct
        Assert.True(result.AssA is > 0.2 and < 0.8, $"AssA {result.AssA}");
        Assert.True(result.Hota < 0.8, $"HOTA {result.Hota}");
    }

    [Fact]
    public void MissedAndFalseDetections_LowerDetA()
    {
        var gt = Frames(
            new[] { Player(1, 0, 0), Player(2, 20, 0, "right", "7") },
            new[] { Player(1, 0, 0), Player(2, 20, 0, "right", "7") });
        var pred = Frames(
            new[] { Player(1, 0, 0) },                                     // one miss
            new[] { Player(1, 0, 0), Player(2, 20, 0, "right", "7"), Player(3, 40, 0, "left", "5") }); // one FP
        var result = GsHota.Evaluate(gt, pred);
        // TP=3, FN=1, FP=1 at every alpha -> DetA = 3/5
        Assert.Equal(0.6, result.DetA, 6);
    }
}
