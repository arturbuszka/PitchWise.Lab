using PitchWise.App.Core.Analysis;
using PitchWise.Tracking.Engine;

namespace PitchWise.App.Core.Tests;

public class ModuleCostModelTests
{
    /// <summary>The real pipeline, in order, as PipelineConfig assembles it.</summary>
    private static readonly string[] FullPipeline =
    [
        "YoloOnnxDetector", "PrtReidOnnx", "StrongSortModule", "NbjwCalibKeypoints",
        "NbjwCalib", "MmocrOnnx", "MajorityVoteTracklet", "TrackletTeamClustering",
        "TrackletTeamSideLabeling",
    ];

    private static TrackingProgress Tick(int moduleIndex, double moduleFraction, string? name = null)
    {
        var items = 100;
        return new TrackingProgress(1L, name ?? FullPipeline[moduleIndex], moduleIndex,
            FullPipeline.Length, (long)(items * moduleFraction), items);
    }

    [Fact]
    public void Fraction_RunsFromZeroToOneAcrossTheWholePipeline()
    {
        var costs = new ModuleCostModel(FullPipeline);

        Assert.Equal(0, costs.RawFraction(Tick(0, 0)));
        Assert.Equal(1, costs.RawFraction(Tick(FullPipeline.Length - 1, 1)), precision: 6);
    }

    [Fact]
    public void Fraction_WeightsExpensiveModulesAboveCheapOnes()
    {
        var costs = new ModuleCostModel(FullPipeline);

        // finishing the detector alone should account for far more of the bar than finishing
        // the three tracklet modules that follow it, which are near-instant
        var afterYolo = costs.RawFraction(Tick(0, 1));
        var lastThree = costs.RawFraction(Tick(8, 1)) - costs.RawFraction(Tick(6, 0));

        Assert.True(afterYolo > 0.3, $"detector only worth {afterYolo:P0} of the run");
        Assert.True(afterYolo > lastThree * 5,
            $"detector {afterYolo:P0} should dwarf the tracklet modules' {lastThree:P0}");
    }

    [Fact]
    public void Advance_NeverGoesBackwardsEvenWhenTheWeightsAreWrong()
    {
        var costs = new ModuleCostModel(FullPipeline);

        var high = costs.Advance(Tick(5, 1.0));
        // a later tick that the weights happen to score lower must not rewind the bar
        var next = costs.Advance(Tick(0, 0.0));

        Assert.Equal(high, next);
        Assert.True(next > 0);
    }

    [Fact]
    public void Advance_IsMonotonicAcrossAFullRun()
    {
        var costs = new ModuleCostModel(FullPipeline);
        var last = 0.0;

        for (var module = 0; module < FullPipeline.Length; module++)
            foreach (var fraction in new[] { 0, 0.25, 0.5, 0.75, 1.0 })
            {
                var value = costs.Advance(Tick(module, fraction));
                Assert.True(value >= last, $"went backwards at module {module} ({fraction:P0})");
                Assert.InRange(value, 0, 1);
                last = value;
            }

        Assert.Equal(1, last, precision: 6);
    }

    [Fact]
    public void UnknownModules_GetTheCheapDefaultRatherThanFailing()
    {
        // a renamed or newly added module must not break the bar
        var costs = new ModuleCostModel(["YoloOnnxDetector", "SomeBrandNewModule"]);

        Assert.Equal(1, ModuleCostModel.WeightOf("SomeBrandNewModule"));
        Assert.True(ModuleCostModel.WeightOf("YoloOnnxDetector") > 1);
        Assert.InRange(costs.RawFraction(Tick(1, 0.5, "SomeBrandNewModule")), 0, 1);
    }

    [Fact]
    public void ToAnalysisProgress_LabelsTheModuleAndWithholdsEtaUntilItMeansSomething()
    {
        var costs = new ModuleCostModel(FullPipeline);

        var early = costs.ToAnalysisProgress(Tick(0, 0.01), TimeSpan.FromSeconds(3));
        Assert.Equal(AnalysisStage.Tracking, early.Stage);
        Assert.Equal("YoloOnnxDetector (1/9)", early.Detail);
        Assert.Null(early.Eta);   // below 5%: model loading still dominates

        var later = costs.ToAnalysisProgress(Tick(1, 0.5), TimeSpan.FromSeconds(60));
        Assert.NotNull(later.Eta);
        Assert.True(later.Eta!.Value > TimeSpan.Zero);

        var done = costs.ToAnalysisProgress(Tick(8, 1.0), TimeSpan.FromSeconds(120));
        Assert.Null(done.Eta);    // finished: nothing left to estimate
    }

    [Fact]
    public void ToStageProgress_KeepsTheBarMonotonicAcrossStages()
    {
        var costs = new ModuleCostModel(FullPipeline);
        costs.Advance(Tick(8, 1.0));

        // a stage claiming less than tracking already reached must not rewind
        var observing = costs.ToStageProgress(AnalysisStage.Observing, "stats", 0.9, TimeSpan.Zero);

        Assert.Equal(AnalysisStage.Observing, observing.Stage);
        Assert.Equal(1.0, observing.Fraction);
    }
}
