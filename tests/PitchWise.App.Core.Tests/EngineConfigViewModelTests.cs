using PitchWise.App.Core.Tuning;
using PitchWise.Engine;

namespace PitchWise.App.Core.Tests;

public class EngineConfigViewModelTests
{
    [Fact]
    public void Knobs_StartAtTheEnginesOwnDefaults()
    {
        var vm = new EngineConfigViewModel(() => { });

        // an untouched panel must describe exactly the config the engine would use on its own,
        // or opening the tuning tab would silently change the results
        Assert.Equal(new EngineConfig(), vm.ToConfig());
        Assert.False(vm.IsModified);
        Assert.All(vm.Knobs, k => Assert.False(k.IsModified));
    }

    [Fact]
    public void ToConfig_MapsEveryKnobToItsOwnField()
    {
        var vm = new EngineConfigViewModel(() => { });

        // move each knob to a value nothing else has, so a mis-wired name cannot pass unnoticed
        for (var i = 0; i < vm.Knobs.Count; i++)
        {
            var knob = vm.Knobs[i];
            knob.Value = knob.Minimum + (knob.Maximum - knob.Minimum) * (i + 1) / (vm.Knobs.Count + 2);
        }
        var config = vm.ToConfig();

        var byName = vm.Knobs.ToDictionary(k => k.Name, k => k.Value);
        Assert.Equal(byName["Min confidence"], config.MinBallConfidence);
        Assert.Equal(byName["Max speed"], config.MaxBallSpeed);
        Assert.Equal(byName["Max coast"], config.MaxCoastSeconds);
        Assert.Equal(byName["Process noise"], config.BallProcessNoise);
        Assert.Equal(byName["Measurement noise"], config.BallMeasurementNoise);
        Assert.Equal(byName["Stick radius"], config.BallStickRadius);
        Assert.Equal(byName["Stick noise"], config.BallStickNoise);
        Assert.Equal(byName["Max stick"], config.MaxStickSeconds);
        Assert.Equal(byName["Stick confidence"], config.BallStickConfidence);
        Assert.Equal(byName["Control radius"], config.ControlRadius);
        Assert.Equal(byName["Capture dwell"], config.CaptureDwell);
        Assert.Equal(byName["Release grace"], config.ReleaseGrace);
        Assert.Equal(byName["Contested radius"], config.ContestedRadius);
        Assert.Equal(byName["Contest margin"], config.ContestMargin);
        Assert.Equal(byName["Pass max flight"], config.PassMaxFlightSeconds);
        Assert.Equal(byName["Pass min distance"], config.PassMinDistance);
        Assert.Equal(byName["Event cooldown"], config.EventCooldownSeconds);
    }

    [Fact]
    public void EveryTunableEngineFieldHasAKnob()
    {
        var vm = new EngineConfigViewModel(() => { });

        // the pitch is a fact about the world, not a threshold: changing it would invalidate the
        // recorded positions rather than re-interpret them
        var notTunable = new[] { nameof(EngineConfig.PitchLength), nameof(EngineConfig.PitchWidth) };
        var fields = typeof(EngineConfig).GetProperties()
            .Where(p => p.PropertyType == typeof(double) && !notTunable.Contains(p.Name))
            .Select(p => p.Name)
            .ToList();

        // if the engine grows a threshold, this fails until the panel exposes it
        Assert.Equal(fields.Count, vm.Knobs.Count);
    }

    [Fact]
    public void Knobs_ReportWhenTheyHaveMovedAndCanBeResetTogether()
    {
        var vm = new EngineConfigViewModel(() => { });
        var control = vm.Knobs.First(k => k.Name == "Control radius");

        control.Value = 3.0;

        Assert.True(control.IsModified);
        Assert.True(vm.IsModified);
        Assert.Equal(3.0, vm.ToConfig().ControlRadius);
        Assert.Equal("3 m", control.ValueText);

        vm.ResetAll();

        Assert.False(vm.IsModified);
        Assert.Equal(new EngineConfig(), vm.ToConfig());
    }

    [Fact]
    public void ChangingAKnob_NotifiesOnceSoARetuneCanBeDebounced()
    {
        var changes = 0;
        var vm = new EngineConfigViewModel(() => changes++);

        vm.Knobs.First(k => k.Name == "Control radius").Value = 2.0;
        Assert.Equal(1, changes);

        // setting the same value again is not a change
        vm.Knobs.First(k => k.Name == "Control radius").Value = 2.0;
        Assert.Equal(1, changes);
    }

    [Fact]
    public void Knobs_CarryTheReasoningFromTheEnginesDocComments()
    {
        var vm = new EngineConfigViewModel(() => { });

        // the tooltips are the best tuning documentation there is; they must survive
        Assert.All(vm.Knobs, k => Assert.False(string.IsNullOrWhiteSpace(k.Description)));

        var stick = vm.Knobs.First(k => k.Name == "Stick radius");
        Assert.Contains("hidden by the carrier's own legs", stick.Description);

        var coast = vm.Knobs.First(k => k.Name == "Max coast");
        Assert.Contains("median", coast.Description);
    }

    [Fact]
    public void Knobs_AreGroupedForTheUiWithoutLosingAny()
    {
        var vm = new EngineConfigViewModel(() => { });

        Assert.Equal(vm.Knobs.Count, vm.Groups.Sum(g => g.Knobs.Count));
        Assert.Equal(["Ball filter", "Ball carry", "Possession", "Rules"],
            vm.Groups.Select(g => g.Name));
    }
}
