using PitchWise.Tracking.Datastruct;

namespace PitchWise.Tracking.Pipeline;

/// <summary>
/// Declares the detection/image columns a module consumes and produces so the
/// pipeline can be validated before running.
/// </summary>
public abstract class Module
{
    /// <summary>Columns required on the detections (key "detection") or image metadata
    /// (key "image") before this module runs.</summary>
    public abstract IReadOnlyDictionary<string, string[]> InputColumns { get; }

    /// <summary>Columns this module adds, per level ("detection" / "image").</summary>
    public abstract IReadOnlyDictionary<string, string[]> OutputColumns { get; }

    public virtual string Name => GetType().Name;

    /// <summary>"image", "detection" or "video" — drives how the engine feeds the module.</summary>
    public abstract string Level { get; }

    public virtual void Reset() { }

    public string[] GetInputColumns(string level)
        => InputColumns.TryGetValue(level, out var cols) ? cols : Array.Empty<string>();

    public string[] GetOutputColumns(string level)
        => OutputColumns.TryGetValue(level, out var cols) ? cols : Array.Empty<string>();

    public void ValidateInput(DataFrame detections)
    {
        foreach (var col in GetInputColumns("detection"))
            if (!detections.Columns.Contains(col) && !detections.IsEmpty)
                throw new InvalidOperationException($"The input detections should contain {col}.");
    }

    public void ValidateOutput(DataFrame detections)
    {
        foreach (var col in GetOutputColumns("detection"))
            if (!detections.Columns.Contains(col) && !detections.IsEmpty)
                throw new InvalidOperationException($"The output detections should contain {col}.");
    }
}

/// <summary>Placeholder module that is dropped from the pipeline (a no-op skip).</summary>
public sealed class Skip : Module
{
    public override IReadOnlyDictionary<string, string[]> InputColumns { get; } = new Dictionary<string, string[]>();
    public override IReadOnlyDictionary<string, string[]> OutputColumns { get; } = new Dictionary<string, string[]>();
    public override string Name => "skip";
    public override string Level => "skip";
}
