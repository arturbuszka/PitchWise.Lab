using Microsoft.Extensions.Logging;

namespace TrackLab.Pipeline;

/// <summary>Port of tracklab.pipeline.module.Pipeline — ordered module list with
/// static validation that every module's inputs are produced upstream.</summary>
public sealed class Pipeline
{
    public IReadOnlyList<Module> Models { get; }
    private readonly ILogger? _log;

    public Pipeline(IEnumerable<Module> models, ILogger? log = null)
    {
        Models = models.Where(m => m.Name != "skip").ToList();
        _log = log;
        _log?.LogInformation("Pipeline: {Pipeline}", ToString());
    }

    /// <param name="loadColumns">Columns already available per level ("image"/"detection"),
    /// e.g. provided by the dataset or a loaded tracker state.</param>
    public void Validate(IReadOnlyDictionary<string, ISet<string>> loadColumns)
    {
        var columns = loadColumns.ToDictionary(kv => kv.Key, kv => new HashSet<string>(kv.Value));
        foreach (var level in new[] { "image", "detection" })
        {
            columns.TryAdd(level, new HashSet<string>());
            foreach (var model in Models)
            {
                var needed = model.GetInputColumns(level);
                if (!needed.All(columns[level].Contains))
                    throw new InvalidOperationException(
                        $"The {model.Name} model doesn't have all the input needed, " +
                        $"needed [{string.Join(", ", needed)}], provided [{string.Join(", ", columns[level])}]");
                columns[level].UnionWith(model.GetOutputColumns(level));
            }
        }
        _log?.LogInformation("Pipeline has been validated");
    }

    public bool IsEmpty => Models.Count == 0;

    public override string ToString() => string.Join(" -> ", Models.Select(m => m.Name));
}
