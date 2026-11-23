using System.Text.Json;
using System.Text.Json.Serialization;
using PitchWise.Engine;

namespace PitchWise.Foosball;

/// <summary>Reads the engine's parameters for a foosball table; its defaults describe a football pitch.</summary>
public static class FoosballEngineConfig
{
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web)
    {
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true,
        // A misspelt key would silently leave the football default in place.
        UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow,
    };

    /// <summary>Pitch size always comes from <paramref name="table"/>, so it is written down only once.</summary>
    public static EngineConfig Load(string path, TableModel table)
    {
        var config = JsonSerializer.Deserialize<EngineConfig>(File.ReadAllText(path), JsonOptions)
            ?? throw new InvalidDataException($"Empty engine config {path}");
        return config with { PitchLength = table.Length, PitchWidth = table.Width };
    }
}
