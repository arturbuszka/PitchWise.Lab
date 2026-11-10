using System.Text.Json;
using System.Text.Json.Serialization;

namespace PitchWise.Foosball;

/// <summary>The player whose goal is at x = 0 (Left) or at x = <see cref="TableModel.Length"/> (Right).</summary>
public enum TableSide { Left, Right }

public enum RodRole { Goalkeeper, Defense, Midfield, Attack }

/// <summary>One rod. Rods run across the table (along y), so a rod is fixed by its x alone.</summary>
/// <param name="FigureSpacing">Centre-to-centre distance between neighbouring figures, in metres.</param>
public record RodSpec(RodRole Role, TableSide Owner, double X, int FigureCount, double FigureSpacing);

/// <summary>
/// The table template: the single source of truth for playfield geometry, like
/// <c>PitchModel</c> for the football pitch, and in the same convention so the engine and the
/// replay tooling need no special case: origin at the top-left corner of the playfield,
/// x along the length (Left goal → Right goal), y across, in metres.
///
/// Values come from a config file rather than constants because tables differ between
/// manufacturers, and the bar's table will be measured once and written down there.
/// </summary>
public record TableModel(double Length, double Width, double GoalWidth, IReadOnlyList<RodSpec> Rods)
{
    /// <summary>Rods per player: goalkeeper, defense, midfield, attack.</summary>
    public const int RodsPerSide = 4;

    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web)
    {
        // Comments let the config say which values are still unmeasured.
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true,
        Converters = { new JsonStringEnumConverter() },
    };

    /// <summary>Reads and validates a table config, so a bad file fails at startup rather than mid-match.</summary>
    public static TableModel Load(string path)
    {
        var model = JsonSerializer.Deserialize<TableModel>(File.ReadAllText(path), JsonOptions)
            ?? throw new InvalidDataException($"Empty table config {path}");
        model.Validate();
        return model;
    }

    public IEnumerable<RodSpec> RodsOf(TableSide side) => Rods.Where(r => r.Owner == side);

    /// <summary>The x a rod would have on the opposite side; the table is point-symmetric along x.</summary>
    public double Mirror(RodSpec rod) => Length - rod.X;

    /// <summary>Throws <see cref="InvalidDataException"/> when the geometry cannot be a real table.</summary>
    public void Validate()
    {
        if (Length <= 0 || Width <= 0)
            throw new InvalidDataException($"Table size must be positive, got {Length} × {Width}");
        if (GoalWidth <= 0 || GoalWidth >= Width)
            throw new InvalidDataException($"Goal width {GoalWidth} must be within table width {Width}");
        if (Rods is null || Rods.Count != 2 * RodsPerSide)
            throw new InvalidDataException($"Expected {2 * RodsPerSide} rods, got {Rods?.Count ?? 0}");

        foreach (var side in Enum.GetValues<TableSide>())
        {
            var roles = RodsOf(side).Select(r => r.Role).ToHashSet();
            if (roles.Count != RodsPerSide || RodsOf(side).Count() != RodsPerSide)
                throw new InvalidDataException($"Player {side} must have exactly one rod of each role");
        }

        foreach (var rod in Rods)
        {
            if (rod.X <= 0 || rod.X >= Length)
                throw new InvalidDataException($"{rod.Owner} {rod.Role} rod at x={rod.X} lies outside the table");
            if (rod.FigureCount < 1 || rod.FigureSpacing < 0)
                throw new InvalidDataException($"{rod.Owner} {rod.Role} rod has invalid figures");
            if ((rod.FigureCount - 1) * rod.FigureSpacing >= Width)
                throw new InvalidDataException($"{rod.Owner} {rod.Role} figures span wider than the table");
        }
    }
}
