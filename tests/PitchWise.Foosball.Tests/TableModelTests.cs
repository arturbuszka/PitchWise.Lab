namespace PitchWise.Foosball.Tests;

/// <summary>Checks the shipped table config is a geometrically sane, symmetric table.</summary>
public class TableModelTests
{
    private const double Tolerance = 0.001; // 1 mm

    private static TableModel LoadShipped() =>
        TableModel.Load(Path.Combine(AppContext.BaseDirectory, "table.json"));

    [Fact]
    public void ShippedConfigLoadsAndValidates()
    {
        var table = LoadShipped();
        Assert.NotEmpty(table.Rods);
    }

    [Fact]
    public void AllRodsLieOnThePlayfield()
    {
        var table = LoadShipped();
        Assert.All(table.Rods, r => Assert.InRange(r.X, Tolerance, table.Length - Tolerance));
    }

    [Fact]
    public void BothPlayersHaveTheSameRods()
    {
        var table = LoadShipped();
        Assert.Equal(
            table.RodsOf(TableSide.Left).Select(r => r.Role).Order(),
            table.RodsOf(TableSide.Right).Select(r => r.Role).Order());
    }

    [Fact]
    public void RodsAreSymmetricBetweenPlayers()
    {
        var table = LoadShipped();
        foreach (var rod in table.RodsOf(TableSide.Left))
        {
            var twin = Assert.Single(table.RodsOf(TableSide.Right), r => r.Role == rod.Role);
            Assert.Equal(table.Mirror(rod), twin.X, Tolerance);
            Assert.Equal(rod.FigureCount, twin.FigureCount);
            Assert.Equal(rod.FigureSpacing, twin.FigureSpacing, Tolerance);
        }
    }

    [Fact]
    public void RodOutsideTableIsRejected()
    {
        var table = LoadShipped();
        var rods = table.Rods.ToList();
        rods[0] = rods[0] with { X = table.Length + 0.1 };
        Assert.Throws<InvalidDataException>(() => (table with { Rods = rods }).Validate());
    }

    [Fact]
    public void MissingRodIsRejected()
    {
        var table = LoadShipped();
        Assert.Throws<InvalidDataException>(() => (table with { Rods = table.Rods.Skip(1).ToList() }).Validate());
    }

    [Fact]
    public void DuplicateRoleIsRejected()
    {
        var table = LoadShipped();
        var rods = table.Rods.ToList();
        var left = rods.FindAll(r => r.Owner == TableSide.Left);
        rods[rods.IndexOf(left[1])] = left[1] with { Role = left[0].Role };
        Assert.Throws<InvalidDataException>(() => (table with { Rods = rods }).Validate());
    }

    [Fact]
    public void FullSizeEightRodTableIsAccepted()
    {
        RodSpec Rod(RodRole role, TableSide side, double x, int n, double s) => new(role, side, x, n, s);
        var table = new TableModel(1.20, 0.68, 0.20,
        [
            Rod(RodRole.Goalkeeper, TableSide.Left, 0.075, 1, 0), Rod(RodRole.Defense, TableSide.Left, 0.225, 2, 0.24),
            Rod(RodRole.Attack, TableSide.Right, 0.375, 3, 0.20), Rod(RodRole.Midfield, TableSide.Left, 0.525, 5, 0.12),
            Rod(RodRole.Midfield, TableSide.Right, 0.675, 5, 0.12), Rod(RodRole.Attack, TableSide.Left, 0.825, 3, 0.20),
            Rod(RodRole.Defense, TableSide.Right, 0.975, 2, 0.24), Rod(RodRole.Goalkeeper, TableSide.Right, 1.125, 1, 0),
        ]);
        table.Validate();
    }
}
