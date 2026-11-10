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
        Assert.Equal(2 * TableModel.RodsPerSide, table.Rods.Count);
    }

    [Fact]
    public void AllRodsLieOnThePlayfield()
    {
        var table = LoadShipped();
        Assert.All(table.Rods, r => Assert.InRange(r.X, Tolerance, table.Length - Tolerance));
    }

    [Fact]
    public void EachPlayerHasOneRodOfEachRole()
    {
        var table = LoadShipped();
        foreach (var side in Enum.GetValues<TableSide>())
            Assert.Equal(Enum.GetValues<RodRole>().Order(), table.RodsOf(side).Select(r => r.Role).Order());
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
}
