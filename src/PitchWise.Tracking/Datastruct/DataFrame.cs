namespace PitchWise.Tracking.Datastruct;

/// <summary>
/// A single row of a <see cref="DataFrame"/> — the C# counterpart of a pandas Series
/// (dynamic, string-keyed values with a row index).
/// </summary>
public sealed class Series
{
    public long Index { get; }
    private readonly Dictionary<string, object?> _values;

    public Series(long index, Dictionary<string, object?>? values = null)
    {
        Index = index;
        _values = values ?? new Dictionary<string, object?>();
    }

    public object? this[string column]
    {
        get => _values.TryGetValue(column, out var v) ? v : null;
        set => _values[column] = value;
    }

    public bool Has(string column) => _values.ContainsKey(column) && _values[column] is not null;

    public T Get<T>(string column) => (T)_values[column]!;

    public T? GetOrDefault<T>(string column, T? fallback = default)
        => _values.TryGetValue(column, out var v) && v is T t ? t : fallback;

    public IReadOnlyCollection<string> Columns => _values.Keys;

    public IReadOnlyDictionary<string, object?> Values => _values;
}

/// <summary>
/// Minimal pandas.DataFrame stand-in covering the operations the pipeline relies on:
/// dynamic columns, row index, filtering, and merge-by-index where module outputs
/// override previous rows with the same index.
/// </summary>
public sealed class DataFrame
{
    private readonly Dictionary<long, Series> _rows = new();
    private readonly List<long> _order = new();
    private readonly HashSet<string> _columns = new();

    public static DataFrame Empty() => new();

    public int Count => _rows.Count;
    public bool IsEmpty => _rows.Count == 0;
    public IReadOnlyCollection<string> Columns => _columns;
    public IEnumerable<Series> Rows => _order.Select(i => _rows[i]);
    public IEnumerable<long> Index => _order;

    public Series this[long index] => _rows[index];
    public bool ContainsIndex(long index) => _rows.ContainsKey(index);

    public long NextIndex { get; set; }

    public Series NewRow()
    {
        var row = new Series(NextIndex++);
        Upsert(row);
        return row;
    }

    /// <summary>Adds a row, or overrides columns of the existing row with the same index
    /// (the aggregation semantics of the engine merge step).</summary>
    public void Upsert(Series row)
    {
        if (_rows.TryGetValue(row.Index, out var existing))
        {
            foreach (var (k, v) in row.Values)
            {
                existing[k] = v;
                _columns.Add(k);
            }
        }
        else
        {
            _rows[row.Index] = row;
            _order.Add(row.Index);
            foreach (var k in row.Columns) _columns.Add(k);
            if (row.Index >= NextIndex) NextIndex = row.Index + 1;
        }
    }

    public void Merge(DataFrame other)
    {
        foreach (var row in other.Rows) Upsert(row);
    }

    public void Merge(IEnumerable<Series> rows)
    {
        foreach (var row in rows) Upsert(row);
    }

    public DataFrame Where(Func<Series, bool> predicate)
    {
        var result = new DataFrame();
        foreach (var row in Rows.Where(predicate)) result.Upsert(row);
        result.NextIndex = NextIndex;
        return result;
    }

    public IEnumerable<T?> Column<T>(string name) => Rows.Select(r => r.GetOrDefault<T>(name));
}
