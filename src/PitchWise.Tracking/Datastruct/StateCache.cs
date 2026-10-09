using System.IO.Compression;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;

namespace PitchWise.Tracking.Datastruct;

/// <summary>
/// Snapshot of one video's pipeline state (detections + image predictions, every column) in a
/// gzipped JSON file, so a later run can skip the modules whose outputs did not change and rerun
/// only the rest — e.g. retune the tracker without re-running detection, calibration and OCR.
/// Values keep their CLR type: each column type is stored once in a type table.
/// </summary>
public static class StateCache
{
    private static readonly JsonSerializerOptions Options = new()
    {
        IncludeFields = true,
        Converters = { new Rank2ArrayConverter<double>(), new Rank2ArrayConverter<float>() },
    };

    public sealed record Snapshot(DataFrame Detections, DataFrame ImagePreds, IReadOnlyDictionary<string, string> Meta);

    public static void Save(string path, DataFrame detections, DataFrame imagePreds,
        IReadOnlyDictionary<string, string> meta)
    {
        var types = new List<string>();
        var typeIds = new Dictionary<Type, int>();
        JsonArray Rows(DataFrame frame)
        {
            var rows = new JsonArray();
            foreach (var row in frame.Rows)
            {
                var values = new JsonObject();
                foreach (var (column, value) in row.Values)
                {
                    if (value is null)
                        continue;
                    var type = value.GetType();
                    if (!typeIds.TryGetValue(type, out var id))
                    {
                        typeIds[type] = id = types.Count;
                        types.Add(type.AssemblyQualifiedName!);
                    }
                    values[column] = new JsonArray(id, JsonSerializer.SerializeToNode(value, type, Options));
                }
                rows.Add(new JsonObject { ["index"] = row.Index, ["values"] = values });
            }
            return rows;
        }

        var root = new JsonObject
        {
            ["meta"] = JsonSerializer.SerializeToNode(meta),
            ["detections"] = Rows(detections),
            ["detectionsNextIndex"] = detections.NextIndex,
            ["imagePreds"] = Rows(imagePreds),
        };
        root["types"] = JsonSerializer.SerializeToNode(types);

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        using var file = File.Create(path);
        using var gzip = new GZipStream(file, CompressionLevel.Fastest);
        using var writer = new Utf8JsonWriter(gzip);
        root.WriteTo(writer);
    }

    public static Snapshot Load(string path)
    {
        using var file = File.OpenRead(path);
        using var gzip = new GZipStream(file, CompressionMode.Decompress);
        var root = JsonNode.Parse(gzip)!.AsObject();
        var types = root["types"]!.AsArray()
            .Select(t => Type.GetType(t!.GetValue<string>(), throwOnError: true)!)
            .ToList();

        DataFrame Rows(JsonArray rows)
        {
            var frame = new DataFrame();
            foreach (var node in rows)
            {
                var row = new Series(node!["index"]!.GetValue<long>());
                foreach (var (column, cell) in node["values"]!.AsObject())
                {
                    var pair = cell!.AsArray();
                    row[column] = pair[1].Deserialize(types[pair[0]!.GetValue<int>()], Options);
                }
                frame.Upsert(row);
            }
            return frame;
        }

        var detections = Rows(root["detections"]!.AsArray());
        detections.NextIndex = root["detectionsNextIndex"]!.GetValue<long>();
        var meta = root["meta"].Deserialize<Dictionary<string, string>>()!;
        return new Snapshot(detections, Rows(root["imagePreds"]!.AsArray()), meta);
    }

    /// <summary>System.Text.Json has no multi-dimensional arrays: store them as nested rows.</summary>
    private sealed class Rank2ArrayConverter<T> : JsonConverter<T[,]>
    {
        public override T[,] Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            var rows = JsonSerializer.Deserialize<T[][]>(ref reader, options)!;
            var result = new T[rows.Length, rows.Length == 0 ? 0 : rows[0].Length];
            for (var i = 0; i < rows.Length; i++)
                for (var j = 0; j < rows[i].Length; j++)
                    result[i, j] = rows[i][j];
            return result;
        }

        public override void Write(Utf8JsonWriter writer, T[,] value, JsonSerializerOptions options)
        {
            var rows = new T[value.GetLength(0)][];
            for (var i = 0; i < rows.Length; i++)
            {
                rows[i] = new T[value.GetLength(1)];
                for (var j = 0; j < rows[i].Length; j++)
                    rows[i][j] = value[i, j];
            }
            JsonSerializer.Serialize(writer, rows, options);
        }
    }
}
