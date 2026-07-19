using System.Text.Json;
using System.Text.Json.Serialization;
using PitchWise.GameState.Calibration;
using PitchWise.Tracking.Datastruct;

namespace PitchWise.App.Core.Workspace;

/// <summary>
/// Reads and writes the detection / image-metadata DataFrames of a finished analysis.
/// </summary>
/// <remarks>
/// TrackerState.Save exists but filters columns to a primitives-only whitelist, which silently
/// drops bbox_pitch and lines — the radar and the pitch overlay — and it has no Load at all.
/// A browsing tool has to round-trip a completed run exactly, so it needs its own store.
///
/// Values are type-tagged because the consumers cast rather than convert: the visualizer does
/// Get&lt;float[]&gt;("bbox_ltwh") and Get&lt;long&gt;("track_id"), so a float[] coming back as
/// double[] is an InvalidCastException at draw time, not a rounding difference. The tag set
/// covers exactly what a real run produces (verified against the pipeline output), and anything
/// unknown fails loudly on write rather than being dropped the way TrackerState.Save drops it.
/// </remarks>
public static class DetectionsStore
{
    /// <summary>
    /// Re-id embeddings are excluded: 512+ floats per detection is hundreds of megabytes on a
    /// full match, and nothing downstream of a finished run reads them back.
    /// </summary>
    private static readonly string[] ExcludedColumns = ["embeddings"];

    private static readonly JsonSerializerOptions Options = new()
    {
        Converters = { new Double2DArrayConverter() },
    };

    public static void WriteDetections(string path, DataFrame detections) => Write(path, detections);

    public static DataFrame ReadDetections(string path) => Read(path);

    public static void WriteImagePreds(string path, DataFrame imagePreds) => Write(path, imagePreds);

    public static DataFrame ReadImagePreds(string path) => Read(path);

    private static void Write(string path, DataFrame frame)
    {
        using var stream = File.Create(path);
        using var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = false });

        writer.WriteStartObject();
        writer.WriteNumber("next_index", frame.NextIndex);
        writer.WriteStartArray("rows");
        foreach (var row in frame.Rows)
        {
            writer.WriteStartObject();
            writer.WriteNumber("index", row.Index);
            writer.WriteStartObject("cols");
            foreach (var (column, value) in row.Values)
            {
                if (ExcludedColumns.Contains(column) || value is null) continue;
                writer.WritePropertyName(column);
                WriteValue(writer, column, value);
            }
            writer.WriteEndObject();
            writer.WriteEndObject();
        }
        writer.WriteEndArray();
        writer.WriteEndObject();
    }

    private static void WriteValue(Utf8JsonWriter writer, string column, object value)
    {
        writer.WriteStartObject();
        switch (value)
        {
            case long v: writer.WriteNumber("l", v); break;
            case int v: writer.WriteNumber("i", v); break;
            case double v: writer.WriteNumber("d", v); break;
            case float v: writer.WriteNumber("f", v); break;
            case bool v: writer.WriteBoolean("b", v); break;
            case string v: writer.WriteString("s", v); break;
            case float[] v:
                writer.WriteStartArray("fa");
                foreach (var f in v) writer.WriteNumberValue(f);
                writer.WriteEndArray();
                break;
            case double[] v:
                writer.WriteStartArray("da");
                foreach (var d in v) writer.WriteNumberValue(d);
                writer.WriteEndArray();
                break;
            case BboxPitch v:
                writer.WritePropertyName("bp");
                JsonSerializer.Serialize(writer, v, Options);
                break;
            case CameraParameters v:
                writer.WritePropertyName("cp");
                JsonSerializer.Serialize(writer, v, Options);
                break;
            case Dictionary<string, List<Keypoint>> v:
                writer.WritePropertyName("ln");
                JsonSerializer.Serialize(writer, v, Options);
                break;
            case Dictionary<int, Keypoint> v:
                writer.WritePropertyName("kp");
                JsonSerializer.Serialize(writer, v, Options);
                break;
            case double[,] v:
                writer.WritePropertyName("d2");
                JsonSerializer.Serialize(writer, v, Options);
                break;
            default:
                throw new NotSupportedException(
                    $"column '{column}' holds an unsupported type {value.GetType()}; add a tag for it " +
                    "rather than letting it be dropped");
        }
        writer.WriteEndObject();
    }

    private static DataFrame Read(string path)
    {
        var frame = new DataFrame();
        using var document = JsonDocument.Parse(File.ReadAllBytes(path));
        var root = document.RootElement;

        foreach (var rowElement in root.GetProperty("rows").EnumerateArray())
        {
            var row = new Series(rowElement.GetProperty("index").GetInt64());
            foreach (var column in rowElement.GetProperty("cols").EnumerateObject())
                row[column.Name] = ReadValue(column.Value);
            frame.Upsert(row);
        }

        // restore the row-allocation cursor, so a frame that is read back and appended to
        // cannot hand out an index that is already taken
        frame.NextIndex = root.GetProperty("next_index").GetInt64();
        return frame;
    }

    private static object ReadValue(JsonElement tagged)
    {
        var tag = tagged.EnumerateObject().First();
        var value = tag.Value;
        return tag.Name switch
        {
            "l" => value.GetInt64(),
            "i" => value.GetInt32(),
            "d" => value.GetDouble(),
            "f" => value.GetSingle(),
            "b" => value.GetBoolean(),
            "s" => value.GetString()!,
            "fa" => value.EnumerateArray().Select(e => e.GetSingle()).ToArray(),
            "da" => value.EnumerateArray().Select(e => e.GetDouble()).ToArray(),
            "bp" => value.Deserialize<BboxPitch>(Options)!,
            "cp" => value.Deserialize<CameraParameters>(Options)!,
            "ln" => value.Deserialize<Dictionary<string, List<Keypoint>>>(Options)!,
            "kp" => value.Deserialize<Dictionary<int, Keypoint>>(Options)!,
            "d2" => value.Deserialize<double[,]>(Options)!,
            _ => throw new InvalidDataException($"unknown value tag '{tag.Name}'"),
        };
    }

    /// <summary>
    /// System.Text.Json has no built-in support for rank-2 arrays, which the homography and the
    /// camera rotation matrix both are. Stored as its dimensions plus row-major data.
    /// </summary>
    private sealed class Double2DArrayConverter : JsonConverter<double[,]>
    {
        public override void Write(Utf8JsonWriter writer, double[,] value, JsonSerializerOptions options)
        {
            writer.WriteStartObject();
            writer.WriteNumber("rows", value.GetLength(0));
            writer.WriteNumber("cols", value.GetLength(1));
            writer.WriteStartArray("data");
            foreach (var d in value) writer.WriteNumberValue(d);   // foreach walks row-major
            writer.WriteEndArray();
            writer.WriteEndObject();
        }

        public override double[,] Read(ref Utf8JsonReader reader, Type typeToConvert,
            JsonSerializerOptions options)
        {
            using var document = JsonDocument.ParseValue(ref reader);
            var root = document.RootElement;
            var rows = root.GetProperty("rows").GetInt32();
            var cols = root.GetProperty("cols").GetInt32();
            var result = new double[rows, cols];
            var index = 0;
            foreach (var element in root.GetProperty("data").EnumerateArray())
            {
                result[index / cols, index % cols] = element.GetDouble();
                index++;
            }
            if (index != rows * cols)
                throw new InvalidDataException($"expected {rows * cols} values, got {index}");
            return result;
        }
    }
}
