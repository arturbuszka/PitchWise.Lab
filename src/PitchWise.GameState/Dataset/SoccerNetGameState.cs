using System.Text.Json;
using PitchWise.GameState.Calibration;
using PitchWise.Tracking.Datastruct;
using PitchWise.Tracking.Utils;

namespace PitchWise.GameState.Dataset;

/// <summary>
/// Loads the SoccerNet Game State dataset:
/// loads the SoccerNet-GSR dataset layout (train/valid/test/challenge splits, one folder
/// per video with img1/ frames and Labels-GameState.json annotations). Videos without
/// the labels file (challenge split) yield image metadata only.
/// </summary>
public sealed class SoccerNetGameState : TrackingDataset
{
    public override string Name => "SoccerNetGameState";

    public SoccerNetGameState(string datasetPath, int nvid = -1) : base(datasetPath)
    {
        if (!Directory.Exists(datasetPath))
            throw new DirectoryNotFoundException(
                $"'{datasetPath}' does not exist. Download SoccerNetGS following https://github.com/SoccerNet/sn-gamestate");

        foreach (var split in new[] { "train", "valid", "test", "challenge" })
        {
            var splitPath = Path.Combine(datasetPath, split);
            if (Directory.Exists(splitPath))
                Sets[split] = LoadSet(splitPath, nvid);
        }
    }

    public static TrackingSet LoadSet(string splitPath, int nvid = -1)
    {
        var videoMetadatas = new DataFrame();
        var imageMetadatas = new DataFrame();
        var detectionsGt = new DataFrame();

        var videoFolders = Directory.GetDirectories(splitPath).OrderBy(p => p, StringComparer.Ordinal).ToList();
        if (nvid > 0)
            videoFolders = videoFolders.Take(nvid).ToList();

        foreach (var videoFolder in videoFolders)
            LoadVideoDir(videoFolder, videoMetadatas, imageMetadatas, detectionsGt);

        return new TrackingSet
        {
            VideoMetadatas = videoMetadatas,
            ImageMetadatas = imageMetadatas,
            DetectionsGt = detectionsGt,
        };
    }

    /// <summary>Loads a single video folder as its own TrackingSet (fast path for tools
    /// that operate on one video, e.g. the render command).</summary>
    public static TrackingSet LoadVideo(string videoFolder)
    {
        var videoMetadatas = new DataFrame();
        var imageMetadatas = new DataFrame();
        var detectionsGt = new DataFrame();
        LoadVideoDir(videoFolder, videoMetadatas, imageMetadatas, detectionsGt);
        return new TrackingSet
        {
            VideoMetadatas = videoMetadatas,
            ImageMetadatas = imageMetadatas,
            DetectionsGt = detectionsGt,
        };
    }

    private static void LoadVideoDir(
        string videoFolder, DataFrame videoMetadatas, DataFrame imageMetadatas, DataFrame detectionsGt)
    {
        var folderName = Path.GetFileName(videoFolder);
        var labelsPath = Path.Combine(videoFolder, "Labels-GameState.json");

        if (!File.Exists(labelsPath))
        {
            // unlabeled video (challenge split): metadata from the img1 folder only
            var videoId = long.Parse(folderName.Split('-')[^1]);
            var imgFolder = Path.Combine(videoFolder, "img1");
            var files = Directory.GetFiles(imgFolder).OrderBy(f => f, StringComparer.Ordinal).ToList();

            var video = videoMetadatas.NewRow();
            video["id"] = videoId;
            video["name"] = folderName;
            video["nframes"] = (long)files.Count;

            for (var i = 0; i < files.Count; i++)
            {
                var image = imageMetadatas.NewRow();
                image["id"] = image.Index;
                image["video_id"] = videoId;
                image["frame"] = (long)i;
                image["file_path"] = files[i];
                image["nframes"] = (long)files.Count;
            }
            return;
        }

        using var doc = JsonDocument.Parse(File.ReadAllText(labelsPath));
        var root = doc.RootElement;
        var info = root.GetProperty("info");
        var images = root.GetProperty("images");
        var annotations = root.GetProperty("annotations");

        var vid = info.TryGetProperty("id", out var idProp)
            ? AsLong(idProp) : long.Parse(folderName.Split('-')[^1]);
        var nframes = AsLong(info.GetProperty("seq_length"));
        var imDir = info.TryGetProperty("im_dir", out var imDirProp) ? imDirProp.GetString()! : "img1";
        var imgFolderPath = Path.Combine(videoFolder, imDir);

        var videoRow = videoMetadatas.NewRow();
        videoRow["id"] = vid;
        videoRow["name"] = info.TryGetProperty("name", out var name) ? name.GetString() : folderName;
        videoRow["nframes"] = nframes;
        videoRow["frame_rate"] = info.TryGetProperty("frame_rate", out var fr) ? AsLong(fr) : 0L;
        videoRow["im_width"] = images[0].TryGetProperty("width", out var w) ? AsLong(w) : 0L;
        videoRow["im_height"] = images[0].TryGetProperty("height", out var h) ? AsLong(h) : 0L;

        // image metadata; index = numeric image_id from the annotation file
        var frame = 0L;
        foreach (var img in images.EnumerateArray())
        {
            var imageId = AsLong(img.GetProperty("image_id"));
            var image = new Series(imageId);
            image["id"] = imageId;
            image["video_id"] = vid;
            image["frame"] = frame++;
            image["file_path"] = Path.Combine(imgFolderPath, img.GetProperty("file_name").GetString()!);
            image["is_labeled"] = img.TryGetProperty("is_labeled", out var lab) && lab.GetBoolean();
            // v1.3 labelling flags; GS-HOTA only evaluates frames where all three are set
            foreach (var flag in new[] { "has_labeled_pitch", "has_labeled_camera", "has_labeled_person" })
                image[flag] = !img.TryGetProperty(flag, out var f) || f.ValueKind != JsonValueKind.False;
            image["nframes"] = nframes;
            imageMetadatas.Upsert(image);
        }

        foreach (var ann in annotations.EnumerateArray())
        {
            var supercategory = ann.GetProperty("supercategory").GetString();
            if (supercategory == "pitch" && ann.TryGetProperty("lines", out var linesProp))
            {
                // pitch GT: per-image line polylines (normalized), used by the visualizer
                var lines = new Dictionary<string, List<Keypoint>>();
                foreach (var line in linesProp.EnumerateObject())
                    lines[line.Name] = line.Value.EnumerateArray()
                        .Select(p => new Keypoint(p.GetProperty("x").GetDouble(), p.GetProperty("y").GetDouble(), 1.0))
                        .ToList();
                var imageRow = new Series(AsLong(ann.GetProperty("image_id")));
                imageRow["lines"] = lines;
                imageMetadatas.Upsert(imageRow);
                continue;
            }
            if (supercategory != "object")
                continue; // camera ground truth rows are not detections

            var det = detectionsGt.NewRow();
            det["id"] = ann.GetProperty("id").ToString();
            det["image_id"] = AsLong(ann.GetProperty("image_id"));
            det["video_id"] = vid;
            det["track_id"] = AsLong(ann.GetProperty("track_id"));
            det["category_id"] = AsLong(ann.GetProperty("category_id"));
            det["bbox_conf"] = 1.0;
            det["visibility"] = 1.0;

            var bbox = ann.GetProperty("bbox_image");
            det["bbox_ltwh"] = Coordinates.XywhToLtwh(new[]
            {
                bbox.GetProperty("x_center").GetSingle(), bbox.GetProperty("y_center").GetSingle(),
                bbox.GetProperty("w").GetSingle(), bbox.GetProperty("h").GetSingle(),
            });

            if (ann.TryGetProperty("bbox_pitch", out var bp) && bp.ValueKind == JsonValueKind.Object)
                det["bbox_pitch"] = new BboxPitch(
                    bp.GetProperty("x_bottom_left").GetDouble(), bp.GetProperty("y_bottom_left").GetDouble(),
                    bp.GetProperty("x_bottom_right").GetDouble(), bp.GetProperty("y_bottom_right").GetDouble(),
                    bp.GetProperty("x_bottom_middle").GetDouble(), bp.GetProperty("y_bottom_middle").GetDouble());

            var attributes = ann.GetProperty("attributes");
            var role = attributes.GetProperty("role").GetString();
            var team = attributes.TryGetProperty("team", out var t) ? t.GetString() : null;
            var jersey = attributes.TryGetProperty("jersey", out var j) ? j.GetString() : null;
            det["role"] = role;
            det["team"] = team;
            det["team_cluster"] = team == "left" ? 1 : 0;
            det["jersey_number"] = jersey;
            det["category"] = ExtractCategory(role, team, jersey);
        }
    }

    /// <summary>The annotation files mix number and string encodings between versions —
    /// accept both.</summary>
    private static long AsLong(JsonElement element)
        => element.ValueKind == JsonValueKind.Number ? element.GetInt64() : long.Parse(element.GetString()!);

    /// <summary>extract_category: unique GT identity label role_team_jersey.</summary>
    internal static string ExtractCategory(string? role, string? team, string? jersey)
        => role switch
        {
            "goalkeeper" or "player" => jersey is not null && jersey.All(char.IsDigit) && jersey.Length > 0
                ? $"{role}_{team}_{int.Parse(jersey)}"
                : $"{role}_{team}",
            _ => role ?? "other",
        };
}
