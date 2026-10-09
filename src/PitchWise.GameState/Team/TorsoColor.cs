using OpenCvSharp;
using PitchWise.Tracking.Datastruct;
using PitchWise.Tracking.Pipeline;

namespace PitchWise.GameState.Team;

/// <summary>
/// Jersey colour of each detection, the feature <see cref="TrackletTeamClustering"/> splits teams by:
/// the per-channel median Lab colour of the torso region (25–75% of the box width, 15–50% of its
/// height) with grass pixels masked out. Not part of the Python sn-gamestate pipeline, which
/// clusters ReID embeddings; those encode identity more than kit and split teams far less cleanly
/// (see docs/eval-baseline.md, "Diagnoza: drużyny i strony").
/// </summary>
public sealed class TorsoColor : DetectionLevelModule
{
    /// <summary>Fewer non-grass pixels than this and the crop gives no colour.</summary>
    private const int MinPixels = 10;

    public override IReadOnlyDictionary<string, string[]> InputColumns { get; } = new Dictionary<string, string[]>
    {
        ["detection"] = new[] { "bbox_ltwh" },
    };

    public override IReadOnlyDictionary<string, string[]> OutputColumns { get; } = new Dictionary<string, string[]>
    {
        ["detection"] = new[] { "torso_color" },
    };

    public TorsoColor(int batchSize = 64) : base(batchSize) { }

    public override object Preprocess(Mat image, Series detection, Series metadata)
        => new Sample(Measure(image, detection.Get<float[]>("bbox_ltwh")), detection.Index);

    public override IReadOnlyList<Series> Process(IReadOnlyList<object> batch, DataFrame detections, IReadOnlyList<Series> metadatas)
        => batch.Cast<Sample>().Select(s => new Series(s.DetectionIndex) { ["torso_color"] = s.Color }).ToList();

    /// <summary>Median Lab (OpenCV 8-bit scale) of the torso's non-grass pixels, or null.</summary>
    internal static float[]? Measure(Mat bgr, float[] ltwh)
    {
        var (x, y, w, h) = (ltwh[0], ltwh[1], ltwh[2], ltwh[3]);
        var x0 = (int)Math.Max(0, x + 0.25f * w);
        var x1 = (int)Math.Min(bgr.Width, x + 0.75f * w);
        var y0 = (int)Math.Max(0, y + 0.15f * h);
        var y1 = (int)Math.Min(bgr.Height, y + 0.5f * h);
        if (x1 - x0 < 2 || y1 - y0 < 2)
            return null;

        using var crop = new Mat(bgr, new Rect(x0, y0, x1 - x0, y1 - y0));
        using var hsv = new Mat();
        using var lab = new Mat();
        Cv2.CvtColor(crop, hsv, ColorConversionCodes.BGR2HSV);
        Cv2.CvtColor(crop, lab, ColorConversionCodes.BGR2Lab);
        hsv.GetArray(out Vec3b[] hsvPx);
        lab.GetArray(out Vec3b[] labPx);

        var channels = new[] { new List<float>(), new List<float>(), new List<float>() };
        for (var i = 0; i < hsvPx.Length; i++)
        {
            var p = hsvPx[i];
            if (p.Item0 is >= 35 and <= 85 && p.Item1 > 60)   // grass: green hue, saturated
                continue;
            var l = labPx[i];
            channels[0].Add(l.Item0);
            channels[1].Add(l.Item1);
            channels[2].Add(l.Item2);
        }
        return channels[0].Count < MinPixels ? null : channels.Select(Median).ToArray();
    }

    /// <summary>numpy.median: the mean of the two middle values for an even count.</summary>
    internal static float Median(List<float> values)
    {
        values.Sort();
        var mid = values.Count / 2;
        return values.Count % 2 == 1 ? values[mid] : (values[mid - 1] + values[mid]) / 2;
    }

    private sealed record Sample(float[]? Color, long DetectionIndex);
}
