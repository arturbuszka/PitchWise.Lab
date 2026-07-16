using OpenCvSharp;
using PitchWise.GameState.Calibration;
using PitchWise.Tracking.Datastruct;
using PitchWise.Tracking.Utils;

namespace PitchWise.GameState.Visualization;

/// <summary>
/// Game-state overlay rendering: pitch line overlay,
/// per-player ellipses with id/jersey/role labels, and the 2D radar minimap.
/// Colors follow the baseline: team left red / right blue (BGR), referee yellow.
/// </summary>
public static class GameStateVisualizer
{
    /// <summary>SoccerPitch.palette (sn_calibration_baseline), BGR order as used by cv2.</summary>
    public static readonly IReadOnlyDictionary<string, Scalar> LinePalette = new Dictionary<string, Scalar>
    {
        ["Big rect. left bottom"] = new(127, 0, 0),
        ["Big rect. left main"] = new(102, 102, 102),
        ["Big rect. left top"] = new(0, 0, 127),
        ["Big rect. right bottom"] = new(86, 32, 39),
        ["Big rect. right main"] = new(48, 77, 0),
        ["Big rect. right top"] = new(14, 97, 100),
        ["Circle central"] = new(0, 0, 255),
        ["Circle left"] = new(255, 127, 0),
        ["Circle right"] = new(0, 255, 255),
        ["Goal left crossbar"] = new(255, 255, 200),
        ["Goal left post left"] = new(165, 255, 0),
        ["Goal left post right"] = new(155, 119, 45),
        ["Goal right crossbar"] = new(86, 32, 139),
        ["Goal right post left"] = new(196, 120, 153),
        ["Goal right post right"] = new(166, 36, 52),
        ["Middle line"] = new(255, 255, 0),
        ["Side line bottom"] = new(255, 0, 255),
        ["Side line left"] = new(0, 255, 150),
        ["Side line right"] = new(0, 230, 0),
        ["Side line top"] = new(230, 0, 0),
        ["Small rect. left bottom"] = new(0, 150, 255),
        ["Small rect. left main"] = new(254, 173, 225),
        ["Small rect. left top"] = new(87, 72, 39),
        ["Small rect. right bottom"] = new(122, 0, 255),
        ["Small rect. right main"] = new(255, 255, 255),
        ["Small rect. right top"] = new(153, 23, 153),
    };

    private static readonly Scalar TeamLeft = new(0, 0, 255);
    private static readonly Scalar TeamRight = new(255, 0, 0);
    private static readonly Scalar Referee = new(2, 210, 238);
    private static readonly Scalar Other = new(0, 255, 0);
    private static readonly Scalar NoTeam = new(0, 0, 0);

    /// <summary>draw_pitch: overlays the detected pitch lines (normalized coordinates);
    /// the central circle is rendered with a fitted ellipse when enough points exist.</summary>
    public static void DrawPitch(Mat image, IReadOnlyDictionary<string, List<Keypoint>> lines, int lineThickness = 3)
    {
        foreach (var (name, line) in lines)
        {
            var color = LinePalette.GetValueOrDefault(name, Scalar.Black);
            var points = line.Select(p => new Point((int)(p.X * image.Width), (int)(p.Y * image.Height))).ToList();
            if (name == "Circle central" && points.Count > 4)
            {
                var ellipse = Cv2.FitEllipse(points.Select(p => new Point2f(p.X, p.Y)));
                Cv2.Ellipse(image, ellipse, color, lineThickness);
            }
            else
            {
                for (var j = 0; j < points.Count - 1; j++)
                    Cv2.Line(image, points[j], points[j + 1], color, lineThickness);
            }
        }
    }

    /// <summary>CompletePlayerEllipse.draw_detection: team-colored ground ellipse under
    /// each detection plus a compact label (ID / jersey number / role).</summary>
    public static void DrawPlayerEllipse(Mat image, Series detection,
        bool displayTrackId = true, bool displayJersey = true, bool displayRole = true)
    {
        if (!detection.Has("bbox_ltwh")) return;
        var bbox = Coordinates.LtwhToLtrb(detection.Get<float[]>("bbox_ltwh"));
        var color = TeamColor(detection);

        var center = new Point((int)((bbox[0] + bbox[2]) / 2), (int)bbox[3]);
        var width = bbox[2] - bbox[0];
        Cv2.Ellipse(image, center, new Size((int)width, (int)(0.35 * width)),
            0.0, -45.0, 235.0, color, 2, LineTypes.AntiAlias);

        var parts = new List<string>();
        if (displayTrackId && detection.Has("track_id"))
            parts.Add($"ID: {detection.Get<long>("track_id")}");
        if (displayJersey && detection.GetOrDefault<string>("jersey_number") is { Length: > 0 } jn && jn.All(char.IsDigit))
            parts.Add($"JN: {int.Parse(jn)}");
        if (displayRole)
        {
            var role = detection.GetOrDefault<string>("role") switch
            {
                "referee" => "R: RE", "player" => "R: P", "goalkeeper" => "R: GK",
                "other" => "OT", "ball" => "R: B", _ => "",
            };
            if (role.Length > 0) parts.Add(role);
        }
        if (parts.Count > 0)
            Cv2Draw.DrawText(image, string.Join('\n', parts), center,
                HersheyFonts.HersheyPlain, 0.75, 1, colorTxt: null, colorBg: color, alignH: 'c', alignV: 'c');
    }

    /// <summary>draw_radar_view: blends the pitch minimap into the frame and places each
    /// detection at its pitch coordinates (jersey number / GK / RE label, team color).</summary>
    public static void DrawRadarView(Mat image, IEnumerable<Series> detections, Mat radarTemplate,
        string group = "Predictions", int scale = 4, int delta = 32)
    {
        const int pitchWidth = 105 + 2 * 10;
        const int pitchHeight = 68 + 2 * 5;
        var sign = group == "Ground Truth" ? -1 : +1;
        const int yDelta = 3;
        var radarCenterX = (int)(image.Width / 2.0 - pitchWidth * scale / 2.0 * sign - delta * sign);
        var radarCenterY = (int)(image.Height - pitchHeight * scale / 2.0 - yDelta);
        var radarTopX = radarCenterX - pitchWidth * scale / 2;
        var radarTopY = image.Height - pitchHeight * scale - yDelta;

        using var radar = radarTemplate.Resize(new Size(pitchWidth * scale, pitchHeight * scale));
        Cv2.Line(radar, new Point(0, 0), new Point(0, radar.Height), new Scalar(0, 0, 255), 6);
        Cv2.Line(radar, new Point(radar.Width, 0), new Point(radar.Width, radar.Height), new Scalar(255, 0, 0), 6);

        const double alpha = 0.3;
        var roiRect = new Rect(radarTopX, radarTopY, radar.Width, radar.Height)
            .Intersect(new Rect(0, 0, image.Width, image.Height));
        if (roiRect.Width <= 0 || roiRect.Height <= 0) return;
        using (var roi = new Mat(image, roiRect))
        using (var radarRoi = new Mat(radar, new Rect(roiRect.X - radarTopX, roiRect.Y - radarTopY, roiRect.Width, roiRect.Height)))
        {
            Cv2.AddWeighted(roi, 1 - alpha, radarRoi, alpha, 0, roi);
        }
        Cv2Draw.DrawText(image, group, new Point(radarCenterX, radarTopY - 5),
            HersheyFonts.HersheySimplex, 1, 1, colorTxt: Scalar.White, alignH: 'c', alignV: 'b');

        foreach (var detection in detections)
        {
            if (detection.GetOrDefault<string>("role") == "ball") continue;
            if (detection["bbox_pitch"] is not BboxPitch pitch) continue;

            var color = TeamColor(detection);
            var x = Math.Clamp(pitch.XBottomMiddle, -10000, 10000);
            var y = Math.Clamp(pitch.YBottomMiddle, -10000, 10000);
            var pos = new Point(radarCenterX + (int)(x * scale), radarCenterY + (int)(y * scale));

            var label = detection.GetOrDefault<string>("role") switch
            {
                "goalkeeper" => "GK",
                "referee" => "RE",
                "other" => "OT",
                _ => detection.GetOrDefault<string>("jersey_number") is { Length: > 0 } jn && jn.All(char.IsDigit)
                    ? $"{int.Parse(jn)}" : null,
            };
            if (label is not null)
                Cv2Draw.DrawText(image, label, pos, HersheyFonts.HersheyPlain, 0.2 * scale, 1,
                    colorTxt: color, alignH: 'c', alignV: 'b');
            else
                Cv2.Circle(image, pos, scale, color, -1);
        }
    }

    private static Scalar TeamColor(Series detection)
        => detection.GetOrDefault<string>("role") switch
        {
            "referee" => Referee,
            "other" => Other,
            _ => detection.GetOrDefault<string>("team") switch
            {
                "left" => TeamLeft,
                "right" => TeamRight,
                _ => NoTeam,
            },
        };
}
