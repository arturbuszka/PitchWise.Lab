using OpenCvSharp;

namespace TrackLab.Utils;

/// <summary>Port of tracklab.utils.cv2.draw_text: aligned text with an optional
/// semi-transparent background box.</summary>
public static class Cv2Draw
{
    public static void DrawText(
        Mat image, string text, Point position,
        HersheyFonts fontFace = HersheyFonts.HersheyPlain, double fontScale = 1, int thickness = 1,
        Scalar? colorTxt = null, Scalar? colorBg = null, double alphaBg = 0.6,
        char alignH = 'l', char alignV = 't')
    {
        if (string.IsNullOrEmpty(text)) return;
        var lines = text.Split('\n');
        var sizes = lines.Select(l => Cv2.GetTextSize(l, fontFace, fontScale, thickness, out _)).ToList();
        var blockWidth = sizes.Max(s => s.Width);
        var lineHeight = sizes.Max(s => s.Height) + 4;
        var blockHeight = lineHeight * lines.Length;

        var x0 = alignH switch { 'c' => position.X - blockWidth / 2, 'r' => position.X - blockWidth, _ => position.X };
        var y0 = alignV switch { 'c' => position.Y - blockHeight / 2, 'b' => position.Y - blockHeight, _ => position.Y };

        if (colorBg is not null)
        {
            var rect = new Rect(x0 - 2, y0 - 2, blockWidth + 4, blockHeight + 4);
            rect = rect.Intersect(new Rect(0, 0, image.Width, image.Height));
            if (rect.Width > 0 && rect.Height > 0)
            {
                using var roi = new Mat(image, rect);
                using var overlay = new Mat(rect.Size, image.Type(), colorBg.Value);
                Cv2.AddWeighted(roi, 1 - alphaBg, overlay, alphaBg, 0, roi);
            }
        }

        var txtColor = colorTxt ?? (colorBg is not null ? PickContrastColor(colorBg.Value) : Scalar.White);
        for (var i = 0; i < lines.Length; i++)
            Cv2.PutText(image, lines[i], new Point(x0 + (blockWidth - sizes[i].Width) / 2, y0 + (i + 1) * lineHeight - 3),
                fontFace, fontScale, txtColor, thickness, LineTypes.AntiAlias);
    }

    private static Scalar PickContrastColor(Scalar bg)
        => 0.114 * bg.Val0 + 0.587 * bg.Val1 + 0.299 * bg.Val2 > 127 ? Scalar.Black : Scalar.White;
}
