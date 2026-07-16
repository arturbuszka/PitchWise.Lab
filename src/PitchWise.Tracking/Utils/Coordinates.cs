namespace PitchWise.Tracking.Utils;

/// <summary>Bbox format conversions and clipping.
/// All bboxes are float[4]; ltwh = left, top, width, height; ltrb = left, top, right, bottom;
/// xywh = center x, center y, width, height.</summary>
public static class Coordinates
{
    public static float[] LtwhToLtrb(float[] b) => new[] { b[0], b[1], b[0] + b[2], b[1] + b[3] };
    public static float[] LtrbToLtwh(float[] b) => new[] { b[0], b[1], b[2] - b[0], b[3] - b[1] };
    public static float[] LtwhToXywh(float[] b) => new[] { b[0] + b[2] / 2, b[1] + b[3] / 2, b[2], b[3] };
    public static float[] XywhToLtwh(float[] b) => new[] { b[0] - b[2] / 2, b[1] - b[3] / 2, b[2], b[3] };
    public static float[] LtrbToXywh(float[] b) => LtwhToXywh(LtrbToLtwh(b));
    public static float[] XywhToLtrb(float[] b) => LtwhToLtrb(XywhToLtwh(b));

    public static float[] ClipLtwhToImageDim(float[] bbox, int imgW, int imgH)
    {
        var l = Math.Clamp(bbox[0], 0, imgW - 1);
        var t = Math.Clamp(bbox[1], 0, imgH - 1);
        var w = Math.Clamp(bbox[2], 0, imgW - 1 - l);
        var h = Math.Clamp(bbox[3], 0, imgH - 1 - t);
        return new[] { l, t, w, h };
    }

    /// <summary>Ground point of a person bbox in image coordinates (bottom-center),
    /// used as the point projected onto the pitch by the calibration modules.</summary>
    public static (float x, float y) BboxLtwhToGroundPoint(float[] b) => (b[0] + b[2] / 2, b[1] + b[3]);
}
