using OpenCvSharp;

namespace PitchWise.GameState;

/// <summary>
/// Vectorized image-to-tensor conversion — the .NET counterpart of torchvision's
/// ToTensor + Normalize (which are native vectorized ops in the original pipeline).
/// Replaces per-pixel C# loops: per channel a single fused native ConvertTo computes
/// (px/255 - mean)/std, then one block copy into the CHW tensor buffer.
/// </summary>
public static class TensorOps
{
    /// <summary>Fills <paramref name="dst"/> (length 3*H*W) with the normalized RGB CHW
    /// representation of a BGR image. mean/std omitted = plain /255 scaling.</summary>
    public static void CopyNormalizedChw(Mat bgr, Span<float> dst, float[]? mean = null, float[]? std = null)
    {
        int height = bgr.Rows, width = bgr.Cols, plane = height * width;
        if (dst.Length < 3 * plane)
            throw new ArgumentException($"destination too small: {dst.Length} < {3 * plane}");

        var planes = bgr.Split(); // B, G, R
        try
        {
            using var f32 = new Mat();
            for (var c = 0; c < 3; c++)
            {
                var m = mean?[c] ?? 0f;
                var s = std?[c] ?? 1f;
                // dst = px * alpha + beta == (px/255 - mean) / std
                planes[2 - c].ConvertTo(f32, MatType.CV_32FC1, 1.0 / (255.0 * s), -m / s);
                f32.GetArray(out float[] data);
                data.CopyTo(dst.Slice(c * plane, plane));
            }
        }
        finally
        {
            foreach (var p in planes) p.Dispose();
        }
    }
}
