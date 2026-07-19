using Avalonia;
using Avalonia.Media.Imaging;
using Avalonia.Platform;
using OpenCvSharp;

namespace PitchWise.App.Infrastructure;

/// <summary>
/// Blits an OpenCV frame into an Avalonia bitmap.
/// </summary>
/// <remarks>
/// Two mismatches to reconcile: OpenCV frames are 3-channel BGR while Avalonia wants 4-channel
/// BGRA, and a Mat's row stride may exceed width*3 for aligned allocations while the bitmap has a
/// stride of its own. Constructing a Mat *over the locked bitmap buffer* and letting CvtColor
/// write straight into it handles both in one pass of OpenCV's SIMD path — no managed per-pixel
/// loop and no intermediate copy.
///
/// The only file in the app that touches both OpenCV and Avalonia, which is why it lives here
/// rather than in App.Core.
/// </remarks>
internal static class MatBitmapConverter
{
    /// <summary>Copies a BGR CV_8UC3 Mat into a Bgra8888 bitmap of the same size.</summary>
    public static void Blit(Mat bgr, WriteableBitmap target)
    {
        using var buffer = target.Lock();
        using var destination = Mat.FromPixelData(bgr.Rows, bgr.Cols, MatType.CV_8UC4,
            buffer.Address, buffer.RowBytes);
        Cv2.CvtColor(bgr, destination, ColorConversionCodes.BGR2BGRA);
    }

    /// <summary>
    /// One bitmap per resolution, reused for the life of the view — allocating per frame would
    /// make the GC saw at 25 fps.
    /// </summary>
    public static WriteableBitmap Create(int width, int height) => new(
        new PixelSize(width, height),
        new Vector(96, 96),
        PixelFormat.Bgra8888,
        AlphaFormat.Opaque);
}
