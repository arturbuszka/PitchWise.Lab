using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;
using Avalonia.Media.Imaging;
using Avalonia.Threading;
using PitchWise.App.Infrastructure;

// Avalonia and OpenCvSharp both define Size/Point/Rect; this file is geometry in Avalonia's
// world and only borrows a Mat, so Mat is the one that gets qualified.
using Mat = OpenCvSharp.Mat;

namespace PitchWise.App.Controls;

/// <summary>
/// Shows frames handed to it by the playback loop.
/// </summary>
/// <remarks>
/// <see cref="Show"/> is called on the playback thread and blits synchronously, because the loop
/// reuses the Mat as soon as the call returns. Only the invalidate is marshalled — a blit is
/// ~1-2 ms, and doing it here rather than on the UI thread keeps the frame budget free.
/// </remarks>
public sealed class FrameView : Control
{
    private WriteableBitmap? _bitmap;
    private readonly Lock _gate = new();

    /// <summary>Copies a frame in. Safe to call from the playback thread.</summary>
    public void Show(Mat frame)
    {
        if (frame.Empty()) return;

        lock (_gate)
        {
            if (_bitmap is null
                || _bitmap.PixelSize.Width != frame.Cols
                || _bitmap.PixelSize.Height != frame.Rows)
            {
                _bitmap?.Dispose();
                _bitmap = MatBitmapConverter.Create(frame.Cols, frame.Rows);
            }
            MatBitmapConverter.Blit(frame, _bitmap);
        }

        Dispatcher.UIThread.Post(InvalidateVisual, DispatcherPriority.Render);
    }

    public override void Render(DrawingContext context)
    {
        lock (_gate)
        {
            if (_bitmap is null) return;

            // letterbox: fill the control without distorting the frame
            var source = new Rect(0, 0, _bitmap.PixelSize.Width, _bitmap.PixelSize.Height);
            var scale = Math.Min(Bounds.Width / source.Width, Bounds.Height / source.Height);
            var size = new Size(source.Width * scale, source.Height * scale);
            var origin = new Point((Bounds.Width - size.Width) / 2, (Bounds.Height - size.Height) / 2);

            context.DrawImage(_bitmap, source, new Rect(origin, size));
        }
    }

    protected override void OnDetachedFromVisualTree(VisualTreeAttachmentEventArgs e)
    {
        base.OnDetachedFromVisualTree(e);
        lock (_gate)
        {
            _bitmap?.Dispose();
            _bitmap = null;
        }
    }
}
