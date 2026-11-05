using OpenCvSharp;
using PitchWise.Tracking.Datastruct;

namespace PitchWise.App.Core.Playback;

/// <summary>A frame that has been decoded and drawn, ready to be shown.</summary>
/// <param name="Image">
/// BGR CV_8UC3, owned by the loop. Valid only for the duration of the callback: copy it out
/// (blit it) synchronously and never store the Mat, which the loop reuses for a later frame.
/// </param>
/// <param name="Index">Frame index, for the scrub bar.</param>
/// <param name="Metadata">That frame's image metadata.</param>
public readonly record struct RenderedFrame(Mat Image, int Index, Series Metadata);

/// <summary>
/// Decodes and draws frames on its own thread, handing finished ones to the UI.
/// </summary>
/// <remarks>
/// <para>
/// Not on the UI thread: decoding a jpg plus drawing the overlay is ~5-15 ms, enough to make a
/// 40 ms frame budget stutter.
/// </para>
/// <para>
/// Two long-lived Mats swapped by pointer, never reallocated — one per frame at 25 fps would make
/// the GC saw. The handler is called synchronously and must blit before returning, which is what
/// keeps the buffers safe without a lock: only this thread ever touches them, and it does not
/// resume decoding until the blit is done.
/// </para>
/// </remarks>
public sealed class PlaybackLoop : IDisposable
{
    private readonly IFrameSource _source;
    private readonly OverlayRenderer _overlay;
    private readonly IFrameClock _clock;
    private readonly Action<RenderedFrame> _onFrame;

    // swapped by reference each frame rather than reallocated; see RenderOne
    private Mat _decode = new();
    private Mat _ready = new();

    private readonly CancellationTokenSource _cts = new();
    private readonly SemaphoreSlim _wake = new(0);
    private readonly Thread _thread;

    private volatile bool _playing;
    private volatile int _requestedIndex;
    private volatile bool _seekPending;
    private double _rate = 1;
    private OverlayOptions _options = OverlayOptions.Default;
    private int _currentIndex = -1;

    /// <param name="onFrame">
    /// Called on the loop's thread once a frame is drawn. Marshal to the UI inside it; the Mat is
    /// only valid for the duration of the call.
    /// </param>
    public PlaybackLoop(IFrameSource source, OverlayRenderer overlay, IFrameClock clock,
        Action<RenderedFrame> onFrame)
    {
        _source = source;
        _overlay = overlay;
        _clock = clock;
        _onFrame = onFrame;

        _thread = new Thread(Run)
        {
            IsBackground = true,
            Name = "pitchwise-playback",
        };
        _thread.Start();
    }

    public bool IsPlaying => _playing;

    public int CurrentIndex => _currentIndex;

    public void Play()
    {
        _playing = true;
        _wake.Release();
    }

    public void Pause() => _playing = false;

    /// <summary>
    /// Jumps to a frame. Only the newest request survives, so dragging a scrub bar never queues a
    /// backlog of decodes to work through.
    /// </summary>
    public void SeekTo(int index)
    {
        _requestedIndex = Math.Clamp(index, 0, Math.Max(0, _source.FrameCount - 1));
        _seekPending = true;
        _wake.Release();
    }

    /// <summary>Steps one frame and stops, for frame-by-frame inspection.</summary>
    public void Step(int delta)
    {
        _playing = false;
        SeekTo((_currentIndex < 0 ? 0 : _currentIndex) + delta);
    }

    /// <summary>Playback speed; 1 is real time.</summary>
    public void SetRate(double rate) => _rate = Math.Clamp(rate, 0.1, 8);

    /// <summary>Changes what is drawn and redraws the current frame, so a toggle is immediate
    /// even while paused.</summary>
    public void SetOverlay(OverlayOptions options)
    {
        _options = options;
        if (!_playing) SeekTo(_currentIndex < 0 ? 0 : _currentIndex);
    }

    private void Run()
    {
        var ct = _cts.Token;
        try
        {
            while (!ct.IsCancellationRequested)
            {
                if (!_playing && !_seekPending)
                {
                    _wake.Wait(ct);   // idle: nothing to draw and nothing burning CPU
                    continue;
                }

                int index;
                if (_seekPending)
                {
                    _seekPending = false;
                    index = _requestedIndex;
                }
                else
                {
                    index = _currentIndex + 1;
                    if (index >= _source.FrameCount)
                    {
                        _playing = false;   // stop at the end rather than wrapping
                        continue;
                    }
                }

                if (RenderOne(index) && _playing)
                    _clock.WaitNextFrame(_source.Fps, _rate, ct);
            }
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            // Dispose was called. Nothing above this thread could catch it, so it would end the process.
        }
    }

    private bool RenderOne(int index)
    {
        var metadata = _source.Metadata(index);
        if (metadata is null || !_source.TryRead(index, _decode)) return false;

        _overlay.Draw(_decode, metadata, _options);
        _currentIndex = index;

        // OpenCvSharp has no Cv2.Swap, but swapping the references is the same idea and just as
        // cheap: no pixels move, and both Mats stay alive to be reused for the next frames
        (_decode, _ready) = (_ready, _decode);
        _onFrame(new RenderedFrame(_ready, index, metadata));
        return true;
    }

    public void Dispose()
    {
        _cts.Cancel();
        _wake.Release();
        _thread.Join(TimeSpan.FromSeconds(2));
        _cts.Dispose();
        _wake.Dispose();
        _decode.Dispose();
        _ready.Dispose();
    }
}
