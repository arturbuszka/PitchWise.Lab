namespace PitchWise.Foosball.Vision.Frames;

/// <summary>
/// A stream of frames, from a recording now and from the camera later (step 6.1), so the rest of the
/// pipeline does not care where pixels come from. Sequential only: a live camera cannot seek either.
/// </summary>
public interface IFrameSource : IDisposable
{
    double Fps { get; }
    int Width { get; }
    int Height { get; }

    /// <summary>Frames in order until the stream ends. Each frame owns its image; dispose it when done.</summary>
    IEnumerable<VideoFrame> ReadFrames();
}
