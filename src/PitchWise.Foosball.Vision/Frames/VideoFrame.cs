using OpenCvSharp;

namespace PitchWise.Foosball.Vision.Frames;

/// <summary>One decoded frame. The image never leaves PitchWise.Foosball.Vision; downstream gets metres.</summary>
/// <param name="Index">0-based position in the stream.</param>
/// <param name="Timestamp">Presentation time of the frame in the stream, not Index / Fps: phone
/// recordings do not keep perfectly even frame spacing, and match time is the recording position.</param>
public sealed record VideoFrame(long Index, TimeSpan Timestamp, Mat Image) : IDisposable
{
    public void Dispose() => Image.Dispose();
}
