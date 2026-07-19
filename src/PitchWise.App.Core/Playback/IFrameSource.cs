using OpenCvSharp;
using PitchWise.Tracking.Datastruct;

namespace PitchWise.App.Core.Playback;

/// <summary>
/// Random access to the frames of one analysed video, addressed by the same index the analysis
/// used — the index that keys the detections.
/// </summary>
/// <remarks>
/// NOT thread-safe: the playback loop owns one and no one else touches it.
/// </remarks>
public interface IFrameSource : IDisposable
{
    int FrameCount { get; }

    double Fps { get; }

    Size FrameSize { get; }

    /// <summary>True when seeking lands exactly on the requested frame.</summary>
    /// <remarks>
    /// False for video files: OpenCV seeks to the nearest keyframe and decodes forward, so the
    /// frame returned can be off by one or more. The overlay is keyed by image_id, so an
    /// inexact seek draws the wrong players — worth telling the user about.
    /// </remarks>
    bool SeekIsExact { get; }

    /// <summary>Reads a frame into <paramref name="destination"/> as BGR CV_8UC3.</summary>
    /// <returns>False if the index is out of range or the frame could not be decoded.</returns>
    bool TryRead(int index, Mat destination);

    /// <summary>Image metadata of a frame; its Index is the image_id the detections are keyed by.</summary>
    Series? Metadata(int index);
}
