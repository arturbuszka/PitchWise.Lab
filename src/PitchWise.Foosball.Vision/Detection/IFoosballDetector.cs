using OpenCvSharp;

namespace PitchWise.Foosball.Vision.Detection;

/// <summary>
/// Finds the ball and the figures on a single frame. An interface because the bar box may run the
/// model on a different runtime than ONNX Runtime on a PC.
/// </summary>
public interface IFoosballDetector : IDisposable
{
    /// <param name="image">BGR frame, as OpenCV decodes it.</param>
    FrameDetections Detect(Mat image);
}
