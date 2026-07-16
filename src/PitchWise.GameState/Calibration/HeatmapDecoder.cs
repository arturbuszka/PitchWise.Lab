using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace PitchWise.GameState.Calibration;

/// <summary>
/// Extraction of keypoint/line-extremity
/// coordinates from HRNet heatmaps via max-pool non-maximum suppression, and the
/// thresholding of coords_to_dict. The max-pool is computed natively (cv2 dilate),
/// mirroring the vectorized torch max_pool2d of the original.
/// </summary>
public static class HeatmapDecoder
{
    /// <summary>
    /// get_keypoints_from_heatmap_batch_maxpool + coords_to_dict for the keypoint model:
    /// one peak per channel, 31x31 NMS window, borders excluded (the Python code pads
    /// with 1.0 so any window touching the border can't be a local maximum).
    /// Returns per batch item: keypoint id (channel + 1) -> (x, y, score) in input pixels.
    /// </summary>
    public static List<Dictionary<int, Keypoint>> DecodeKeypoints(
        Tensor<float> heatmap, double threshold = 0.1449, int scale = 2, int minPixelDistance = 15)
    {
        int batch = heatmap.Dimensions[0], channels = heatmap.Dimensions[1];
        int height = heatmap.Dimensions[2], width = heatmap.Dimensions[3];
        var buffer = ((DenseTensor<float>)heatmap).Buffer;
        var results = new List<Dictionary<int, Keypoint>>(batch);

        for (var n = 0; n < batch; n++)
        {
            var keypoints = new Dictionary<int, Keypoint>();
            // the last channel is the background heatmap (Python slices [:, :-1])
            for (var c = 0; c < channels - 1; c++)
            {
                var plane = buffer.Slice(((n * channels) + c) * height * width, height * width);
                var best = FindPeaks(plane, height, width, minPixelDistance,
                                     excludeBorder: true, maxPeaks: 1);
                if (best.Count == 1 && best[0].Score > threshold)
                    keypoints[c + 1] = new Keypoint(best[0].X * scale, best[0].Y * scale, best[0].Score);
            }
            results.Add(keypoints);
        }
        return results;
    }

    /// <summary>
    /// get_keypoints_from_heatmap_batch_maxpool_l + coords_to_dict for the line model:
    /// two peaks per channel (the two line extremities), 21x21 NMS window, borders allowed.
    /// Returns per batch item: line id (channel + 1) -> segment, only when both
    /// extremity scores exceed the threshold.
    /// </summary>
    public static List<Dictionary<int, LineSegment>> DecodeLines(
        Tensor<float> heatmap, double threshold = 0.2983, int scale = 2, int minPixelDistance = 10)
    {
        int batch = heatmap.Dimensions[0], channels = heatmap.Dimensions[1];
        int height = heatmap.Dimensions[2], width = heatmap.Dimensions[3];
        var buffer = ((DenseTensor<float>)heatmap).Buffer;
        var results = new List<Dictionary<int, LineSegment>>(batch);

        for (var n = 0; n < batch; n++)
        {
            var lines = new Dictionary<int, LineSegment>();
            for (var c = 0; c < channels - 1; c++)
            {
                var plane = buffer.Slice(((n * channels) + c) * height * width, height * width);
                var peaks = FindPeaks(plane, height, width, minPixelDistance,
                                      excludeBorder: false, maxPeaks: 2);
                if (peaks.Count == 2 && peaks[0].Score > threshold && peaks[1].Score > threshold)
                    lines[c + 1] = new LineSegment(
                        new Keypoint(peaks[0].X * scale, peaks[0].Y * scale, peaks[0].Score),
                        new Keypoint(peaks[1].X * scale, peaks[1].Y * scale, peaks[1].Score));
            }
            results.Add(lines);
        }
        return results;
    }

    /// <summary>Local maxima of one channel within a (2*minDist+1)^2 window, best first.
    /// max-pool via native dilate, then a single scan for pixels equal to their window max.</summary>
    private static List<Keypoint> FindPeaks(
        Memory<float> plane, int height, int width, int minDist, bool excludeBorder, int maxPeaks)
    {
        var data = plane.ToArray();
        var kernel = 2 * minDist + 1;
        using var src = Mat.FromPixelData(height, width, MatType.CV_32FC1, data);
        using var dilated = new Mat();
        using var element = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(kernel, kernel));
        Cv2.Dilate(src, dilated, element);
        dilated.GetArray(out float[] windowMax);

        var peaks = new List<Keypoint>();
        int y0 = excludeBorder ? minDist : 0, y1 = excludeBorder ? height - minDist : height;
        int x0 = excludeBorder ? minDist : 0, x1 = excludeBorder ? width - minDist : width;
        for (var y = y0; y < y1; y++)
        {
            var row = y * width;
            for (var x = x0; x < x1; x++)
            {
                var value = data[row + x];
                if (value > 0 && value == windowMax[row + x])
                    peaks.Add(new Keypoint(x, y, value));
            }
        }
        return peaks.OrderByDescending(p => p.Score).Take(maxPeaks).ToList();
    }
}
