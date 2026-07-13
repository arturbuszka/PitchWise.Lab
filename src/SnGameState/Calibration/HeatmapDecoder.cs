using Microsoft.ML.OnnxRuntime.Tensors;

namespace SnGameState.Calibration;

/// <summary>
/// Port of nbjw_calib.utils.utils_heatmap: extraction of keypoint/line-extremity
/// coordinates from HRNet heatmaps via max-pool non-maximum suppression, and the
/// thresholding of coords_to_dict.
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
        var results = new List<Dictionary<int, Keypoint>>(batch);

        for (var n = 0; n < batch; n++)
        {
            var keypoints = new Dictionary<int, Keypoint>();
            // the last channel is the background heatmap (Python slices [:, :-1])
            for (var c = 0; c < channels - 1; c++)
            {
                var best = FindPeaks(heatmap, n, c, height, width, minPixelDistance,
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
        var results = new List<Dictionary<int, LineSegment>>(batch);

        for (var n = 0; n < batch; n++)
        {
            var lines = new Dictionary<int, LineSegment>();
            for (var c = 0; c < channels - 1; c++)
            {
                var peaks = FindPeaks(heatmap, n, c, height, width, minPixelDistance,
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

    /// <summary>Local maxima of one channel within a (2*minDist+1)^2 window, best first.</summary>
    private static List<Keypoint> FindPeaks(
        Tensor<float> heatmap, int n, int c, int height, int width,
        int minDist, bool excludeBorder, int maxPeaks)
    {
        var peaks = new List<Keypoint>();
        for (var y = 0; y < height; y++)
        {
            if (excludeBorder && (y < minDist || y >= height - minDist)) continue;
            for (var x = 0; x < width; x++)
            {
                if (excludeBorder && (x < minDist || x >= width - minDist)) continue;
                var value = heatmap[n, c, y, x];
                if (value <= 0) continue;
                if (!IsLocalMax(heatmap, n, c, y, x, height, width, minDist, value)) continue;
                peaks.Add(new Keypoint(x, y, value));
            }
        }
        return peaks.OrderByDescending(p => p.Score).Take(maxPeaks).ToList();
    }

    private static bool IsLocalMax(
        Tensor<float> heatmap, int n, int c, int y, int x,
        int height, int width, int minDist, float value)
    {
        var y0 = Math.Max(0, y - minDist);
        var y1 = Math.Min(height - 1, y + minDist);
        var x0 = Math.Max(0, x - minDist);
        var x1 = Math.Min(width - 1, x + minDist);
        for (var yy = y0; yy <= y1; yy++)
            for (var xx = x0; xx <= x1; xx++)
                if (heatmap[n, c, yy, xx] > value)
                    return false;
        return true;
    }
}
