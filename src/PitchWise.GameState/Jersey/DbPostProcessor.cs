using OpenCvSharp;

namespace PitchWise.GameState.Jersey;

/// <summary>
/// Postprocessing of a DBNet text-detection probability map (mmocr DBPostprocessor):
/// binarize, find contours, score-filter, and unclip each region by the DB offset
/// delta = area * unclip_ratio / perimeter. The mmocr_api pipeline flattens polygons
/// to axis-aligned boxes anyway (bbox2poly(poly2bbox(...))), so boxes are returned.
/// </summary>
public static class DbPostProcessor
{
    /// <summary>Extracts text region boxes from the probability map (values 0..1).</summary>
    /// <returns>Boxes in probability-map pixel coordinates with their mean score.</returns>
    public static List<(Rect Box, float Score)> ExtractTextBoxes(
        float[,] probabilityMap, double maskThreshold = 0.3, double minScore = 0.3,
        double unclipRatio = 1.5, int maxCandidates = 1000)
    {
        var height = probabilityMap.GetLength(0);
        var width = probabilityMap.GetLength(1);

        using var prob = Mat.FromPixelData(height, width, MatType.CV_32FC1, probabilityMap);
        using var mask = new Mat();
        Cv2.Threshold(prob, mask, maskThreshold, 1, ThresholdTypes.Binary);
        using var mask8 = new Mat();
        mask.ConvertTo(mask8, MatType.CV_8UC1, 255);

        Cv2.FindContours(mask8, out var contours, out _, RetrievalModes.List, ContourApproximationModes.ApproxSimple);

        var results = new List<(Rect, float)>();
        foreach (var contour in contours.Take(maxCandidates))
        {
            var area = Cv2.ContourArea(contour);
            var perimeter = Cv2.ArcLength(contour, closed: true);
            if (perimeter < 1e-3 || area < 4) continue;

            var rect = Cv2.BoundingRect(contour);
            var score = MeanScore(probabilityMap, rect);
            if (score < minScore) continue;

            // DB unclip: expand the region by delta on each side
            var delta = (int)Math.Round(area * unclipRatio / perimeter);
            var expanded = new Rect(
                Math.Max(0, rect.X - delta), Math.Max(0, rect.Y - delta),
                Math.Min(width - Math.Max(0, rect.X - delta), rect.Width + 2 * delta),
                Math.Min(height - Math.Max(0, rect.Y - delta), rect.Height + 2 * delta));
            results.Add((expanded, score));
        }
        return results;
    }

    private static float MeanScore(float[,] map, Rect rect)
    {
        var sum = 0.0;
        for (var y = rect.Top; y < rect.Bottom; y++)
            for (var x = rect.Left; x < rect.Right; x++)
                sum += map[y, x];
        return (float)(sum / Math.Max(1, rect.Width * rect.Height));
    }
}
