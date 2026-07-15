namespace PitchWise.Engine;

/// <summary>
/// The pitch template: the fixed real-world positions of the lines and intersections a keypoint
/// model detects, in metres. This is the "known" side of every pixel↔pitch correspondence that
/// feeds <c>Homography.FromPoints</c>.
///
/// <b>Coordinate convention</b> — pinned here once, because three places disagreed and a
/// homography fitted in one convention and rendered in another silently mirrors the pitch:
/// <list type="bullet">
/// <item>Pitch is <b>105 m × 68 m</b> (the real FIFA dimensions the frontend and
///       <c>Homography.cs</c> already use — NOT Roboflow's default 120×70).</item>
/// <item>Origin (0,0) is the <b>top-left</b> corner. x runs 0→105 along the length (left goal
///       line to right goal line); y runs 0→68 down the width (top touchline to bottom
///       touchline). This matches <c>web/components/PitchMinimap.tsx</c>, which renders the dots,
///       so it is the source of truth. (The prose comment in <c>Homography.cs</c> says "bottom
///       touchline"; that comment is wrong relative to what actually renders — the fit and the
///       render must share this convention, and the frontend wins because the user sees it.)</item>
/// </list>
///
/// The 32 keypoints and their ordering are the Roboflow <c>SoccerPitchConfiguration</c> set (so a
/// model trained against it maps index-for-index onto this array), with positions given by the
/// REAL Laws-of-the-Game geometry on a 105×68 m pitch — see the comment above
/// <see cref="Keypoints"/> for why they must not be produced by rescaling Roboflow's template.
/// A model's output channel <c>i</c> corresponds to <see cref="Keypoints"/>[i].
/// </summary>
public static class PitchModel
{
    public const double Length = 105.0;   // m, along x
    public const double Width = 68.0;      // m, along y

    // TRUE FIFA geometry, in metres, NOT a rescale of Roboflow's nominal 120×70 template.
    //
    // Two hard-won lessons are baked into these numbers:
    //
    // 1. The model predicts where the PAINTED LINES intersect in a real image — its training
    //    annotations were humans clicking real pitch markings. The Laws of the Game fix the box
    //    geometry ABSOLUTELY, not proportionally to the pitch: the penalty area is 16.5 m deep and
    //    40.32 m wide on every pitch, the goal area 5.5 m × 18.32 m, the penalty spot 11 m out,
    //    the centre circle radius 9.15 m. Rescaling Roboflow's 120×70 template onto 105×68 (as
    //    this file previously did) therefore put the template's penalty box at 17.63 m and the
    //    penalty spot at 9.63 m — over a metre of systematic error on the very correspondences the
    //    homography is fitted from.
    //
    // 2. The INDEXING must match roboflow/sports SoccerPitchConfiguration.vertices exactly,
    //    because the model's output channel i is that list's vertex i. Three entries here were
    //    corrupted relative to the source (21: 11100 for 10900; 30: 4085 for 5085; 31: 7915 for
    //    6915 — the centre-circle points off by a full 10 m). Points 21 and 31 appear in nearly
    //    every broadcast frame's fit, so every homography was fitted against a template lying by
    //    metres. Measured on Croatia–Spain 720p: median held-out reprojection error 2.75 m, and
    //    possession attribution collapsed to 2.8 s per hour of play. If you touch this list,
    //    re-verify against https://github.com/roboflow/sports sports/configs/soccer.py.
    private const double HalfW = Width / 2.0;              // 34
    private const double PenD = 16.5, PenW = 40.32;        // penalty area depth / width
    private const double GoalD = 5.5, GoalW = 18.32;       // goal area depth / width
    private const double Spot = 11.0, Circle = 9.15;       // penalty spot distance, circle radius
    private const double PenTop = HalfW - PenW / 2.0, PenBot = HalfW + PenW / 2.0;    // 13.84 / 54.16
    private const double GoalTop = HalfW - GoalW / 2.0, GoalBot = HalfW + GoalW / 2.0; // 24.84 / 43.16

    /// <summary>The 32 pitch keypoints in metres, indexed to match the Roboflow model's output
    /// channel order. Origin top-left, x→right (length), y→down (width).</summary>
    public static readonly IReadOnlyList<(double X, double Y)> Keypoints = new[]
    {
        (0.0, 0.0),                          //  0  left goal line × top touchline (top-left corner)
        (0.0, PenTop),                       //  1  left goal line × penalty-box top edge extended
        (0.0, GoalTop),                      //  2  left goal line × 6-yd box top
        (0.0, GoalBot),                      //  3  left goal line × 6-yd box bottom
        (0.0, PenBot),                       //  4  left goal line × penalty-box bottom edge extended
        (0.0, Width),                        //  5  left goal line × bottom touchline (bottom-left corner)
        (GoalD, GoalTop),                    //  6  left 6-yd box top-inner
        (GoalD, GoalBot),                    //  7  left 6-yd box bottom-inner
        (Spot, HalfW),                       //  8  left penalty spot
        (PenD, PenTop),                      //  9  left penalty box top-outer
        (PenD, GoalTop),                     // 10  left penalty box top-inner
        (PenD, GoalBot),                     // 11  left penalty box bottom-inner
        (PenD, PenBot),                      // 12  left penalty box bottom-outer
        (Length / 2, 0.0),                   // 13  halfway line × top touchline
        (Length / 2, HalfW - Circle),        // 14  centre circle top
        (Length / 2, HalfW + Circle),        // 15  centre circle bottom
        (Length / 2, Width),                 // 16  halfway line × bottom touchline
        (Length - PenD, PenTop),             // 17  right penalty box top-outer
        (Length - PenD, GoalTop),            // 18  right penalty box top-inner
        (Length - PenD, GoalBot),            // 19  right penalty box bottom-inner
        (Length - PenD, PenBot),             // 20  right penalty box bottom-outer
        (Length - Spot, HalfW),              // 21  right penalty spot
        (Length - GoalD, GoalTop),           // 22  right 6-yd box top-inner
        (Length - GoalD, GoalBot),           // 23  right 6-yd box bottom-inner
        (Length, 0.0),                       // 24  right goal line × top touchline (top-right corner)
        (Length, PenTop),                    // 25  right goal line × penalty-box top edge extended
        (Length, GoalTop),                   // 26  right goal line × 6-yd box top
        (Length, GoalBot),                   // 27  right goal line × 6-yd box bottom
        (Length, PenBot),                    // 28  right goal line × penalty-box bottom edge extended
        (Length, Width),                     // 29  right goal line × bottom touchline (bottom-right corner)
        (Length / 2 - Circle, HalfW),        // 30  centre circle left point
        (Length / 2 + Circle, HalfW),        // 31  centre circle right point
    };

    public static int Count => Keypoints.Count;
}
