using PitchWise.Engine;
using SnGameState.Calibration;
using SnGameState.Detection;
using TrackLab.Datastruct;

namespace SnGameState.WorldState;

/// <summary>
/// The seam between the vision pipeline and the football engine. Turns the enriched detection
/// <see cref="DataFrame"/> — the output of a track run — into the stream of
/// <see cref="FrameObservation"/> the dependency-free <c>PitchWise.Engine</c> consumes. This is
/// the sn-gamestate counterpart of pitchwise's <c>ObservationMapper</c>: the last place anything
/// knows about bboxes, tracks or ONNX; below it everything is football.
///
/// Positions come straight from <see cref="BboxPitch.XBottomMiddle"/>/<c>YBottomMiddle</c> — the
/// player's foot on the pitch, in metres. sn-gamestate's pitch origin is the centre spot rather
/// than a corner, but every threshold in <see cref="EngineConfig"/> is a relative distance, so the
/// origin is irrelevant to possession and passing; only the units (metres) matter.
/// </summary>
public static class ObservationAdapter
{
    /// <summary>Streams one <see cref="FrameObservation"/> per image, in frame order. Detections
    /// without a pitch position are dropped (no homography that frame → nothing to place).</summary>
    /// <param name="detections">The per-detection rows from a completed track run.</param>
    /// <param name="imageMetadatas">Image rows, used for frame ordering and timestamps.</param>
    /// <param name="fps">Video frame rate, to turn a frame index into a timestamp.</param>
    public static IEnumerable<FrameObservation> ToObservations(
        DataFrame detections, DataFrame imageMetadatas, double fps)
    {
        double secondsPerFrame = fps > 0 ? 1.0 / fps : 1.0 / 25.0;

        // Group detections by their image once, so each frame is an O(1) lookup below.
        var byImage = new Dictionary<long, List<Series>>();
        foreach (var d in detections.Rows)
        {
            var imageId = Convert.ToInt64(d["image_id"]);
            if (!byImage.TryGetValue(imageId, out var list))
                byImage[imageId] = list = new List<Series>();
            list.Add(d);
        }

        var orderedImages = imageMetadatas.Rows
            .OrderBy(m => m.GetOrDefault<long>("frame"))
            .ToList();

        foreach (var image in orderedImages)
        {
            long frame = image.GetOrDefault<long>("frame");
            double timestamp = frame * secondsPerFrame;

            byImage.TryGetValue(image.Index, out var rows);
            rows ??= new List<Series>();

            var players = new List<PlayerObservation>(rows.Count);
            var ball = BallObservation.Missing;

            foreach (var row in rows)
            {
                if (row["bbox_pitch"] is not BboxPitch pitch) continue; // no homography this frame
                double x = pitch.XBottomMiddle;
                double y = pitch.YBottomMiddle;
                float conf = row.GetOrDefault<float>("bbox_conf");

                if (Convert.ToInt64(row["category_id"]) == YoloOnnxDetector.BallCategoryId)
                {
                    // Keep the most-confident ball if the frame somehow carries more than one.
                    if (!ball.Detected || conf > ball.Confidence)
                        ball = new BallObservation(x, y, conf, Detected: true);
                    continue;
                }

                // Only tracked people become players: an untracked detection shares no stable id,
                // and the engine refuses possession to PlayerIdentity.None anyway.
                if (!row.Has("track_id")) continue;
                int trackId = (int)row.Get<long>("track_id");

                players.Add(new PlayerObservation(
                    PlayerIdentity.FromTrack(trackId),
                    MapRole(row.GetOrDefault<string>("role")),
                    MapTeam(row.GetOrDefault<string>("team")),
                    x, y, conf));
            }

            // We have pitch positions (metres) whenever bbox_pitch resolved; the engine is told so.
            yield return new FrameObservation((int)frame, timestamp, players, ball, NormalizedCoords: false);
        }
    }

    /// <summary>sn-gamestate roles: ball / goalkeeper / other / player / referee.
    /// "other" and anything unrecognised map to Player — a detection on the pitch that isn't a
    /// referee is, for possession purposes, a player.</summary>
    private static PlayerRole MapRole(string? role) => role switch
    {
        "goalkeeper" => PlayerRole.Goalkeeper,
        "referee" => PlayerRole.Referee,
        _ => PlayerRole.Player,
    };

    /// <summary>TrackletTeamSideLabeling writes "left"/"right"; the engine's two sides are A/B.
    /// An unresolved team stays Unknown, which bars the detection from possessing the ball.</summary>
    private static TeamId MapTeam(string? team) => team switch
    {
        "left" => TeamId.A,
        "right" => TeamId.B,
        _ => TeamId.Unknown,
    };
}
