using PitchWise.GameState.Calibration;
using PitchWise.GameState.Team;
using PitchWise.Tracking.Datastruct;

namespace PitchWise.GameState.Tests;

public class TrackletRoleCorrectionTests
{
    private static readonly float[] White = { 220, 125, 130 };
    private static readonly float[] Yellow = { 190, 116, 180 };
    private static readonly float[] Black = { 20, 128, 128 };
    private static readonly float[] Green = { 130, 90, 160 };

    /// <summary>One video; image ids equal frame numbers.</summary>
    private sealed class Video
    {
        public DataFrame Detections { get; } = new();

        /// <summary>Detections of <paramref name="trackId"/> in frames [from, to) at pitch x, with a voted role.</summary>
        public void Track(long trackId, string role, float[] color, double x = 0, long from = 0, long to = 100)
        {
            for (var f = from; f < to; f++)
                Detections.Upsert(new Series(Detections.NextIndex)
                {
                    ["image_id"] = f,
                    ["track_id"] = trackId,
                    ["role"] = role,
                    ["torso_color"] = color,
                    ["bbox_pitch"] = new BboxPitch(x, 0, x, 0, x, 0),
                });
        }

        /// <summary>A team of <paramref name="count"/> tracklets from <paramref name="firstId"/>,
        /// spread over midfield, every one with the same voted role.</summary>
        public void Team(long firstId, int count, string role, float[] color)
        {
            for (var i = 0; i < count; i++)
                Track(firstId + i, role, color, x: -20 + 4 * i);
        }

        public Dictionary<long, string> Run(TrackletRoleCorrectionOptions? options = null)
        {
            new TrackletRoleCorrection(options).Process(Detections, new DataFrame());
            return Detections.Rows.GroupBy(r => r.Get<long>("track_id"))
                .ToDictionary(g => g.Key, g => g.Select(r => r.Get<string>("role")).Distinct().Single());
        }
    }

    [Fact]
    public void ACrowdedRefereeGroupIsATeam_AndPlayersFarFromBothTeamsAreReferees()
    {
        var video = new Video();
        video.Team(1, 10, "player", White);
        video.Team(11, 10, "referee", Yellow);   // 10 "referees" in every frame
        video.Track(21, "player", Black);         // the real referee

        var roles = video.Run();

        Assert.All(Enumerable.Range(1, 20), id => Assert.Equal("player", roles[id]));
        Assert.Equal("referee", roles[21]);
    }

    [Fact]
    public void AFewRefereesAtATimeAreLeftAlone()
    {
        var video = new Video();
        video.Team(1, 10, "player", White);
        video.Team(11, 10, "player", Yellow);
        video.Track(21, "referee", Black);
        video.Track(22, "referee", Yellow);   // a referee in a team's colour stays a referee
        video.Track(23, "referee", Black);

        var roles = video.Run();

        Assert.Equal(new[] { "referee", "referee", "referee" }, new[] { roles[21], roles[22], roles[23] });
        Assert.All(Enumerable.Range(1, 20), id => Assert.Equal("player", roles[id]));
    }

    [Fact]
    public void AGoalkeeperVotedPlayerIsFoundByColourAndPosition()
    {
        var video = new Video();
        video.Team(1, 10, "player", White);
        video.Team(11, 10, "player", Yellow);
        video.Track(21, "player", Green, x: 48);    // in a colour of its own near the right goal
        video.Track(22, "player", Black, x: 5);     // odd colour in midfield: not a goalkeeper

        var roles = video.Run();

        Assert.Equal("goalkeeper", roles[21]);
        Assert.Equal("player", roles[22]);
        Assert.All(Enumerable.Range(1, 20), id => Assert.Equal("player", roles[id]));
    }

    [Fact]
    public void OnlyTheFarthestColourBecomesGoalkeeperInEachHalf()
    {
        var video = new Video();
        video.Team(1, 10, "player", White);
        video.Team(11, 10, "player", Yellow);
        video.Track(21, "player", Green, x: -45);
        video.Track(22, "player", new float[] { 210, 115, 150 }, x: -44);   // between the teams, closer to them

        var roles = video.Run(new TrackletRoleCorrectionOptions { GoalkeeperColorDistance = 5 });

        Assert.Equal("goalkeeper", roles[21]);
        Assert.Equal("player", roles[22]);
    }

    [Fact]
    public void AnOutsiderNearAGoalIsTheGoalkeeperNotAReferee()
    {
        var video = new Video();
        video.Team(1, 10, "player", White);
        video.Team(11, 10, "referee", Yellow);
        video.Track(21, "player", Green, x: -50);
        video.Track(22, "player", Black, x: 10);

        var roles = video.Run();

        Assert.Equal("goalkeeper", roles[21]);
        Assert.Equal("referee", roles[22]);
    }

    [Fact]
    public void ACrowdedGoalkeeperGroupIsATeamToo()
    {
        var video = new Video();
        video.Team(1, 10, "player", White);
        video.Team(11, 10, "goalkeeper", Yellow);

        var roles = video.Run();

        Assert.All(Enumerable.Range(1, 20), id => Assert.Equal("player", roles[id]));
    }

    [Fact]
    public void ShadowChangesLightnessNotTeam()
    {
        var video = new Video();
        video.Team(1, 10, "player", White);
        video.Track(11, "player", new float[] { 150, 125, 130 });   // a white shirt in the shade
        video.Team(12, 10, "referee", Yellow);

        var roles = video.Run();

        Assert.Equal("player", roles[11]);
    }

    [Fact]
    public void ShortTrackletsKeepTheirRole()
    {
        var video = new Video();
        video.Team(1, 10, "player", White);
        video.Team(11, 10, "referee", Yellow);
        video.Track(21, "referee", Yellow, from: 0, to: 5);   // too few coloured detections

        var roles = video.Run();

        Assert.Equal("referee", roles[21]);
        Assert.Equal("player", roles[11]);
    }
}
