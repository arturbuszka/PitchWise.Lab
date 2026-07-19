using PitchWise.App.Core.Analysis;
using PitchWise.App.Core.Workspace;

namespace PitchWise.App.Core.Tests;

/// <summary>
/// Paths must not depend on the working directory: the CLI runs from the repo root while the app
/// runs from its own bin folder, and a job outlives the process that queued it.
/// </summary>
public class AnalysisRequestPathTests
{
    [Fact]
    public void Paths_AreAbsoluteWhateverTheWorkingDirectoryIs()
    {
        var request = new AnalysisRequest
        {
            SourceKind = SourceKind.FrameSequence,
            SourcePath = "data/SoccerNetGS/valid/SNGS-021/img1",
            ConfigPath = "configs/cpu.yaml",
            ModelsDir = "pretrained_models",
        };

        Assert.True(Path.IsPathRooted(request.SourcePath), request.SourcePath);
        Assert.True(Path.IsPathRooted(request.ConfigPath!), request.ConfigPath);
        Assert.True(Path.IsPathRooted(request.ModelsDir!), request.ModelsDir);
    }

    [Fact]
    public void AbsolutePaths_SurviveUntouched()
    {
        var absolute = Path.GetFullPath(Path.Combine(Path.GetTempPath(), "clip.mp4"));

        var request = new AnalysisRequest
        {
            SourceKind = SourceKind.VideoFile,
            SourcePath = absolute,
        };

        Assert.Equal(absolute, request.SourcePath);
    }

    [Fact]
    public void BlankOptionalPaths_BecomeNullRatherThanTheWorkingDirectory()
    {
        // an empty text box must mean "unset", not "the process's current directory"
        var request = new AnalysisRequest
        {
            SourceKind = SourceKind.FrameSequence,
            SourcePath = Path.GetTempPath(),
            ConfigPath = "   ",
            ModelsDir = "",
        };

        Assert.Null(request.ConfigPath);
        Assert.Null(request.ModelsDir);
    }

    [Fact]
    public void SurroundingWhitespace_IsTrimmedOffPastedPaths()
    {
        var request = new AnalysisRequest
        {
            SourceKind = SourceKind.VideoFile,
            SourcePath = $"  {Path.Combine(Path.GetTempPath(), "clip.mp4")}  ",
        };

        Assert.Equal(Path.Combine(Path.GetTempPath(), "clip.mp4"), request.SourcePath);
    }

    [Fact]
    public void DisplayName_StillComesFromTheSourceAfterAbsolutising()
    {
        var request = new AnalysisRequest
        {
            SourceKind = SourceKind.FrameSequence,
            SourcePath = "data/SoccerNetGS/valid/SNGS-021/img1",
        };

        // absolutising must not defeat the img1-means-its-parent rule
        Assert.Equal("SNGS-021", request.ResolveDisplayName());
    }
}
