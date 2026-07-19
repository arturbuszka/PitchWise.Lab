using PitchWise.App.Core.Workspace;
using PitchWise.Tracking.Datastruct;

namespace PitchWise.App.Core.Playback;

/// <summary>Chooses how to read a finished job's frames back.</summary>
public interface IFrameSourceFactory
{
    /// <exception cref="InvalidOperationException">Neither the frames nor the source video are readable.</exception>
    IFrameSource Create(AnalysisManifest manifest, DataFrame imagePreds);
}

/// <summary>
/// Prefers the frames the analysis worked from, and falls back to the source video only when
/// they are gone — the fallback seeks approximately, which the UI has to warn about.
/// </summary>
public sealed class FrameSourceFactory : IFrameSourceFactory
{
    public IFrameSource Create(AnalysisManifest manifest, DataFrame imagePreds)
    {
        var framesExist = imagePreds.Rows
            .FirstOrDefault(r => r.Has("file_path")) is { } first
            && File.Exists(first.Get<string>("file_path"));

        if (framesExist)
            return new FrameSequenceSource(imagePreds, manifest.Fps);

        if (manifest.SourceKind == SourceKind.VideoFile && File.Exists(manifest.SourcePath))
            return new VideoFileSource(manifest.SourcePath, imagePreds, manifest.Fps);

        throw new InvalidOperationException(
            $"the frames of '{manifest.DisplayName}' are gone and its source ({manifest.SourcePath}) " +
            "is not a readable video");
    }
}
