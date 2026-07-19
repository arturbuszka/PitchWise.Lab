using OpenCvSharp;
using PitchWise.GameState.Dataset;

namespace PitchWise.GameState.Tests;

public class FrameSequenceSetTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), $"frameseq-test-{Guid.NewGuid():N}");

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    private string CreateFrames(int count, int width = 64, int height = 36)
    {
        Directory.CreateDirectory(_root);
        using var frame = new Mat(new Size(width, height), MatType.CV_8UC3, new Scalar(20, 90, 20));
        for (var i = 1; i <= count; i++)
            Cv2.ImWrite(Path.Combine(_root, $"{i:D6}.jpg"), frame);
        return _root;
    }

    [Fact]
    public void Load_ProducesTheSameColumnContractAsTheOtherLoaders()
    {
        var set = FrameSequenceSet.Load(CreateFrames(3), fps: 25);

        var video = set.VideoMetadatas.Rows.Single();
        Assert.Equal(1L, video.Get<long>("id"));
        Assert.Equal(25L, video.Get<long>("frame_rate"));
        Assert.Equal(64L, video.Get<long>("im_width"));
        Assert.Equal(36L, video.Get<long>("im_height"));
        Assert.Equal(3L, video.Get<long>("nframes"));

        Assert.Equal(3, set.ImageMetadatas.Count);
        var images = set.ImageMetadatas.Rows.OrderBy(r => r.Get<long>("frame")).ToList();
        Assert.Equal([0L, 1L, 2L], images.Select(i => i.Get<long>("frame")));
        foreach (var image in images)
        {
            Assert.Equal(1L, image.Get<long>("video_id"));
            Assert.Equal(image.Index, image.Get<long>("id"));
            Assert.True(File.Exists(image.Get<string>("file_path")));
        }
    }

    [Fact]
    public void Load_PointsAtTheOriginalsRatherThanCopyingThem()
    {
        var dir = CreateFrames(2);
        var set = FrameSequenceSet.Load(dir);

        // the whole point of this loader: no re-encoding, no second copy on disk
        foreach (var image in set.ImageMetadatas.Rows)
            Assert.Equal(dir, Path.GetDirectoryName(image.Get<string>("file_path")));
        Assert.Equal(2, Directory.GetFiles(dir).Length);
    }

    [Fact]
    public void Load_HonoursStrideAndMaxFrames()
    {
        var dir = CreateFrames(10);

        var strided = FrameSequenceSet.Load(dir, frameStride: 3);
        Assert.Equal(4, strided.ImageMetadatas.Count);   // 1st, 4th, 7th, 10th
        Assert.EndsWith("000001.jpg", strided.ImageMetadatas.Rows.First().Get<string>("file_path"));
        Assert.EndsWith("000004.jpg", strided.ImageMetadatas.Rows.Skip(1).First().Get<string>("file_path"));

        var capped = FrameSequenceSet.Load(dir, maxFrames: 3);
        Assert.Equal(3, capped.ImageMetadatas.Count);
        Assert.Equal(3L, capped.VideoMetadatas.Rows.Single().Get<long>("nframes"));

        // stride applies before the cap, so the two compose
        var both = FrameSequenceSet.Load(dir, frameStride: 2, maxFrames: 3);
        Assert.Equal(3, both.ImageMetadatas.Count);
        Assert.EndsWith("000005.jpg", both.ImageMetadatas.Rows.Last().Get<string>("file_path"));
    }

    [Fact]
    public void Load_OrdersFramesOrdinallyAndIgnoresNonImages()
    {
        var dir = CreateFrames(2);
        File.WriteAllText(Path.Combine(dir, "notes.txt"), "not a frame");

        var set = FrameSequenceSet.Load(dir);

        Assert.Equal(2, set.ImageMetadatas.Count);
        Assert.All(set.ImageMetadatas.Rows,
            i => Assert.EndsWith(".jpg", i.Get<string>("file_path")));
    }

    [Fact]
    public void Load_ThrowsWhenThereIsNothingToLoad()
    {
        Assert.Throws<DirectoryNotFoundException>(
            () => FrameSequenceSet.Load(Path.Combine(_root, "nope")));

        Directory.CreateDirectory(_root);
        Assert.Throws<FileNotFoundException>(() => FrameSequenceSet.Load(_root));
    }
}
