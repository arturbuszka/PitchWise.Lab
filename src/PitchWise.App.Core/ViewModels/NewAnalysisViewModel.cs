using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using PitchWise.App.Core.Analysis;
using PitchWise.App.Core.Services;
using PitchWise.App.Core.Workspace;

namespace PitchWise.App.Core.ViewModels;

/// <summary>The "analyse something" form.</summary>
public sealed partial class NewAnalysisViewModel : ViewModelBase
{
    private readonly IFileDialogService _dialogs;
    private readonly JobQueueViewModel _queue;

    public NewAnalysisViewModel(IFileDialogService dialogs, JobQueueViewModel queue)
    {
        _dialogs = dialogs;
        _queue = queue;

        // Prefill from the repo the app was built in, if it is still there. Without this the
        // fields are empty, the config's own `models_dir: pretrained_models` resolves against the
        // app's bin directory, and the first run fails with a missing-model error that says
        // nothing about why.
        if (FindRepoRoot() is not { } repo) return;

        var config = Path.Combine(repo, "configs", "soccernet.yaml");
        if (File.Exists(config)) _configPath = config;

        var models = Path.Combine(repo, "pretrained_models");
        if (Directory.Exists(models)) _modelsDir = models;
    }

    /// <summary>The repo the binaries came from, identified by the solution file.</summary>
    private static string? FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "PitchWise.slnx")))
            dir = dir.Parent;
        return dir?.FullName;
    }

    public IReadOnlyList<SourceKind> SourceKinds { get; } = Enum.GetValues<SourceKind>();

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(SourceHint))]
    [NotifyPropertyChangedFor(nameof(IsVideoSource))]
    [NotifyCanExecuteChangedFor(nameof(QueueCommand))]
    private SourceKind _sourceKind = SourceKind.FrameSequence;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(QueueCommand))]
    private string _sourcePath = "";

    [ObservableProperty] private string? _configPath;
    [ObservableProperty] private string? _modelsDir;
    [ObservableProperty] private int _maxFrames;
    [ObservableProperty] private double _startSeconds;
    [ObservableProperty] private double _sourceFps = 25;
    [ObservableProperty] private bool _exportVideo;

    /// <summary>Only video sources can be seeked into before extraction.</summary>
    public bool IsVideoSource => SourceKind == SourceKind.VideoFile;

    public string SourceHint => SourceKind switch
    {
        SourceKind.VideoFile => "A video file; its frames are extracted into the job folder.",
        SourceKind.FrameSequence => "A folder of numbered images, used where they are — no copy.",
        SourceKind.SoccerNetVideo => "A SoccerNet video folder; also loads ground truth for comparison.",
        _ => "",
    };

    [RelayCommand]
    private async Task BrowseAsync()
    {
        var picked = SourceKind == SourceKind.VideoFile
            ? await _dialogs.PickFileAsync("Choose a video", ["*.mp4", "*.mkv", "*.avi", "*.mov"])
            : await _dialogs.PickFolderAsync(SourceKind == SourceKind.FrameSequence
                ? "Choose a folder of frames"
                : "Choose a SoccerNet video folder");
        if (picked is not null) SourcePath = picked;
    }

    [RelayCommand]
    private async Task BrowseConfigAsync()
    {
        if (await _dialogs.PickFileAsync("Choose a pipeline config", ["*.yaml", "*.yml"]) is { } picked)
            ConfigPath = picked;
    }

    [RelayCommand]
    private async Task BrowseModelsAsync()
    {
        if (await _dialogs.PickFolderAsync("Choose the models directory") is { } picked)
            ModelsDir = picked;
    }

    private bool CanQueue => !string.IsNullOrWhiteSpace(SourcePath);

    [RelayCommand(CanExecute = nameof(CanQueue))]
    private void Queue()
    {
        // AnalysisRequest trims and absolutises the paths itself
        _queue.Enqueue(new AnalysisRequest
        {
            SourceKind = SourceKind,
            SourcePath = SourcePath,
            ConfigPath = ConfigPath,
            ModelsDir = ModelsDir,
            MaxFrames = Math.Max(0, MaxFrames),
            StartSeconds = IsVideoSource ? Math.Max(0, StartSeconds) : 0,
            SourceFps = SourceFps > 0 ? SourceFps : 25,
            ExportVideo = ExportVideo,
            // FrameStride stays at its default of 1: see AnalysisRequest.FrameStride
        });
        SourcePath = "";
    }
}
