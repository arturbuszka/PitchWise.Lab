using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using PitchWise.App.Core.Analysis;
using PitchWise.App.Core.Playback;
using PitchWise.App.Core.Services;
using PitchWise.App.Core.Tuning;
using PitchWise.App.Core.ViewModels;
using PitchWise.App.Core.Workspace;
using PitchWise.App.Infrastructure;

namespace PitchWise.App;

/// <summary>
/// The composition root.
/// </summary>
/// <remarks>
/// The libraries and App.Core stay container-free — every type in them is plain-new-able, which
/// is what keeps the view-models testable and lets the CLI go on wiring by hand in Program.cs.
/// A container earns its place only here, because a GUI has lifetimes a CLI verb does not: the
/// store, the runner and the dispatcher live as long as the process, while view-models are
/// created on user action from inside views. Threading those by hand would mean a static service
/// locator, which is worse.
/// </remarks>
internal static class ServiceRegistration
{
    public static ServiceProvider Build() => new ServiceCollection()
        .AddLogging(b => b.AddSimpleConsole())   // mirrors the CLI's logging
        .AddSingleton<IUiDispatcher, AvaloniaUiDispatcher>()
        .AddSingleton<FileDialogService>()
        .AddSingleton<IFileDialogService>(sp => sp.GetRequiredService<FileDialogService>())
        .AddSingleton<IWorkspaceStore>(sp => new WorkspaceStore())
        .AddSingleton<IAnalysisJobRunner, AnalysisJobRunner>()
        .AddSingleton<IFrameSourceFactory, FrameSourceFactory>()
        // a factory, not an instance: FrameClock paces from its own last-frame timestamp, so
        // every playback loop needs a fresh one — a singleton clock injected into the singleton
        // shell would be shared by every analysis opened in the session
        .AddSingleton<Func<IFrameClock>>(_ => () => new FrameClock())
        .AddSingleton<ITuningService, TuningService>()
        .AddSingleton<JobQueueViewModel>()
        .AddSingleton<NewAnalysisViewModel>()
        .AddSingleton<ShellViewModel>()
        .BuildServiceProvider();
}
