namespace PitchWise.App.Core.Services;

/// <summary>
/// Marshals work onto the UI thread.
/// </summary>
/// <remarks>
/// View-models depend on this rather than Avalonia's Dispatcher.UIThread, which is what keeps
/// this project free of an Avalonia reference and lets view-model tests run with a dispatcher
/// that simply invokes inline.
/// </remarks>
public interface IUiDispatcher
{
    bool IsOnUiThread { get; }

    /// <summary>Queues work on the UI thread and returns immediately.</summary>
    void Post(Action action);

    /// <summary>Runs work on the UI thread and completes when it has run.</summary>
    Task InvokeAsync(Action action);
}
