using Avalonia.Threading;
using PitchWise.App.Core.Services;

namespace PitchWise.App.Infrastructure;

/// <summary>The real dispatcher. The only place the app's threading touches Avalonia.</summary>
public sealed class AvaloniaUiDispatcher : IUiDispatcher
{
    public bool IsOnUiThread => Dispatcher.UIThread.CheckAccess();

    public void Post(Action action) => Dispatcher.UIThread.Post(action);

    public Task InvokeAsync(Action action) => Dispatcher.UIThread.InvokeAsync(action).GetTask();
}
