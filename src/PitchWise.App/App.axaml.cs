using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using Microsoft.Extensions.DependencyInjection;
using PitchWise.App.Core.ViewModels;
using PitchWise.App.Infrastructure;
using PitchWise.App.Views;

namespace PitchWise.App;

public partial class App : Application
{
    private ServiceProvider? _services;

    public override void Initialize() => AvaloniaXamlLoader.Load(this);

    public override void OnFrameworkInitializationCompleted()
    {
        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        {
            _services = ServiceRegistration.Build();

            var window = new ShellWindow { DataContext = _services.GetRequiredService<ShellViewModel>() };
            // the file pickers need a top-level window, which only exists now
            _services.GetRequiredService<FileDialogService>().Attach(window);
            desktop.MainWindow = window;

            desktop.ShutdownRequested += (_, _) => _services.Dispose();
        }

        base.OnFrameworkInitializationCompleted();
    }
}
