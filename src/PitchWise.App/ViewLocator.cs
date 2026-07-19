using Avalonia.Controls;
using Avalonia.Controls.Templates;
using PitchWise.App.Core.ViewModels;

namespace PitchWise.App;

/// <summary>Maps FooViewModel to PitchWise.App.Views.FooView by name.</summary>
public sealed class ViewLocator : IDataTemplate
{
    public Control Build(object? data)
    {
        if (data is null) return new TextBlock { Text = "null" };

        var name = data.GetType().FullName!
            .Replace("PitchWise.App.Core.ViewModels.", "PitchWise.App.Views.")
            .Replace("ViewModel", "View");

        return Type.GetType(name) is { } type
            ? (Control)Activator.CreateInstance(type)!
            : new TextBlock { Text = $"no view for {name}" };
    }

    public bool Match(object? data) => data is ViewModelBase;
}
