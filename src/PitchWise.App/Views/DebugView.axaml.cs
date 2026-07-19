using Avalonia.Controls;
using Avalonia.Input;
using PitchWise.App.Core.ViewModels;

namespace PitchWise.App.Views;

public partial class DebugView : UserControl
{
    public DebugView() => InitializeComponent();

    /// <summary>Double-clicking a tracklet jumps to where it starts.</summary>
    private void OnTrackletDoubleTapped(object? sender, TappedEventArgs e)
    {
        if (sender is DataGrid { SelectedItem: TrackletSummary tracklet }
            && DataContext is DebugViewModel vm)
            vm.GoToTrackletCommand.Execute(tracklet);
    }
}
