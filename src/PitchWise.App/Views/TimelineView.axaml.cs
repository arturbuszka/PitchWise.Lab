using Avalonia.Controls;
using PitchWise.App.Core.ViewModels;

namespace PitchWise.App.Views;

public partial class TimelineView : UserControl
{
    public TimelineView()
    {
        InitializeComponent();
        // the bar reports a pick; routing it through the view-model's command keeps the seek
        // decision there rather than in the control
        Bar.MarkerPicked += marker =>
        {
            if (DataContext is TimelineViewModel vm) vm.GoToCommand.Execute(marker);
        };
    }
}
