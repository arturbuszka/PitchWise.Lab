using Avalonia.Controls;
using PitchWise.App.Core.Playback;
using PitchWise.App.Core.ViewModels;

namespace PitchWise.App.Views;

/// <summary>
/// Wires the playback loop's frames to the control that draws them.
/// </summary>
/// <remarks>
/// The one piece of code-behind with any logic: frames are Mats handed over on a background
/// thread with a copy-before-return contract, which is not something binding can express.
/// </remarks>
public partial class PlayerView : UserControl
{
    private PlayerViewModel? _subscribed;

    public PlayerView()
    {
        InitializeComponent();
        DataContextChanged += (_, _) => Rebind();
        DetachedFromVisualTree += (_, _) => Unsubscribe();
    }

    private void Rebind()
    {
        Unsubscribe();
        if (DataContext is not PlayerViewModel vm) return;

        vm.FrameReady += OnFrameReady;
        _subscribed = vm;
        // only now that something is listening is it worth drawing the first frame
        vm.ShowFirstFrame();
    }

    private void Unsubscribe()
    {
        if (_subscribed is null) return;
        _subscribed.FrameReady -= OnFrameReady;
        _subscribed = null;
    }

    /// <summary>Called on the playback thread; blits before returning, as the loop requires.</summary>
    private void OnFrameReady(RenderedFrame frame) => Frame.Show(frame.Image);
}
