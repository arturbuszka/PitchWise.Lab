using Avalonia.Controls;
using Avalonia.Platform.Storage;
using PitchWise.App.Core.Services;

namespace PitchWise.App.Infrastructure;

/// <summary>
/// File and folder pickers, backed by Avalonia's storage provider.
/// </summary>
/// <remarks>
/// The provider hangs off a top-level window, which does not exist when the container is built,
/// so the window is supplied later by the shell rather than injected.
/// </remarks>
public sealed class FileDialogService : IFileDialogService
{
    private Window? _owner;

    public void Attach(Window owner) => _owner = owner;

    public async Task<string?> PickFileAsync(string title, IReadOnlyList<string>? patterns = null)
    {
        if (_owner?.StorageProvider is not { } storage) return null;

        var files = await storage.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = title,
            AllowMultiple = false,
            FileTypeFilter = patterns is null
                ? null
                : [new FilePickerFileType("Supported") { Patterns = [.. patterns] }],
        });
        return files.Count > 0 ? files[0].TryGetLocalPath() : null;
    }

    public async Task<string?> PickFolderAsync(string title)
    {
        if (_owner?.StorageProvider is not { } storage) return null;

        var folders = await storage.OpenFolderPickerAsync(new FolderPickerOpenOptions
        {
            Title = title,
            AllowMultiple = false,
        });
        return folders.Count > 0 ? folders[0].TryGetLocalPath() : null;
    }
}
