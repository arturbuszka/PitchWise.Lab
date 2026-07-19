namespace PitchWise.App.Core.Services;

/// <summary>
/// Asks the user to pick a file or folder. Abstracted so view-models that offer a "Browse…"
/// button stay testable and free of a window reference.
/// </summary>
public interface IFileDialogService
{
    /// <param name="title">Dialog title.</param>
    /// <param name="patterns">Glob patterns to offer, e.g. ["*.mp4", "*.mkv"].</param>
    /// <returns>The chosen path, or null if the user cancelled.</returns>
    Task<string?> PickFileAsync(string title, IReadOnlyList<string>? patterns = null);

    /// <returns>The chosen directory, or null if the user cancelled.</returns>
    Task<string?> PickFolderAsync(string title);
}
