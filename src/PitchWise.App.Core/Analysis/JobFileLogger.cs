using Microsoft.Extensions.Logging;

namespace PitchWise.App.Core.Analysis;

/// <summary>
/// Appends a job's log lines to its own file, so the debug panel can show what a run did long
/// after the console it was started from is gone.
/// </summary>
/// <remarks>
/// A whole logging provider would be the tidy answer, but a job log is a single file with a
/// single writer and a known lifetime; wiring one up through ILoggerFactory would be more
/// ceremony than the thing being built. Logging failures are swallowed: a job must not die
/// because its log file is locked.
/// </remarks>
public sealed class JobFileLogger : ILogger, IDisposable
{
    private readonly StreamWriter? _writer;
    private readonly ILogger? _inner;
    private readonly object _lock = new();

    /// <param name="path">File to append to.</param>
    /// <param name="inner">Also forwarded here, so a run still logs to the console it was started from.</param>
    public JobFileLogger(string path, ILogger? inner = null)
    {
        _inner = inner;
        try
        {
            _writer = new StreamWriter(path, append: true) { AutoFlush = true };
        }
        catch (IOException)
        {
            _writer = null;   // best-effort, exactly like WorldStateJsonl.TryCreate
        }
        catch (UnauthorizedAccessException)
        {
            _writer = null;
        }
    }

    public IDisposable? BeginScope<TState>(TState state) where TState : notnull
        => _inner?.BeginScope(state);

    public bool IsEnabled(LogLevel logLevel) => logLevel >= LogLevel.Information;

    public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception,
        Func<TState, Exception?, string> formatter)
    {
        _inner?.Log(logLevel, eventId, state, exception, formatter);
        if (_writer is null || !IsEnabled(logLevel)) return;

        var line = $"{DateTime.Now:HH:mm:ss} {Abbreviate(logLevel)} {formatter(state, exception)}";
        try
        {
            lock (_lock)
            {
                _writer.WriteLine(line);
                if (exception is not null) _writer.WriteLine(exception);
            }
        }
        catch (IOException)
        {
        }
        catch (ObjectDisposedException)
        {
        }
    }

    private static string Abbreviate(LogLevel level) => level switch
    {
        LogLevel.Trace => "trce",
        LogLevel.Debug => "dbug",
        LogLevel.Information => "info",
        LogLevel.Warning => "warn",
        LogLevel.Error => "fail",
        LogLevel.Critical => "crit",
        _ => "none",
    };

    public void Dispose()
    {
        lock (_lock) _writer?.Dispose();
    }
}
