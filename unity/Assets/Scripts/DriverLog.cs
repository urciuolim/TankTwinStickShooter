using System;
using System.Globalization;
using System.Text;

// DriverLog: PURE log-line formatting for the DriverController/GameController observability lines
// (the reset/restart-handshake hang investigation). Mirrors the established pure-helper pattern in
// this codebase (WallMessage.Build / FrameCapture.BuildFrameMessage): static, no MonoBehaviour /
// scene / socket dependency, so it is fully unit-testable in an EditMode test.
//
// This is OBSERVABILITY ONLY. None of these strings ever touch the socket -- they go to
// Debug.Log, gated behind DriverController.verbose. They do NOT affect the wire, the 52-float
// state layout, message ordering, or any control flow.
//
// LINE SHAPE (single line, key=value, space-separated, stable & grep-able):
//   "[tag] wall=<ISO-8601 UtcNow 'o'> k1=v1 k2=v2 ..."
// The leading "[tag]" is a stable event tag (e.g. deadzone_enter, ingame_flip). The wall=<...>
// field is the wall-clock stamp (DateTime.UtcNow.ToString("o")) so the C# log cross-correlates
// with the Python files by wall-clock (Python stamps time.time()). DURATIONS are passed in as
// already-measured values from a MONOTONIC source (Stopwatch / realtimeSinceStartup) by the
// caller -- this formatter never reads any clock except the wall-clock stamp it is handed.
//
// Numbers are formatted with InvariantCulture so a comma-decimal locale never corrupts a line.
public static class DriverLog
{
    // Compose one log line: a stable event tag, the wall-clock stamp, and zero or more key/value
    // pairs (keys[i] -> values[i]). keys/values may be null/empty (tag + wall only). The caller
    // supplies utcNow (DateTime.UtcNow) so this method stays pure and deterministically testable.
    public static string Format(string tag, DateTime utcNow, string[] keys, string[] values)
    {
        var sb = new StringBuilder();
        sb.Append('[').Append(tag ?? string.Empty).Append(']');
        sb.Append(" wall=").Append(utcNow.ToString("o", CultureInfo.InvariantCulture));

        if (keys != null && values != null)
        {
            int n = Math.Min(keys.Length, values.Length);
            for (int i = 0; i < n; i++)
            {
                sb.Append(' ').Append(keys[i]).Append('=').Append(values[i]);
            }
        }

        return sb.ToString();
    }

    // Convenience: tag + wall only (no key/values).
    public static string Format(string tag, DateTime utcNow)
    {
        return Format(tag, utcNow, (string[])null, (string[])null);
    }

    // Convenience: one key/value pair (the common case -- e.g. a duration).
    public static string Format(string tag, DateTime utcNow, string key, string value)
    {
        return Format(tag, utcNow, new[] { key }, new[] { value });
    }

    // Format a duration (milliseconds) with InvariantCulture and a fixed 3-decimal precision so
    // sub-millisecond per-step (ReadPixels) timings are legible and locale-independent.
    public static string Ms(double milliseconds)
    {
        return milliseconds.ToString("F3", CultureInfo.InvariantCulture);
    }
}
