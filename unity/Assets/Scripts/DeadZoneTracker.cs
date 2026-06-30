// DeadZoneTracker: PURE state-machine helper that detects ENTER / EXIT of the FixedUpdate
// "dead-zone" -- the window where (ingame == true && state == null), during which the existing
// DriverController.FixedUpdate branch services NO socket I/O. It computes the DURATION spent in
// that window from MONOTONIC timestamps injected by the caller (Stopwatch.ElapsedMilliseconds or
// Time.realtimeSincestartup-derived ms). It reads NO clock and touches NO Unity / socket state, so
// it is fully unit-testable in an EditMode test (mirrors the pure-helper bar in pop-unity).
//
// OBSERVABILITY ONLY: this class merely OBSERVES the condition values the caller already evaluates
// at the top of FixedUpdate's body. It does not alter the if/else branch behavior in any way.
//
// Usage (per FixedUpdate tick):
//   var ev = tracker.Observe(ingame && state == null, nowMs);
//   if (ev == DeadZoneTracker.Event.Entered) Debug.Log(... deadzone_enter ...);
//   if (ev == DeadZoneTracker.Event.Exited)  Debug.Log(... deadzone_exit duration_ms=tracker.LastDurationMs ...);
public struct DeadZoneTracker
{
    public enum Event
    {
        None,     // no transition this tick
        Entered,  // condition went false -> true this tick (entered the dead-zone)
        Exited    // condition went true -> false this tick (left the dead-zone)
    }

    private bool inDeadZone;
    private double enterMs;
    private double lastDurationMs;

    // True while currently inside the dead-zone (between an Entered and the next Exited).
    public bool InDeadZone => inDeadZone;

    // Duration (ms) of the most recently completed dead-zone window. Valid after an Exited event.
    public double LastDurationMs => lastDurationMs;

    // Feed the current dead-zone condition and a monotonic timestamp (ms). Returns the transition
    // that occurred this tick (None / Entered / Exited). Idempotent while the condition holds:
    // repeated true (or repeated false) yields None, so the caller logs only on transitions -- no
    // per-tick spam. On Exited, LastDurationMs = nowMs - (the enterMs captured at Entered).
    public Event Observe(bool conditionActive, double nowMs)
    {
        if (conditionActive)
        {
            if (!inDeadZone)
            {
                inDeadZone = true;
                enterMs = nowMs;
                return Event.Entered;
            }
            return Event.None;
        }

        if (inDeadZone)
        {
            inDeadZone = false;
            lastDurationMs = nowMs - enterMs;
            return Event.Exited;
        }
        return Event.None;
    }
}
