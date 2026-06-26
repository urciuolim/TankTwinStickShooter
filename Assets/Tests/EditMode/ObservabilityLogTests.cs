using System;
using System.Globalization;
using NUnit.Framework;

namespace TankTwinStickShooter.EditModeTests
{
    // Pins the PURE observability helpers used by DriverController/GameController for the reset/
    // restart-handshake hang investigation: DriverLog (log-line formatting) and DeadZoneTracker
    // (FixedUpdate dead-zone enter/exit/duration). Both are pure (no MonoBehaviour / scene / socket),
    // so they are fully unit-testable here -- mirroring FrameFramingTests / WallMessageTests. These
    // helpers are OBSERVABILITY ONLY: they never touch the wire, the 52-float state, ordering, or
    // control flow.
    public class ObservabilityLogTests
    {
        // --- DriverLog ---------------------------------------------------------------------------

        [Test]
        public void Format_TagAndWall_ProducesStableShape()
        {
            DateTime t = new DateTime(2026, 6, 26, 12, 0, 0, DateTimeKind.Utc);
            string line = DriverLog.Format("deadzone_enter", t);
            Assert.AreEqual("[deadzone_enter] wall=" + t.ToString("o", CultureInfo.InvariantCulture), line);
        }

        [Test]
        public void Format_UsesRoundTripIso8601WallClock()
        {
            // The "o" round-trip format is what lets Python cross-correlate by wall-clock.
            DateTime t = new DateTime(2026, 1, 2, 3, 4, 5, 678, DateTimeKind.Utc);
            string line = DriverLog.Format("evt", t);
            StringAssert.Contains("wall=2026-01-02T03:04:05.6780000Z", line);
        }

        [Test]
        public void Format_SingleKeyValue_AppendsAfterWall()
        {
            DateTime t = new DateTime(2026, 6, 26, 0, 0, 0, DateTimeKind.Utc);
            string line = DriverLog.Format("deadzone_exit", t, "duration_ms", "30000.000");
            StringAssert.StartsWith("[deadzone_exit] wall=", line);
            StringAssert.EndsWith(" duration_ms=30000.000", line);
        }

        [Test]
        public void Format_MultipleKeyValues_PreserveOrder()
        {
            DateTime t = new DateTime(2026, 6, 26, 0, 0, 0, DateTimeKind.Utc);
            string line = DriverLog.Format("ingame_flip", t,
                new[] { "from", "to", "cause" },
                new[] { "false", "true", "start" });
            StringAssert.EndsWith(" from=false to=true cause=start", line);
        }

        [Test]
        public void Format_MismatchedKeyValueLengths_UsesShorter()
        {
            DateTime t = DateTime.UtcNow;
            // 2 keys, 1 value => only the first pair is emitted (no exception, no partial garbage).
            string line = DriverLog.Format("evt", t, new[] { "a", "b" }, new[] { "1" });
            StringAssert.EndsWith(" a=1", line);
        }

        [Test]
        public void Format_NullKeysOrValues_TagAndWallOnly()
        {
            DateTime t = DateTime.UtcNow;
            string line = DriverLog.Format("evt", t, (string[])null, (string[])null);
            Assert.AreEqual("[evt] wall=" + t.ToString("o", CultureInfo.InvariantCulture), line);
        }

        [Test]
        public void Ms_FormatsInvariantThreeDecimals()
        {
            // Locale-independent: '.' decimal, 3 places (legible for sub-ms ReadPixels timings).
            Assert.AreEqual("1.500", DriverLog.Ms(1.5));
            Assert.AreEqual("0.001", DriverLog.Ms(0.001));
            Assert.AreEqual("30000.000", DriverLog.Ms(30000.0));
        }

        // --- DeadZoneTracker ---------------------------------------------------------------------

        [Test]
        public void DeadZone_FirstActiveTick_Enters()
        {
            var dz = new DeadZoneTracker();
            Assert.AreEqual(DeadZoneTracker.Event.Entered, dz.Observe(true, 100d));
            Assert.IsTrue(dz.InDeadZone);
        }

        [Test]
        public void DeadZone_RepeatedActive_NoSpam()
        {
            var dz = new DeadZoneTracker();
            dz.Observe(true, 100d);
            // Subsequent active ticks must NOT re-fire Entered (would spam every FixedUpdate).
            Assert.AreEqual(DeadZoneTracker.Event.None, dz.Observe(true, 200d));
            Assert.AreEqual(DeadZoneTracker.Event.None, dz.Observe(true, 300d));
        }

        [Test]
        public void DeadZone_Exit_ReportsDurationFromInjectedTimestamps()
        {
            var dz = new DeadZoneTracker();
            dz.Observe(true, 1000d);           // enter at t=1000ms
            dz.Observe(true, 5000d);           // still in (no event)
            var ev = dz.Observe(false, 31000d); // exit at t=31000ms -> 30000ms window
            Assert.AreEqual(DeadZoneTracker.Event.Exited, ev);
            Assert.IsFalse(dz.InDeadZone);
            Assert.AreEqual(30000d, dz.LastDurationMs, 1e-9);
        }

        [Test]
        public void DeadZone_InactiveWhenNeverEntered_NoEvent()
        {
            var dz = new DeadZoneTracker();
            Assert.AreEqual(DeadZoneTracker.Event.None, dz.Observe(false, 0d));
            Assert.AreEqual(DeadZoneTracker.Event.None, dz.Observe(false, 100d));
            Assert.IsFalse(dz.InDeadZone);
        }

        [Test]
        public void DeadZone_ReEnter_TracksSecondWindowIndependently()
        {
            var dz = new DeadZoneTracker();
            dz.Observe(true, 0d);
            dz.Observe(false, 10d);            // first window = 10ms
            Assert.AreEqual(10d, dz.LastDurationMs, 1e-9);

            Assert.AreEqual(DeadZoneTracker.Event.Entered, dz.Observe(true, 100d));
            Assert.AreEqual(DeadZoneTracker.Event.Exited, dz.Observe(false, 250d));
            Assert.AreEqual(150d, dz.LastDurationMs, 1e-9); // second window = 150ms
        }
    }
}
