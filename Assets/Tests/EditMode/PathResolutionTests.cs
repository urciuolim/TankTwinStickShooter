using System.IO;
using System.Runtime.Serialization;
using NUnit.Framework;
using UnityEngine;

namespace TankTwinStickShooter.EditModeTests
{
    // Regression coverage for DriverController's config/arena path-resolution (M0 cleanup B4).
    // These pin the documented precedence in docs/game-architecture.md ("Config / arena
    // source-of-truth map"). The two methods under test are instance methods but use NO
    // instance state (only their args + Application/File/Path statics), so we exercise them
    // on an uninitialized DriverController created WITHOUT running Awake -- Awake reads/parses
    // the real config file and sets the static singleton, neither of which this test wants.
    // No production logic is changed; the methods were made internal (+ InternalsVisibleTo)
    // purely for visibility.
    public class PathResolutionTests
    {
        private DriverController driver;

        [SetUp]
        public void SetUp()
        {
            // GetUninitializedObject skips the constructor/Awake. Valid for invoking instance
            // methods that don't touch instance fields (both methods here qualify).
            driver = (DriverController)FormatterServices.GetUninitializedObject(typeof(DriverController));
        }

        // ---- ResolveConfigPath: precedence (1) explicit CLI arg ----

        [Test]
        public void ResolveConfigPath_DoubleDashConfigArg_TakesPrecedence()
        {
            string[] args = { "Tank.exe", "50000", "--config", "C:/custom/myconfig.json" };
            Assert.AreEqual("C:/custom/myconfig.json", driver.ResolveConfigPath(args));
        }

        [Test]
        public void ResolveConfigPath_SingleDashConfigArg_TakesPrecedence()
        {
            string[] args = { "Tank.exe", "50000", "-config", "C:/custom/myconfig.json" };
            Assert.AreEqual("C:/custom/myconfig.json", driver.ResolveConfigPath(args));
        }

        [Test]
        public void ResolveConfigPath_ConfigArg_ReturnsValueImmediatelyAfterFlag()
        {
            // The arg right after the flag is the path, regardless of surrounding args.
            string[] args = { "Tank.exe", "-batchmode", "--config", "relative/preset.json", "-extra" };
            Assert.AreEqual("relative/preset.json", driver.ResolveConfigPath(args));
        }

        [Test]
        public void ResolveConfigPath_FirstConfigFlagWins()
        {
            // Loop returns on the first match, so the earliest --config/-config arg wins.
            string[] args = { "Tank.exe", "--config", "first.json", "-config", "second.json" };
            Assert.AreEqual("first.json", driver.ResolveConfigPath(args));
        }

        [Test]
        public void ResolveConfigPath_ConfigFlagAsLastArgWithNoValue_IsIgnored()
        {
            // Loop bound is args.Length - 1, so a trailing flag with no following value is not
            // treated as a match; resolution falls through to the StreamingAssets/legacy chain.
            string[] args = { "Tank.exe", "50000", "--config" };
            string expectedStreaming = Path.Combine(Application.streamingAssetsPath, "config.json");
            string result = driver.ResolveConfigPath(args);
            Assert.IsTrue(
                result == expectedStreaming || result == "Assets/config.json",
                "Expected the StreamingAssets canonical path or the legacy fallback, got: " + result);
        }

        // ---- ResolveConfigPath: precedence (2) StreamingAssets / (3) legacy fallback ----

        [Test]
        public void ResolveConfigPath_NoCliArg_FallsBackToStreamingOrLegacy()
        {
            // With no --config arg, precedence is StreamingAssets/config.json (if present) then
            // the legacy Assets/config.json. In-editor the canonical StreamingAssets file ships,
            // so this normally resolves to the StreamingAssets path; the legacy path is the
            // documented last resort. We accept either to stay robust to fixture presence.
            string[] args = { "Tank.exe", "50000" };
            string expectedStreaming = Path.Combine(Application.streamingAssetsPath, "config.json");
            string result = driver.ResolveConfigPath(args);

            if (File.Exists(expectedStreaming))
                Assert.AreEqual(expectedStreaming, result,
                    "StreamingAssets/config.json exists, so it must take precedence over the legacy fallback.");
            else
                Assert.AreEqual("Assets/config.json", result,
                    "No StreamingAssets/config.json, so the legacy fallback must be returned.");
        }

        // ---- ResolveArenaPath: all three branches ----

        [Test]
        public void ResolveArenaPath_AbsolutePath_UsedAsIs()
        {
            string absolute = Path.GetFullPath("C:/arenas/default.json");
            Assert.AreEqual(absolute,
                driver.ResolveArenaPath("C:/anything/config.json", absolute));
        }

        [Test]
        public void ResolveArenaPath_ExistingRelativePath_UsedAsIs()
        {
            // Legacy "Assets/Arenas/..." form: a relative path that exists relative to the
            // working dir is returned unchanged (not re-anchored to the config dir).
            string tempName = "ttss_arena_test_" + System.Guid.NewGuid().ToString("N") + ".json";
            string tempRelative = tempName; // resolves against the current working dir
            File.WriteAllText(tempRelative, "{}");
            try
            {
                Assert.IsTrue(File.Exists(tempRelative), "precondition: temp arena file exists");
                Assert.AreEqual(tempRelative,
                    driver.ResolveArenaPath("C:/somewhere/else/config.json", tempRelative));
            }
            finally
            {
                File.Delete(tempRelative);
            }
        }

        [Test]
        public void ResolveArenaPath_NonExistingRelativePath_ResolvedAgainstConfigDir()
        {
            // A relative path that does NOT exist on the working dir is anchored to the
            // directory of the resolved config file, so config + its Arenas/ travel together.
            string configPath = "C:/game/StreamingAssets/config.json";
            string arenaRelative = "Arenas/nonexistent_" + System.Guid.NewGuid().ToString("N") + ".json";

            string configDir = Path.GetDirectoryName(Path.GetFullPath(configPath));
            string expected = Path.Combine(configDir, arenaRelative);

            Assert.AreEqual(expected, driver.ResolveArenaPath(configPath, arenaRelative));
        }

        [Test]
        public void ResolveArenaPath_RelativeConfig_AnchorIsConfigFullPathDir()
        {
            // The config path is run through GetFullPath before its directory is taken, so a
            // relative config path is anchored under the current working dir.
            string arenaRelative = "Arenas/missing_" + System.Guid.NewGuid().ToString("N") + ".json";
            string configPath = "configs/preset.json";

            string configDir = Path.GetDirectoryName(Path.GetFullPath(configPath));
            string expected = Path.Combine(configDir, arenaRelative);

            Assert.AreEqual(expected, driver.ResolveArenaPath(configPath, arenaRelative));
        }
    }
}
