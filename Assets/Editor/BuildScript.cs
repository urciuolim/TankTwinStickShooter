using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;

// Headless build entry point for the Tank Twin Stick Shooter.
// Invoke from the command line, e.g.:
//   Unity.exe -batchmode -quit -projectPath <proj> \
//     -executeMethod BuildScript.BuildWindows -logFile -
//
// Output goes to <project>/build/TankTwinStickShooter.exe (git-ignored). This is the
// game_path the Python env launches via subprocess.Popen([game_path, str(port)]).
// StreamingAssets (config.json + Arenas/*.json) ships automatically with the player.
public static class BuildScript
{
    private const string ProductExe = "TankTwinStickShooter.exe";

    [MenuItem("Build/Build Windows (Standalone)")]
    public static void BuildWindows()
    {
        string projectRoot = Directory.GetParent(Application.dataPath).FullName;
        string outputDir = Path.Combine(projectRoot, "build");
        string locationPathName = Path.Combine(outputDir, ProductExe);

        Directory.CreateDirectory(outputDir);

        string[] scenes = EditorBuildSettings.scenes
            .Where(s => s.enabled)
            .Select(s => s.path)
            .ToArray();

        if (scenes.Length == 0)
            throw new Exception("BuildScript: no enabled scenes in EditorBuildSettings; aborting.");

        Debug.Log("BuildScript: building " + scenes.Length + " scene(s) -> " + locationPathName);
        foreach (string s in scenes)
            Debug.Log("BuildScript: scene -> " + s);

        BuildPlayerOptions options = new BuildPlayerOptions
        {
            scenes = scenes,
            locationPathName = locationPathName,
            target = BuildTarget.StandaloneWindows64,
            targetGroup = BuildTargetGroup.Standalone,
            options = BuildOptions.None
        };

        var report = BuildPipeline.BuildPlayer(options);
        var summary = report.summary;

        Debug.Log("BuildScript: result=" + summary.result
            + " totalErrors=" + summary.totalErrors
            + " sizeBytes=" + summary.totalSize
            + " output=" + summary.outputPath);

        if (summary.result != UnityEditor.Build.Reporting.BuildResult.Succeeded)
        {
            // Non-zero exit so -batchmode callers can detect failure.
            EditorApplication.Exit(1);
        }
    }
}
