using System.Collections;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace TankTwinStickShooter.PlayModeTests
{
    // FULL-VIEW (no side-crop) GUARD for FrameCapture's supersample path.
    //
    // FrameCapture renders the orthographic "Main Camera" at a 16:9 intermediate and bilinear-
    // downscales to the obs resolution, so the captured frame must always show the WHOLE 16:9 view --
    // even at a SQUARE obs (the 16:9 view is SQUISHED into the square, never side-cropped). The OLD
    // code pointed the camera straight at a square RT without setting cam.aspect, which shrank the
    // orthographic horizontal extent and CROPPED the arena's left/right; this test fails on that bug.
    //
    // This drives the REAL GPU path (cam.Render + Graphics.Blit + ReadPixels) on a self-contained
    // orthographic rig (no Driver/socket needed): a "Main Camera" tagged like the Arena's, framing a
    // 16:9 region whose FULL WIDTH is covered by bright content that reaches the live view's left and
    // right edges. After a SQUARE capture we assert bright content survives in BOTH the leftmost and
    // rightmost obs columns -- impossible if the sides were cropped. A 16:9 capture is also checked to
    // confirm the proportional (undistorted) downscale still reaches both edges.
    public class FrameCaptureViewTests
    {
        private GameObject camGo;
        private GameObject contentGo;
        private FrameCapture capture;
        private GameObject captureHost;

        private const float OrthoSize = 4.6701684f;          // matches Arena.unity's Main Camera
        private const float LiveAspect = 16f / 9f;            // the live 16:9 game view

        [SetUp]
        public void SetUp()
        {
            // Orthographic camera tagged MainCamera so FrameCapture.Camera.main / "Main Camera" finds
            // it. Background BLACK so "bright content" is unambiguous against it.
            camGo = new GameObject("Main Camera");
            camGo.tag = "MainCamera";
            var cam = camGo.AddComponent<Camera>();
            cam.orthographic = true;
            cam.orthographicSize = OrthoSize;
            cam.aspect = LiveAspect;
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = Color.black;
            cam.transform.position = new Vector3(0f, 0f, -10f);

            // A bright quad sized to the FULL 16:9 view: half-width = OrthoSize*aspect spans from the
            // left edge to the right edge of the live framing, full height. If the capture crops the
            // sides, the obs left/right columns go black; if it shows the full view, they stay bright.
            float halfH = OrthoSize;
            float halfW = OrthoSize * LiveAspect;
            contentGo = GameObject.CreatePrimitive(PrimitiveType.Quad);
            contentGo.name = "FullWidthContent";
            Object.DestroyImmediate(contentGo.GetComponent<Collider>());
            contentGo.transform.position = new Vector3(0f, 0f, 0f);
            contentGo.transform.localScale = new Vector3(halfW * 2f, halfH * 2f, 1f);
            var mr = contentGo.GetComponent<MeshRenderer>();
            // Unlit-ish bright material (default sprite/standard may be dark without a light); force a
            // fully-bright unlit color so the readback is unambiguous regardless of scene lighting.
            mr.sharedMaterial = new Material(Shader.Find("Unlit/Color")) { color = Color.white };

            captureHost = new GameObject("CaptureHost");
            capture = captureHost.AddComponent<FrameCapture>();
        }

        [TearDown]
        public void TearDown()
        {
            if (contentGo != null) Object.DestroyImmediate(contentGo);
            if (camGo != null) Object.DestroyImmediate(camGo);
            if (captureHost != null) Object.DestroyImmediate(captureHost); // OnDestroy releases RTs
        }

        // Reshape the bottom-up RGB24 payload and return whether ANY pixel in obs column `col` is
        // bright (clearly above the black background). Rows are bottom-up; column indexing is the
        // same either way since we scan the whole column.
        private static bool ColumnHasBrightPixel(byte[] rgb, int w, int h, int col)
        {
            for (int row = 0; row < h; row++)
            {
                int idx = (row * w + col) * 3;
                // Bright = any channel well above black. The bilinear edge may dim the rim, so use a
                // modest threshold rather than pure white.
                if (rgb[idx] > 64 || rgb[idx + 1] > 64 || rgb[idx + 2] > 64)
                    return true;
            }
            return false;
        }

        [UnityTest]
        public IEnumerator SquareObs_CapturesFullWidth_NoSideCrop()
        {
            const int n = 64; // SQUARE obs -- the case the old crop bug broke.
            capture.Init(n, n);

            // Let the rig render at least one frame before the manual capture render.
            yield return null;

            byte[] rgb = capture.CaptureRGB(out int w, out int h);
            Assert.IsNotNull(rgb, "CaptureRGB returned null (no camera found?)");
            Assert.AreEqual(n, w);
            Assert.AreEqual(n, h);
            Assert.AreEqual(n * n * 3, rgb.Length, "payload must be W*H*3");

            // The full-width content must reach BOTH edge columns -> proves no left/right crop.
            Assert.IsTrue(ColumnHasBrightPixel(rgb, w, h, 0),
                "leftmost obs column is background -> the left side was CROPPED (full 16:9 view not captured)");
            Assert.IsTrue(ColumnHasBrightPixel(rgb, w, h, w - 1),
                "rightmost obs column is background -> the right side was CROPPED (full 16:9 view not captured)");
        }

        [UnityTest]
        public IEnumerator WideObs_CapturesFullWidth_NoSideCrop()
        {
            // 16:9 obs: the proportional (undistorted) downscale must still reach both edges.
            capture.Init(64, 36);
            yield return null;

            byte[] rgb = capture.CaptureRGB(out int w, out int h);
            Assert.IsNotNull(rgb, "CaptureRGB returned null (no camera found?)");
            Assert.AreEqual(64, w);
            Assert.AreEqual(36, h);

            Assert.IsTrue(ColumnHasBrightPixel(rgb, w, h, 0),
                "leftmost obs column is background -> left side cropped at 16:9 obs");
            Assert.IsTrue(ColumnHasBrightPixel(rgb, w, h, w - 1),
                "rightmost obs column is background -> right side cropped at 16:9 obs");
        }
    }
}
