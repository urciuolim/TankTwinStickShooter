using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;

namespace TankTwinStickShooter.EditModeTests
{
    // Pins the AREA-AVERAGE downscale chain (FrameCapture.ComputeHalvingChain), the pure logic behind
    // the progressive 2x bilinear-halving path. The chain replaces the old single large-reduction
    // Graphics.Blit (which 4-tapped per output pixel and skipped >90% of source pixels on a ~6-11x
    // shrink, washing out the tank's ~1px wireframe). These tests assert the chain is a TRUE area
    // filter -- every source pixel contributes -- by checking:
    //   * each step at most halves the still-too-large axis (so no step under-samples), and
    //   * after the chain, each axis is in [dst, 2*dst), so the final bilinear Blit to the exact obs
    //     dims is at most a 2x reduction (a clean 2x2 average), not a large skip-sampling blit.
    // Pure + GPU-free, so it runs in the headless EditMode suite. The actual GPU Blit chain + readback
    // (CaptureRGB) is exercised by the PlayMode FrameCaptureViewTests and the Director's native
    // re-capture verify.
    public class FrameDownscaleChainTests
    {
        private const int SrcW = FrameCapture.IntermediateWidth;  // 1280
        private const int SrcH = FrameCapture.IntermediateHeight; // 720

        // Walks one chain and asserts the area-filter invariants. After the chain, the next bilinear
        // Blit lands on (dstW, dstH).
        private static void AssertValidChain(int dstW, int dstH)
        {
            List<Vector2Int> chain = FrameCapture.ComputeHalvingChain(SrcW, SrcH, dstW, dstH);

            int prevW = SrcW;
            int prevH = SrcH;
            foreach (Vector2Int step in chain)
            {
                // Each axis either stayed (already within 2x of dst) or was exactly halved (floored).
                // It must never grow and never drop below the destination.
                Assert.IsTrue(step.x == prevW || step.x == prevW / 2,
                    $"W step {step.x} from {prevW} is neither unchanged nor a clean halving (dst={dstW})");
                Assert.IsTrue(step.y == prevH || step.y == prevH / 2,
                    $"H step {step.y} from {prevH} is neither unchanged nor a clean halving (dst={dstH})");
                Assert.GreaterOrEqual(step.x, dstW, "a halving step dropped W below the destination");
                Assert.GreaterOrEqual(step.y, dstH, "a halving step dropped H below the destination");
                prevW = step.x;
                prevH = step.y;
            }

            // Post-chain size must be in [dst, 2*dst] on BOTH axes: the final Blit to dst is then at
            // most a 2x reduction (a clean 2x2 average), so no large-reduction skip-sampling occurs.
            // The upper bound is inclusive: an exact 2x final reduction IS the clean halving case we
            // want (e.g. 1280x720 -> 640x360 needs no halving step and finals on a single 2x Blit).
            Assert.GreaterOrEqual(prevW, dstW);
            Assert.LessOrEqual(prevW, 2 * dstW, $"post-chain W={prevW} is > 2*dst -> final Blit would skip-sample (dst={dstW})");
            Assert.GreaterOrEqual(prevH, dstH);
            Assert.LessOrEqual(prevH, 2 * dstH, $"post-chain H={prevH} is > 2*dst -> final Blit would skip-sample (dst={dstH})");
        }

        [Test]
        public void Chain_Square64_StaysWithinAreaFilterBounds()
        {
            AssertValidChain(64, 64);
        }

        [Test]
        public void Chain_Square96_StaysWithinAreaFilterBounds()
        {
            AssertValidChain(96, 96);
        }

        [Test]
        public void Chain_Square128_StaysWithinAreaFilterBounds()
        {
            AssertValidChain(128, 128);
        }

        [Test]
        public void Chain_Square256_StaysWithinAreaFilterBounds()
        {
            AssertValidChain(256, 256);
        }

        [Test]
        public void Chain_Wide64x36_StaysWithinAreaFilterBounds()
        {
            AssertValidChain(64, 36);
        }

        [Test]
        public void Chain_Wide640x360_StaysWithinAreaFilterBounds()
        {
            AssertValidChain(640, 360);
        }

        [Test]
        public void Chain_LargeObs_NoHalvingSteps()
        {
            // A destination already >= half the source on both axes needs NO halving -- a single
            // bilinear Blit is at most a 2x reduction. 1280x720 -> 700x400: 700 > 640 and 400 > 360,
            // so the chain must be empty.
            List<Vector2Int> chain = FrameCapture.ComputeHalvingChain(SrcW, SrcH, 700, 400);
            Assert.AreEqual(0, chain.Count, "destination within one 2x step should produce an empty chain");
        }

        [Test]
        public void Chain_Square128_ProducesExpectedSteps()
        {
            // Spot-check the concrete chain for the Director's 128x128 re-capture target. Each axis
            // halves only while it is still > 2*dst (=256), independently:
            //   start (1280,720)
            //   step1 (640,360)   w:1280>256->640   h:720>256->360
            //   step2 (320,180)   w:640>256->320    h:360>256->180
            //   step3 (160,180)   w:320>256->160    h:180>256? no -> 180 unchanged
            //   stop  (160 in [128,256), 180 in [128,256)); final Blit 160x180 -> 128x128.
            List<Vector2Int> chain = FrameCapture.ComputeHalvingChain(SrcW, SrcH, 128, 128);
            Assert.AreEqual(3, chain.Count);
            Assert.AreEqual(new Vector2Int(640, 360), chain[0]);
            Assert.AreEqual(new Vector2Int(320, 180), chain[1]);
            Assert.AreEqual(new Vector2Int(160, 180), chain[2]);
        }

        [Test]
        public void Chain_EveryStepReadsEntirePredecessor()
        {
            // The area-filter guarantee: across ALL obs sizes, no chain step reduces an axis by more
            // than 2x (a >2x bilinear step would skip source pixels). Verified by AssertValidChain's
            // "unchanged or exactly halved" check; here we additionally confirm the FINAL Blit (chain
            // tail -> dst) is also <= 2x on both axes for a representative spread.
            int[][] obs =
            {
                new[] { 64, 64 }, new[] { 96, 96 }, new[] { 128, 128 }, new[] { 256, 256 },
                new[] { 64, 36 }, new[] { 84, 84 }, new[] { 200, 150 },
            };
            foreach (int[] o in obs)
            {
                List<Vector2Int> chain = FrameCapture.ComputeHalvingChain(SrcW, SrcH, o[0], o[1]);
                int tailW = chain.Count > 0 ? chain[chain.Count - 1].x : SrcW;
                int tailH = chain.Count > 0 ? chain[chain.Count - 1].y : SrcH;
                Assert.LessOrEqual(tailW, 2 * o[0], $"final Blit W reduction > 2x for obs {o[0]}x{o[1]}");
                Assert.LessOrEqual(tailH, 2 * o[1], $"final Blit H reduction > 2x for obs {o[0]}x{o[1]}");
            }
        }
    }
}
