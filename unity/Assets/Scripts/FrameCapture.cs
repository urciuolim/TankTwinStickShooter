using System.Collections.Generic;
using UnityEngine;

// FrameCapture: renders the Arena "Main Camera" at a high-res 16:9 intermediate and AREA-AVERAGE
// downsamples (progressive 2x halving) to the obs resolution, returning the frame as raw RGB24
// bytes for the OPTIONAL pixel-observation channel (config key "obs_pixels", default OFF).
// Rendering the full 16:9 view (then resizing) keeps the captured frame undistorted-by-crop: the obs
// always shows the WHOLE arena (a square obs squishes the 16:9 view into the square, never crops its
// sides), and the area-average downscale preserves the tank's thin (~1px) wireframe + barrel down to
// small obs sizes. Additive Stage-1 seam: NOTHING here runs unless DriverController has
// obs_pixels=true.
//
// =====================================================================================
// FRAME WIRE CONTRACT (must match the Python framed-read byte-for-byte)
// =====================================================================================
// When obs_pixels is enabled, DriverController.SendAndReceiveData() writes the existing raw
// `state` JSON UNCHANGED, then writes ONE contiguous framed binary message built by
// BuildFrameMessage() below. Layout of that message (all multi-byte ints BIG-ENDIAN):
//
//   offset 0 : 1 byte  : magic/type tag = ASCII 'F' (0x46)
//   offset 1 : 4 bytes : uint32 PAYLOAD length = W*H*3 (the RGB bytes ONLY; does NOT
//                        include the tag, this length field, or the W/H/C header below)
//   offset 5 : 2 bytes : uint16 width  W
//   offset 7 : 2 bytes : uint16 height H
//   offset 9 : 1 byte  : channels C (always 3)
//   offset 10: W*H*3   : raw uint8 RGB payload (3 bytes/pixel, R,G,B order)
//
// So total message length = 1 (tag) + 4 (len) + 5 (W/H/C header) + W*H*3 (payload) = 10 + W*H*3.
// The length field counts ONLY the W*H*3 payload (it is the value Python uses for its
// read-exactly loop on the pixel bytes; the fixed 10-byte prefix is read first). The W/H/C
// header sits AFTER the length field but is fixed-size, so Python reads the 10-byte prefix,
// asserts tag=='F', parses W/H/C and len, then reads exactly `len` (== W*H*3) payload bytes.
//
// PIXEL ORIGIN: Texture2D.ReadPixels / GetRawTextureData on a RGB24 texture returns rows
// BOTTOM-UP (row 0 = bottom of the image), left-to-right within a row. We ship those bytes
// AS-IS (no flip) for C#-side simplicity; Python reshapes to (H, W, 3) and flips vertically
// (e.g. np.flipud) to get a top-left-origin image. Documented so Python matches exactly.
// =====================================================================================
//
// NOTE: ReadPixels is a SYNCHRONOUS GPU readback. Acceptable for Stage 1 because the sim is
// Python-clocked with Time.timeScale==0 at capture time (correctness/determinism over
// throughput). AsyncGPUReadback is a possible future optimization but is intentionally NOT
// used here. No UnityEditor APIs are referenced, so this compiles into the shipping player.
public class FrameCapture : MonoBehaviour
{
    public const byte FrameTag = 0x46; // ASCII 'F'
    public const byte Channels = 3;    // RGB24

    // AREA-AVERAGE DOWNSCALE (resize, never crop). The camera renders the FULL 16:9 view into a
    // HIGH-res 16:9 intermediate RenderTexture (so the orthographic horizontal FOV matches the live
    // game and NOTHING is cropped). A single large-reduction bilinear Blit only 4-taps per OUTPUT
    // pixel, so on a ~6-11x shrink it samples the source ~2 texels apart and SKIPS >90% of source
    // pixels -- the tank's thin wireframe lines fall between the samples and wash out. Instead we
    // PROGRESSIVELY HALVE (720 -> 360 -> 180 -> ... ) with bilinear Blits into cached half-size
    // ping-pong RTs until just above the obs size, then a final bilinear Blit to the exact obs dims.
    // Each 2x bilinear step IS a 2x2 box average, so the chain is a true area filter where EVERY
    // source pixel contributes -- matching a proper image-library resize (PIL BILINEAR) and keeping
    // the ~1px wireframe + barrel legible at small obs sizes. The obs frame always shows the FULL
    // view: a 16:9 obs downscales proportionally; a SQUARE obs squishes the whole arena into the
    // square (full content + AA, never a side crop). 16:9 because DriverController's pixel defaults
    // are 16:9 to match the build's render aspect (undistorted vs the live game view).
    public const int IntermediateWidth = 1280;  // 720p, 16:9; high enough that the area-average
    public const int IntermediateHeight = 720;  // downscale is a clean filter for any obs <= 720p.

    private int width;
    private int height;
    private bool initialized;

    private Camera cam;
    private RenderTexture intermediateRt;            // 16:9 full-view render target (high-res)
    private RenderTexture obsRt;                      // obs-sized final downscale target
    private readonly List<RenderTexture> halfRts = new List<RenderTexture>(); // ping-pong halving chain
    private Texture2D readbackTex;

    // MonoBehaviours cannot take constructor args; DriverController creates this via AddComponent
    // and immediately calls Init() with the configured obs_pixels_width / obs_pixels_height.
    public void Init(int width, int height)
    {
        this.width = width;
        this.height = height;
        initialized = true;
    }

    // Compute the intermediate-size dimensions of each progressive-halving step for a downscale from
    // (srcW, srcH) to (dstW, dstH). Pure + static so the chain length/sizes are unit-testable without
    // a GPU. Each step halves whichever axis is still more than 2x the destination (floored, never
    // below the destination); halving stops once BOTH axes are within 2x of the destination, after
    // which a single final bilinear Blit (NOT in this list) lands on the exact destination. Returns
    // the intermediate sizes in order (excludes the source and the final destination); empty if the
    // destination is already within one bilinear step (>= half the source on both axes).
    public static List<Vector2Int> ComputeHalvingChain(int srcW, int srcH, int dstW, int dstH)
    {
        var chain = new List<Vector2Int>();
        int w = srcW;
        int h = srcH;
        // Halve an axis only while doing so stays at/above the destination, i.e. while it is still
        // more than 2x the destination. This guarantees the post-chain size is in [dst, 2*dst) on
        // each axis, so the final bilinear Blit to dst is at most a 2x reduction (a clean 2x2 average).
        while (w > 2 * dstW || h > 2 * dstH)
        {
            if (w > 2 * dstW)
                w = w / 2;
            if (h > 2 * dstH)
                h = h / 2;
            chain.Add(new Vector2Int(w, h));
        }
        return chain;
    }

    private void EnsureResources()
    {
        if (cam == null)
        {
            // The Arena scene has a single "Main Camera" (orthographic). Camera.main resolves it
            // via the MainCamera tag; fall back to a tag-independent search if needed.
            cam = Camera.main;
            if (cam == null)
            {
                GameObject go = GameObject.Find("Main Camera");
                if (go != null)
                    cam = go.GetComponent<Camera>();
            }
        }

        if (intermediateRt == null)
        {
            // 16:9 high-res full-view target. 24-bit depth so the orthographic sprite/tilemap render
            // has a depth buffer. Bilinear so each Blit step averages neighbours.
            intermediateRt = new RenderTexture(IntermediateWidth, IntermediateHeight, 24, RenderTextureFormat.ARGB32);
            intermediateRt.filterMode = FilterMode.Bilinear;
            intermediateRt.Create();
        }

        if (halfRts.Count == 0)
        {
            // Cache the ping-pong half-size RTs for the progressive-halving chain (computed once from
            // the fixed intermediate + obs dims; obs dims do not change at runtime). No depth needed:
            // each step is a full-screen Blit, not a scene render. Bilinear so every 2x step is a 2x2
            // box average -- the chain == a true area filter.
            foreach (Vector2Int size in ComputeHalvingChain(IntermediateWidth, IntermediateHeight, width, height))
            {
                var rt = new RenderTexture(size.x, size.y, 0, RenderTextureFormat.ARGB32);
                rt.filterMode = FilterMode.Bilinear;
                rt.Create();
                halfRts.Add(rt);
            }
        }

        if (obsRt == null)
        {
            // Obs-sized final downscale target. No depth needed (full-screen Blit). Bilinear too, for
            // any later sampling of the result.
            obsRt = new RenderTexture(width, height, 0, RenderTextureFormat.ARGB32);
            obsRt.filterMode = FilterMode.Bilinear;
            obsRt.Create();
        }

        if (readbackTex == null)
        {
            readbackTex = new Texture2D(width, height, TextureFormat.RGB24, false);
        }
    }

    // Render the Arena camera offscreen and return raw RGB24 bytes (W*H*3). Returns null (and
    // logs once) if no camera is available. Rendering the camera into a RenderTexture excludes
    // Screen-Space-Overlay canvases (the HUD draws only to the screen backbuffer), so the HUD is
    // culled for free even when present; in AI-vs-AI mode GameController also SetActive(false)s it.
    //
    // The scene renders at the 16:9 intermediate resolution (full view, no crop), then a progressive
    // bilinear-halving chain area-averages it down to the SMALL obs dims (the throughput win is
    // preserved -- only the render + the cheap GPU Blits are high-res; the readback stays small).
    public byte[] CaptureRGB(out int w, out int h)
    {
        w = width;
        h = height;

        if (!initialized)
        {
            Debug.LogError("FrameCapture: CaptureRGB called before Init(); no frame produced.");
            return null;
        }

        EnsureResources();

        if (cam == null)
        {
            Debug.LogError("FrameCapture: no Main Camera found in the active scene; cannot capture frame.");
            return null;
        }

        RenderTexture prevTarget = cam.targetTexture;
        RenderTexture prevActive = RenderTexture.active;
        try
        {
            // Force the camera aspect to the intermediate's 16:9 for the capture render so the
            // orthographic horizontal extent matches the live game view (no side crop), independent
            // of the obs RT's aspect. Cleared with ResetAspect() in finally (reverts the camera to
            // automatic screen-aspect, its state before this override).
            cam.aspect = (float)IntermediateWidth / IntermediateHeight;
            cam.targetTexture = intermediateRt;
            cam.Render();

            // Progressive area-average downscale: bilinear-Blit through each cached half-size step
            // (each a 2x2 box average so EVERY source pixel contributes), then a final bilinear Blit
            // onto the exact obs dims. The full 16:9 content is preserved (a square obs squishes it,
            // never side-crops) and the thin wireframe survives the shrink.
            RenderTexture src = intermediateRt;
            for (int i = 0; i < halfRts.Count; i++)
            {
                Graphics.Blit(src, halfRts[i]);
                src = halfRts[i];
            }
            Graphics.Blit(src, obsRt);

            RenderTexture.active = obsRt;
            readbackTex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
            readbackTex.Apply(false);
        }
        finally
        {
            cam.targetTexture = prevTarget;
            cam.ResetAspect();
            RenderTexture.active = prevActive;
        }

        // RGB24 Texture2D -> exactly W*H*3 raw bytes (rows bottom-up; see PIXEL ORIGIN note above).
        return readbackTex.GetRawTextureData();
    }

    // Pure header+message assembler. Extracted as a static method so the framing/byte layout is
    // unit-testable WITHOUT a camera (EditMode test). Takes the already-captured RGB payload and
    // wraps it per the wire contract above. Assembles ONE contiguous byte[] so DriverController can
    // issue a single nwStream.Write (TCP may fragment; one Write keeps the bytes contiguous on our
    // side and the Python read-exactly loop reassembles).
    public static byte[] BuildFrameMessage(byte[] rgb, int w, int h)
    {
        int payloadLen = w * h * 3;
        byte[] msg = new byte[1 + 4 + 5 + payloadLen];
        int o = 0;

        msg[o++] = FrameTag;

        // 4-byte BIG-ENDIAN payload length (W*H*3 only). Most-significant byte first, written
        // explicitly so the byte order is unambiguous regardless of host endianness.
        msg[o++] = (byte)((payloadLen >> 24) & 0xFF);
        msg[o++] = (byte)((payloadLen >> 16) & 0xFF);
        msg[o++] = (byte)((payloadLen >> 8) & 0xFF);
        msg[o++] = (byte)(payloadLen & 0xFF);

        // 2-byte BIG-ENDIAN width, 2-byte BIG-ENDIAN height, 1-byte channels.
        msg[o++] = (byte)((w >> 8) & 0xFF);
        msg[o++] = (byte)(w & 0xFF);
        msg[o++] = (byte)((h >> 8) & 0xFF);
        msg[o++] = (byte)(h & 0xFF);
        msg[o++] = Channels;

        if (rgb != null)
            System.Buffer.BlockCopy(rgb, 0, msg, o, payloadLen);

        return msg;
    }

    private void OnDestroy()
    {
        if (intermediateRt != null)
        {
            intermediateRt.Release();
            Object.Destroy(intermediateRt);
            intermediateRt = null;
        }
        foreach (RenderTexture rt in halfRts)
        {
            if (rt != null)
            {
                rt.Release();
                Object.Destroy(rt);
            }
        }
        halfRts.Clear();
        if (obsRt != null)
        {
            obsRt.Release();
            Object.Destroy(obsRt);
            obsRt = null;
        }
        if (readbackTex != null)
        {
            Object.Destroy(readbackTex);
            readbackTex = null;
        }
    }
}
