using UnityEngine;

// FrameCapture: renders the Arena "Main Camera" to an offscreen RenderTexture and returns the
// frame as raw RGB24 bytes, for the OPTIONAL pixel-observation channel (config key "obs_pixels",
// default OFF). Additive Stage-1 seam: NOTHING here runs unless DriverController has obs_pixels=true.
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

    private int width;
    private int height;
    private bool initialized;

    private Camera cam;
    private RenderTexture rt;
    private Texture2D readbackTex;

    // MonoBehaviours cannot take constructor args; DriverController creates this via AddComponent
    // and immediately calls Init() with the configured obs_pixels_width / obs_pixels_height.
    public void Init(int width, int height)
    {
        this.width = width;
        this.height = height;
        initialized = true;
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

        if (rt == null)
        {
            // 24-bit depth so the orthographic sprite/tilemap render has a depth buffer.
            rt = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32);
            rt.Create();
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
            cam.targetTexture = rt;
            cam.Render();

            RenderTexture.active = rt;
            readbackTex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
            readbackTex.Apply(false);
        }
        finally
        {
            cam.targetTexture = prevTarget;
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
        if (rt != null)
        {
            rt.Release();
            Object.Destroy(rt);
            rt = null;
        }
        if (readbackTex != null)
        {
            Object.Destroy(readbackTex);
            readbackTex = null;
        }
    }
}
