using NUnit.Framework;

namespace TankTwinStickShooter.EditModeTests
{
    // Pins the FRAME WIRE CONTRACT byte layout produced by FrameCapture.BuildFrameMessage so the
    // Python framed-read stays in lockstep. The header/length assembly is pure (no camera/GPU), so
    // it is fully unit-testable here; the actual GPU readback (CaptureRGB) needs a live PlayMode run
    // against the rebuilt player and is owned by the training-engineer's capture verify.
    //
    // Contract under test (all multi-byte ints BIG-ENDIAN):
    //   [0]    tag 'F' (0x46)
    //   [1..4] uint32 payload length = W*H*3 (RGB bytes only)
    //   [5..6] uint16 width
    //   [7..8] uint16 height
    //   [9]    channels = 3
    //   [10..] W*H*3 raw RGB payload
    public class FrameFramingTests
    {
        [Test]
        public void BuildFrameMessage_HasCorrectTotalLength()
        {
            int w = 4, h = 3;
            byte[] rgb = new byte[w * h * 3];
            byte[] msg = FrameCapture.BuildFrameMessage(rgb, w, h);
            // 1 (tag) + 4 (len) + 5 (W/H/C header) + W*H*3 payload.
            Assert.AreEqual(10 + w * h * 3, msg.Length);
        }

        [Test]
        public void BuildFrameMessage_FirstByteIsFrameTag()
        {
            byte[] rgb = new byte[2 * 2 * 3];
            byte[] msg = FrameCapture.BuildFrameMessage(rgb, 2, 2);
            Assert.AreEqual(0x46, msg[0]);
            Assert.AreEqual(FrameCapture.FrameTag, msg[0]);
        }

        [Test]
        public void BuildFrameMessage_LengthFieldIsBigEndianPayloadLength()
        {
            // 640x360x3 = 691200 = 0x000A8C00. Big-endian => bytes 00 0A 8C 00.
            int w = 640, h = 360;
            byte[] rgb = new byte[w * h * 3];
            byte[] msg = FrameCapture.BuildFrameMessage(rgb, w, h);

            int payloadLen = w * h * 3;
            Assert.AreEqual((byte)((payloadLen >> 24) & 0xFF), msg[1]);
            Assert.AreEqual((byte)((payloadLen >> 16) & 0xFF), msg[2]);
            Assert.AreEqual((byte)((payloadLen >> 8) & 0xFF), msg[3]);
            Assert.AreEqual((byte)(payloadLen & 0xFF), msg[4]);

            Assert.AreEqual(0x00, msg[1]);
            Assert.AreEqual(0x0A, msg[2]);
            Assert.AreEqual(0x8C, msg[3]);
            Assert.AreEqual(0x00, msg[4]);
        }

        [Test]
        public void BuildFrameMessage_WidthHeightChannelsAreBigEndian()
        {
            // W=640=0x0280, H=360=0x0168, C=3.
            int w = 640, h = 360;
            byte[] rgb = new byte[w * h * 3];
            byte[] msg = FrameCapture.BuildFrameMessage(rgb, w, h);

            Assert.AreEqual(0x02, msg[5]); // width hi
            Assert.AreEqual(0x80, msg[6]); // width lo
            Assert.AreEqual(0x01, msg[7]); // height hi
            Assert.AreEqual(0x68, msg[8]); // height lo
            Assert.AreEqual(3, msg[9]);    // channels
            Assert.AreEqual(FrameCapture.Channels, msg[9]);
        }

        [Test]
        public void BuildFrameMessage_PayloadIsCopiedVerbatimAfterHeader()
        {
            int w = 2, h = 2; // 12 RGB bytes
            byte[] rgb = new byte[w * h * 3];
            for (int i = 0; i < rgb.Length; i++)
                rgb[i] = (byte)(i + 1); // 1..12, distinct, non-zero

            byte[] msg = FrameCapture.BuildFrameMessage(rgb, w, h);

            for (int i = 0; i < rgb.Length; i++)
                Assert.AreEqual(rgb[i], msg[10 + i], "payload byte mismatch at index " + i);
        }

        [Test]
        public void BuildFrameMessage_LengthFieldMatchesActualPayloadBytes()
        {
            // The declared length (bytes 1..4) must equal the number of payload bytes that follow
            // the 10-byte prefix -- this is exactly what the Python read-exactly loop relies on.
            int w = 7, h = 5;
            byte[] rgb = new byte[w * h * 3];
            byte[] msg = FrameCapture.BuildFrameMessage(rgb, w, h);

            int declared = (msg[1] << 24) | (msg[2] << 16) | (msg[3] << 8) | msg[4];
            Assert.AreEqual(w * h * 3, declared);
            Assert.AreEqual(declared, msg.Length - 10);
        }
    }
}
