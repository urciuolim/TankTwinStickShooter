using Newtonsoft.Json.Linq;
using NUnit.Framework;

namespace TankTwinStickShooter.EditModeTests
{
    // Pins the OUTBOUND LENGTH-PREFIX FRAMING + message classification in DriverProtocol so the
    // Unity read-exactly path stays byte-for-byte in lockstep with the Python sender
    // (pop_trainer.core.protocol.encode_framed / Connection.send). Both helpers are pure (no
    // MonoBehaviour, socket, or scene), so they are fully unit-testable here.
    //
    // Length-prefix contract under test (matches pop_trainer.core.protocol):
    //   [0..3] uint32 BIG-ENDIAN = UTF-8 JSON byte length (counts ONLY the JSON payload)
    //   [4..]  the JSON payload
    public class DriverProtocolTests
    {
        [Test]
        public void LengthPrefixBytes_IsFour()
        {
            Assert.AreEqual(4, DriverProtocol.LengthPrefixBytes);
        }

        [Test]
        public void EncodeLengthPrefix_IsBigEndian()
        {
            // 691200 = 0x000A8C00 -> big-endian bytes 00 0A 8C 00.
            byte[] prefix = DriverProtocol.EncodeLengthPrefix(691200);
            Assert.AreEqual(4, prefix.Length);
            Assert.AreEqual(0x00, prefix[0]);
            Assert.AreEqual(0x0A, prefix[1]);
            Assert.AreEqual(0x8C, prefix[2]);
            Assert.AreEqual(0x00, prefix[3]);
        }

        [Test]
        public void EncodeLengthPrefix_SmallValue()
        {
            // 13 = 0x0000000D -> 00 00 00 0D.
            byte[] prefix = DriverProtocol.EncodeLengthPrefix(13);
            Assert.AreEqual(0x00, prefix[0]);
            Assert.AreEqual(0x00, prefix[1]);
            Assert.AreEqual(0x00, prefix[2]);
            Assert.AreEqual(0x0D, prefix[3]);
        }

        [Test]
        public void DecodeLengthPrefix_InvertsEncode()
        {
            foreach (int n in new[] { 0, 1, 13, 255, 256, 65535, 65536, 691200, 16777215 })
            {
                byte[] prefix = DriverProtocol.EncodeLengthPrefix(n);
                Assert.AreEqual(n, DriverProtocol.DecodeLengthPrefix(prefix), "round-trip failed for " + n);
            }
        }

        [Test]
        public void DecodeLengthPrefix_MatchesPythonBigEndianBytes()
        {
            // Bytes exactly as Python's int.to_bytes(4, "big") would emit for 691200.
            byte[] fromPython = { 0x00, 0x0A, 0x8C, 0x00 };
            Assert.AreEqual(691200, DriverProtocol.DecodeLengthPrefix(fromPython));
        }

        [Test]
        public void DecodeLengthPrefix_HonorsOffset()
        {
            // Prefix sits after 2 leading bytes; decode from offset 2.
            byte[] buffer = { 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x0D };
            Assert.AreEqual(13, DriverProtocol.DecodeLengthPrefix(buffer, 2));
        }

        [Test]
        public void DecodeLengthPrefix_OfEncodedJsonLength_MatchesPayloadByteCount()
        {
            // The end-to-end framing invariant the Unity read-exactly loop relies on: the prefix
            // value equals the number of UTF-8 JSON bytes that follow it.
            JObject message = JObject.Parse("{\"1\":[0.5,-0.5,0.0,0.0,1.0],\"2\":[0.0,0.0,0.0,0.0,0.0]}");
            byte[] payload = System.Text.Encoding.UTF8.GetBytes(message.ToString(Newtonsoft.Json.Formatting.None));
            byte[] prefix = DriverProtocol.EncodeLengthPrefix(payload.Length);
            Assert.AreEqual(payload.Length, DriverProtocol.DecodeLengthPrefix(prefix));
        }

        [Test]
        public void Classify_Start()
        {
            Assert.AreEqual(DriverProtocol.MessageKind.Start,
                DriverProtocol.Classify(JObject.Parse("{\"start\":true}")));
        }

        [Test]
        public void Classify_End()
        {
            Assert.AreEqual(DriverProtocol.MessageKind.End,
                DriverProtocol.Classify(JObject.Parse("{\"end\":true}")));
        }

        [Test]
        public void Classify_Restart()
        {
            Assert.AreEqual(DriverProtocol.MessageKind.Restart,
                DriverProtocol.Classify(JObject.Parse("{\"restart\":true}")));
        }

        [Test]
        public void Classify_SwitchArena()
        {
            Assert.AreEqual(DriverProtocol.MessageKind.SwitchArena,
                DriverProtocol.Classify(JObject.Parse("{\"switch_arena\":\"maps/center_block.json\"}")));
        }

        [Test]
        public void Classify_StepActionMessage_IsAction()
        {
            // The per-step integer-keyed action message carries no control key.
            Assert.AreEqual(DriverProtocol.MessageKind.Action,
                DriverProtocol.Classify(JObject.Parse("{\"1\":[0,0,0,0,0],\"2\":[0,0,0,0,0]}")));
        }

        [Test]
        public void Classify_FalseControlFlag_IsNotThatControl()
        {
            // A false boolean control flag must NOT be treated as that control (matches the
            // original Value<bool>() truthiness probe); with no other key it is an Action payload.
            Assert.AreEqual(DriverProtocol.MessageKind.Action,
                DriverProtocol.Classify(JObject.Parse("{\"start\":false}")));
        }

        [Test]
        public void Classify_Null_IsUnknown()
        {
            Assert.AreEqual(DriverProtocol.MessageKind.Unknown, DriverProtocol.Classify(null));
        }
    }
}
