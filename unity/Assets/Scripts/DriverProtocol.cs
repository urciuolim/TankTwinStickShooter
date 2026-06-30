using Newtonsoft.Json.Linq;

// DriverProtocol: PURE protocol helpers extracted from DriverController so they are unit-testable
// in an EditMode test (no MonoBehaviour, socket, or Unity-scene dependency). Two responsibilities:
//
//  1. OUTBOUND LENGTH-PREFIX FRAMING (Python -> Unity). Every control/step message Python sends is
//     prefixed with a 4-byte BIG-ENDIAN uint32 = the UTF-8 JSON byte length, followed by that JSON
//     (see pop_trainer.core.protocol.encode_framed). DriverController reads EXACTLY those 4 bytes,
//     then reads EXACTLY that many bytes, then JObject.Parse -- a read-exactly frame, NOT the old
//     single nwStream.Read(ReceiveBufferSize) that assumed one JSON per recv. DecodeLengthPrefix
//     parses the prefix; EncodeLengthPrefix is the symmetric writer (pinned by the EditMode test so
//     the byte order stays in lockstep with the Python side). The prefix counts ONLY the JSON
//     payload (it excludes the 4 prefix bytes). The Unity->Python pixel FRAME channel keeps its own
//     separate 10-byte header (FrameCapture); inbound JSON Unity writes stays UNFRAMED (Python's
//     receive() brace-scans it).
//
//  2. MESSAGE CLASSIFICATION. Classify() maps a parsed inbound JObject to the control action it
//     requests, so the waiting-state read and the (safety-cap) mid-round read share ONE tested
//     dispatch instead of duplicating the key-probing if/else chain.
public static class DriverProtocol
{
    // 4-byte big-endian length prefix; matches pop_trainer.core.protocol.SEND_LENGTH_PREFIX_LEN.
    public const int LengthPrefixBytes = 4;

    // The control action an inbound Python message requests. Action = a per-step {"1":..,"2":..}
    // message (no control key); Unknown = anything else (ignored, never desyncs the wire).
    public enum MessageKind
    {
        Unknown,
        Start,
        End,
        Restart,
        SwitchArena,
        Action
    }

    // Write a 4-byte BIG-ENDIAN uint32 length prefix. Most-significant byte first, written
    // explicitly so the order is unambiguous regardless of host endianness (mirrors
    // FrameCapture.BuildFrameMessage's length field and the Python big-endian prefix).
    public static byte[] EncodeLengthPrefix(int payloadLen)
    {
        return new byte[]
        {
            (byte)((payloadLen >> 24) & 0xFF),
            (byte)((payloadLen >> 16) & 0xFF),
            (byte)((payloadLen >> 8) & 0xFF),
            (byte)(payloadLen & 0xFF)
        };
    }

    // Parse a 4-byte BIG-ENDIAN uint32 length prefix starting at `offset`. The inverse of
    // EncodeLengthPrefix and of the Python int.to_bytes(4, "big") prefix.
    public static int DecodeLengthPrefix(byte[] prefix, int offset = 0)
    {
        return (prefix[offset] << 24)
            | (prefix[offset + 1] << 16)
            | (prefix[offset + 2] << 8)
            | prefix[offset + 3];
    }

    // Classify a parsed inbound message by its control key (the booleans must be truthy, matching
    // the original Value<bool>() probes). A {"1":..,"2":..} step message carries no control key and
    // is classified Action; null / an unrecognized object is Unknown (ignored, no wire desync).
    public static MessageKind Classify(JObject message)
    {
        if (message == null)
            return MessageKind.Unknown;
        if (message["start"] != null && message["start"].Value<bool>())
            return MessageKind.Start;
        if (message["end"] != null && message["end"].Value<bool>())
            return MessageKind.End;
        if (message["restart"] != null && message["restart"].Value<bool>())
            return MessageKind.Restart;
        if (message["switch_arena"] != null)
            return MessageKind.SwitchArena;
        return MessageKind.Action;
    }
}
