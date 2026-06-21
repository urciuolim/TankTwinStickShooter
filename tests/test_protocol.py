"""Regression net for tank_twin.protocol (extracted from PythonScripts/tank_env.py, M1 1.4).

Round-trips the strict-JSON wire messages through a FAKE in-process transport (no
Unity, no real socket). Pins the BYTE-IDENTICAL wire contract:

* handshake/control shapes ``{"restart":True}`` / ``{"start":True}`` / ``{"end":True}``,
* a step message ``{1: action, 2: opp_action}`` whose INTEGER keys serialize to the
  strings ``"1"`` / ``"2"`` (what the Unity side reads),
* inbound ``state`` / ``winner`` / ``done`` messages,
* STRICT encode/decode (a trailing-comma payload RAISES),
* the legacy ``socket.timeout -> ConnectionError`` translation,
* the unframed ``recv(1024)`` buffer size (framing is a later Plan phase).

stdlib only; no torch/sb3/gym/Unity/numpy.
"""

import json

import pytest

from tank_twin.protocol import RECV_BUFSIZE, Connection, decode, encode


class FakeTransport:
    """In-process stand-in for a socket: records sends, replays queued recvs.

    ``sendall`` appends to ``sent``; ``recv(n)`` pops the next queued reply (and
    records the buffer size it was asked for in ``recv_sizes``). Mirrors the
    socket surface ``Connection`` uses — nothing more.
    """

    def __init__(self, replies=None):
        self.sent = []
        self.recv_sizes = []
        self._replies = list(replies or [])

    def sendall(self, data):
        assert isinstance(data, bytes | bytearray)
        self.sent.append(bytes(data))

    def recv(self, bufsize):
        self.recv_sizes.append(bufsize)
        return self._replies.pop(0)

    def queue(self, raw_bytes):
        self._replies.append(raw_bytes)


class TimeoutTransport:
    """Transport whose send/recv raise ``socket.timeout`` (drives the error path)."""

    def sendall(self, data):
        raise TimeoutError()

    def recv(self, bufsize):
        raise TimeoutError()


# --- pure encode / decode -------------------------------------------------


def test_encode_is_utf8_strict_json_bytes():
    assert encode({"restart": True}) == b'{"restart": true}'


def test_encode_integer_keys_serialize_as_string_keys():
    # The 2021 wire contract: {1: [...], 2: [...]} -> "1"/"2" on the wire.
    data = encode({1: [0.0, 1.0], 2: [-1.0, 0.5]})
    obj = json.loads(data.decode("utf-8"))
    assert set(obj.keys()) == {"1", "2"}
    assert obj["1"] == [0.0, 1.0]
    assert obj["2"] == [-1.0, 0.5]


def test_decode_accepts_bytes_and_str():
    assert decode(b'{"state": [1, 2, 3]}') == {"state": [1, 2, 3]}
    assert decode('{"winner": 0}') == {"winner": 0}


def test_decode_strict_rejects_trailing_comma():
    # Newtonsoft tolerates it; the Python side must RAISE (no tolerant fallback).
    with pytest.raises(json.JSONDecodeError):
        decode('{"state": [1, 2, 3],}')


def test_encode_decode_round_trip_preserves_payload():
    msg = {"state": [0.1, -0.2, 3.0], "winner": 1, "done": True}
    assert decode(encode(msg)) == msg


# --- Connection over the fake transport -----------------------------------


def test_send_writes_strict_json_bytes_to_transport():
    t = FakeTransport()
    conn = Connection(t)
    conn.send({"restart": True})
    assert t.sent == [b'{"restart": true}']


def test_send_step_message_integer_keys_on_the_wire():
    t = FakeTransport()
    Connection(t).send({1: [1.0, 0.0, 0.0, 0.0, 0.0], 2: [0.0, 0.0, 0.0, 0.0, 0.0]})
    obj = json.loads(t.sent[0].decode("utf-8"))
    assert list(obj.keys()) == ["1", "2"]


def test_receive_decodes_state_message():
    t = FakeTransport(replies=[b'{"state": [0.0, 1.0, 2.0]}'])
    assert Connection(t).receive() == {"state": [0.0, 1.0, 2.0]}


def test_receive_uses_recv_bufsize_1024():
    # Unframed recv(1024) is preserved (framing is out of scope for this wave).
    t = FakeTransport(replies=[b'{"starting": true}'])
    Connection(t).receive()
    assert t.recv_sizes == [RECV_BUFSIZE]
    assert RECV_BUFSIZE == 1024


# --- a full handshake / step / end exchange -------------------------------


def test_reset_handshake_then_first_state_exchange():
    # Replays the legacy reset(): restart -> ack, start -> "starting" ack, first state.
    t = FakeTransport(
        replies=[
            b'{"restarting": true}',
            b'{"starting": true}',
            b'{"state": [0.0, 0.0, 0.0]}',
        ]
    )
    conn = Connection(t)
    conn.send({"restart": True})
    ack1 = conn.receive()
    conn.send({"start": True})
    ack2 = conn.receive()
    first = conn.receive()

    assert ack1 == {"restarting": True}
    assert "starting" in ack2
    assert first == {"state": [0.0, 0.0, 0.0]}
    # Outbound bytes are the exact strict-JSON control messages.
    assert t.sent == [b'{"restart": true}', b'{"start": true}']


def test_step_exchange_returns_winner_and_done():
    t = FakeTransport(replies=[b'{"state": [0.0, 0.0], "winner": 0, "done": true}'])
    conn = Connection(t)
    conn.send({1: [0.0] * 5, 2: [0.0] * 5})
    received = conn.receive()
    assert received["winner"] == 0
    assert received["done"] is True
    assert received["state"] == [0.0, 0.0]


def test_close_handshake_end_message_shape():
    t = FakeTransport(replies=[b'{"restarting": true}', b'{"ending": true}'])
    conn = Connection(t)
    conn.send({"restart": True})
    conn.receive()
    conn.send({"end": True})
    end_ack = conn.receive()
    assert "ending" in end_ack
    assert t.sent == [b'{"restart": true}', b'{"end": true}']


# --- timeout -> ConnectionError translation (legacy behavior) --------------


def test_send_timeout_raises_connection_error():
    with pytest.raises(ConnectionError):
        Connection(TimeoutTransport()).send({"restart": True})


def test_receive_timeout_raises_connection_error():
    with pytest.raises(ConnectionError):
        Connection(TimeoutTransport()).receive()


def test_receive_strict_rejects_malformed_reply():
    # A malformed inbound packet raises through receive() (strict decode).
    t = FakeTransport(replies=[b'{"state": [1, 2, 3],}'])
    with pytest.raises(json.JSONDecodeError):
        Connection(t).receive()


# --- buffer-aware receive(): the coalesced control-ack + state + frame landmine ----
#
# These pin the correctness fix: on the pixels-ON path the build writes a control ack
# (e.g. {"starting":true}) and then IMMEDIATELY the first state JSON + its binary frame.
# TCP coalesces them, so a single recv can return {ack}{state}<frame>. The old receive()
# json.loads'd the WHOLE recv chunk and raised "Extra data" on the trailing bytes. The
# buffer-aware receive() must return ONLY the ack and retain {state}<frame> in _buffer for
# the following receive_state_and_frame(). stdlib + raw bytes for the frame; the frame is
# decoded by the REAL receive_frame so we don't reshape by hand.

FRAME_TAG = 0x46  # ASCII 'F' — must match protocol.FRAME_TAG


def _build_frame_message(w, h):
    """Assemble one on-wire pixel frame exactly as FrameCapture.BuildFrameMessage does.

    ``[ tag=0x46 | uint32_BE payloadLen=W*H*3 | uint16_BE W | uint16_BE H | uint8 C=3 |
    W*H*3 RGB bytes ]`` — all multi-byte ints BIG-ENDIAN. Returns (message_bytes,
    payload_bytes); the payload is a deterministic per-row gradient (row r filled with the
    byte value r, BOTTOM-UP as Unity ships) so the np.flipud top-left flip is observable.
    """
    c = 3
    payload_len = w * h * c
    # Bottom-up rows: wire row 0 == all 0s, wire row (h-1) == all (h-1).
    payload = bytearray()
    for r in range(h):
        payload += bytes([r]) * (w * c)
    header = (
        bytes([FRAME_TAG])
        + payload_len.to_bytes(4, "big")
        + w.to_bytes(2, "big")
        + h.to_bytes(2, "big")
        + bytes([c])
    )
    return header + bytes(payload), bytes(payload)


def test_receive_buffers_trailing_state_and_frame_after_coalesced_ack():
    # (a) COALESCED: one recv returns {"starting":true} + {state} + <frame> glued together.
    # receive() must return ONLY the ack and leave {state}<frame> in _buffer; then
    # receive_state_and_frame() must drain that buffer into the right (state, frame) pair.
    w, h = 4, 4  # tiny so the whole blob fits one FakeTransport recv
    frame_msg, _payload = _build_frame_message(w, h)
    state = [0.1, -2.0, 3.5] + [0.0] * 49  # 52 floats
    ack_bytes = encode({"starting": True})
    state_bytes = encode({"state": state})
    coalesced = ack_bytes + state_bytes + frame_msg

    t = FakeTransport(replies=[coalesced])
    conn = Connection(t)

    ack = conn.receive()
    assert ack == {"starting": True}
    # Everything after the ack's closing brace is retained for the next read.
    assert conn._buffer == state_bytes + frame_msg

    got_state, got_frame = conn.receive_state_and_frame()
    assert got_state == {"state": state}
    assert got_frame.shape == (h, w, 3)
    assert str(got_frame.dtype) == "uint8"
    # Verify the flip + content: array row 0 (top-left origin) is the LAST wire row (h-1),
    # array row -1 is wire row 0 (all 0s).
    assert int(got_frame[0].min()) == int(got_frame[0].max()) == h - 1
    assert int(got_frame[-1].min()) == int(got_frame[-1].max()) == 0
    assert conn._buffer == b""  # frame fully drained


def test_receive_reassembles_ack_split_across_two_recvs():
    # (b) SPLIT-ACROSS-RECV: the ack JSON arrives in two recv chunks (partial, remainder).
    # receive() must loop recv until the top-level object closes and reassemble it.
    ack_bytes = encode({"restarting": True})
    cut = len(ack_bytes) // 2
    t = FakeTransport(replies=[ack_bytes[:cut], ack_bytes[cut:]])
    conn = Connection(t)

    ack = conn.receive()
    assert ack == {"restarting": True}
    assert conn._buffer == b""
    assert len(t.recv_sizes) == 2  # took two recvs to assemble the one object


def test_receive_split_ack_with_trailing_state_frame_is_buffered():
    # (b, third variant) The ack is split across two recvs AND chunk-2 carries trailing
    # state/frame bytes. receive() returns just the ack; the trailing bytes are buffered
    # for receive_state_and_frame().
    w, h = 3, 3
    frame_msg, _payload = _build_frame_message(w, h)
    state = [1.5, -1.5, 0.25] + [0.0] * 49
    ack_bytes = encode({"starting": True})
    state_bytes = encode({"state": state})
    cut = len(ack_bytes) // 2
    chunk1 = ack_bytes[:cut]
    chunk2 = ack_bytes[cut:] + state_bytes + frame_msg

    t = FakeTransport(replies=[chunk1, chunk2])
    conn = Connection(t)

    ack = conn.receive()
    assert ack == {"starting": True}
    assert conn._buffer == state_bytes + frame_msg

    got_state, got_frame = conn.receive_state_and_frame()
    assert got_state == {"state": state}
    assert got_frame.shape == (h, w, 3)
