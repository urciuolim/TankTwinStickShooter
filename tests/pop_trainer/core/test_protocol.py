"""Contract tests for ``pop_trainer.core.protocol`` (the strict TCP-JSON wire client)."""

import json

import numpy as np
import pytest

from pop_trainer.core import protocol as P
from pop_trainer.core import state as S


class FakeTransport:
    """In-process socket-like fake: feeds queued recv chunks, records sendall bytes.

    ``recv_chunks`` is a list of byte blobs; each ``recv(n)`` returns the head of the next
    queued chunk, capped at ``n`` bytes (a real socket NEVER returns more than requested).
    Any unconsumed tail of a chunk is re-queued so a single oversized chunk is delivered
    across multiple ``recv`` calls — this exercises both the coalescing path (one chunk
    holding several objects) and the read-exactly path. When exhausted, ``recv`` returns
    ``b""`` (peer closed).
    """

    def __init__(self, recv_chunks=None, raise_timeout_on_recv=False):
        self.recv_chunks = list(recv_chunks or [])
        self.sent = bytearray()
        self.raise_timeout_on_recv = raise_timeout_on_recv

    def sendall(self, data):
        self.sent += data

    def recv(self, bufsize):
        if self.raise_timeout_on_recv:
            raise TimeoutError("simulated socket timeout")
        if not self.recv_chunks:
            return b""
        chunk = self.recv_chunks[0]
        if len(chunk) <= bufsize:
            return self.recv_chunks.pop(0)
        # Honor the requested size: return the head, re-queue the tail.
        self.recv_chunks[0] = chunk[bufsize:]
        return chunk[:bufsize]


def test_encode_decode_round_trip():
    msg = {"state": [1.0, 2.0, 3.0], "winner": -1, "done": True}
    assert P.decode(P.encode(msg)) == msg


def test_integer_keys_serialize_to_string_keys():
    wire = P.encode({1: [0.1, 0.2], 2: [0.3, 0.4]})
    text = wire.decode("utf-8")
    assert '"1"' in text and '"2"' in text
    # And they decode back as string keys (JSON has no integer keys).
    decoded = P.decode(wire)
    assert set(decoded) == {"1", "2"}


def test_decode_strict_rejects_trailing_comma():
    with pytest.raises(json.JSONDecodeError):
        P.decode('{"state": [1, 2, 3,]}')
    with pytest.raises(json.JSONDecodeError):
        P.decode('{"a": 1,}')


def test_decode_accepts_bytes_and_str():
    assert P.decode(b'{"x": 1}') == {"x": 1}
    assert P.decode('{"x": 1}') == {"x": 1}


def test_send_writes_strict_json_bytes():
    t = FakeTransport()
    conn = P.Connection(t)
    conn.send({"start": True})
    assert bytes(t.sent) == P.encode({"start": True})


def test_receive_splits_two_coalesced_objects():
    """Two JSON objects in ONE recv: receive() returns only the first; second is buffered."""
    first = P.encode({"starting": True})
    second = P.encode({"state": [0.0] * S.STATE_LEN})
    t = FakeTransport(recv_chunks=[first + second])
    conn = P.Connection(t)
    assert conn.receive() == {"starting": True}
    # The second object comes from the internal buffer with NO further recv.
    assert conn.receive() == {"state": [0.0] * S.STATE_LEN}


def test_receive_handles_object_split_across_recvs():
    """One object delivered across two recv chunks is reassembled."""
    blob = P.encode({"state": [1.0, 2.0], "done": False})
    half = len(blob) // 2
    t = FakeTransport(recv_chunks=[blob[:half], blob[half:]])
    conn = P.Connection(t)
    assert conn.receive() == {"state": [1.0, 2.0], "done": False}


def test_receive_brace_inside_string_does_not_miscount():
    blob = P.encode({"note": "a } brace { in a string", "v": 1})
    t = FakeTransport(recv_chunks=[blob])
    conn = P.Connection(t)
    assert conn.receive() == {"note": "a } brace { in a string", "v": 1}


def test_receive_empty_recv_raises_connection_error():
    t = FakeTransport(recv_chunks=[])  # immediately returns b""
    conn = P.Connection(t)
    with pytest.raises(ConnectionError):
        conn.receive()


def test_recv_timeout_translated_to_connection_error():
    t = FakeTransport(raise_timeout_on_recv=True)
    conn = P.Connection(t)
    with pytest.raises(ConnectionError):
        conn.receive()


def test_send_timeout_translated_to_connection_error():
    class TimeoutSend:
        def sendall(self, _data):
            raise TimeoutError("send timeout")

        def recv(self, _n):
            return b""

    conn = P.Connection(TimeoutSend())
    with pytest.raises(ConnectionError):
        conn.send({"start": True})


# --- pixel frame channel ----------------------------------------------------------------


def _build_frame(w, h, rgb_bytes, tag=P.FRAME_TAG, channels=P.FRAME_CHANNELS):
    payload_len = len(rgb_bytes)
    header = (
        bytes([tag])
        + payload_len.to_bytes(4, "big")
        + w.to_bytes(2, "big")
        + h.to_bytes(2, "big")
        + bytes([channels])
    )
    return header + rgb_bytes


def test_parse_frame_header_round_trip():
    header = _build_frame(5, 4, b"")[: P.FRAME_HEADER_LEN]
    w, h, c, payload_len = P.parse_frame_header(header)
    assert (w, h, c, payload_len) == (5, 4, 3, 0)


def test_parse_frame_header_rejects_bad_magic():
    bad = bytes([0x47]) + (0).to_bytes(4, "big") + (1).to_bytes(2, "big") * 2 + bytes([3])
    with pytest.raises(ValueError):
        P.parse_frame_header(bad)


def test_parse_frame_header_rejects_wrong_length():
    with pytest.raises(ValueError):
        P.parse_frame_header(b"\x46\x00\x00")


def test_receive_frame_flips_to_top_left_origin():
    w, h = 2, 2
    # Bottom-up RGB rows: row0 (bottom) red, row1 (top) blue.
    bottom_row = bytes([255, 0, 0]) * w
    top_row = bytes([0, 0, 255]) * w
    payload = bottom_row + top_row  # bottom-up
    t = FakeTransport(recv_chunks=[_build_frame(w, h, payload)])
    conn = P.Connection(t)
    frame = conn.receive_frame()
    assert frame.shape == (h, w, 3)
    assert frame.dtype == np.uint8
    # After flipud, the FIRST row (top-left origin) is the LAST bottom-up row (blue).
    assert np.array_equal(frame[0, 0], [0, 0, 255])
    assert np.array_equal(frame[1, 0], [255, 0, 0])


def test_receive_state_and_frame_coalesced():
    """A state JSON glued to its frame in one recv: both are recovered, in order."""
    raw = [float(i) for i in range(S.STATE_LEN)]
    state_bytes = P.encode({"state": raw})
    w, h = 2, 1
    payload = bytes([10, 20, 30]) * (w * h)
    frame_bytes = _build_frame(w, h, payload)
    t = FakeTransport(recv_chunks=[state_bytes + frame_bytes])
    conn = P.Connection(t)
    state, frame = conn.receive_state_and_frame()
    assert state["state"] == raw
    assert frame.shape == (h, w, 3)


def test_state_round_trip_through_protocol():
    """DoD: 52-float state -> {"state": ...} -> encode -> transport -> receive -> accessors.

    A known state (including a -100 sentinel bullet) survives the full wire path and reads
    back EXACTLY via the state.py accessors.
    """
    raw = [round(0.1 * i, 4) for i in range(S.STATE_LEN)]
    # Make P2 bullet slot 2 absent via the sentinel.
    sentinel_idx = S.BULLET_START[S.PLAYER_2] + 2 * S.BULLET_STRIDE
    raw[sentinel_idx] = S.ABSENT_BULLET_SENTINEL

    wire = P.encode({"state": raw})
    t = FakeTransport(recv_chunks=[wire])
    conn = P.Connection(t)
    message = conn.receive()
    out = message["state"]

    S.validate(out)
    assert out == raw
    # Read fields back via accessors and confirm they match the source layout.
    assert S.position(out, S.PLAYER_1) == (raw[0], raw[1])
    assert S.aim(out, S.PLAYER_2) == (raw[30], raw[31])
    # The sentinel bullet is absent; the others are present.
    p2_bullets_all = list(S.iter_bullets(out, S.PLAYER_2, include_absent=True))
    assert p2_bullets_all[2][0] == S.ABSENT_BULLET_SENTINEL
    assert not S.bullet_present(p2_bullets_all[2][0])
    assert len(list(S.iter_bullets(out, S.PLAYER_2))) == S.NUM_BULLETS - 1


# --- state_message_is_valid (A2: kept inbound-state contract check; branch coverage) ----


def test_state_message_is_valid_true_for_52_float_state():
    assert P.state_message_is_valid({"state": [0.0] * S.STATE_LEN}) is True
    # A tuple value is accepted too.
    assert P.state_message_is_valid({"state": tuple(range(S.STATE_LEN))}) is True


def test_state_message_is_valid_false_for_wrong_length_state():
    assert P.state_message_is_valid({"state": [0.0] * (S.STATE_LEN - 1)}) is False
    assert P.state_message_is_valid({"state": [0.0] * (S.STATE_LEN + 1)}) is False


def test_state_message_is_valid_false_when_state_key_missing():
    assert P.state_message_is_valid({"winner": -1, "done": True}) is False
    assert P.state_message_is_valid({}) is False


def test_state_message_is_valid_false_for_non_dict_input():
    assert P.state_message_is_valid(None) is False
    assert P.state_message_is_valid([0.0] * S.STATE_LEN) is False
    assert P.state_message_is_valid("state") is False


def test_state_message_is_valid_false_when_state_not_a_sequence():
    # Present but wrong type: a dict / a scalar / a string are all rejected.
    assert P.state_message_is_valid({"state": {"x": 1}}) is False
    assert P.state_message_is_valid({"state": 52}) is False
    # A 52-char string is a Sequence but NOT a list/tuple -> rejected.
    assert P.state_message_is_valid({"state": "x" * S.STATE_LEN}) is False


# --- A3: inbound size guards ------------------------------------------------------------


class RecordingTransport:
    """Transport that records every ``recv`` size requested (to prove no payload read)."""

    def __init__(self, recv_chunks=None):
        self.recv_chunks = list(recv_chunks or [])
        self.recv_sizes = []
        self.sent = bytearray()

    def sendall(self, data):
        self.sent += data

    def recv(self, bufsize):
        self.recv_sizes.append(bufsize)
        if not self.recv_chunks:
            return b""
        chunk = self.recv_chunks[0]
        if len(chunk) <= bufsize:
            return self.recv_chunks.pop(0)
        self.recv_chunks[0] = chunk[bufsize:]
        return chunk[:bufsize]


def test_unclosed_object_exceeding_max_object_bytes_raises_connection_error():
    """A JSON object that never closes and grows past the cap raises ConnectionError."""
    # Open brace then a flood of non-closing bytes, never a top-level '}'.
    flood = b'{"state": [' + b"0," * 10_000
    t = FakeTransport(recv_chunks=[flood])
    conn = P.Connection(t, bufsize=64, max_object_bytes=256)
    with pytest.raises(ConnectionError):
        conn.receive()


def test_max_object_bytes_does_not_clip_a_valid_leading_object():
    """A valid small object followed by large trailing bytes is returned, not rejected."""
    first = P.encode({"starting": True})
    # Trailing bytes that on their own would exceed the cap, but the first object closes.
    trailing = b"x" * 1000
    t = FakeTransport(recv_chunks=[first + trailing])
    conn = P.Connection(t, max_object_bytes=256)
    assert conn.receive() == {"starting": True}


def test_frame_payload_over_max_frame_bytes_raises_before_reading_payload():
    """An over-cap advertised payload_len raises ValueError without reading the payload."""
    w, h = 1000, 1000  # advertises 3_000_000 payload bytes
    header = (
        bytes([P.FRAME_TAG])
        + (w * h * P.FRAME_CHANNELS).to_bytes(4, "big")
        + w.to_bytes(2, "big")
        + h.to_bytes(2, "big")
        + bytes([P.FRAME_CHANNELS])
    )
    # Only the header is queued; NO payload follows. If receive_frame tried to read the
    # payload it would hit b"" and raise ConnectionError. We assert ValueError instead,
    # proving the cap rejected it before any payload read.
    t = RecordingTransport(recv_chunks=[header])
    conn = P.Connection(t, max_frame_bytes=1024)
    with pytest.raises(ValueError):
        conn.receive_frame()
    # Exactly the 10-byte header was read; the giant payload was never requested.
    assert sum(t.recv_sizes) <= P.FRAME_HEADER_LEN
