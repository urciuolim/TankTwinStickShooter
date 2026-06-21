"""Regression net for the ADDITIVE pixel-frame channel (Stage 1, default-OFF in the build).

Pins the binary frame wire contract that ``FrameCapture.cs`` ships byte-for-byte, exercised
through a FAKE in-process transport (no Unity, no real socket):

* ``parse_frame_header`` (pure): a valid 10-byte header round-trips W/H/C/len; a bad magic
  tag raises; a short prefix raises.
* ``Connection.receive_frame``: the whole-message-in-one-recv case yields the right
  ``(H, W, 3)`` uint8 array AND applies the bottom-up -> top-left vertical flip.
* FRAGMENTATION: a transport that dribbles the message out in tiny recv chunks is
  reassembled by ``_recv_exactly`` (the load-bearing TCP-fragmentation guarantee).
* DROPPED CONNECTION: a ``recv`` returning ``b''`` mid-frame raises ``ConnectionError``.

The frame channel is SEPARATE from the JSON ``send`` / ``receive`` path: nothing here
touches ``RECV_BUFSIZE`` / ``recv(1024)`` (those invariants are pinned in test_protocol.py
and must stay green). numpy is used here only because the frame path returns a numpy array.
"""

import numpy as np
import pytest

from tank_twin.protocol import (
    FRAME_CHANNELS,
    FRAME_HEADER_LEN,
    FRAME_TAG,
    Connection,
    parse_frame_header,
)


def build_frame_message(rgb_bytes, w, h, *, tag=FRAME_TAG, channels=FRAME_CHANNELS):
    """Assemble the on-wire frame message exactly as ``FrameCapture.BuildFrameMessage`` does.

    ``[ tag | uint32_BE payloadLen=W*H*C | uint16_BE W | uint16_BE H | uint8 C | payload ]``.
    All multi-byte ints BIG-ENDIAN. ``tag`` / ``channels`` are overridable so the
    bad-header tests can ship a wrong magic byte.
    """
    payload_len = w * h * channels
    assert len(rgb_bytes) == payload_len, (len(rgb_bytes), payload_len)
    header = (
        bytes([tag])
        + payload_len.to_bytes(4, "big")
        + w.to_bytes(2, "big")
        + h.to_bytes(2, "big")
        + bytes([channels])
    )
    return header + bytes(rgb_bytes)


class ChunkedTransport:
    """Replays a single byte stream in fixed-size ``recv`` chunks (TCP-fragmentation fake).

    ``recv(n)`` returns at most ``min(n, chunk_size)`` bytes from the remaining stream,
    advancing a cursor — so a small ``chunk_size`` forces ``_recv_exactly`` to loop. When
    the stream is exhausted, ``recv`` returns ``b''`` (peer closed) — which lets the
    dropped-connection test fall out of a truncated stream naturally.
    """

    def __init__(self, stream, chunk_size):
        self._stream = bytes(stream)
        self._pos = 0
        self.chunk_size = chunk_size
        self.recv_sizes = []

    def sendall(self, data):  # pragma: no cover - not used by the frame read path
        raise AssertionError("ChunkedTransport is recv-only")

    def recv(self, bufsize):
        self.recv_sizes.append(bufsize)
        take = min(bufsize, self.chunk_size, len(self._stream) - self._pos)
        chunk = self._stream[self._pos : self._pos + take]
        self._pos += take
        return chunk


class OneShotTransport:
    """Single-packet fake: each recv returns up to ``bufsize`` from the buffered message.

    Respects the requested ``bufsize`` like a real socket (a real ``recv(n)`` NEVER returns
    more than ``n`` bytes). With the default 1024 bufsize this still delivers the small test
    header in one recv and the (sub-1024) payload in one recv — the "whole message arrives
    promptly, no fragmentation" case — while never over-returning. Returns ``b''`` once the
    stream is exhausted (peer closed), so a too-short message surfaces as a dropped read.
    """

    def __init__(self, message):
        self._message = bytes(message)
        self._pos = 0
        self.recv_sizes = []

    def recv(self, bufsize):
        self.recv_sizes.append(bufsize)
        take = min(bufsize, len(self._message) - self._pos)
        chunk = self._message[self._pos : self._pos + take]
        self._pos += take
        return chunk


# --- pure header parse ----------------------------------------------------


def test_parse_frame_header_round_trips_w_h_c_len():
    msg = build_frame_message(bytes(640 * 360 * 3), 640, 360)
    w, h, c, payload_len = parse_frame_header(msg[:FRAME_HEADER_LEN])
    assert (w, h, c) == (640, 360, 3)
    assert payload_len == 640 * 360 * 3 == 691200
    # The default RT total message size the contract advertises.
    assert len(msg) == FRAME_HEADER_LEN + payload_len == 691210


def test_parse_frame_header_small_dims():
    msg = build_frame_message(bytes(4 * 2 * 3), 4, 2)
    assert parse_frame_header(msg[:FRAME_HEADER_LEN]) == (4, 2, 3, 24)


def test_parse_frame_header_bad_tag_raises():
    msg = build_frame_message(bytes(2 * 2 * 3), 2, 2, tag=0x47)  # 'G', not 'F'
    with pytest.raises(ValueError, match="magic tag"):
        parse_frame_header(msg[:FRAME_HEADER_LEN])


def test_parse_frame_header_short_prefix_raises():
    with pytest.raises(ValueError, match="exactly 10 bytes"):
        parse_frame_header(b"F\x00\x00\x00")  # 4 bytes, not 10


def test_parse_frame_header_too_long_prefix_raises():
    with pytest.raises(ValueError, match="exactly 10 bytes"):
        parse_frame_header(b"\x46" + bytes(10))  # 11 bytes


# --- receive_frame: single recv -------------------------------------------


def gradient_rgb_bottom_up(w, h):
    """A per-row gradient so the vertical flip is OBSERVABLE.

    Row r (bottom-up, as Unity ships) is filled with the constant byte value ``r``, so the
    BOTTOM row (index 0 on the wire) is all 0s and the TOP wire row is all (h-1). After
    ``receive_frame``'s ``np.flipud``, the array's row 0 (top-left origin) must be the LAST
    wire row (value h-1) and the array's last row must be the FIRST wire row (value 0).
    """
    rows = [np.full((w, 3), r, dtype=np.uint8) for r in range(h)]
    bottom_up = np.stack(rows, axis=0)  # (h, w, 3), row r == value r
    return bottom_up.tobytes(), bottom_up


def test_receive_frame_single_recv_shape_dtype_and_flip():
    w, h = 5, 4
    payload, bottom_up = gradient_rgb_bottom_up(w, h)
    msg = build_frame_message(payload, w, h)
    conn = Connection(OneShotTransport(msg))

    frame = conn.receive_frame()

    assert frame.shape == (h, w, 3)
    assert frame.dtype == np.uint8
    # Vertical flip applied: array row 0 is the LAST wire row (value h-1); the round-trip
    # of flud(received) must equal the original bottom-up payload.
    assert np.array_equal(frame, np.flipud(bottom_up))
    assert np.all(frame[0] == h - 1)  # top of the top-left-origin image
    assert np.all(frame[-1] == 0)  # bottom row is the bottom-up wire row 0
    # And it is writable (a copy, not a read-only frombuffer view).
    frame[0, 0, 0] = 200
    assert frame[0, 0, 0] == 200


def test_receive_frame_realistic_values_min_max():
    # A non-trivial payload so min/max are distinct (the live-verify "not blank" check).
    w, h = 8, 6
    payload = np.arange(w * h * 3, dtype=np.uint8).tobytes()
    msg = build_frame_message(payload, w, h)
    frame = Connection(OneShotTransport(msg)).receive_frame()
    assert frame.min() != frame.max()


# --- receive_frame: TCP fragmentation -------------------------------------


@pytest.mark.parametrize("chunk_size", [1, 7, 13, 64])
def test_receive_frame_reassembles_across_fragmented_recv(chunk_size):
    w, h = 9, 7
    payload, bottom_up = gradient_rgb_bottom_up(w, h)
    msg = build_frame_message(payload, w, h)
    transport = ChunkedTransport(msg, chunk_size=chunk_size)

    frame = Connection(transport).receive_frame()

    assert frame.shape == (h, w, 3)
    assert frame.dtype == np.uint8
    assert np.array_equal(frame, np.flipud(bottom_up))
    # Proof the read actually fragmented: more than one recv was needed for the small chunk.
    if chunk_size < FRAME_HEADER_LEN + w * h * 3:
        assert len(transport.recv_sizes) > 1
    # _recv_exactly never asks for more than what remains for the current read (<= bufsize).
    assert all(s <= Connection(transport).bufsize for s in transport.recv_sizes)


def test_recv_exactly_loops_until_n_bytes():
    # Direct _recv_exactly check: 100 bytes dribbled 7 at a time reassemble exactly.
    stream = bytes(range(100)) * 1  # 0..99
    conn = Connection(ChunkedTransport(stream, chunk_size=7))
    got = conn._recv_exactly(100)
    assert got == stream
    assert len(got) == 100


# --- receive_frame: dropped connection ------------------------------------


def test_receive_frame_dropped_mid_header_raises_connection_error():
    # Only 4 of the 10 header bytes arrive, then the peer closes (recv -> b'').
    conn = Connection(ChunkedTransport(b"\x46\x00\x00\x00", chunk_size=4))
    with pytest.raises(ConnectionError, match="closed mid-frame"):
        conn.receive_frame()


def test_receive_frame_dropped_mid_payload_raises_connection_error():
    # Full header + part of the payload, then close. _recv_exactly on the payload sees b''.
    w, h = 4, 4
    full = build_frame_message(bytes(w * h * 3), w, h)
    truncated = full[: FRAME_HEADER_LEN + 5]  # header + only 5 payload bytes
    conn = Connection(ChunkedTransport(truncated, chunk_size=64))
    with pytest.raises(ConnectionError, match="closed mid-frame"):
        conn.receive_frame()


def test_recv_exactly_empty_recv_raises_connection_error():
    conn = Connection(OneShotTransport(b""))  # first recv already returns b''
    with pytest.raises(ConnectionError, match="closed mid-frame"):
        conn._recv_exactly(10)


# --- receive_state_and_frame: the coalesced state-JSON + frame split (the landmine) ---


def _state_json_bytes(state):
    """Encode a state dict the way the wire ships it (strict JSON utf-8)."""
    import json

    return json.dumps({"state": state}).encode("utf-8")


def test_receive_state_and_frame_split_in_one_recv():
    # The LANDMINE case: TCP coalesces the state JSON and the trailing binary frame into ONE
    # recv. receive_state_and_frame must read exactly the JSON object, then the frame from the
    # buffered leftover — the plain receive() recv(1024) would UTF-8-fail on the frame bytes.
    w, h = 6, 5
    payload, bottom_up = gradient_rgb_bottom_up(w, h)
    frame_msg = build_frame_message(payload, w, h)
    state = [0.1, -2.0, 3.5] + [0.0] * 49
    coalesced = _state_json_bytes(state) + frame_msg

    conn = Connection(OneShotTransport(coalesced))  # one recv returns JSON+frame glued
    got_state, got_frame = conn.receive_state_and_frame()

    assert got_state == {"state": state}
    assert got_frame.shape == (h, w, 3)
    assert np.array_equal(got_frame, np.flipud(bottom_up))


def test_receive_state_and_frame_state_and_frame_separate_recvs():
    # The other delivery order: state JSON in its own recv, frame in the next (what the live
    # build did). ChunkedTransport with a chunk that splits exactly between them.
    w, h = 4, 4
    payload, bottom_up = gradient_rgb_bottom_up(w, h)
    state = [9.0, 8.0] + [0.0] * 50
    state_bytes = _state_json_bytes(state)
    frame_msg = build_frame_message(payload, w, h)
    # chunk_size == len(state_bytes) so the first recv yields exactly the JSON, the next the frame.
    transport = ChunkedTransport(state_bytes + frame_msg, chunk_size=len(state_bytes))

    conn = Connection(transport)
    got_state, got_frame = conn.receive_state_and_frame()
    assert got_state == {"state": state}
    assert np.array_equal(got_frame, np.flipud(bottom_up))


def test_receive_state_and_frame_fragmented_everywhere():
    # Worst case: every byte dribbled one at a time. Brace-scan + read-exactly must still split.
    w, h = 3, 3
    payload, bottom_up = gradient_rgb_bottom_up(w, h)
    state = [1.5, -1.5, 0.25] + [0.0] * 49
    stream = _state_json_bytes(state) + build_frame_message(payload, w, h)
    conn = Connection(ChunkedTransport(stream, chunk_size=1))
    got_state, got_frame = conn.receive_state_and_frame()
    assert got_state == {"state": state}
    assert np.array_equal(got_frame, np.flipud(bottom_up))


def test_receive_state_and_frame_handles_winner_done_and_braces_in_no_strings():
    # A terminal state message with winner/done; the brace scanner must find the right '}'.
    w, h = 2, 2
    payload, bottom_up = gradient_rgb_bottom_up(w, h)
    import json

    obj = {"state": [0.0, 1.0], "winner": 0, "done": True}
    stream = json.dumps(obj).encode("utf-8") + build_frame_message(payload, w, h)
    conn = Connection(OneShotTransport(stream))
    got_state, got_frame = conn.receive_state_and_frame()
    assert got_state == obj
    assert np.array_equal(got_frame, np.flipud(bottom_up))


def test_receive_one_json_leaves_no_buffer_when_no_frame_follows():
    # A lone JSON object (no frame) is read whole and leaves the buffer empty.
    conn = Connection(OneShotTransport(b'{"state": [1, 2, 3]}'))
    got = conn._receive_one_json()
    assert got == b'{"state": [1, 2, 3]}'
    assert conn._buffer == b""


def test_receive_one_json_dropped_midway_raises():
    conn = Connection(ChunkedTransport(b'{"state": [1, 2', chunk_size=4))
    with pytest.raises(ConnectionError, match="closed mid-state-json"):
        conn._receive_one_json()


def test_recv_exactly_drains_buffer_before_recv():
    # If the buffer already holds part of the frame (from a coalesced state read), _recv_exactly
    # must consume it FIRST, then recv the rest.
    conn = Connection(ChunkedTransport(b"WORLD", chunk_size=64))
    conn._buffer = b"HELLO"
    got = conn._recv_exactly(10)
    assert got == b"HELLOWORLD"
    assert conn._buffer == b""


# --- regression: the frame channel did NOT change the JSON-path constants ---


def test_frame_constants_match_wire_contract():
    assert FRAME_TAG == 0x46  # ASCII 'F'
    assert FRAME_CHANNELS == 3
    assert FRAME_HEADER_LEN == 10
