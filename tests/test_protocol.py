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
