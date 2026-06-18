"""TCP-JSON wire protocol: strict (de)serialization over an injectable transport.

Extracted from ``TankEnv.send`` / ``TankEnv.receive`` (``PythonScripts/tank_env.py``).
The wire contract is BYTE-IDENTICAL to 2021 — this is the RL seam, the Unity side
parses these exact bytes:

* handshake / control: ``{"restart": True}``, ``{"start": True}``, ``{"end": True}``
* a step message: ``{1: action_list, 2: opp_action_list}`` — note the INTEGER keys.
  ``json.dumps`` coerces them to the strings ``"1"`` / ``"2"`` on the wire, exactly
  as the legacy relied on. Unity reads ``"1"``/``"2"``.
* inbound state / terminal: ``{"state": [...]}`` plus optional ``"winner"`` / ``"done"``.

Two things are deliberately preserved from 2021 and OUT OF SCOPE to change here:

* the read is still unframed ``recv(1024)`` (assumes one JSON object per packet).
  Length-prefix framing is a later Plan phase; do not add it here.
* a ``socket.timeout`` on send or recv is re-raised as ``ConnectionError`` (the env's
  reconnect path keys off ``ConnectionError``).

What IS new: the socket is INJECTED, not constructed. ``encode`` / ``decode`` are pure
(testable with no socket at all) and ``Connection`` wraps any object exposing
``sendall(bytes)`` / ``recv(int)`` — a real socket in production, a fake in-process
pipe in tests. JSON is STRICT in both directions (Python's ``json`` rejects trailing
commas / leading-dot floats; we do NOT add a tolerant fallback).

stdlib only; no sb3/gym/torch/numpy. (``socket.timeout`` is an alias of the builtin
``TimeoutError`` since Python 3.10, so the timeout catch needs no ``socket`` import;
a real ``socket.socket`` still raises it on timeout.)
"""

import json

RECV_BUFSIZE = 1024  # legacy recv(1024); framing is a later Plan phase (do not change here)


def encode(message):
    """Serialize a message dict to UTF-8 wire bytes (STRICT JSON).

    Byte-identical to ``json.dumps(message)`` encoded utf-8 — integer keys become
    ``"1"``/``"2"`` strings, exactly as the 2021 protocol relied on. Pure: no socket.
    """
    return bytes(json.dumps(message), encoding="utf-8")


def decode(data):
    """Parse UTF-8 wire bytes (or str) into a dict with STRICT JSON.

    Strict ``json.loads`` — a trailing comma / leading-dot float RAISES
    ``json.JSONDecodeError`` rather than being tolerated. Pure: no socket.
    """
    if isinstance(data, bytes | bytearray):
        data = data.decode("utf-8")
    return json.loads(data)


class Connection:
    """Strict-JSON send/receive over an injected transport (socket-like).

    ``transport`` must expose ``sendall(bytes)`` and ``recv(int)`` — a real
    ``socket.socket`` in production, or any in-process fake in tests. ``send`` /
    ``receive`` mirror the legacy ``TankEnv`` methods byte-for-byte, including the
    ``socket.timeout -> ConnectionError`` translation.
    """

    def __init__(self, transport, bufsize=RECV_BUFSIZE):
        self.transport = transport
        self.bufsize = bufsize

    def send(self, message):
        """Encode ``message`` (strict JSON) and write it to the transport.

        A ``socket.timeout`` is translated to ``ConnectionError`` (the env's
        reconnect logic keys off that), matching the legacy ``send``.
        """
        data = encode(message)
        try:
            self.transport.sendall(data)
        except TimeoutError as exc:
            raise ConnectionError from exc

    def receive(self):
        """Read one ``recv(bufsize)`` packet and strict-decode it to a dict.

        Unframed by design (one JSON object per packet, as in 2021). A
        ``socket.timeout`` is translated to ``ConnectionError``, matching the
        legacy ``receive``.
        """
        try:
            received = self.transport.recv(self.bufsize)
        except TimeoutError as exc:
            raise ConnectionError from exc
        return decode(received)
