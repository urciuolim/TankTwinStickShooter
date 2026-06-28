"""TCP-JSON wire protocol: strict (de)serialization over an injected transport.

This is the Python side of the Unity wire seam. The Unity simulator parses these exact
bytes, so the wire format is preserved EXACTLY:

* control / handshake messages: ``{"restart": True}``, ``{"start": True}``,
  ``{"end": True}``. The start-ack Unity sends back contains the substring ``"starting"``.
* a step message: ``{1: action_list, 2: opp_action_list}`` — note the INTEGER keys.
  ``json.dumps`` coerces them to the strings ``"1"`` / ``"2"`` on the wire; Unity reads
  ``"1"`` / ``"2"``.
* an inbound state message: ``{"state": [...52 floats...]}`` plus optional ``"winner"``
  (an int; ``PLAYER_1`` == 0 won, ``-1`` == draw) and ``"done"`` keys.

Design notes:

* The transport is INJECTED, not constructed here. :class:`Connection` wraps any object
  exposing ``sendall(bytes)`` / ``recv(int)`` — a real ``socket.socket`` in production, an
  in-process fake in tests. :func:`encode` / :func:`decode` are PURE (no socket) so the
  serialization is testable with no I/O.
* JSON is STRICT in both directions. Python's ``json`` rejects trailing commas /
  leading-dot floats; we add NO tolerant fallback (Unity's Newtonsoft tolerates such JSON;
  this side must not).
* Reads are FRAME-AWARE, not ``recv(1024)``-as-the-contract: :meth:`Connection.receive`
  reads EXACTLY one complete top-level JSON object via a string/escape-aware brace-depth
  scan and RETAINS any trailing bytes in an internal buffer for the next read (TCP can
  coalesce back-to-back writes into one ``recv``). It never feeds extra bytes into
  ``json.loads``.
* A ``socket.timeout`` (an alias of the builtin ``TimeoutError`` since Python 3.10) on
  send or recv is re-raised as ``ConnectionError``; a ``recv`` returning ``b""`` mid-read
  also raises ``ConnectionError`` (the env's reconnect path keys off ``ConnectionError``).
* Inbound reads are SIZE-GUARDED against a runaway / desynced peer: a single JSON object
  that never closes is capped at ``Connection.max_object_bytes`` (raising
  ``ConnectionError`` as bytes accumulate), and an over-cap advertised pixel-frame
  ``payload_len`` is rejected with ``ValueError`` BEFORE any payload allocation
  (``Connection.max_frame_bytes``). Both caps are constructor parameters with generous
  defaults (:data:`DEFAULT_MAX_OBJECT_BYTES` / :data:`DEFAULT_MAX_FRAME_BYTES`).

PIXEL FRAME CHANNEL (additive). When the build runs with pixel observations, Unity writes
one length-prefixed binary RGB frame on the SAME socket IMMEDIATELY AFTER each state JSON.
:func:`parse_frame_header` decodes the 10-byte prefix and :meth:`Connection.receive_frame`
reads + reshapes the payload into a ``(H, W, 3)`` uint8 array (vertically flipped to a
top-left origin). numpy is used ONLY on this frame path; the JSON path stays numpy-free in
behavior.

WALL-LAYOUT MESSAGE (additive). On a map load / change Unity sends ONE strict-JSON object
tagged ``{"type": "walls", ...}`` as its own discrete write (AFTER the handshake-confirmation
write, never bundled with it; skipped entirely when no arena is configured). It carries the
static wall layout so Python tracks map-state instead of re-parsing the arena JSON.
:func:`is_walls_message` discriminates it by the ``"type"`` tag and :func:`parse_walls_message`
strict-parses it into an immutable :class:`WallLayout`. The ``columns`` keys are STRINGS of the
column x integer (JSON object keys are always strings) and are converted back to ints; only
occupied columns are present. This is a PURE parse (no socket, no numpy).

SWITCH-ARENA HANDSHAKE (additive OUTBOUND, reset-time only). Between the restart ack and the
``{"start": True}`` send (Unity's ``!ingame`` window) the caller MAY request a map change with
``{"switch_arena": <arena_path>}``. Unity replies ``{"arena_switched": true}`` FIRST, THEN (only
when the NEW arena carries a "Walls" block) writes a walls message as its own discrete write —
identical ordering to the start branch. :meth:`Connection.switch_arena` sends the request and
reads + validates ONLY the ack; the OPTIONAL trailing walls message is left for the caller to
route with the SAME :meth:`receive` + :func:`is_walls_message` logic used after the start ack, so
a walls-absent switch never consumes the following state. This is the one additive write on the
otherwise-frozen wire; the per-step state/frame/action path is untouched.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

import numpy as np

from pop_trainer.core.logging_setup import LAYER_PROTOCOL
from pop_trainer.core.state import STATE_LEN

# Default recv chunk size. Reads are frame-aware (the scan below), so this is just the
# per-``recv`` cap, not a framing assumption.
RECV_BUFSIZE = 1024

# --- inbound size guards (defense against a runaway / desynced peer flooding memory) -----
# Upper bound on a SINGLE top-level JSON object. A real state message is ~52 floats (well
# under a few KB); this is deliberately generous headroom so legitimate traffic is never
# clipped, while a peer that never closes a brace cannot grow the read buffer without bound.
# Enforced as bytes accumulate (not only after a complete object). Tunable per-Connection.
DEFAULT_MAX_OBJECT_BYTES = 4 * 1024 * 1024  # 4 MiB

# Upper bound on a pixel-frame payload (the RGB bytes only). A 1024x1024x3 RGB frame is
# ~3 MiB; this leaves headroom above that while rejecting an absurd advertised length
# BEFORE any allocation/read of the payload. Tunable per-Connection.
DEFAULT_MAX_FRAME_BYTES = 8 * 1024 * 1024  # 8 MiB

# --- pixel frame wire contract (additive; big-endian, matches the Unity frame writer) ---
FRAME_TAG = 0x46  # ASCII 'F' — magic/type byte at offset 0 of every frame message
FRAME_CHANNELS = 3  # RGB24; the C byte at offset 9 must equal this
FRAME_HEADER_LEN = 10  # 1 (tag) + 4 (uint32 payload len) + 2 (W) + 2 (H) + 1 (C)

# Brace-scan byte constants (kept readable; the scan operates on raw bytes).
_QUOTE = 0x22  # "
_BACKSLASH = 0x5C  # \
_OPEN_BRACE = 0x7B  # {
_CLOSE_BRACE = 0x7D  # }

# --- wall-layout message wire contract (additive; matches the Unity WallMessage writer) ---
WALLS_TYPE_TAG = "walls"  # the fixed value of the top-level "type" tag on a walls message

# --- switch-arena handshake wire contract (additive OUTBOUND; matches DriverController) ---
SWITCH_ARENA_KEY = "switch_arena"  # outbound request key; value is the arena path (a str)
ARENA_SWITCHED_KEY = "arena_switched"  # the bool ack key Unity replies with (must be True)

# Outbound control/handshake keys (logged at INFO; everything else is a per-step send at DEBUG).
_CONTROL_SEND_KEYS = frozenset({"restart", "start", "end", SWITCH_ARENA_KEY})

__all__ = [
    "RECV_BUFSIZE",
    "DEFAULT_MAX_OBJECT_BYTES",
    "DEFAULT_MAX_FRAME_BYTES",
    "FRAME_TAG",
    "FRAME_CHANNELS",
    "FRAME_HEADER_LEN",
    "WALLS_TYPE_TAG",
    "SWITCH_ARENA_KEY",
    "ARENA_SWITCHED_KEY",
    "encode",
    "decode",
    "parse_frame_header",
    "Connection",
    "state_message_is_valid",
    "WallDims",
    "WallLayout",
    "is_walls_message",
    "parse_walls_message",
]


def encode(message) -> bytes:
    """Serialize a message dict to UTF-8 wire bytes (STRICT JSON). Pure: no socket.

    Integer keys ``{1: ..., 2: ...}`` become the strings ``"1"`` / ``"2"`` on the wire,
    exactly as the step protocol relies on (``json.dumps`` does this coercion).
    """
    return json.dumps(message).encode("utf-8")


def decode(data) -> dict:
    """Parse UTF-8 wire bytes (or a str) into a dict with STRICT JSON. Pure: no socket.

    Strict ``json.loads`` — a trailing comma / leading-dot float RAISES
    ``json.JSONDecodeError`` rather than being tolerated.
    """
    if isinstance(data, bytes | bytearray):
        data = bytes(data).decode("utf-8")
    return json.loads(data)


def parse_frame_header(prefix):
    """Parse the 10-byte pixel-frame header -> ``(w, h, c, payload_len)``. Pure (no socket).

    Layout (all multi-byte ints BIG-ENDIAN):

    * offset 0, 1 byte:  magic tag, must equal :data:`FRAME_TAG` (0x46, ASCII 'F').
    * offset 1, 4 bytes: uint32 payload length = ``W*H*3`` (the RGB bytes ONLY; excludes
      the tag, this length field, and the W/H/C header).
    * offset 5, 2 bytes: uint16 width W.
    * offset 7, 2 bytes: uint16 height H.
    * offset 9, 1 byte:  uint8 channels C.

    Raises ``ValueError`` if ``prefix`` is not exactly :data:`FRAME_HEADER_LEN` (10) bytes
    or if the magic tag is not 0x46 (an early, clear failure on a desynced / wrong-channel
    read). Does NOT assert C == 3 here — :meth:`Connection.receive_frame` does that once it
    knows the shape. Returns plain Python ints.
    """
    if len(prefix) != FRAME_HEADER_LEN:
        raise ValueError(
            f"frame header must be exactly {FRAME_HEADER_LEN} bytes, got {len(prefix)}"
        )
    tag = prefix[0]
    if tag != FRAME_TAG:
        raise ValueError(f"bad frame magic tag: expected {FRAME_TAG:#04x} ('F'), got {tag:#04x}")
    payload_len = int.from_bytes(prefix[1:5], "big")
    w = int.from_bytes(prefix[5:7], "big")
    h = int.from_bytes(prefix[7:9], "big")
    c = prefix[9]
    return w, h, c, payload_len


class Connection:
    """Strict-JSON send/receive over an injected socket-like transport.

    ``transport`` must expose ``sendall(bytes)`` and ``recv(int)`` — a real
    ``socket.socket`` in production, or any in-process fake in tests. ``send`` / ``receive``
    translate a ``socket.timeout`` (builtin ``TimeoutError``) to ``ConnectionError``.

    Two upper-bound guards protect against a runaway / desynced peer flooding memory:

    * ``max_object_bytes`` — cap on ONE top-level JSON object. Enforced as bytes accumulate
      in :meth:`receive`; exceeding it raises ``ConnectionError`` (the env's reconnect path
      keys off ``ConnectionError``). Default :data:`DEFAULT_MAX_OBJECT_BYTES`.
    * ``max_frame_bytes`` — cap on a pixel-frame payload. Checked in :meth:`receive_frame`
      right after the header is parsed, BEFORE allocating/reading the payload; an over-cap
      advertised length raises ``ValueError`` (consistent with the other malformed-header
      rejections). Default :data:`DEFAULT_MAX_FRAME_BYTES`.

    OBSERVABILITY (purely additive). An OPTIONAL ``logger`` (default ``None`` = no logging, zero
    overhead, byte-identical behavior) routes wire activity to the per-connection JSONL file: each
    SEND logs its event + byte count (INFO for control/handshake sends, DEBUG per step), each RECV
    logs begin -> done with bytes + ``elapsed_ms``, and every timeout/closed/over-cap point logs
    BEFORE re-raising. The frame PAYLOAD bytes are never logged — only the byte COUNT. Logging WRAPS
    the wire; it never alters the brace-scan, the buffer logic, the size guards, or any raise.
    """

    def __init__(
        self,
        transport,
        bufsize: int = RECV_BUFSIZE,
        max_object_bytes: int = DEFAULT_MAX_OBJECT_BYTES,
        max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
        *,
        logger: logging.Logger | None = None,
    ):
        self.transport = transport
        self.bufsize = bufsize
        self.max_object_bytes = max_object_bytes
        self.max_frame_bytes = max_frame_bytes
        # OPTIONAL observability logger. None = no logging (the default, byte-identical path).
        self.logger = logger
        # Bytes that arrived after one complete read (TCP coalescing glues the next state
        # JSON, or a trailing pixel frame, onto the current read). Drained FIRST on the next
        # JSON or frame read so nothing is lost and binary frame bytes never reach json.loads.
        self._buffer = b""

    # --- observability helpers (no-op when self.logger is None) --------------------------

    def _log(self, level: int, event: str, **detail) -> None:
        """Emit one observability record at ``level`` if a logger is attached; else a no-op.

        The ``self.logger is None`` early return keeps the default (no-logger) path allocation-free.
        ``detail`` is forwarded as the JSONL formatter's per-record detail (e.g. ``bytes`` /
        ``elapsed_ms``); the per-record ``layer`` is overridden to ``protocol`` so protocol records
        are distinguishable from the env records sharing the same file.
        """
        if self.logger is None:
            return
        self.logger.log(level, event, extra={"detail": {"layer": LAYER_PROTOCOL, **detail}})

    def send(self, message) -> None:
        """Encode ``message`` (strict JSON) and write it to the transport.

        A ``socket.timeout`` is translated to ``ConnectionError`` (the env's reconnect
        logic keys off that). Observational logging (when a logger is attached): a control/handshake
        send (restart / start / end / switch_arena) logs at INFO, a per-step send at DEBUG, each
        with the outgoing byte count; a send timeout logs at WARNING before the re-raise. The wire
        bytes are unchanged.
        """
        data = encode(message)
        if self.logger is not None:
            is_control = isinstance(message, dict) and any(k in _CONTROL_SEND_KEYS for k in message)
            self._log(
                logging.INFO if is_control else logging.DEBUG,
                "send",
                bytes=len(data),
                control=is_control,
            )
        try:
            self.transport.sendall(data)
        except TimeoutError as exc:
            self._log(logging.WARNING, "send_timeout", bytes=len(data))
            raise ConnectionError("send timed out") from exc

    def receive(self) -> dict:
        """Read EXACTLY one complete JSON object and strict-decode it to a dict.

        BUFFER-AWARE: returns one COMPLETE top-level JSON object and RETAINS any bytes that
        arrived after its closing brace in ``self._buffer`` for the next read. So a coalesced
        ``{ack}{state}`` recv yields ONLY the ``{ack}`` here, with ``{state}`` buffered for
        the next :meth:`receive`. On a clean single-object-per-recv wire the scan stops at
        the closing ``}`` and leaves ``self._buffer`` empty. A ``socket.timeout`` mid-read is
        translated to ``ConnectionError``.

        Observational logging (when a logger is attached): a ``recv_begin`` DEBUG, then on success a
        ``recv_done`` with the object's byte count + ``elapsed_ms`` — INFO for a handshake ack,
        DEBUG for a per-step state object. The timeout/closed re-raises are logged inside
        :meth:`_receive_one_json`. The decoded value is unchanged.
        """
        if self.logger is None:
            return decode(self._receive_one_json())
        self._log(logging.DEBUG, "recv_begin")
        t0 = time.monotonic()
        raw = self._receive_one_json()
        message = decode(raw)
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        is_step = state_message_is_valid(message)
        self._log(
            logging.DEBUG if is_step else logging.INFO,
            "recv_done",
            bytes=len(raw),
            elapsed_ms=elapsed_ms,
            is_state=is_step,
        )
        return message

    def switch_arena(self, arena_path: str) -> dict:
        """Request a map change and read + validate the ``{"arena_switched": true}`` ack.

        Sends ``{"switch_arena": <arena_path>}`` (strict JSON; the value is a string, so the
        encoder needs no special handling) and reads EXACTLY one JSON object: Unity's
        ``{"arena_switched": true}`` confirmation. Returns the decoded ack dict.

        WIRE CONTRACT (see ``DriverController.cs`` ``switch_arena`` branch): Unity writes the
        ``{"arena_switched": true}`` ack FIRST and ONLY THEN — and ONLY when the NEW arena
        carries a "Walls" block — writes a walls message as its OWN discrete write. This method
        deliberately reads ONLY the ack: the OPTIONAL trailing walls message is left in the wire
        for the caller to route with the SAME :meth:`receive` + :func:`is_walls_message` logic it
        already uses after the start ack. Reading the ack alone is what guarantees a walls-ABSENT
        switch never consumes the following ``state``.

        Unity handles ``switch_arena`` only while ``!ingame`` — i.e. AFTER the restart ack and
        BEFORE the ``{"start": True}`` send. Calling it outside that window desyncs the wire.

        Raises ``ValueError`` if the ack is missing the ``arena_switched`` key or it is not
        truthy (a clear protocol error). A ``socket.timeout`` is translated to ``ConnectionError``
        via the underlying send/receive, consistent with the rest of the class.
        """
        self.send({SWITCH_ARENA_KEY: arena_path})
        ack = self.receive()
        if not (isinstance(ack, dict) and ack.get(ARENA_SWITCHED_KEY)):
            raise ValueError(
                f"expected {{{ARENA_SWITCHED_KEY!r}: true}} ack after {SWITCH_ARENA_KEY!r}, "
                f"got {ack!r}"
            )
        return ack

    # --- pixel frame channel (additive; shares ``self._buffer`` with the JSON read) ------

    def _recv_exactly(self, n: int) -> bytes:
        """Read EXACTLY ``n`` bytes, draining ``self._buffer`` first then looping over recv.

        ``recv`` may return fewer bytes than requested (TCP is a stream, not framed), so this
        loops until ``n`` bytes are collected, never reading past ``n``. A ``recv`` returning
        ``b""`` (peer closed mid-read) raises ``ConnectionError``; a ``socket.timeout`` is
        likewise translated to ``ConnectionError``. Returns exactly ``n`` bytes.
        """
        chunks = []
        remaining = n
        if self._buffer:
            take = min(remaining, len(self._buffer))
            chunks.append(self._buffer[:take])
            self._buffer = self._buffer[take:]
            remaining -= take
        while remaining > 0:
            try:
                chunk = self.transport.recv(min(remaining, self.bufsize))
            except TimeoutError as exc:
                self._log(logging.WARNING, "recv_timeout", wanted=n, got=n - remaining)
                raise ConnectionError("recv timed out") from exc
            if not chunk:
                self._log(logging.WARNING, "recv_closed_mid_read", wanted=n, got=n - remaining)
                raise ConnectionError(
                    f"connection closed mid-read: wanted {n} bytes, got {n - remaining}"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def receive_frame(self):
        """Read ONE length-prefixed pixel frame and return a ``(H, W, 3)`` uint8 array.

        Reads exactly the 10-byte header, parses it (:func:`parse_frame_header` — asserts the
        0x46 magic), then GUARDS the advertised ``payload_len`` against ``max_frame_bytes``
        BEFORE allocating/reading any payload (an over-cap length raises ``ValueError`` — a
        malformed/oversized header is a protocol error). Asserts ``C == 3`` and
        ``payload_len == W*H*C``, then reads exactly ``W*H*3`` payload bytes. The payload is
        RGB24 rows in BOTTOM-UP order (Unity ``ReadPixels`` origin), so the array is built
        ``(H, W, 3)`` and VERTICALLY FLIPPED (``np.flipud``) to a conventional top-left
        origin. Returns a contiguous uint8 copy. A dropped connection mid-frame raises
        ``ConnectionError`` (via ``_recv_exactly``).
        """
        header = self._recv_exactly(FRAME_HEADER_LEN)
        w, h, c, payload_len = parse_frame_header(header)
        # Reject an absurd advertised length BEFORE allocating/reading the payload.
        if payload_len > self.max_frame_bytes:
            self._log(
                logging.WARNING,
                "frame_over_max_bytes",
                payload_len=payload_len,
                max_frame_bytes=self.max_frame_bytes,
            )
            raise ValueError(
                f"frame payload length {payload_len} exceeds max_frame_bytes {self.max_frame_bytes}"
            )
        if c != FRAME_CHANNELS:
            raise ValueError(f"unexpected frame channel count: expected {FRAME_CHANNELS}, got {c}")
        if payload_len != w * h * c:
            raise ValueError(
                f"frame payload length {payload_len} != W*H*C ({w}*{h}*{c} = {w * h * c})"
            )
        payload = self._recv_exactly(payload_len)
        self._log(logging.DEBUG, "recv_frame", bytes=payload_len, w=w, h=h)
        frame = np.frombuffer(payload, dtype=np.uint8).reshape(h, w, c)
        return np.flipud(frame).copy()

    def receive_state_and_frame(self):
        """Read one step's state JSON dict AND its trailing pixel frame -> ``(dict, ndarray)``.

        Reads EXACTLY one complete state JSON object (buffering any frame bytes glued on by
        TCP coalescing), strict-decodes it, then reads the length-prefixed frame (which drains
        the buffered bytes first). Returns the decoded state dict and the ``(H, W, 3)`` uint8
        frame, time-aligned (the frame Unity captured for THIS state).
        """
        state = decode(self._receive_one_json())
        frame = self.receive_frame()
        return state, frame

    # --- internal: read exactly one top-level JSON object, buffer the rest --------------

    def _receive_one_json(self) -> bytes:
        """Read EXACTLY one complete top-level JSON object as bytes; buffer any trailing bytes.

        ``recv`` chunks are accumulated (draining ``self._buffer`` first) while a brace-depth
        scan — string/escape-aware so braces inside strings don't count — tracks the depth of
        ``{`` / ``}`` OUTSIDE strings. When the top-level object closes (depth returns to 0)
        everything AFTER its closing brace is stashed into ``self._buffer`` for the next read,
        and the exact object bytes are returned. A dropped connection mid-object raises
        ``ConnectionError``; a ``socket.timeout`` is translated to ``ConnectionError``.

        GUARD: if the accumulated bytes for a SINGLE object exceed ``self.max_object_bytes``
        before the object closes (e.g. a peer that never sends a closing brace), this raises
        ``ConnectionError`` rather than letting the buffer grow without bound. The cap is
        checked as bytes accumulate, not only after a complete object.
        """
        out = bytearray()
        scan_from = 0
        # Scanner state, persisted across recv chunks via the closure below.
        state = {"depth": 0, "in_string": False, "escaped": False, "started": False}

        def scan(buf) -> int | None:
            """Advance the scanner over ``buf``; return the index AFTER the top-level object's
            closing brace, or ``None`` if it is not yet complete."""
            for i in range(len(buf)):
                byte = buf[i]
                if state["in_string"]:
                    if state["escaped"]:
                        state["escaped"] = False
                    elif byte == _BACKSLASH:
                        state["escaped"] = True
                    elif byte == _QUOTE:
                        state["in_string"] = False
                    continue
                if byte == _QUOTE:
                    state["in_string"] = True
                elif byte == _OPEN_BRACE:
                    state["depth"] += 1
                    state["started"] = True
                elif byte == _CLOSE_BRACE:
                    state["depth"] -= 1
                    if state["started"] and state["depth"] == 0:
                        return i + 1
            return None

        if self._buffer:
            chunk = self._buffer
            self._buffer = b""
        else:
            chunk = b""
        while True:
            if not chunk:
                try:
                    chunk = self.transport.recv(self.bufsize)
                except TimeoutError as exc:
                    self._log(logging.WARNING, "recv_timeout", read=len(out))
                    raise ConnectionError("recv timed out") from exc
                if not chunk:
                    self._log(logging.WARNING, "recv_closed_mid_json", read=len(out))
                    raise ConnectionError("connection closed mid-json-object")
            out += chunk
            end = scan(out[scan_from:])
            if end is not None:
                end_abs = scan_from + end
                self._buffer = bytes(out[end_abs:])  # trailing bytes (next object / frame)
                return bytes(out[:end_abs])
            # No top-level object has closed yet — guard the still-open object's size as it
            # grows so a peer that never sends a closing brace cannot flood memory.
            if len(out) > self.max_object_bytes:
                self._log(
                    logging.WARNING,
                    "recv_over_max_object_bytes",
                    read=len(out),
                    max_object_bytes=self.max_object_bytes,
                )
                raise ConnectionError(
                    f"incoming JSON object exceeds max_object_bytes "
                    f"{self.max_object_bytes} (read {len(out)} bytes with no top-level close)"
                )
            scan_from = len(out)
            chunk = b""


def state_message_is_valid(message) -> bool:
    """Whether ``message`` is an inbound state message with a 52-float ``state`` array.

    Convenience contract check for consumers: ``True`` iff ``message`` has a ``"state"`` key
    whose value is a length-:data:`STATE_LEN` sequence. Does not mutate or raise.
    """
    state = message.get("state") if isinstance(message, dict) else None
    return isinstance(state, list | tuple) and len(state) == STATE_LEN


# --- wall-layout message (additive; the one-time map-layout message from Unity) ----------


@dataclass(frozen=True)
class WallDims:
    """The integer grid bounds of a map's wall layout (inclusive cell ranges)."""

    min_x: int
    max_x: int
    min_y: int
    max_y: int


@dataclass(frozen=True)
class WallLayout:
    """An immutable parsed wall-layout message (the static map geometry Unity emits).

    ``columns`` maps a column x (int) to the ascending tuple of occupied y-cells in that
    column; only OCCUPIED columns are present (a wall-free map yields an empty mapping). The
    derived :attr:`occupied` set of ``(x, y)`` cells is computed once at construction.
    """

    map_id: str
    tile_id: int
    dims: WallDims
    columns: dict[int, tuple[int, ...]]
    occupied: frozenset[tuple[int, int]] = field(init=False)

    def __post_init__(self) -> None:
        cells = frozenset((x, y) for x, ys in self.columns.items() for y in ys)
        # frozen dataclass: bypass the immutability guard to cache the derived set once.
        object.__setattr__(self, "occupied", cells)


def is_walls_message(message) -> bool:
    """Whether ``message`` is a wall-layout message, by its ``"type"`` tag.

    ``True`` iff ``message`` is a dict whose ``"type"`` equals :data:`WALLS_TYPE_TAG`
    (``"walls"``). Mirrors :func:`state_message_is_valid`: it discriminates by tag only and
    does NOT validate the full shape (that is :func:`parse_walls_message`'s job). Does not
    mutate or raise.
    """
    return isinstance(message, dict) and message.get("type") == WALLS_TYPE_TAG


def _require_int(value, what: str) -> int:
    """Coerce a JSON number to ``int`` for a wall-message field, rejecting non-ints.

    Wall coordinates are INTS on the wire. A JSON bool is an ``int`` subclass but is never a
    valid coordinate, so it is rejected; a float with a fractional part is rejected. An exact
    integral float (``5.0``) is accepted and narrowed, since strict JSON for an int-valued
    field may still decode as a float. Raises ``ValueError`` on anything else.
    """
    if isinstance(value, bool):
        raise ValueError(f"{what} must be an int, got bool {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ValueError(f"{what} must be an int, got {value!r}")


def parse_walls_message(message) -> WallLayout:
    """Strict-parse a wall-layout message dict into an immutable :class:`WallLayout`.

    Expects the exact Unity ``WallMessage`` shape::

        {"type": "walls", "map_id": <str>, "tileID": <int>,
         "dims": {"minX": <int>, "maxX": <int>, "minY": <int>, "maxY": <int>},
         "columns": {"<x>": [y0, y1, ...], ...}}

    The ``columns`` keys are STRINGS of the column x integer (JSON object keys are always
    strings, so a negative column is ``"-1"``) and are converted back to int. All cell
    coordinates must be ints. Raises ``ValueError`` on any malformed walls message (wrong
    tag, missing/!int ``tileID`` or ``dims`` keys, a non-int column key, a non-list column
    value, or a non-int cell).
    """
    if not is_walls_message(message):
        raise ValueError(f"not a walls message (missing type=={WALLS_TYPE_TAG!r}): {message!r}")

    map_id = message.get("map_id", "")
    if not isinstance(map_id, str):
        raise ValueError(f"walls map_id must be a string, got {map_id!r}")

    tile_id = _require_int(message.get("tileID"), "walls tileID")

    dims_raw = message.get("dims")
    if not isinstance(dims_raw, dict):
        raise ValueError(f"walls dims must be an object, got {dims_raw!r}")
    dims = WallDims(
        min_x=_require_int(dims_raw.get("minX"), "walls dims.minX"),
        max_x=_require_int(dims_raw.get("maxX"), "walls dims.maxX"),
        min_y=_require_int(dims_raw.get("minY"), "walls dims.minY"),
        max_y=_require_int(dims_raw.get("maxY"), "walls dims.maxY"),
    )

    columns_raw = message.get("columns")
    if not isinstance(columns_raw, dict):
        raise ValueError(f"walls columns must be an object, got {columns_raw!r}")
    columns: dict[int, tuple[int, ...]] = {}
    for key, ys in columns_raw.items():
        try:
            x = int(key)  # JSON object keys are strings; negative x like "-1" parses
        except (TypeError, ValueError) as exc:
            raise ValueError(f"walls column key is not an int: {key!r}") from exc
        if not isinstance(ys, list):
            raise ValueError(f"walls column {key!r} value must be a list, got {ys!r}")
        columns[x] = tuple(_require_int(y, f"walls column {key!r} cell") for y in ys)

    return WallLayout(map_id=map_id, tile_id=tile_id, dims=dims, columns=columns)
