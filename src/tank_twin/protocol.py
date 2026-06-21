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

* the JSON channel has no length-prefix framing — ``receive`` reads via ``recv(1024)``.
  Length-prefix framing of the JSON channel is a later Plan phase; do not add it here.
  ``receive`` IS buffer-aware (it returns exactly one complete JSON object via a
  brace-depth scan and retains any trailing bytes in ``Connection._buffer`` for the next
  read) — a correctness fix so a coalesced control-ack + state-JSON + binary-frame recv
  on the pixels path does not feed frame bytes into ``json.loads``. On the no-pixel wire
  (one clean JSON object per recv) it stays byte-identical: the scan stops at the closing
  ``}`` and leaves ``_buffer`` empty after a single ``recv(1024)``. (The ADDITIVE binary
  pixel-frame channel below IS length-prefixed; it shares the same ``_buffer``.)
* a ``socket.timeout`` on send or recv is re-raised as ``ConnectionError`` (the env's
  reconnect path keys off ``ConnectionError``).

What IS new: the socket is INJECTED, not constructed. ``encode`` / ``decode`` are pure
(testable with no socket at all) and ``Connection`` wraps any object exposing
``sendall(bytes)`` / ``recv(int)`` — a real socket in production, a fake in-process
pipe in tests. JSON is STRICT in both directions (Python's ``json`` rejects trailing
commas / leading-dot floats; we do NOT add a tolerant fallback).

PIXEL FRAME CHANNEL (Stage 1, additive, default-OFF in the build). When the build runs
with ``obs_pixels`` true it writes — IMMEDIATELY AFTER each raw ``state`` JSON, on the
SAME socket — one length-prefixed binary frame. ``parse_frame_header`` (pure) decodes
the 10-byte prefix and ``Connection.receive_frame`` does a read-exactly over the
transport and returns a ``(H, W, 3)`` uint8 numpy array (vertical-flipped to a
top-left origin). This path is entirely separate from ``encode`` / ``decode`` /
``send`` / ``receive``, whose byte behavior is UNCHANGED — when ``obs_pixels`` is off
the build sends no frame and ``receive`` reads exactly the ``state`` JSON as before.

stdlib only for the JSON path. ``numpy`` is imported at module top ONLY for the pixel
frame path (the frame reshape) — numpy is already a project dependency (via the env);
``encode`` / ``decode`` / ``send`` / ``receive`` remain numpy-free in behavior.
(``socket.timeout`` is an alias of the builtin ``TimeoutError`` since Python 3.10, so
the timeout catch needs no ``socket`` import; a real ``socket.socket`` still raises it
on timeout.)
"""

import json

import numpy as np

RECV_BUFSIZE = 1024  # legacy recv(1024); framing is a later Plan phase (do not change here)

# --- pixel frame wire contract (Stage 1, additive; matches FrameCapture.cs byte-for-byte) ---
FRAME_TAG = 0x46  # ASCII 'F' — magic/type byte at offset 0 of every frame message
FRAME_CHANNELS = 3  # RGB24; the C byte at offset 9 must equal this
FRAME_HEADER_LEN = 10  # 1 (tag) + 4 (uint32 payload len) + 2 (W) + 2 (H) + 1 (C)


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


def parse_frame_header(prefix):
    """Parse the 10-byte pixel-frame header -> ``(w, h, c, payload_len)``. Pure (no socket).

    Layout (all multi-byte ints BIG-ENDIAN), matching ``FrameCapture.BuildFrameMessage``:

    * offset 0, 1 byte:  magic tag, must equal ``FRAME_TAG`` (0x46, ASCII 'F').
    * offset 1, 4 bytes: uint32 payload length = ``W*H*3`` (the RGB bytes ONLY; excludes
      the tag, this length field, and the W/H/C header).
    * offset 5, 2 bytes: uint16 width W.
    * offset 7, 2 bytes: uint16 height H.
    * offset 9, 1 byte:  uint8 channels C.

    Raises ``ValueError`` if ``prefix`` is not exactly ``FRAME_HEADER_LEN`` (10) bytes or
    if the magic tag is not 0x46 (a clear, early failure on a desynced / wrong-channel
    read). Does NOT assert C == 3 here (``receive_frame`` does, after it knows the shape);
    keeping the header parse permissive about C lets a caller log the raw header on a
    mismatch. Returns plain Python ints.
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
    """Strict-JSON send/receive over an injected transport (socket-like).

    ``transport`` must expose ``sendall(bytes)`` and ``recv(int)`` — a real
    ``socket.socket`` in production, or any in-process fake in tests. ``send`` /
    ``receive`` mirror the legacy ``TankEnv`` methods byte-for-byte, including the
    ``socket.timeout -> ConnectionError`` translation.
    """

    def __init__(self, transport, bufsize=RECV_BUFSIZE):
        self.transport = transport
        self.bufsize = bufsize
        # Leftover bytes from a greedy read on the PIXEL path only. ``receive`` /
        # ``receive_frame`` / ``_recv_exactly`` for the JSON-only path do NOT touch this;
        # it is filled/drained solely by ``receive_state_and_frame`` (which must read one
        # complete state JSON object and may over-read into the trailing binary frame, since
        # TCP can coalesce the two back-to-back C# writes into one segment — the confirmed
        # unframed-recv landmine). ``_recv_exactly`` drains this buffer FIRST so the frame
        # read picks up any frame bytes that arrived glued to the state JSON.
        self._buffer = b""

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
        """Read EXACTLY one complete JSON object and strict-decode it to a dict.

        BUFFER-AWARE: returns one COMPLETE top-level JSON object and RETAINS any bytes
        that arrived after its closing brace in ``self._buffer`` for the next read
        (``receive_state_and_frame`` / ``receive_frame`` / ``_recv_exactly`` all drain
        ``self._buffer`` first). This delegates to ``_receive_one_json`` — the same
        brace-depth scan (string/escape-aware) that ``receive_state_and_frame`` uses —
        so a coalesced ``{ack}{state}<frame>`` recv yields ONLY the ack here (with
        ``{state}<frame>`` buffered), instead of feeding the binary frame bytes into
        ``json.loads`` and raising ``json.JSONDecodeError: Extra data`` (the confirmed
        pixels-ON handshake landmine).

        On the no-pixel wire (a clean single JSON object per ``recv``) this is
        byte-identical to the legacy behavior: the brace-scan stops at the closing
        ``}`` and leaves ``self._buffer`` empty, calling ``recv(bufsize)`` exactly once.
        A ``socket.timeout`` is translated to ``ConnectionError`` (inside
        ``_receive_one_json``'s recv loop), matching the legacy ``receive``.
        """
        return decode(self._receive_one_json())

    # --- pixel frame channel (Stage 1, additive; does NOT touch send/receive above) ---

    def _recv_exactly(self, n):
        """Read EXACTLY ``n`` bytes from the transport, looping over ``recv`` (TCP fragments).

        ``recv`` may return fewer bytes than requested (TCP delivers a stream, not framed
        messages), so this loops until ``n`` bytes are collected, capping each ``recv`` at
        ``self.bufsize`` for the (smaller) JSON-path parity but never reading PAST ``n``.
        Drains ``self._buffer`` (bytes a prior greedy state read over-read into the frame)
        FIRST, then ``recv``s the remainder. A ``recv`` returning ``b''`` means the peer
        closed the connection mid-read: raised as ``ConnectionError`` (the env's reconnect
        logic keys off ``ConnectionError``, same as the ``socket.timeout`` translation). A
        ``socket.timeout`` is likewise translated to ``ConnectionError``. Returns a ``bytes``
        of length exactly ``n``.
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
                raise ConnectionError from exc
            if not chunk:
                raise ConnectionError(
                    f"connection closed mid-frame: wanted {n} bytes, got {n - remaining}"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def receive_frame(self):
        """Read ONE length-prefixed pixel frame and return a ``(H, W, 3)`` uint8 array.

        Reads exactly the 10-byte header (``_recv_exactly``), parses it
        (``parse_frame_header`` — asserts the 0x46 magic tag), asserts ``C == 3``, then
        reads exactly ``W*H*3`` payload bytes. The payload is interpreted as RGB24 rows in
        BOTTOM-UP order (Unity ``ReadPixels`` origin), so the array is built as ``(H, W, 3)``
        and then VERTICALLY FLIPPED (``np.flipud``) to a conventional top-left origin. The
        returned array is a contiguous ``uint8`` copy (the flip + frombuffer would otherwise
        be a read-only view onto the recv buffer). A dropped connection mid-frame raises
        ``ConnectionError`` (via ``_recv_exactly``).

        This is the additive pixel channel: it is only ever called when the build runs with
        ``obs_pixels`` true. It does NOT change ``receive`` / ``recv(1024)`` — the JSON
        ``state`` is still read by ``receive`` first; ``receive_frame`` reads the frame that
        follows it on the same socket.
        """
        header = self._recv_exactly(FRAME_HEADER_LEN)
        w, h, c, payload_len = parse_frame_header(header)
        if c != FRAME_CHANNELS:
            raise ValueError(f"unexpected frame channel count: expected {FRAME_CHANNELS}, got {c}")
        if payload_len != w * h * c:
            raise ValueError(
                f"frame payload length {payload_len} != W*H*C ({w}*{h}*{c} = {w * h * c})"
            )
        payload = self._recv_exactly(payload_len)
        frame = np.frombuffer(payload, dtype=np.uint8).reshape(h, w, c)
        return np.flipud(frame).copy()

    def _receive_one_json(self):
        """Read EXACTLY one complete JSON object as bytes, buffering any trailing bytes.

        The pixels-ON build writes the ``state`` JSON and then the binary frame as two
        back-to-back socket writes; TCP can coalesce them, so a single ``recv`` may return
        the whole state JSON PLUS the start of the frame (the confirmed unframed-recv
        landmine — the plain ``recv(1024)`` in ``receive`` cannot separate them and would
        UTF-8-fail on the binary frame bytes). This reads ``recv`` chunks (draining the
        leftover ``self._buffer`` first), tracking the depth of curly braces OUTSIDE of JSON
        strings, until the top-level object closes (depth returns to 0). Everything AFTER
        that closing brace is stashed back into ``self._buffer`` for ``receive_frame`` /
        ``_recv_exactly``. Returns the exact JSON object bytes.

        The 52-float ``state`` JSON Unity emits has NO nested objects and no braces inside
        strings (the keys are ``"state"`` / ``"winner"`` / ``"done"`` and the values are
        numbers / a number array), so the brace-depth scan is exact for this wire. A dropped
        connection mid-object raises ``ConnectionError``; a ``socket.timeout`` is translated
        to ``ConnectionError`` (matching ``receive``).
        """
        out = bytearray()
        depth = 0
        in_string = False
        escaped = False
        started = False
        scan_from = 0

        def scan(buf):
            """Advance the brace scanner over ``buf``; return the index AFTER the closing
            brace of the top-level object, or ``None`` if not yet complete."""
            nonlocal depth, in_string, escaped, started
            for i in range(len(buf)):
                byte = buf[i]
                if in_string:
                    if escaped:
                        escaped = False
                    elif byte == 0x5C:  # backslash
                        escaped = True
                    elif byte == 0x22:  # closing quote
                        in_string = False
                    continue
                if byte == 0x22:  # opening quote
                    in_string = True
                elif byte == 0x7B:  # {
                    depth += 1
                    started = True
                elif byte == 0x7D:  # }
                    depth -= 1
                    if started and depth == 0:
                        return i + 1
            return None

        # Drain any buffered bytes first (a previous frame read should leave the buffer empty,
        # but be safe), then recv.
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
                    raise ConnectionError from exc
                if not chunk:
                    raise ConnectionError("connection closed mid-state-json")
            out += chunk
            end = scan(out[scan_from:])
            if end is not None:
                end_abs = scan_from + end
                self._buffer = bytes(out[end_abs:])  # trailing frame bytes, if any
                return bytes(out[:end_abs])
            scan_from = len(out)
            chunk = b""

    def receive_state_and_frame(self):
        """Read one step's ``state`` JSON dict AND its trailing pixel frame -> ``(dict, ndarray)``.

        The pixels-ON read primitive: reads EXACTLY one complete ``state`` JSON object
        (``_receive_one_json``, which buffers any frame bytes glued on by TCP coalescing),
        strict-decodes it (same ``decode`` as ``receive``), then reads the length-prefixed
        frame (``receive_frame``, which drains the buffered bytes first). Returns the decoded
        state dict and the ``(H, W, 3)`` uint8 frame array, time-aligned (the frame Unity
        captured for THIS state, with ``Time.timeScale == 0``).

        This is the method the pixel-observation path should call instead of ``receive`` +
        ``receive_frame`` separately, BECAUSE the unframed ``receive``'s ``recv(1024)`` cannot
        cleanly split the coalesced state-JSON + binary-frame on the wire. ``receive`` itself
        is UNCHANGED and remains correct on the no-pixel path (where no frame follows).
        """
        state_bytes = self._receive_one_json()
        state = decode(state_bytes)
        frame = self.receive_frame()
        return state, frame
