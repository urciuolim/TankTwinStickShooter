"""``HumanAgent`` — a keyboard-driven :class:`pop_trainer.core.agent.Agent` for live human play.

Two players share ONE physical keyboard (human-vs-human testing): a SINGLE
:class:`KeyboardListener` tracks the pressed-key set into a shared :class:`KeyboardState`, and
TWO :class:`HumanAgent`s (one per :class:`KeyMapping`) read that SAME state. Each agent's ``act``
IGNORES its observation — the action comes from the keyboard, not the pixels/state — and emits the
env's 5-float ``[move_x, move_y, aim_x, aim_y, fire]`` (``move``/``aim`` in ``[-1, 1]``, fire 0/1).

Layering, by design, so the pressed-key -> action MAPPING is a PURE function of (mapping, pressed
key set) and is unit-tested with NO hardware and NO ``pynput``:

* :class:`KeyMapping` (frozen) names ONE player's keys as KEY TOKENS — a single character
  (``"w"``, case-insensitive), a Windows virtual-key code ``int`` (numpad), or a named special
  (``"shift"`` / ``"enter"`` / arrow names). Built WITHOUT importing ``pynput``.
* :class:`KeyboardState` holds the currently-pressed tokens in a NORMALIZED form and answers
  "is this token down?". :class:`HumanAgent` reads ONLY this object.
* :class:`KeyboardListener` wraps a ``pynput`` ``keyboard.Listener``, owns one
  :class:`KeyboardState`, and on press/release mutates the set. **``pynput`` is an OPTIONAL, LAZY
  dependency** (the ``human`` extra): imported INSIDE the listener — never at module top — so
  ``import pop_trainer.agents`` / ``pop_trainer.agents.human_agent`` stay importable core-only
  without it. Constructing/starting the listener without ``pynput`` raises a clear error naming the
  extra.

NUMPAD-BY-VK (load-bearing): the P2 aim keys are detected by VIRTUAL-KEY CODE so numpad-8/4/5/6
work regardless of Num Lock — with Num Lock OFF a numpad key arrives as an arrow/``Key`` (no vk),
so each numpad token is a SET that accepts EITHER the vk OR the arrow-name fallback. Numpad-Enter
shares ``VK_RETURN`` (13) with main Enter and is not separable via ``pynput``'s public API, so P2
fire matches ``Key.enter`` (named ``"enter"``); the fire key is a configurable mapping token, not
hard-coded.

Boundary: imports :mod:`pop_trainer.core` (the Agent contract via :mod:`pop_trainer.agents.base`)
plus ``pynput`` (LAZY, listener only). NO torch. The ``agents/`` package stays importable without
``pynput``.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pop_trainer.agents.base import validate_action

if TYPE_CHECKING:
    from types import TracebackType

__all__ = [
    "KeyToken",
    "KeyMapping",
    "KeyboardState",
    "KeyboardListener",
    "HumanAgent",
    "player1_mapping",
    "player2_mapping",
    "PYNPUT_MISSING_MSG",
]

# A normalized key token: a lowercase character, a virtual-key code int, or a named special key.
KeyToken = str | int

# Windows numpad virtual-key code range (VK_NUMPAD0=96 .. VK_DIVIDE=111). A KeyCode whose vk is in
# this range is a numpad key and is normalized BY VK even when Num Lock gives it a digit ``.char``,
# so numpad-8/4/5/6 detection is Num-Lock-independent.
_NUMPAD_VK_MIN = 96
_NUMPAD_VK_MAX = 111

# Actionable message when the OPTIONAL listener dependency is absent.
PYNPUT_MISSING_MSG = (
    "HumanAgent keyboard input needs pynput; install the 'human' extra: "
    "uv sync --extra human (or pip install 'tank-twin[human]')"
)


def _normalize_token(token: KeyToken) -> KeyToken:
    """Normalize a single key token: lowercase a character, pass an int/name through.

    A character compares case-insensitively (``"W"`` and ``"w"`` are the same key); a virtual-key
    code compares by its ``int``; a named special / arrow compares by its (already-normalized)
    name string. Multi-character strings (named specials like ``"shift"``) are lowercased too.
    """
    if isinstance(token, str):
        return token.lower()
    return token


@dataclass(frozen=True)
class KeyMapping:
    """ONE player's directional + fire keys, each a frozenset of accepted (normalized) tokens.

    A direction is satisfied when ANY token in its set is down — which lets a single key map to
    multiple physical encodings (e.g. a numpad key as a vk OR its Num-Lock-OFF arrow name). Build
    via :func:`player1_mapping` / :func:`player2_mapping`; constructible WITHOUT ``pynput``.
    """

    up: frozenset[KeyToken]
    down: frozenset[KeyToken]
    left: frozenset[KeyToken]
    right: frozenset[KeyToken]
    aim_up: frozenset[KeyToken]
    aim_down: frozenset[KeyToken]
    aim_left: frozenset[KeyToken]
    aim_right: frozenset[KeyToken]
    fire: frozenset[KeyToken]
    name: str = ""

    def __post_init__(self) -> None:
        # Normalize every token once, at construction, so the matcher never re-normalizes.
        for slot in (
            "up",
            "down",
            "left",
            "right",
            "aim_up",
            "aim_down",
            "aim_left",
            "aim_right",
            "fire",
        ):
            normalized = frozenset(_normalize_token(t) for t in getattr(self, slot))
            object.__setattr__(self, slot, normalized)


def _tokens(*tokens: KeyToken) -> frozenset[KeyToken]:
    """A frozenset of one direction's accepted tokens (sugar for the mapping factories)."""
    return frozenset(tokens)


def player1_mapping() -> KeyMapping:
    """Player 1: move WASD, aim TFGH, fire LEFT SHIFT.

    W=+y up / S=-y down / A=-x left / D=+x right; T=+y / G=-y / F=-x / H=+x; fire = left shift
    (``Key.shift_l`` -> ``"shift_l"``, with the generic ``"shift"`` accepted as a fallback).
    """
    return KeyMapping(
        up=_tokens("w"),
        down=_tokens("s"),
        left=_tokens("a"),
        right=_tokens("d"),
        aim_up=_tokens("t"),
        aim_down=_tokens("g"),
        aim_left=_tokens("f"),
        aim_right=_tokens("h"),
        fire=_tokens("shift", "shift_l"),
        name="player1",
    )


def player2_mapping() -> KeyMapping:
    """Player 2: move IJKL, aim NUMPAD 8/4/5/6, fire NUMPAD ENTER.

    I=+y / K=-y / J=-x / L=+x; numpad 8=+y / 5=-y / 4=-x / 6=+x detected by VIRTUAL-KEY CODE
    (NUMPAD4=100, NUMPAD5=101, NUMPAD6=102, NUMPAD8=104) with the Num-Lock-OFF fallbacks: the
    arrow keys (8->up, 4->left, 6->right) and numpad-5, which has no arrow — it arrives as
    ``VK_CLEAR`` (vk 12) on Windows, or a ``Key.begin`` / ``Key.clear`` on builds that expose one.
    Numpad-Enter shares VK_RETURN with main Enter and is not separable, so fire matches
    ``Key.enter`` (``"enter"``).
    """
    return KeyMapping(
        up=_tokens("i"),
        down=_tokens("k"),
        left=_tokens("j"),
        right=_tokens("l"),
        aim_up=_tokens(104, "up"),
        aim_down=_tokens(101, 12, "begin", "clear"),
        aim_left=_tokens(100, "left"),
        aim_right=_tokens(102, "right"),
        fire=_tokens("enter"),
        name="player2",
    )


class KeyboardState:
    """The SHARED pressed-key set, normalized, with a thread-safe down-query.

    The listener thread mutates the set via :meth:`press` / :meth:`release`; the agent thread reads
    via :meth:`any_down`. A :class:`threading.Lock` guards the set so a read never races a mutate
    on the OS listener thread. Tests inject a state directly (``KeyboardState({"w"})``) — no
    listener and no ``pynput`` required.
    """

    def __init__(self, pressed: Iterable[KeyToken] | None = None):
        self._lock = threading.Lock()
        self._pressed: set[KeyToken] = set()
        if pressed is not None:
            for token in pressed:
                self._pressed.add(_normalize_token(token))

    def press(self, token: KeyToken) -> None:
        """Mark a (raw) key token down; normalized before insertion."""
        with self._lock:
            self._pressed.add(_normalize_token(token))

    def release(self, token: KeyToken) -> None:
        """Mark a (raw) key token up; a token never pressed is a no-op."""
        with self._lock:
            self._pressed.discard(_normalize_token(token))

    def any_down(self, tokens: Iterable[KeyToken]) -> bool:
        """True iff ANY of ``tokens`` (assumed already normalized) is currently down."""
        with self._lock:
            return any(t in self._pressed for t in tokens)

    def snapshot(self) -> frozenset[KeyToken]:
        """A frozen copy of the currently-pressed tokens (for debugging / tests)."""
        with self._lock:
            return frozenset(self._pressed)

    def clear(self) -> None:
        """Drop all pressed keys (e.g. on listener stop, to avoid a stuck key)."""
        with self._lock:
            self._pressed.clear()


class HumanAgent:
    """A keyboard-driven :class:`pop_trainer.core.agent.Agent`; ``act`` ignores the observation.

    Holds ONE player's :class:`KeyMapping` and a SHARED :class:`KeyboardState` (the demo builds one
    state + listener and hands the state to both players). Does NOT construct or touch the
    listener — and never imports ``pynput`` — so it is pure and unit-testable from an injected
    state.
    """

    def __init__(self, mapping: KeyMapping, keyboard_state: KeyboardState):
        self.mapping = mapping
        self.keyboard_state = keyboard_state

    def act(self, obs) -> list[float]:  # noqa: ARG002 (keyboard-driven: ignores obs by contract)
        """Map the CURRENT pressed keys to the env action; the observation is ignored.

        ``move_x = right − left``, ``move_y = up − down``, the same for aim, and ``fire = 1.0`` iff
        the fire key is down. Diagonals are left UN-NORMALIZED (e.g. W+D -> ``move=(1, 1)``): the
        wire / :func:`validate_action` tolerates it and the env coerces, so this keeps the mapping
        a trivial, exactly-testable per-channel sum.
        """
        m = self.mapping
        s = self.keyboard_state
        move_x = float(s.any_down(m.right)) - float(s.any_down(m.left))
        move_y = float(s.any_down(m.up)) - float(s.any_down(m.down))
        aim_x = float(s.any_down(m.aim_right)) - float(s.any_down(m.aim_left))
        aim_y = float(s.any_down(m.aim_up)) - float(s.any_down(m.aim_down))
        fire = 1.0 if s.any_down(m.fire) else 0.0
        return validate_action([move_x, move_y, aim_x, aim_y, fire])

    def __repr__(self) -> str:
        return f"HumanAgent(mapping={self.mapping.name!r})"


@dataclass
class KeyboardListener:
    """Owns ONE shared :class:`KeyboardState` fed by a ``pynput`` ``keyboard.Listener``.

    ``pynput`` is imported LAZILY inside :meth:`start` (and probed in :meth:`__init__`) — never at
    module import — so this module stays importable without the ``human`` extra. The press/release
    callbacks run on an OS thread (Windows): they are CHEAP — they only normalize the raw key and
    mutate the shared set's guarded set. Use as a context manager or call :meth:`start` /
    :meth:`stop`.
    """

    state: KeyboardState = field(default_factory=KeyboardState)
    _listener: object | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Fail fast with the actionable message if the optional extra is not installed, rather than
        # deferring the ImportError to start() after the demo has already launched the build.
        self._require_pynput()

    @staticmethod
    def _require_pynput():
        """Import ``pynput.keyboard`` LAZILY; raise the actionable error if it is absent.

        This is the ONLY ``pynput`` import site in the package — keeping it inside the listener is
        what lets ``import pop_trainer.agents`` succeed without the ``human`` extra installed.
        """
        try:
            from pynput import keyboard  # noqa: PLC0415 (lazy by design: optional 'human' extra)
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatched import
            raise RuntimeError(PYNPUT_MISSING_MSG) from exc
        return keyboard

    def _on_press(self, key) -> None:
        token = _token_from_pynput_key(key)
        if token is not None:
            self.state.press(token)

    def _on_release(self, key) -> None:
        token = _token_from_pynput_key(key)
        if token is not None:
            self.state.release(token)

    def start(self) -> KeyboardState:
        """Start the background ``pynput`` listener; returns the shared :class:`KeyboardState`."""
        keyboard = self._require_pynput()
        if self._listener is None:
            self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
            self._listener.start()
        return self.state

    def stop(self) -> None:
        """Stop the listener (if running) and clear the pressed set so no key stays stuck."""
        listener = self._listener
        self._listener = None
        if listener is not None:
            listener.stop()
        self.state.clear()

    def __enter__(self) -> KeyboardState:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()


def _token_from_pynput_key(key) -> KeyToken | None:
    """Normalize a ``pynput`` ``Key`` / ``KeyCode`` to a :class:`KeyToken`, or ``None`` to ignore.

    * a ``KeyCode`` whose ``.vk`` is in the NUMPAD range -> the vk ``int`` (the numpad-by-vk path:
      numpad-8/4/5/6 with Num Lock ON arrive as ``KeyCode(vk=104, char='8')`` — preferring the vk
      over the digit char is what makes detection Num-Lock-independent);
    * any other ``KeyCode`` with a ``.char`` -> the lowercase char (the WASD/IJKL/TFGH letters);
    * a ``Key`` (shift / enter / arrows: the numpad keys with Num Lock OFF, and P2 fire) -> its
      ``.name`` (e.g. ``"shift"``, ``"enter"``, ``"up"``, ``"left"``).

    Both the vk and the arrow-name paths are recorded by the P2 mapping, so a numpad aim key
    matches regardless of Num Lock. Returns ``None`` for a key carrying none of these.
    """
    # Numpad keys FIRST: a KeyCode whose vk is in the numpad range is mapped by vk even when Num
    # Lock has also given it a digit .char — the vk is the Num-Lock-independent identity.
    vk = getattr(key, "vk", None)
    if isinstance(vk, int) and _NUMPAD_VK_MIN <= vk <= _NUMPAD_VK_MAX:
        return vk
    # A character key (WASD/IJKL/TFGH): a KeyCode whose .char is a printable character.
    char = getattr(key, "char", None)
    if isinstance(char, str) and char:
        return char.lower()
    # Any other vk-bearing KeyCode with no usable char (rare): fall back to the vk.
    if isinstance(vk, int):
        return vk
    # A named special / arrow Key (shift, enter, up/down/left/right): use its name.
    name = getattr(key, "name", None)
    if isinstance(name, str) and name:
        return name.lower()
    return None
