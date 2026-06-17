# Boardroom

A tiny local presentation surface for CTO ↔ director board sessions during the revival.

- `server.py` — zero-dependency Python stdlib HTTP server. Serves the board page and records decisions.
- `index.html` — the current board session (org, research plan, milestones, decision controls).
- `decisions.json` — written when you submit a decision (git-ignored). The director watches this file.

## Run

Uses an isolated venv (stdlib only — no packages installed), never the system Python:

```
python -m venv boardroom/.venv            # once
boardroom/.venv/Scripts/python boardroom/server.py    # Windows
```

Then open http://127.0.0.1:8777 (set `BOARDROOM_PORT` to change the port).

Make your selections, hit **Submit decision**, and either let the director's file-watcher pick it up or ping in chat.
