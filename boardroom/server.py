"""Tiny zero-dependency boardroom server.

Serves the interactive board page and records CTO decisions to decisions.json.
Run:  python boardroom/server.py        (optional: set BOARDROOM_PORT)
"""
import http.server
import socketserver
import json
import os
import datetime
from http import HTTPStatus

PORT = int(os.environ.get("BOARDROOM_PORT", "8777"))
HERE = os.path.dirname(os.path.abspath(__file__))
DECISIONS = os.path.join(HERE, "decisions.json")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=HERE, **kwargs)

    def do_POST(self):
        if self.path.rstrip("/") == "/submit":
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length).decode("utf-8")
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = {"raw": raw}
            data["received_at"] = datetime.datetime.now().isoformat(timespec="seconds")
            with open(DECISIONS, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            body = b'{"ok": true}'
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            print("DECISION RECEIVED:", json.dumps(data), flush=True)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, *args):
        pass


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    with Server(("127.0.0.1", PORT), Handler) as httpd:
        print(f"Boardroom server running at http://127.0.0.1:{PORT}", flush=True)
        print(f"Decisions will be written to {DECISIONS}", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("Shutting down boardroom server.", flush=True)
