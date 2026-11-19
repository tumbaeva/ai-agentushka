#!/usr/bin/env python3
"""Standalone AI-agent launcher (chat-only)."""

import socket
import webbrowser
from threading import Timer

from rag_app import create_app

app = create_app(standalone=True)


def _free_port(host: str, preferred: int) -> int:
    """Prefer the configured port; if it is busy, take any free port."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, preferred))
        return preferred
    except OSError:
        pass
    finally:
        s.close()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, 0))
        return s.getsockname()[1]
    finally:
        s.close()


if __name__ == "__main__":
    settings = app.config["SETTINGS"]
    host = settings.host or "127.0.0.1"
    port = _free_port(host, settings.port)
    url = f"http://{host}:{port}"

    print("=" * 52)
    print("  {{AGENT_NAME}}")
    print("  Откройте в браузере:  " + url)
    print("  (окно не закрывайте, пока работаете с агентом)")
    print("=" * 52)

    if host in ("127.0.0.1", "localhost"):
        Timer(1.5, lambda: webbrowser.open(url)).start()
    app.run(host=host, port=port, debug=False)
