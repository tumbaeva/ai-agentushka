#!/usr/bin/env python3
"""Development / product launcher."""

import webbrowser
from threading import Timer

from rag_app import create_app

app = create_app()

if __name__ == "__main__":
    settings = app.config["SETTINGS"]
    url = f"http://{settings.host}:{settings.port}"
    print(f"RAG Dashboard: {url}")
    if settings.host in ("127.0.0.1", "localhost"):
        Timer(1.5, lambda: webbrowser.open(url)).start()
    app.run(host=settings.host, port=settings.port, debug=False)
