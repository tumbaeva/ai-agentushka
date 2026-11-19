#!/usr/bin/env python3
"""Thin entrypoint kept for backwards compatibility.

Use ``python run.py`` (recommended) or ``python app.py``.
"""

from rag_app import create_app

app = create_app()

if __name__ == "__main__":
    settings = app.config["SETTINGS"]
    print(f"RAG Dashboard: http://{settings.host}:{settings.port}")
    app.run(host=settings.host, port=settings.port, debug=False)
