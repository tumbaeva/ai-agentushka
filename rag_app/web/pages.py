"""HTML pages blueprint."""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, current_app, render_template, send_file

from .. import get_ctx
from ..markdown import render_md

pages_bp = Blueprint("pages", __name__)


@pages_bp.get("/")
def index():
    settings = current_app.config["SETTINGS"]
    ctx = get_ctx(current_app)
    return render_template(
        "index.html",
        default_temp=settings.default_temp,
        default_temp_mq=settings.default_temp_mq,
        title=ctx.title or "База знаний",
    )


@pages_bp.get("/recommendations")
def recommendations():
    ctx = get_ctx(current_app)
    return render_template("recommendations.html", title=ctx.title or "База знаний")


@pages_bp.get("/help")
def help_page():
    ctx = get_ctx(current_app)
    help_file = Path(current_app.root_path).parent / "HELP.md"
    md = help_file.read_text(encoding="utf-8") if help_file.is_file() else "# Справка\n\nФайл HELP.md не найден."
    return render_template("help.html", content=render_md(md), title=ctx.title or "База знаний")


@pages_bp.get("/heatmap")
def heatmap():
    path = get_ctx(current_app).settings.heatmap_path
    if path.is_file():
        return send_file(str(path), mimetype="image/png")
    return "Тепловая карта ещё не построена", 404
