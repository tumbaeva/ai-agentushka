from rag_app.markdown import render_md


def test_headings_and_inline():
    html = render_md("# Заголовок\n\nТекст с **жирным** и `кодом`.")
    assert "<h1>Заголовок</h1>" in html
    assert "<strong>жирным</strong>" in html
    assert "<code>кодом</code>" in html


def test_table():
    md = "| A | B |\n| --- | --- |\n| 1 | 2 |\n"
    html = render_md(md)
    assert "<table" in html
    assert "<th>A</th>" in html and "<td>1</td>" in html


def test_code_block_escaped():
    html = render_md("```\n<x> & y\n```")
    assert "<pre><code>" in html
    assert "&lt;x&gt;" in html


def test_list_and_blockquote():
    html = render_md("- один\n- два\n\n> цитата")
    assert "<ul>" in html and "<li>один</li>" in html
    assert "<blockquote>" in html


def test_hr():
    assert "<hr>" in render_md("---")
