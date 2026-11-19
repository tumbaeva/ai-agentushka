from pathlib import Path

from rag_app.chat_store import ChatStore


def test_session_and_message_roundtrip(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.db")
    s = store.create_session()
    store.add_message(s["id"], "user", "Какие номинации?")
    store.add_message(s["id"], "assistant", "Три номинации.", [{"rank": 1, "file_name": "a.docx"}])

    msgs = store.get_messages(s["id"])
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["citations"][0]["file_name"] == "a.docx"


def test_autotitle_from_first_user_message(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.db")
    s = store.create_session()
    store.add_message(s["id"], "user", "Сроки приёма заявок на конкурс")
    sessions = store.list_sessions()
    assert sessions[0]["title"].startswith("Сроки")


def test_delete_session(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.db")
    s = store.create_session()
    store.delete_session(s["id"])
    assert store.list_sessions() == []
