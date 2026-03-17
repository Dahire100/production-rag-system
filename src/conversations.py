"""
Persistent conversation storage for Enterprise AI Assistant.
Each conversation is saved as a JSON file under ./conversations/<user>/.
"""
import os
import json
import uuid
import datetime
from typing import List, Dict, Optional
from src.config import CONVERSATIONS_DIR


def _user_dir(user: str) -> str:
    path = os.path.join(CONVERSATIONS_DIR, user)
    os.makedirs(path, exist_ok=True)
    return path


def new_conversation(user: str, title: str = "") -> str:
    """Create and return a new conversation ID."""
    conv_id = str(uuid.uuid4())[:8]
    metadata = {
        "id":      conv_id,
        "title":   title or f"Chat {datetime.datetime.now().strftime('%b %d %H:%M')}",
        "created": datetime.datetime.now().isoformat(),
        "user":    user,
        "messages": [],
    }
    path = os.path.join(_user_dir(user), f"{conv_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return conv_id


def save_conversation(user: str, conv_id: str, messages: List[Dict], title: str = ""):
    path = os.path.join(_user_dir(user), f"{conv_id}.json")
    existing = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)

    # Strip heavy context data for storage (keep answer + sources only)
    slim_messages = []
    for m in messages:
        slim = {"role": m["role"], "content": m["content"]}
        if m.get("sources"):
            slim["sources"]   = m["sources"]
            slim["filenames"] = m.get("filenames", [])
            slim["pages"]     = m.get("pages", [])
            slim["contexts"]  = [c[:300] for c in m.get("contexts", [])]
        slim_messages.append(slim)

    existing.update({
        "title":    title or existing.get("title", conv_id),
        "updated":  datetime.datetime.now().isoformat(),
        "messages": slim_messages,
    })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)


def load_conversation(user: str, conv_id: str) -> Optional[Dict]:
    path = os.path.join(_user_dir(user), f"{conv_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_conversations(user: str) -> List[Dict]:
    """Return list of conversation metadata, sorted newest first."""
    base = _user_dir(user)
    convs = []
    for fname in os.listdir(base):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(base, fname), "r", encoding="utf-8") as f:
                    data = json.load(f)
                    convs.append({
                        "id":      data.get("id", fname[:-5]),
                        "title":   data.get("title", fname[:-5]),
                        "updated": data.get("updated", data.get("created", "")),
                        "count":   len(data.get("messages", [])),
                    })
            except Exception:
                pass
    return sorted(convs, key=lambda x: x["updated"], reverse=True)


def delete_conversation(user: str, conv_id: str):
    path = os.path.join(_user_dir(user), f"{conv_id}.json")
    if os.path.exists(path):
        os.remove(path)
