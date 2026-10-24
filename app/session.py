"""In-memory conversation state and bounded session storage."""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class Session:
    """Conversation context and latest generated delivery for one browser session."""

    id: str
    history: list = field(default_factory=list)
    plan: dict | None = None
    files: dict = field(default_factory=dict)
    open_issues: list = field(default_factory=list)
    summary: str = ""
    lang: str = "en"
    runs: list = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    pending_clarification: dict | None = None


class SessionStore:
    """Keep a least-recently-used set of sessions in process memory."""

    def __init__(self, max_sessions: int = 200):
        self._items: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_sessions

    def get_or_create(self, sid: str) -> Session:
        with self._lock:
            session = self._items.get(sid)
            if session is None:
                session = Session(id=sid)
                self._items[sid] = session
            self._items.move_to_end(sid)
            while len(self._items) > self._max:
                self._items.popitem(last=False)
            return session

    def get(self, sid: str) -> Session | None:
        with self._lock:
            return self._items.get(sid)
