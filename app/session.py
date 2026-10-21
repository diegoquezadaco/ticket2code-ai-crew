"""In-memory conversation state. Free hosts have ephemeral disks, so memory is enough for the MVP."""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class Session:
    id: str
    history: list = field(default_factory=list)       # compact [{"role","content"}]
    plan: dict | None = None                           # plan of the last delivery
    files: dict = field(default_factory=dict)          # path -> content of the last delivery
    open_issues: list = field(default_factory=list)    # unresolved reviewer issues
    summary: str = ""                                  # one-line description of the last delivery
    lang: str = "en"
    runs: list = field(default_factory=list)           # trace of every turn
    lock: threading.Lock = field(default_factory=threading.Lock)


class SessionStore:
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
