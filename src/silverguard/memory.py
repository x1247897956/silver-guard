"""记忆层（SQLite）。

长期记忆：老人档案 / 联系人白名单 / 历史被诱导事件 / 近期支出。
短期状态：会话状态机（当前等级、已触发动作、已确认事实）。

存储刻意用 SQLite 单文件：本项目是**单机合成数据试验**，不需要 PostgreSQL，
更不做向量检索——检索不是这个项目要回答的问题（见 docs/design-notes.md）。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS elders (
    elder_id     TEXT PRIMARY KEY,
    name         TEXT,
    age          INTEGER,
    city         TEXT,
    note         TEXT,
    updated_at   REAL
);

CREATE TABLE IF NOT EXISTS contacts (
    elder_id      TEXT NOT NULL,
    identifier    TEXT NOT NULL,
    label         TEXT,
    is_whitelist  INTEGER DEFAULT 0,
    first_seen_at REAL,
    last_seen_at  REAL,
    report_count  INTEGER DEFAULT 0,
    note          TEXT,
    PRIMARY KEY (elder_id, identifier)
);

CREATE TABLE IF NOT EXISTS events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    elder_id  TEXT NOT NULL,
    level     TEXT,
    summary   TEXT,
    ts        REAL
);

CREATE TABLE IF NOT EXISTS transactions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    elder_id     TEXT NOT NULL,
    amount       REAL,
    counterparty TEXT,
    ts           REAL
);

-- 会话状态机：等级只会升（单调不降）由应用层保证，库里存当前值
CREATE TABLE IF NOT EXISTS session_state (
    session_id     TEXT PRIMARY KEY,
    elder_id       TEXT,
    case_id        TEXT,
    current_level  TEXT DEFAULT 'L0',
    actions_taken  TEXT DEFAULT '[]',
    confirmed      TEXT DEFAULT '[]',
    updated_at     REAL,
    policy_version TEXT
);

CREATE TABLE IF NOT EXISTS interventions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    elder_id     TEXT NOT NULL,
    action       TEXT NOT NULL,
    session_id   TEXT,
    idem_key     TEXT,
    detail       TEXT,
    ts           REAL
);
CREATE INDEX IF NOT EXISTS idx_interventions_idem ON interventions(idem_key);

-- 可观测：每案一行，含 prompt / 模型 / 策略三个版本号
CREATE TABLE IF NOT EXISTS runs (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id         TEXT,
    split           TEXT,
    config          TEXT,
    elder_id        TEXT,
    model           TEXT,
    report_model    TEXT,
    prompt_version  TEXT,
    policy_version  TEXT,
    policy_tier     TEXT,
    dataset_sha     TEXT,
    max_level       TEXT,
    proposed_level  TEXT,
    suggested_level TEXT,
    first_l2_turn   INTEGER,
    final_action    TEXT,
    unauthorized    INTEGER DEFAULT 0,
    suppressed      INTEGER DEFAULT 0,
    llm_calls       INTEGER DEFAULT 0,
    tool_calls      INTEGER DEFAULT 0,
    prompt_tokens   INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    latency_ms      INTEGER DEFAULT 0,
    degraded_dims   TEXT,
    trace_path      TEXT,
    created_at      REAL
);

-- 动作审计：每次状态迁移 / 每次工具副作用都落一行
CREATE TABLE IF NOT EXISTS action_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id     TEXT,
    session_id  TEXT,
    elder_id    TEXT,
    turn_index  INTEGER,
    kind        TEXT,
    level_from  TEXT,
    level_to    TEXT,
    action      TEXT,
    payload     TEXT,
    ts          REAL
);
CREATE INDEX IF NOT EXISTS idx_action_log_case ON action_log(case_id);
"""


@dataclass
class SessionState:
    session_id: str
    elder_id: str
    case_id: str
    current_level: str = "L0"
    actions_taken: list[str] = None  # type: ignore[assignment]
    confirmed: list[str] = None  # type: ignore[assignment]
    policy_version: str = ""

    def __post_init__(self) -> None:
        if self.actions_taken is None:
            self.actions_taken = []
        if self.confirmed is None:
            self.confirmed = []


class MemoryStore:
    """全部记忆与运行痕迹的唯一读写入口（线程安全）。"""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _exec(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, args)
            self._conn.commit()
            return cur

    def _query(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, args).fetchall())

    # ── 档案 ────────────────────────────────────────────────────────
    def upsert_elder(self, elder_id: str, *, name: str = "", age: int | None = None,
                     city: str = "", note: str = "") -> None:
        self._exec(
            """INSERT INTO elders(elder_id,name,age,city,note,updated_at) VALUES(?,?,?,?,?,?)
               ON CONFLICT(elder_id) DO UPDATE SET name=excluded.name, age=excluded.age,
                 city=excluded.city, note=excluded.note, updated_at=excluded.updated_at""",
            (elder_id, name, age, city, note, time.time()),
        )

    def get_elder(self, elder_id: str) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM elders WHERE elder_id=?", (elder_id,))
        return dict(rows[0]) if rows else None

    # ── 联系人 ──────────────────────────────────────────────────────
    def upsert_contact(self, elder_id: str, identifier: str, *, label: str = "",
                       is_whitelist: bool = False, report_count: int = 0, note: str = "") -> None:
        now = time.time()
        self._exec(
            """INSERT INTO contacts(elder_id,identifier,label,is_whitelist,first_seen_at,last_seen_at,report_count,note)
               VALUES(?,?,?,?,?,?,?,?)
               ON CONFLICT(elder_id,identifier) DO UPDATE SET last_seen_at=excluded.last_seen_at,
                 label=excluded.label""",
            (elder_id, identifier, label, int(is_whitelist), now, now, report_count, note),
        )

    def get_contact(self, elder_id: str, identifier: str) -> dict[str, Any] | None:
        rows = self._query(
            "SELECT * FROM contacts WHERE elder_id=? AND identifier=?", (elder_id, identifier)
        )
        return dict(rows[0]) if rows else None

    def list_contacts(self, elder_id: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self._query(
            "SELECT * FROM contacts WHERE elder_id=? ORDER BY last_seen_at DESC", (elder_id,))]

    def whitelist_members(self, elder_id: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self._query(
            "SELECT * FROM contacts WHERE elder_id=? AND is_whitelist=1", (elder_id,))]

    # ── 历史事件 / 支出 ─────────────────────────────────────────────
    def add_event(self, elder_id: str, level: str, summary: str, ts: float | None = None) -> None:
        self._exec(
            "INSERT INTO events(elder_id,level,summary,ts) VALUES(?,?,?,?)",
            (elder_id, level, summary, ts or time.time()),
        )

    def recent_events(self, elder_id: str, limit: int = 5) -> list[dict[str, Any]]:
        return [dict(r) for r in self._query(
            "SELECT * FROM events WHERE elder_id=? ORDER BY ts DESC LIMIT ?", (elder_id, limit))]

    def add_transaction(self, elder_id: str, amount: float, counterparty: str,
                        ts: float | None = None) -> None:
        self._exec(
            "INSERT INTO transactions(elder_id,amount,counterparty,ts) VALUES(?,?,?,?)",
            (elder_id, float(amount), counterparty, ts or time.time()),
        )

    def recent_transactions(self, elder_id: str, limit: int = 5) -> list[dict[str, Any]]:
        return [dict(r) for r in self._query(
            "SELECT * FROM transactions WHERE elder_id=? ORDER BY ts DESC LIMIT ?", (elder_id, limit))]

    # ── 会话状态机 ──────────────────────────────────────────────────
    def get_session(self, session_id: str) -> SessionState | None:
        rows = self._query("SELECT * FROM session_state WHERE session_id=?", (session_id,))
        if not rows:
            return None
        r = rows[0]
        return SessionState(
            session_id=r["session_id"], elder_id=r["elder_id"], case_id=r["case_id"],
            current_level=r["current_level"],
            actions_taken=json.loads(r["actions_taken"] or "[]"),
            confirmed=json.loads(r["confirmed"] or "[]"),
            policy_version=r["policy_version"] or "",
        )

    def ensure_session(self, session_id: str, elder_id: str, case_id: str,
                       policy_version: str = "") -> SessionState:
        existing = self.get_session(session_id)
        if existing:
            return existing
        self._exec(
            """INSERT INTO session_state(session_id,elder_id,case_id,current_level,actions_taken,
                 confirmed,updated_at,policy_version) VALUES(?,?,?,?,?,?,?,?)""",
            (session_id, elder_id, case_id, "L0", "[]", "[]", time.time(), policy_version),
        )
        state = self.get_session(session_id)
        assert state is not None
        return state

    def save_session(self, state: SessionState) -> None:
        self._exec(
            """UPDATE session_state SET current_level=?, actions_taken=?, confirmed=?,
                 updated_at=?, policy_version=? WHERE session_id=?""",
            (state.current_level, json.dumps(state.actions_taken, ensure_ascii=False),
             json.dumps(state.confirmed, ensure_ascii=False), time.time(),
             state.policy_version, state.session_id),
        )

    # ── 干预幂等 ────────────────────────────────────────────────────
    def record_intervention(self, elder_id: str, action: str, *, session_id: str = "",
                            idem_key: str = "", detail: dict[str, Any] | None = None,
                            ts: float | None = None) -> None:
        self._exec(
            """INSERT INTO interventions(elder_id,action,session_id,idem_key,detail,ts)
               VALUES(?,?,?,?,?,?)""",
            (elder_id, action, session_id, idem_key,
             json.dumps(detail or {}, ensure_ascii=False), ts or time.time()),
        )

    def intervention_count(self, idem_key: str) -> int:
        rows = self._query("SELECT COUNT(*) AS c FROM interventions WHERE idem_key=?", (idem_key,))
        return int(rows[0]["c"]) if rows else 0

    def interventions_for(self, elder_id: str, action: str | None = None) -> list[dict[str, Any]]:
        if action:
            rows = self._query(
                "SELECT * FROM interventions WHERE elder_id=? AND action=? ORDER BY ts", (elder_id, action))
        else:
            rows = self._query("SELECT * FROM interventions WHERE elder_id=? ORDER BY ts", (elder_id,))
        return [dict(r) for r in rows]

    # ── 可观测 ──────────────────────────────────────────────────────
    def log_action(self, *, case_id: str, session_id: str, elder_id: str, turn_index: int | None,
                   kind: str, level_from: str = "", level_to: str = "", action: str = "",
                   payload: dict[str, Any] | None = None) -> None:
        self._exec(
            """INSERT INTO action_log(case_id,session_id,elder_id,turn_index,kind,level_from,
                 level_to,action,payload,ts) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (case_id, session_id, elder_id, turn_index, kind, level_from, level_to, action,
             json.dumps(payload or {}, ensure_ascii=False), time.time()),
        )

    def save_run(self, row: dict[str, Any]) -> int:
        cols = ("case_id", "split", "config", "elder_id", "model", "report_model", "prompt_version",
                "policy_version", "policy_tier", "dataset_sha", "max_level", "proposed_level",
                "suggested_level", "first_l2_turn", "final_action", "unauthorized", "suppressed",
                "llm_calls", "tool_calls", "prompt_tokens", "completion_tokens", "latency_ms",
                "degraded_dims", "trace_path", "created_at")
        payload = {c: row.get(c) for c in cols}
        payload["created_at"] = payload["created_at"] or time.time()
        if isinstance(payload.get("degraded_dims"), (list, dict)):
            payload["degraded_dims"] = json.dumps(payload["degraded_dims"], ensure_ascii=False)
        sql = f"INSERT INTO runs({','.join(cols)}) VALUES({','.join('?' * len(cols))})"
        cur = self._exec(sql, tuple(payload[c] for c in cols))
        return int(cur.lastrowid or 0)

    def fetch_runs(self, *, config: str | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        if config:
            rows = self._query(
                "SELECT * FROM runs WHERE config=? ORDER BY run_id DESC LIMIT ?", (config, limit))
        else:
            rows = self._query("SELECT * FROM runs ORDER BY run_id DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    def fetch_actions(self, case_id: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self._query(
            "SELECT * FROM action_log WHERE case_id=? ORDER BY id", (case_id,))]


def seed_demo_profile(store: MemoryStore, elder_id: str = "elder-0001") -> None:
    """给演示/服务用的最小档案（假数据）。"""
    store.upsert_elder(elder_id, name="李阿姨", age=71, city="示例市", note="独居，子女在外地")
    store.upsert_contact(elder_id, "+86-138-0000-0001", label="女儿（白名单）", is_whitelist=True)
    store.upsert_contact(elder_id, "wechat:daughter-01", label="女儿微信（白名单）", is_whitelist=True)
    store.upsert_contact(elder_id, "+86-400-000-0000", label="历史涉案号码", is_whitelist=False,
                         report_count=3, note="曾被举报")
    store.add_transaction(elder_id, 20000.0, "银行柜台取现", ts=time.time() - 86400 * 3)
    store.add_event(elder_id, "L3", "上月曾接到冒充客服电话，被家人拦下", ts=time.time() - 86400 * 21)
