"""记忆层与上下文预算测试。"""

from __future__ import annotations

import time

from silverguard.context import ContextBudget, rough_tokens, signal_state_block, trim_tool_result
from silverguard.memory import MemoryStore, seed_demo_profile
from silverguard.models import Signal


def test_session_state_roundtrip(store: MemoryStore):
    s = store.ensure_session("sess-1", "elder-0001", "case-1", "v1")
    assert s.current_level == "L0"
    s.current_level = "L3"
    s.actions_taken.append("notify_family")
    store.save_session(s)
    again = store.get_session("sess-1")
    assert again is not None
    assert again.current_level == "L3"
    assert "notify_family" in again.actions_taken


def test_recent_events_and_transactions(store: MemoryStore):
    store.add_event("elder-0001", "L3", "上周被诱导转账", ts=time.time())
    store.add_transaction("elder-0001", 20000.0, "银行柜台", ts=time.time())
    assert store.recent_events("elder-0001")[0]["level"] == "L3"
    assert store.recent_transactions("elder-0001")[0]["amount"] == 20000.0


def test_whitelist_isolation_between_elders(store: MemoryStore):
    store.upsert_elder("elder-9999", name="王大爷")
    assert store.whitelist_members("elder-9999") == []
    assert store.whitelist_members("elder-0001"), "白名单必须按老人隔离"


def test_trim_tool_result_keeps_only_structured_fields():
    raw = {"summary": "命中 2 条规则", "pattern_hit": True, "score": 4,
           "hits": [{"id": "fp_money_action", "matched": ["转账"] * 50}],
           "debug_blob": "x" * 5000}
    trimmed = trim_tool_result(raw)
    assert trimmed["score"] == 4 and trimmed["hit_ids"] == ["fp_money_action"]
    assert "debug_blob" not in trimmed
    assert rough_tokens(str(trimmed)) < rough_tokens(str(raw)) / 5


def test_context_budget_compaction_reduces_tokens():
    turns = [{"role": "fraud" if i % 2 else "elder", "text": f"第 {i} 轮说明 " + "内容" * 40}
             for i in range(12)]
    compact = ContextBudget(enabled=True, max_history_turns=2, summarize_every=6)
    raw, compressed = compact.measure(turns)
    assert compressed < raw
    # 代理侧的占比统计（Agent 每轮 observe_raw + observe_tool）
    for t in turns:
        compact.observe_raw(t["text"])
    assert compact.token_saving_pct > 0


def test_signal_state_block_keeps_signals_verbatim():
    """压缩纪律：风险信号是结论，必须逐条保留，不能被摘要糊掉。"""
    signals = [Signal(type="money_action", quote="请您把资金转到核查账户", confidence=0.9),
               Signal(type="secrecy", quote="不能告诉任何人", confidence=0.8)]
    block = signal_state_block(signals, current_level="L3",
                               actions=["notify_family"], confirmed=["历史 L3 事件"])
    assert "money_action" in block and "请您把资金转到核查账户" in block
    assert "secrecy" in block and "不能告诉任何人" in block
    assert "L3" in block and "notify_family" in block


def test_memory_store_reopen_persists(tmp_path):
    path = tmp_path / "mem.db"
    s1 = MemoryStore(path)
    s1.upsert_elder("e1", name="测试")
    s1.close()
    s2 = MemoryStore(path)
    assert s2.get_elder("e1")["name"] == "测试"
    s2.close()


def test_seed_demo_profile_has_whitelist(store: MemoryStore):
    seed_demo_profile(store, "elder-xyz")
    wl = store.whitelist_members("elder-xyz")
    assert len(wl) >= 2
    assert all(w["is_whitelist"] for w in wl)
