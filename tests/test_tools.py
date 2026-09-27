"""工具层测试：schema 校验、权限最小化、幂等、容错降级、参数注入拦截。"""

from __future__ import annotations

import pytest

from silverguard.memory import MemoryStore
from silverguard.tools import (
    ToolRegistry,
    ToolRuntime,
    ToolSchemaError,
    idempotency_key,
    validate_args,
)


@pytest.fixture()
def registry(tool_runtime: ToolRuntime, policy) -> ToolRegistry:
    return ToolRegistry(tool_runtime, policy=policy)


def test_schema_rejects_missing_and_wrong_type(registry: ToolRegistry):
    a = registry.call("check_contact", {})
    assert not a.ok and a.rejected_by == "schema" and "identifier" in a.error
    b = registry.call("check_contact", {"identifier": 123})
    assert not b.ok and b.rejected_by == "schema"


def test_validate_args_truncates_over_long_text():
    out = validate_args("check_fraud_pattern", {"text": "x" * 5000})
    assert len(out["text"]) == 4000


def test_contact_whitelist_lookup(registry: ToolRegistry):
    ok = registry.call("check_contact", {"identifier": "+86-138-0000-0001", "elder_id": "elder-0001"})
    assert ok.ok and ok.result["is_whitelist"] is True and ok.result["first_seen"] is False


def test_contact_rejects_model_invented_identifier(registry: ToolRegistry):
    """⚠️ 参数注入防护的核心用例：模型自造标识必须被拒。"""
    bad = registry.call("check_contact", {"identifier": "+86-999-9999-9999", "elder_id": "elder-0001"})
    assert not bad.ok
    assert bad.rejected_by == "semantic"
    assert "未在输入轨迹或记忆层中出现" in bad.error
    assert registry.stats["schema_rejected"] >= 1


def test_get_elder_profile_returns_whitelist_and_history(registry: ToolRegistry):
    r = registry.call("get_elder_profile", {"elder_id": "elder-0001"})
    assert r.ok
    assert r.result["exists"] is True
    assert any(m["label"].startswith("女儿") for m in r.result["whitelist"])
    assert r.result["recent_transactions"]


def test_notify_family_requires_authorization_level(registry: ToolRegistry):
    """不可逆动作在低等级下必须被权限层拒绝（不是靠调用方自觉）。"""
    denied = registry.call("notify_family", {"elder_id": "elder-0001", "summary": "s"}, level="L1")
    assert not denied.ok and denied.rejected_by == "privilege"
    assert registry.stats["privilege_denied"] == 1

    allowed = registry.call("notify_family", {"elder_id": "elder-0001", "summary": "s"}, level="L3")
    assert allowed.ok and allowed.result["sent"] is True
    assert allowed.result["mock"] is True, "接口是真的，通道是 mock——不许真的发出去"


def test_notify_family_rejects_non_whitelist_member(registry: ToolRegistry):
    r = registry.call("notify_family",
                      {"elder_id": "elder-0001", "summary": "s", "member_id": "m-deadbeef"},
                      level="L4")
    assert not r.ok and r.rejected_by == "semantic"
    assert "不在白名单内" in r.error


def test_notify_family_is_idempotent_within_ttl(registry: ToolRegistry):
    first = registry.call("notify_family", {"elder_id": "elder-0001", "summary": "s"}, level="L3")
    second = registry.call("notify_family", {"elder_id": "elder-0001", "summary": "s"}, level="L3")
    assert first.ok and not first.idempotent_skip
    assert second.ok and second.idempotent_skip, "TTL 内不得重复打扰家属"
    key = idempotency_key("elder-0001", "notify_family", "all")
    assert registry.rt.store.intervention_count(key) == 1


def test_tool_failure_degrades_explicitly(store: MemoryStore, tool_runtime: ToolRuntime, policy):
    """工具永久失败 → 显式降级并标注，而不是"猜一个安全结论"。"""
    tool_runtime.fail_forever = {"check_fraud_pattern"}
    reg = ToolRegistry(tool_runtime, policy=policy)
    call = reg.call("check_fraud_pattern", {"text": "随便一段话"})
    assert not call.ok
    assert call.degraded is True
    assert "未知" in call.degraded_reason
    assert call.attempts == tool_runtime.max_retries + 1
    assert reg.stats["retries"] >= 1


def test_tool_retry_succeeds_after_transient_failure(store: MemoryStore, tool_runtime: ToolRuntime, policy):
    tool_runtime.fail_times = {"check_fraud_pattern": 1}
    reg = ToolRegistry(tool_runtime, policy=policy)
    call = reg.call("check_fraud_pattern", {"text": "转账 验证码"})
    assert call.ok and call.attempts == 2


def test_pattern_rules_hit_and_miss(registry: ToolRegistry):
    hit = registry.call("check_fraud_pattern", {"text": "您别跟孩子说，先把资金转到核查账户"})
    assert hit.ok and hit.result["pattern_hit"] is True
    assert hit.result["score"] > 0 and hit.result["rule_version"]

    miss = registry.call("check_fraud_pattern", {"text": "妈，降压药还有吗？我周末回去给你带两盒。"})
    assert miss.ok and miss.result["pattern_hit"] is False


def test_unknown_tool_rejected(registry: ToolRegistry):
    call = registry.call("drop_database", {})
    assert not call.ok and call.rejected_by == "schema"


def test_validate_args_rejects_nul_byte():
    with pytest.raises(ToolSchemaError):
        validate_args("check_fraud_pattern", {"text": "abc\x00def"})
