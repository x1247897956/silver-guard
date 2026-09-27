"""策略引擎测试：热加载、失败保留旧策略、LLM 不可越过、单调不降、授权检查。"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import yaml

from silverguard.policy import PolicyContext, PolicyEngine, PolicyLoadError

from .conftest import POLICY_PATH


def ctx(**kw):
    kw.setdefault("signal_confidence", {s: 0.9 for s in kw.get("signal_types", set())})
    return PolicyContext(**kw)


def test_loads_and_exposes_version(policy: PolicyEngine):
    assert policy.loaded
    assert policy.version.startswith("2026-")
    assert policy.tier == "balanced"
    assert policy.mode == "policy"


def test_l4_requires_victim_compromise(policy: PolicyEngine):
    """资金动作 + 保密 + 老人已配合 → L4；缺配合 → 只到 L3。

    这条断言的意义：L4 是"误阻断"，最贵的错误，
    所以它必须比 L3 多一个证据维度，而不是只看关键词。
    """
    d3 = policy.decide(ctx(signal_types={"money_action", "secrecy"}, pattern_hit=True))
    assert d3.level == "L3"
    assert d3.action == "notify_family"

    d4 = policy.decide(ctx(signal_types={"money_action", "secrecy"}, pattern_hit=True,
                           victim_compromise=True))
    assert d4.level == "L4"
    assert d4.action == "block_assist"


def test_normal_transfer_with_compromise_is_not_l4(policy: PolicyEngine):
    """家属正常要钱 + 老人一口答应：只有 money_action + 配合，**不许**判 L4。"""
    d = policy.decide(ctx(signal_types={"money_action"}, victim_compromise=True))
    assert d.level in ("L2", "L3")
    assert d.level != "L4"


def test_secrecy_only_is_low(policy: PolicyEngine):
    """只有"保密"一个信号（家属正常要求保密的样子）不该惊动家属。"""
    d = policy.decide(ctx(signal_types={"secrecy"}))
    assert d.level == "L1"


def test_llm_suggestion_cannot_escalate(policy: PolicyEngine):
    """LLM 建议 L4，但证据只够 L1 → 最终仍是 L1，且理由里写明拒升级。"""
    d = policy.decide(ctx(signal_types={"channel_anomaly"}, suggested_level="L4"))
    assert d.level == "L1"
    assert any("不予升级" in r for r in d.reasons)


def test_llm_only_mode_adopts_suggestion(policy: PolicyEngine):
    """关掉策略引擎（对照实验）时，建议等级被直接采纳。"""
    policy.set_mode("llm_only")
    d = policy.decide(ctx(signal_types={"channel_anomaly"}, suggested_level="L4"))
    assert d.level == "L4"
    assert any("llm_only" in r for r in d.reasons)
    policy.set_mode("policy")
    assert policy.mode == "policy"


def test_monotonic_no_downgrade(policy: PolicyEngine):
    """多轮把等级"劝"回去：规则只给出 L0/L1，但会话当前是 L3 → 维持 L3。"""
    d = policy.decide(ctx(signal_types=set(), current_level="L3"))
    assert d.level == "L3"
    assert any("单调不降" in r for r in d.reasons)


def test_monotonic_can_be_disabled_by_policy_table(tmp_path: Path):
    """要降级必须改策略表——这是"降级路径在架构上不存在"的机械证明。"""
    target = tmp_path / "policy.yaml"
    data = yaml.safe_load(Path(str(POLICY_PATH)).read_text(encoding="utf-8"))
    data["policy"]["monotonic"] = False
    target.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    engine = PolicyEngine(target)
    d = engine.decide(ctx(signal_types=set(), current_level="L3"))
    assert d.level == "L0"


def test_tier_changes_threshold(policy: PolicyEngine):
    sig = {"channel_anomaly", "urgency"}
    assert policy.decide(ctx(signal_types=sig)).level == "L2"
    policy.set_tier("strict")
    assert policy.decide(ctx(signal_types=sig)).level == "L1"
    policy.set_tier("balanced")
    assert policy.decide(ctx(signal_types=sig)).level == "L2"


def test_hot_reload_without_restart(tmp_path: Path):
    target = tmp_path / "policy.yaml"
    target.write_text(Path(str(POLICY_PATH)).read_text(encoding="utf-8"), encoding="utf-8")
    engine = PolicyEngine(target)
    before = engine.decide(ctx(signal_types={"channel_anomaly", "urgency"})).level

    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    data["policy"]["tiers"]["balanced"]["min_signals_l2"] = 4
    data["version"] = "test-hotreload.2"
    time.sleep(0.01)
    target.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")

    assert engine.maybe_reload() is True
    assert engine.version == "test-hotreload.2"
    after = engine.decide(ctx(signal_types={"channel_anomaly", "urgency"})).level
    assert before == "L2" and after == "L1"


def test_broken_yaml_keeps_old_policy(tmp_path: Path):
    """解析失败必须保留旧策略并告警，绝不能变成"无策略"。"""
    target = tmp_path / "policy.yaml"
    target.write_text(Path(str(POLICY_PATH)).read_text(encoding="utf-8"), encoding="utf-8")
    engine = PolicyEngine(target)
    good_version = engine.version
    good_decision = engine.decide(ctx(signal_types={"money_action", "secrecy"})).level

    time.sleep(0.01)
    target.write_text("version: broken\nrules: [this is: not valid\n", encoding="utf-8")
    assert engine.maybe_reload() is False
    assert engine.version == good_version
    assert engine.loaded is True
    assert engine.load_failures, "必须记录告警"
    assert engine.decide(ctx(signal_types={"money_action", "secrecy"})).level == good_decision


def test_missing_required_key_is_rejected(tmp_path: Path):
    target = tmp_path / "policy.yaml"
    data = yaml.safe_load(Path(str(POLICY_PATH)).read_text(encoding="utf-8"))
    del data["level_actions"]
    target.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    with pytest.raises(PolicyLoadError):
        PolicyEngine._validate(data)


def test_validation_rejects_undefined_action():
    data = yaml.safe_load(Path(str(POLICY_PATH)).read_text(encoding="utf-8"))
    data["level_actions"]["L3"] = "no_such_action"
    with pytest.raises(PolicyLoadError):
        PolicyEngine._validate(data)


def test_irreversible_action_requires_authorized_level():
    data = yaml.safe_load(Path(str(POLICY_PATH)).read_text(encoding="utf-8"))
    del data["actions"]["notify_family"]["requires_authorized_level"]
    with pytest.raises(PolicyLoadError):
        PolicyEngine._validate(data)
