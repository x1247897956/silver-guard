"""指标计算测试：公式是被测对象，不许"看起来差不多"。"""

from __future__ import annotations

from silverguard.metrics import (
    CaseMetrics,
    badcase_table,
    compare,
    deltas,
    metrics_from_assessment,
    pct,
    percentile,
    summarize,
)


def row(**kw) -> CaseMetrics:
    base = dict(case_id="c", config="agent", split="dev", is_attack=True, kind="refund_scam",
                max_level="L0", first_l2_turn=None, transfer_turn=None, se_attack=False,
                hard_negative=False, gold_min_level="L2", gold_max_level="L1",
                gold_signals=[], gold_tools=[], called_tools=[], tool_errors=0,
                unauthorized_actions=0, suppressed_actions=0, latency_ms=100, llm_calls=1,
                tool_calls=1, prompt_tokens=10, completion_tokens=5, final_action="none")
    base.update(kw)
    return CaseMetrics(**base)


def test_pct_and_percentile():
    assert pct(1, 4) == 25.0
    assert pct(1, 0) is None
    assert percentile([], 0.95) is None
    assert percentile([10], 0.95) == 10
    # 最近秩法：P50 of 10 → 第 ceil(5) = 5 个数 → 5；P95 → 第 10 个数 → 10
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.5) == 5
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.95) == 10


def test_ir_is_share_at_least_l2():
    rows = [row(case_id=f"a{i}", max_level=lv) for i, lv in enumerate(["L0", "L1", "L2", "L3", "L4"])]
    s = summarize("agent", rows)
    assert s.n_attack == 5
    assert s.ir == 60.0
    assert s.asr == 40.0


def test_pir_requires_strictly_before_transfer():
    """PIR 的分母只算 transfer_turn ≠ null 的样本；判据是**严格早于**。"""
    rows = [
        row(case_id="ok1", max_level="L3", first_l2_turn=2, transfer_turn=4),
        row(case_id="late", max_level="L3", first_l2_turn=4, transfer_turn=4),   # 同轮 → 不算
        row(case_id="after", max_level="L3", first_l2_turn=5, transfer_turn=4),  # 之后 → 不算
        row(case_id="miss", max_level="L1", first_l2_turn=None, transfer_turn=3),
        row(case_id="notransfer", max_level="L3", first_l2_turn=1, transfer_turn=None),  # 不进分母
    ]
    s = summarize("agent", rows)
    assert s.ir == 80.0
    assert s.pir == 25.0, "4 条有 transfer_turn，只有 1 条严格早于"


def test_fpr_is_graded_not_aggregated():
    rows = [
        row(case_id="b0", is_attack=False, max_level="L0"),
        row(case_id="b1", is_attack=False, max_level="L1"),
        row(case_id="b2", is_attack=False, max_level="L2", hard_negative=True),
        row(case_id="b3", is_attack=False, max_level="L3", hard_negative=True),
        row(case_id="b4", is_attack=False, max_level="L4", hard_negative=True),
    ]
    s = summarize("agent", rows)
    assert s.n_benign == 5
    assert s.fpr_l2 == 20.0
    assert s.fpr_l3 == 20.0
    assert s.fpr_l4 == 20.0
    assert s.fpr_hard == 100.0          # 3 条高难里 3 条被误报
    assert s.over_intervention == 40.0  # ≥L3 的两条


def test_se_asr_only_counts_social_engineering_subset():
    rows = [
        row(case_id="s1", se_attack=True, max_level="L3"),
        row(case_id="s2", se_attack=True, max_level="L1"),
        row(case_id="s3", se_attack=True, max_level="L0"),
        row(case_id="n1", se_attack=False, max_level="L0"),
    ]
    s = summarize("agent", rows)
    assert s.se_n == 3
    assert s.se_asr == 66.67
    assert s.ir == 25.0


def test_unauthorized_rate_and_tool_accuracy():
    rows = [
        row(case_id="x", tool_expected_scope=["check_contact", "check_fraud_pattern"],
            gold_tools=["check_contact", "check_fraud_pattern"],
            called_tools=["check_contact", "check_fraud_pattern"], tool_correct=True,
            unauthorized_actions=1, final_action="notify_family"),
        row(case_id="y", tool_expected_scope=["check_contact", "check_fraud_pattern"],
            gold_tools=["check_contact"], called_tools=["check_fraud_pattern"],
            tool_correct=False, tool_missing=["check_contact"], final_action="soft_reminder"),
        row(case_id="z", tool_expected_scope=[], tool_correct=False),
    ]
    s = summarize("agent", rows)
    assert s.tool_accuracy == 50.0
    assert s.unauthorized_count == 1
    assert s.actions_total == 2
    assert s.unauthorized_rate == 33.33


def test_badcase_classification():
    rows = [
        row(case_id="miss", max_level="L1", transfer_turn=3),
        row(case_id="late", max_level="L3", first_l2_turn=5, transfer_turn=3),
        row(case_id="fp3", is_attack=False, max_level="L3", gold_max_level="L1"),
        row(case_id="fp4", is_attack=False, max_level="L4", gold_max_level="L1"),
        row(case_id="fp2", is_attack=False, max_level="L2", gold_max_level="L1"),
        row(case_id="over", is_attack=False, max_level="L1", gold_max_level="L1",
            unauthorized_actions=1),
        row(case_id="clean", max_level="L3", first_l2_turn=1, transfer_turn=3),
    ]
    table = badcase_table(rows)
    assert "漏拦" in table and table["漏拦"] == ["miss"]
    assert "拦截过晚（资金动作之后才拦）" in table and table["拦截过晚（资金动作之后才拦）"] == ["late"]
    assert table["误报-惊动家属(L3)"] == ["fp3"]
    assert table["误报-误阻断(L4)"] == ["fp4"]
    assert table["误报-打扰老人(L2)"] == ["fp2"]
    assert table["越权动作"] == ["over"]
    assert "clean" not in [c for ids in table.values() for c in ids]


def test_compare_and_deltas_shape():
    summaries = {}
    for cfg, ir, pir in (("rule", 40.0, 0.0), ("single_llm", 70.0, 50.0),
                         ("agent", 80.0, 75.0), ("agent_memory", 85.0, 80.0)):
        rows = [row(case_id="a", max_level="L3")] + [row(case_id="b", is_attack=False, max_level="L0")]
        s = summarize(cfg, rows)
        s.ir, s.pir = ir, pir
        summaries[cfg] = s
    table = compare(summaries)
    assert [r["config"] for r in table] == ["rule", "single_llm", "agent", "agent_memory"]
    d = deltas(summaries)
    assert d["B_minus_A_pir"] == 50.0
    assert d["A_to_D_pir"] == 80.0
    assert d["D_minus_C_ir"] == 5.0


def test_metrics_from_assessment_maps_fields(store, policy):
    from silverguard.agent import CaseInput, GuardianAgent
    from silverguard.tools import ToolRuntime

    from .conftest import PATTERNS_PATH

    agent = GuardianAgent(settings=None, store=store, policy=policy, patterns_path=PATTERNS_PATH,
                          llm=None, config="rule", tool_runtime=ToolRuntime(store=store))
    case = CaseInput(case_id="m-1", turns=[
        {"role": "fraud", "text": "我是平台的客服，您先转 5000 元保证金。"},
        {"role": "elder", "text": "好。"},
    ], kind="refund_scam", transfer_turn=1, gold_min_level="L3",
        gold_tools=["check_contact", "check_fraud_pattern"])
    a = agent.assess(case)
    m = metrics_from_assessment(a, case)
    assert m.case_id == "m-1" and m.config == "rule"
    assert m.max_level == a.max_level and m.is_attack is True
    assert m.badcase_type() in (None, "拦截过晚（资金动作之后才拦）", "漏拦")


def test_empty_tool_gold_is_not_a_success():
    s = summarize("agent", [row(tool_expected_scope=["check_contact"], gold_tools=[],
                               tool_correct=True)])
    assert s.tool_accuracy is None
