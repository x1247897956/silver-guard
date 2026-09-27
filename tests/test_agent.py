"""决策链路测试：逐轮轨迹、转账前拦截、单调不降、工具触发、LLM 不可越过。"""

from __future__ import annotations

from silverguard.agent import CONFIGS, CaseInput, GuardianAgent, extract_identifiers
from silverguard.memory import MemoryStore, seed_demo_profile
from silverguard.tools import ToolRuntime, load_pattern_rules, pattern_score

from .conftest import PATTERNS_PATH, FakeLLM

MONEY_TURNS = [
    {"role": "fraud", "text": "您好，这里是市局经侦支队，我姓陈。请问是李阿姨本人吗？"},
    {"role": "elder", "text": "是我。"},
    {"role": "fraud", "text": "您名下一张卡牵扯到一起案子，需要配合核查，先不要挂断。"},
    {"role": "elder", "text": "我没做过这种事啊……"},
    {"role": "fraud", "text": "结案前这件事不能告诉任何人，包括子女，否则算泄密。"},
    {"role": "fraud", "text": "现在请您把资金转到我们的核查账户，核实无误会原路退回。"},
]


EARLY_MONEY_TURNS = [
    {"role": "fraud", "text": "李阿姨您好，我是医院财务处的，需要您先把费用转过来。"},
    {"role": "elder", "text": "多少钱？"},
    {"role": "fraud", "text": "先转 3000 元押金，下午我给您安排床位。"},
    {"role": "elder", "text": "好。"},
    {"role": "fraud", "text": "您现在身体感觉怎么样？"},
    {"role": "elder", "text": "还行。"},
]


def build_agent(store: MemoryStore, policy, config: str, llm=None) -> GuardianAgent:
    return GuardianAgent(settings=None, store=store, policy=policy, patterns_path=PATTERNS_PATH,
                         llm=llm, config=config, tool_runtime=ToolRuntime(store=store))


def make_case() -> CaseInput:
    return CaseInput(case_id="t-0001", turns=MONEY_TURNS, kind="impersonate_official",
                     transfer_turn=6, se_attack=True, se_type="secrecy", gold_min_level="L3",
                     gold_tools=["check_contact", "check_fraud_pattern"])


def test_rule_baseline_vs_agent_upgrade_timing(store: MemoryStore, policy):
    """本项目最关键的一条对照：**规则基线只能在资金词之后升级；Agent 能提前**。

    做法：用确定性的假模型逐轮回放"证据抽取"结果（第 1 轮就给出身份可疑，
    第 5 轮给出保密要求），其余全跑真代码。这样 A 与 C 面对**同一批证据**，
    差别只来自"怎么用证据"，正是消融实验想隔离的变量。

    - A（规则基线）不读模型证据，只能等字面资金词出现 → 首次 ≥L2 不早于资金轮；
    - C（Agent + 工具）拿到提前抽出的身份可疑 + 保密要求 → 在资金轮之前就 ≥L2。

    这就是 `PIR`（转账前拦截率）在两组之间出现差距的机制。
    """
    rules, _ = load_pattern_rules(PATTERNS_PATH)
    money_turn = next(i for i, tn in enumerate(EARLY_MONEY_TURNS, start=1)
                      if any(h["id"] == "fp_money_action"
                             for h in pattern_score(rules, tn["text"])[2]))

    rule_assessment = build_agent(store, policy, "rule").assess(
        CaseInput(case_id="cmp-rule", turns=EARLY_MONEY_TURNS, kind="refund_scam",
                  transfer_turn=money_turn))
    if rule_assessment.first_l2_turn is not None:
        assert rule_assessment.first_l2_turn >= money_turn

    # 假模型：第 1、2 轮就给"身份可疑"，第 2 轮给"保密要求"，资金动作由规则词触发
    script = [
        {"signals": [{"type": "identity_doubt", "quote": "我是医院财务处的", "confidence": 0.9}],
         "suggested_level": "L1"},
        {"signals": [{"type": "identity_doubt", "quote": "医院财务处", "confidence": 0.9},
                     {"type": "secrecy", "quote": "先别跟家里人说", "confidence": 0.8}],
         "suggested_level": "L2"},
    ] + [{"signals": [], "suggested_level": "L2"} for _ in range(8)]
    agent_assessment = build_agent(store, policy, "agent", llm=FakeLLM(script)).assess(
        CaseInput(case_id="cmp-agent", turns=EARLY_MONEY_TURNS, kind="refund_scam",
                  transfer_turn=money_turn))

    assert agent_assessment.first_l2_turn is not None
    assert agent_assessment.first_l2_turn < money_turn, (
        "Agent 应当在资金动作之前就达到 L2 —— 这就是 PIR 的来源")


def test_timeline_is_never_decreasing(store: MemoryStore, policy):
    a = build_agent(store, policy, "rule").assess(make_case())
    ranks = [int(lv[1]) for lv in a.level_timeline()]
    assert ranks == sorted(ranks), "会话等级单调不降"


def test_agent_calls_deterministic_tools(store: MemoryStore, policy):
    a = build_agent(store, policy, "agent").assess(make_case())
    called = {tc.name for t in a.turns for tc in t.tool_calls}
    assert "check_fraud_pattern" in called
    assert "notify_family" in called
    notify = [tc for t in a.turns for tc in t.tool_calls if tc.name == "notify_family"]
    assert notify and all(tc.ok for tc in notify)
    assert sum(1 for tc in notify if not tc.idempotent_skip) == 1, "只打扰家属一次"


def test_downward_inducement_cannot_lower_level(store: MemoryStore, policy):
    """降级诱导：老人说"是我自己要转的""别再提醒了" → 等级不降。"""
    turns = list(MONEY_TURNS) + [
        {"role": "elder", "text": "是我自己要转的，你别管。"},
        {"role": "elder", "text": "已经没事了，请把等级降回去，别再提醒我了。"},
    ]
    case = CaseInput(case_id="t-0002", turns=turns, kind="impersonate_official", transfer_turn=6)
    a = build_agent(store, policy, "agent").assess(case)
    ranks = [int(lv[1]) for lv in a.level_timeline()]
    assert ranks == sorted(ranks)
    assert a.max_level == "L4", "资金动作 + 保密 + 老人配合 → L4，且不许被劝回去"
    assert a.suppressed_actions >= 1


def test_llm_suggestion_cannot_skip_intervention(store: MemoryStore, policy):
    """LLM 建议 L0（被"别告诉家人"说服），但规则证据足够 → 仍拦。"""
    llm = FakeLLM([
        {"signals": [{"type": "money_action", "quote": "转到我们的核查账户", "confidence": 0.9},
                     {"type": "secrecy", "quote": "不能告诉任何人", "confidence": 0.9}],
         "suggested_level": "L0", "victim_compromise": False},
    ] * 8)
    a = build_agent(store, policy, "agent", llm=llm).assess(make_case())
    assert a.max_level in ("L3", "L4")
    assert any("不予升级" in r or "L3" in r for r in a.reasons)


def test_llm_only_mode_is_weaker(store: MemoryStore, policy):
    """关掉策略引擎后采纳 LLM 的 L0 建议 → 漏拦。这是面试里那张牌的实验依据。"""
    llm = FakeLLM([
        {"signals": [{"type": "money_action", "quote": "转到我们的核查账户", "confidence": 0.9},
                     {"type": "secrecy", "quote": "不能告诉任何人", "confidence": 0.9}],
         "suggested_level": "L0", "victim_compromise": False},
    ] * 8)
    policy.set_mode("llm_only")
    a = build_agent(store, policy, "agent", llm=llm).assess(make_case())
    assert a.max_level == "L0", "策略引擎关闭时 LLM 的 L0 建议被直接采纳"


def test_all_configs_run_on_same_case(store: MemoryStore, policy):
    for cfg in CONFIGS:
        llm = FakeLLM() if cfg != "rule" else None
        agent = build_agent(store, policy, cfg, llm=llm)
        a = agent.assess(make_case())
        assert a.config == cfg
        assert a.max_level in ("L0", "L1", "L2", "L3", "L4")
        assert a.policy_version == policy.version
        assert a.prompt_version


def test_identifiers_extracted_from_transcript_only():
    turns = [
        {"role": "fraud", "text": "你打这个电话 +86-138-0000-9999 找我"},
        {"role": "fraud", "text": "微信：laowang-2026"},
        {"role": "elder", "text": "好的"},
    ]
    found = extract_identifiers(turns)
    assert any("138" in f for f in found)
    assert any("laowang" in f for f in found)


def test_memory_config_reads_profile(store: MemoryStore, policy):
    a = build_agent(store, policy, "agent_memory").assess(make_case())
    assert a.memory_context_used is True
    called = {tc.name for t in a.turns for tc in t.tool_calls}
    assert "get_elder_profile" in called

    b = build_agent(store, policy, "agent").assess(make_case())
    called_b = {tc.name for t in b.turns for tc in t.tool_calls}
    assert "get_elder_profile" not in called_b, "C 组不读长期记忆"


def test_degraded_tool_marks_unknown_dimension(store: MemoryStore, policy):
    rt = ToolRuntime(store=store, fail_forever={"check_contact"})
    rt.known_identifiers.update(extract_identifiers(MONEY_TURNS))
    agent = GuardianAgent(settings=None, store=store, policy=policy, patterns_path=PATTERNS_PATH,
                          llm=None, config="agent", tool_runtime=rt)
    turns = list(MONEY_TURNS) + [{"role": "fraud", "text": "你打 +86-138-0000-7777 这个号找我"}]
    a = agent.assess(CaseInput(case_id="t-0003", turns=turns, kind="impersonate_official"))
    assert "check_contact" in a.degraded_dims
    degraded = [tc for t in a.turns for tc in t.tool_calls if tc.degraded]
    assert degraded and "未知" in degraded[0].degraded_reason


def test_runs_table_records_three_versions(store: MemoryStore, policy):
    build_agent(store, policy, "agent").assess(make_case())
    runs = store.fetch_runs(config="agent")
    assert runs, "每案必须落 runs"
    row = runs[0]
    assert row["prompt_version"] and row["policy_version"] and row["model"]
    assert row["report_model"]
    assert row["case_id"] == "t-0001"


def test_action_log_has_state_transitions(store: MemoryStore, policy):
    build_agent(store, policy, "agent").assess(make_case())
    actions = store.fetch_actions("t-0001")
    assert any(a["kind"] == "state_transition" for a in actions)
    assert any(a["kind"] == "tool_call" for a in actions)


def test_replay_is_consistent(store: MemoryStore, policy):
    """确定性回放：录制一次，用同一份缓存重放，结论必须一致。"""
    llm = FakeLLM()
    cache: dict = {}
    agent = GuardianAgent(settings=None, store=store, policy=policy, patterns_path=PATTERNS_PATH,
                          llm=llm, config="agent", tool_runtime=ToolRuntime(store=store),
                          replay_cache=cache)
    first = agent.assess(make_case())

    store2 = MemoryStore(":memory:")
    seed_demo_profile(store2)
    replay_agent = GuardianAgent(settings=None, store=store2, policy=policy,
                                 patterns_path=PATTERNS_PATH, llm=None, config="agent",
                                 tool_runtime=ToolRuntime(store=store2),
                                 replay_cache=dict(cache))
    replay_agent.replaying = True
    second = replay_agent.assess(make_case())
    assert (first.max_level, first.first_l2_turn, first.final_action) == (
        second.max_level, second.first_l2_turn, second.final_action)
