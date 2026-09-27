"""runner 与自动红队的离线测试（用假模型，不访问网络）。"""

from __future__ import annotations

import json
from pathlib import Path

from silverguard.config import Settings
from silverguard.metrics import summarize

from .conftest import PATTERNS_PATH


# ── runner：门禁 ────────────────────────────────────────────────────
def test_baseline_gate_passes_when_metrics_hold():
    from silverguard.runner import baseline_gate

    s = summarize("rule", [])
    s.ir, s.pir, s.fpr_l3 = 50.0, 40.0, 2.0
    baseline = {"summaries": {"rule": {"ir": 48.0, "pir": 38.0, "fpr_l3": 3.0}}}
    ok, msgs = baseline_gate({"rule": s}, baseline)
    assert ok is True
    assert any("✅" in m for m in msgs)


def test_baseline_gate_fails_on_pir_regression():
    from silverguard.runner import baseline_gate

    s = summarize("rule", [])
    s.ir, s.pir, s.fpr_l3 = 50.0, 20.0, 2.0
    baseline = {"summaries": {"rule": {"pir": 40.0, "ir": 50.0, "fpr_l3": 2.0}}}
    ok, msgs = baseline_gate({"rule": s}, baseline)
    assert ok is False
    assert any("pir 掉线" in m for m in msgs)


def test_baseline_gate_fails_on_fpr_regression():
    """FPR-L3 是"惊动家属"，只能变小不能变大——变大喊停。"""
    from silverguard.runner import baseline_gate

    s = summarize("rule", [])
    s.ir, s.pir, s.fpr_l3 = 60.0, 50.0, 12.0
    baseline = {"summaries": {"rule": {"pir": 50.0, "ir": 60.0, "fpr_l3": 5.0}}}
    ok, msgs = baseline_gate({"rule": s}, baseline)
    assert ok is False
    assert any("fpr_l3 越界" in m for m in msgs)


def test_baseline_gate_warns_when_nothing_comparable():
    from silverguard.runner import baseline_gate

    ok, msgs = baseline_gate({"rule": summarize("rule", [])}, {"summaries": {}})
    assert ok is True
    assert any("门禁未生效" in m for m in msgs)


# ── runner：案例装配 ────────────────────────────────────────────────
def test_build_cases_filters_and_limits():
    from silverguard.dataset import load_dataset
    from silverguard.runner import build_cases

    ds = load_dataset(PATHS_DATASET, strict=True)
    cases = build_cases(ds, split="dev", limit=3, kind_filter=None)
    assert len(cases) == 6  # 3 attack + 3 benign
    only_pig = build_cases(ds, split="all", limit=None, kind_filter="romance_pig_butcher")
    assert only_pig and all(c.kind == "romance_pig_butcher" for c in only_pig)


PATHS_DATASET = PATTERNS_PATH.parents[1] / "eval" / "dataset"


def test_run_config_offline_rule(store, policy, settings):
    """rule 配置全程零模型调用：这是 CI 离线门禁能永远跑得起来的原因。"""
    from silverguard.agent import CaseInput
    from silverguard.runner import run_config

    cases = [CaseInput(case_id="r-1", turns=[
        {"role": "fraud", "text": "我是市局的，请您把资金转到核查账户，别告诉子女。"},
        {"role": "elder", "text": "好。"}], kind="impersonate_official", transfer_turn=1,
        gold_min_level="L3")]
    rows, summary, traces, meta = run_config(
        config="rule", cases=cases, settings=settings, policy=policy,
        patterns_path=PATTERNS_PATH, llm_factory=lambda cfg: None, trace_dir=None)
    assert meta["llm_calls"] == 0
    assert summary.ir == 100.0
    assert rows[0].max_level in ("L2", "L3", "L4")


# ── 自动红队 ────────────────────────────────────────────────────────
class MutatorLLM:
    """假攻击方：只负责"变异"。真流程里同一个客户端也做证据抽取，
    所以这里同时实现 complete_json 的两个分支（按 system 提示词区分）。
    """

    def __init__(self, *, turns=None, model="deepseek-chat", recipe=None):
        self.model = model
        self.reported_model = "deepseek-flash"
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cache_hits = 0
        self.turns = turns or [{"role": "fraud", "text": "阿姨您好，我们这边有个内部名额。"},
                               {"role": "elder", "text": "什么名额？"},
                               {"role": "fraud", "text": "您先转 5000 元锁定，别跟孩子讲。"}]
        from .conftest import FakeLLM

        self._evidence = FakeLLM(recipe)

    def complete_json(self, *, system: str = "", user: str = "", **kw):
        from silverguard.llm import LLMResponse

        self.calls += 1
        if "证据抽取模块" in system:
            payload, _ = self._evidence.complete_json(system=system, user=user, **kw)
        else:
            payload = {"turns": self.turns, "transfer_turn": 3, "se_attack": True,
                       "se_type": "secrecy", "mutation": "repackage", "note": "假变异"}
        return payload, LLMResponse(text=json.dumps(payload, ensure_ascii=False),
                                    model_requested=self.model, model_reported="deepseek-flash")

    def complete(self, **kw):
        raise AssertionError("红队流程不应该走到 complete()")

    def close(self):
        return None


def _case():
    from silverguard.agent import CaseInput

    return CaseInput(case_id="atk-x", turns=[
        {"role": "fraud", "text": "我是市局的，请您把资金转到核查账户，别告诉子女。"},
        {"role": "elder", "text": "好。"}], kind="impersonate_official", transfer_turn=1,
        se_attack=True, se_type="secrecy", gold_min_level="L3")


def test_mutate_case_produces_valid_row():
    from silverguard.redteam import mutate_case

    row = mutate_case(MutatorLLM(), _case(), "repackage")
    assert row is not None
    assert row["case_id"].startswith("rt-atk-x-")
    assert row["parent_case_id"] == "atk-x"
    assert row["mutation"] == "repackage"
    assert row["manual_review"]["reviewed"] is True


def test_mutate_case_rejects_noncompliant_variant():
    """攻击方若生成含作案细节的内容，必须被合规闸门挡在数据集之外。"""
    from silverguard.dataset import compliance_scan, validate_case
    from silverguard.redteam import mutate_case

    bad = MutatorLLM(turns=[{"role": "fraud", "text": "你先做四件套，再到 FOFA 上查一下资产。"},
                            {"role": "elder", "text": "好。"}])
    row = mutate_case(bad, _case(), "repackage")
    assert row is not None
    assert compliance_scan([row]), "合规扫描必须命中"
    # 真实流程里这条会被丢弃（见 run_redteam 的预筛）；这里断言闸门确实有效
    assert validate_case(row, is_attack=True) == []


def test_redteam_round_structure_with_fake_paths(tmp_path: Path, monkeypatch):
    """不调真模型地验证共演进骨架：轮次、曲线字段、heldout 两次快照。"""
    import silverguard.redteam as rt

    ds_dir = tmp_path / "dataset"
    ds_dir.mkdir()
    # 直接复制仓库数据集，保证结构与合规都是真的
    for name in ("attack.jsonl", "benign.jsonl"):
        (ds_dir / name).write_bytes((PATHS_DATASET / name).read_bytes())

    settings = Settings(api_key="x", model="deepseek-chat", policy_path=PATHS_POLICY,
                        dataset_dir=ds_dir)

    created: list[MutatorLLM] = []

    def factory(**kw):
        llm = MutatorLLM()
        created.append(llm)
        return llm

    monkeypatch.setattr(rt, "LLMClient", factory)
    result = rt.run_redteam(rounds=1, settings=settings, dataset_dir=ds_dir,
                            out_dir=tmp_path / "rt", max_mutate=3)
    assert result["rounds"], "至少要有 R0"
    assert result["rounds"][0]["round"] == "R0"
    assert "IR" in result["rounds"][0] and "SE_ASR" in result["rounds"][0]
    assert "baseline" in result["heldout"] and "after_evolution" in result["heldout"]
    assert result["dataset"]["attack_sha256"]
    # R0 之后必须真的产生过变异样本，并且单独落盘（不回写 attack.jsonl）
    variants = list((tmp_path / "rt").glob("attack_redteam_R*.jsonl"))
    assert variants, "第二轮之前必须落盘变异样本"
    assert "generated_variants" in result["rounds"][0]
    assert result["rounds"][1]["n_mutated_included"] > 0, "变异样本必须进入下一轮评测"


PATHS_POLICY = PATTERNS_PATH.parent / "policy.yaml"


def test_redteam_rejects_llm_that_returns_too_few_turns():
    from silverguard.redteam import mutate_case

    llm = MutatorLLM(turns=[{"role": "fraud", "text": "只有一轮"}])
    assert mutate_case(llm, _case(), "split_turns") is None


def test_mutation_set_covers_four_documented_manoeuvres():
    from silverguard.redteam import MUTATION_CN, MUTATIONS

    assert set(MUTATIONS) == {"repackage", "split_turns", "add_social_engineering", "elder_voice"}
    assert len(MUTATION_CN) == 4


def test_persuasion_metrics_shape(monkeypatch, settings):
    """老人模拟器：即使不调模型也要给出与无干预基线的对照字段。"""
    import silverguard.redteam as rt

    ds_dir = PATHS_DATASET
    settings = Settings(api_key="x", model="deepseek-chat", policy_path=PATHS_POLICY,
                        dataset_dir=ds_dir)
    monkeypatch.setattr(rt, "elder_simulator",
                        lambda s, p, r: {"will_transfer": "没有提醒" not in r, "reason": "假"})
    import silverguard.config as cfg

    monkeypatch.setattr(cfg, "get_settings", lambda **kw: settings)
    out = rt.persuasion_experiment(settings=settings, limit=2)
    assert out["n"] == 2
    assert out["persuasion_success_rate"] is not None
    assert out["baseline_giveup_rate"] is not None
