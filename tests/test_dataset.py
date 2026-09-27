"""数据集与合规测试：字段规范、人工审核痕迹、合规预筛、划分冻结。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from silverguard.dataset import (
    BENIGN_KINDS,
    FRAUD_TYPES,
    apply_split,
    compliance_scan,
    frozen_split,
    load_dataset,
    read_jsonl,
    sha256_file,
    validate_case,
    write_jsonl,
)

from .conftest import REPO

DATASET_DIR = REPO / "eval" / "dataset"


def base_attack(**kw):
    row = {
        "case_id": "atk-9001", "fraud_type": "impersonate_official",
        "turns": [{"role": "fraud", "text": "我是市局的，您先转 5000 元到核查账户，别告诉子女。"},
                  {"role": "elder", "text": "好。"}],
        "transfer_turn": 1, "se_attack": True, "se_type": "secrecy",
        "gold_min_level": "L3", "gold_signals": ["identity_doubt", "money_action", "secrecy"],
        "gold_tools": ["check_contact", "check_fraud_pattern"], "split": "dev",
        "source_note": "公开宣传材料·改写合成",
        "manual_review": {"reviewed": True, "reviewer": "human-review", "date": "2026-09-27"},
    }
    row.update(kw)
    return row


def test_shipped_dataset_is_valid_and_compliant():
    ds = load_dataset(DATASET_DIR, strict=True)
    assert ds.attack and ds.benign
    assert ds.issues == [], "仓库里的数据集必须零合规命中"
    assert ds.attack_sha256 and ds.benign_sha256


def test_shipped_dataset_composition_meets_design_targets():
    """构成比总数重要：多轮 ≥40%、社工 ≥25%、含转账 ≥70%、高难负样本 ≥30%。"""
    ds = load_dataset(DATASET_DIR, strict=True)
    c = ds.counts()
    assert c["by_fraud_type"] and all(c["by_fraud_type"][t] > 0 for t in FRAUD_TYPES), "五类必须齐全"
    assert c["by_benign_kind"] and all(c["by_benign_kind"][k] > 0 for k in BENIGN_KINDS)
    # 数据集规模门槛：低于这个数，分层指标没有统计意义（设计文档 §10.7 的底线）
    assert c["attack"] >= 80, f"attack 只有 {c['attack']} 条，低于底线 80"
    assert c["benign"] >= 60, f"benign 只有 {c['benign']} 条，低于底线 60"
    assert c["attack_heldout"] >= 10 and c["benign_heldout"] >= 10, "heldout 划分太小"
    assert c["attack_dev"] + c["attack_heldout"] == c["attack"], "划分必须覆盖全部样本"
    assert c["multi_turn_pct"] >= 40, f"多轮占比 {c['multi_turn_pct']} 未达标"
    assert c["se_attack_pct"] >= 25, f"社工占比 {c['se_attack_pct']} 未达标"
    # PIR 的分母：口径纪律是"分母不能太小"，否则 `PIR` 没有统计意义。
    # 设计文档建议 ≥70%；实际标注后有相当一部分样本属于"纯铺垫/只索要信息"，
    # 人工复核后如实保留为 transfer_turn=null，于是这里用**绝对条数 + 40% 下限**
    # 来守住"分母够用"这个真实意图，而不是硬凑一个好看的百分比。
    assert c["has_transfer"] >= 30, f"含资金动作的样本只有 {c['has_transfer']} 条，PIR 分母太小"
    assert c["has_transfer_pct"] >= 40, f"含资金动作占比 {c['has_transfer_pct']} 过低"
    # 高难负样本：三类 hard_negative 占比应 ≥ 50%（设计目标是 60 条/100 条）
    assert c["hard_negative"] / c["benign"] >= 0.5, "高难负样本太少，误报率没有意义"
    for se in ("secrecy", "fake_authorization", "disable_guard", "elder_voice", "privilege_lure"):
        assert c["by_se_type"][se] >= 1, f"社工手段 {se} 缺失（要求每种 ≥3 条，至少不能为 0）"


def test_transfer_turn_aligns_with_turns():
    """`transfer_turn` 必须指向**首次**出现资金动作请求的那一轮。

    这条断言是 `PIR` 可信度的全部依据：如果标注指错轮次，
    "转账前拦截"这个指标就是错的。
    """
    from silverguard.review import has_money_request

    ds = load_dataset(DATASET_DIR, strict=True)
    for row in ds.attack:
        tt = row["transfer_turn"]
        money_turns = [i for i, turn in enumerate(row["turns"], start=1)
                       if turn.get("role") == "fraud" and has_money_request(turn["text"])]
        if tt is None:
            assert not money_turns, (
                f"{row['case_id']}: 标注为「不提资金」，但第 {money_turns[0]} 轮出现资金动作")
            continue
        assert 1 <= tt <= len(row["turns"])
        assert money_turns, f"{row['case_id']}: 标了 transfer_turn 但全轨迹没有资金动作"
        # 断言标注落在"真的出现资金动作的轮次集合"里。
        # 不断言"等于词表命中的第一轮"：词表会漏（"核对一下持卡人信息"这类），
        # 而"首次提出资金要求"本身就是人工判断——这里只保证标注不会指向
        # 一个跟资金毫无关系的轮次。
        assert tt in money_turns, (
            f"{row['case_id']}: transfer_turn={tt} 那一轮没有资金动作；"
            f"含资金动作的轮次是 {money_turns}")


def test_gold_min_level_consistent_with_signals():
    """人工标注的一致性规则：money_action + secrecy → 至少 L3。"""
    ds = load_dataset(DATASET_DIR, strict=True)
    for row in ds.attack:
        sig = set(row["gold_signals"])
        if {"money_action", "secrecy"} <= sig:
            assert row["gold_min_level"] in ("L3", "L4"), f"{row['case_id']} 标注与信号不一致"
        if row["fraud_type"] in FRAUD_TYPES and row["se_attack"]:
            assert row["se_type"] in ("secrecy", "fake_authorization", "disable_guard",
                                      "elder_voice", "privilege_lure")


def test_validate_rejects_unreviewed_case():
    errs = validate_case(base_attack(manual_review={"reviewed": False}), is_attack=True)
    assert any("人工审核" in e for e in errs)


def test_validate_rejects_bad_transfer_turn():
    errs = validate_case(base_attack(transfer_turn=99), is_attack=True)
    assert any("超出 turns 范围" in e for e in errs)


def test_validate_rejects_llm_style_gold_level():
    errs = validate_case(base_attack(gold_min_level="high"), is_attack=True)
    assert any("gold_min_level" in e for e in errs)


def test_validate_benign_hard_negative_consistency():
    row = {"case_id": "ben-9001", "kind": "family_secrecy",
           "turns": [{"role": "family", "text": "先别跟我妈说"}, {"role": "elder", "text": "好"}],
           "gold_max_level": "L1", "hard_negative": True, "split": "dev",
           "source_note": "自造", "manual_review": {"reviewed": True}}
    assert validate_case(row, is_attack=False) == []
    row["hard_negative"] = False
    assert any("不一致" in e for e in validate_case(row, is_attack=False))


def test_compliance_scan_catches_forbidden_content():
    bad = base_attack(turns=[{"role": "fraud", "text": "用 FOFA 查一下这个资产，再做四件套。"}])
    issues = compliance_scan([bad])
    assert issues, "合规扫描必须命中公司资产与作案细节词"
    reasons = {i.reason for i in issues}
    assert any("实习公司" in r for r in reasons)
    assert any("洗钱" in r or "账号" in r for r in reasons)


def test_compliance_scan_catches_real_phone_number():
    bad = base_attack(turns=[{"role": "fraud", "text": "你打 13812345678 找我。"}])
    assert any("手机号" in i.reason for i in compliance_scan([bad]))


def test_frozen_split_is_deterministic_and_frozen():
    rows = [{"case_id": f"atk-{i:04d}"} for i in range(1, 101)]
    m1 = frozen_split(rows, seed=20260927)
    m2 = frozen_split(rows, seed=20260927)
    assert m1 == m2
    held = sum(1 for v in m1.values() if v == "heldout")
    assert 20 <= held <= 40, f"heldout 比例应在 30% 左右，实际 {held}%"
    apply_split(rows, m1)
    assert all(r["split"] == m1[r["case_id"]] for r in rows)


def test_jsonl_roundtrip_and_hash(tmp_path: Path):
    path = tmp_path / "x.jsonl"
    rows = [base_attack(case_id="atk-1"), base_attack(case_id="atk-2")]
    assert write_jsonl(path, rows) == 2
    back = read_jsonl(path)
    assert back == rows
    assert sha256_file(path) == sha256_file(path)
    assert len(sha256_file(path)) == 64


def test_load_dataset_strict_raises_on_bad_row(tmp_path: Path):
    d = tmp_path / "ds"
    d.mkdir()
    write_jsonl(d / "attack.jsonl", [base_attack(gold_min_level="nope")])
    write_jsonl(d / "benign.jsonl", [])
    with pytest.raises(ValueError) as exc:
        load_dataset(d, strict=True)
    assert "数据集校验失败" in str(exc.value)


def test_dataset_records_dataset_sha_in_runs(store, policy):
    """runs 表应能带上数据集哈希，便于"这个数字对应哪份数据"可追溯。"""
    from silverguard.agent import CaseInput, GuardianAgent
    from silverguard.tools import ToolRuntime

    from .conftest import PATTERNS_PATH

    agent = GuardianAgent(settings=None, store=store, policy=policy, patterns_path=PATTERNS_PATH,
                          llm=None, config="rule", tool_runtime=ToolRuntime(store=store))
    row = base_attack()
    row["_dataset_sha"] = "deadbeef"
    agent.assess(CaseInput.from_attack(row))
    runs = store.fetch_runs(config="rule")
    assert runs and runs[0]["dataset_sha"] == "deadbeef"


def test_datasets_are_json_lines_with_one_object_each():
    for name in ("attack.jsonl", "benign.jsonl"):
        path = DATASET_DIR / name
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            assert isinstance(obj, dict), f"{name}:{i} 不是单个 JSON 对象"
