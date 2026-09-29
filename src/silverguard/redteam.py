"""自动红队 / 对抗式评测：对被拦下的样本做变异，看绕过率怎么走。

> 攻击方 LLM 与守护 Agent 构成自动红队 / 对抗式评测流程。

流程（设计文档 §6.3）：

    R0  dev 集跑一遍 → 收集"被正确拦下"的样本
     ↓ 攻击方 LLM 变异（换包装 / 拆轮次 / 加社工 / 本人口吻）
    R1  变异样本并入 dev 集重跑 → 统计绕过率
     ↓ 再变异（对仍然被拦下的）→ 回灌
    R2  再跑一遍 → 出绕过率曲线

两条纪律：
1. **heldout 集全程封存**，由最终评测命令单独运行一次，红队流程禁止访问；
   主结论用 heldout，dev 曲线只作"对抗迭代过程"的过程证据；
2. **变异样本保留为单独文件**（`attack_redteam_R{n}.jsonl`），
   不打散写回 `attack.jsonl`——原始数据集一旦被改动，sha256 就失去意义。
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

from .agent import CaseInput, GuardianAgent
from .config import REPO_ROOT, Settings, get_settings
from .dataset import (
    FRAUD_TYPES,
    compliance_scan,
    load_dataset,
    validate_case,
    write_jsonl,
)
from .llm import LLMClient, LLMError
from .memory import MemoryStore, seed_demo_profile
from .metrics import metrics_from_assessment, summarize
from .policy import load_policy
from .prompts import REDTEAM_SYSTEM, redteam_user_prompt
from .tools import ToolRuntime

log = logging.getLogger("silverguard.redteam")

MUTATIONS = ("repackage", "split_turns", "add_social_engineering", "elder_voice")
MUTATION_CN = {
    "repackage": "换包装（业务术语化）",
    "split_turns": "拆轮次（把资金动作往后推）",
    "add_social_engineering": "加社工（保密/已授权/关提醒）",
    "elder_voice": "本人口吻转述",
}


def evaluate_cases(cases: list[CaseInput], *, settings: Settings, store: MemoryStore, policy,
                   patterns_path: Path, llm: LLMClient, config: str = "agent_memory",
                   trace_dir: Path | None = None) -> tuple[list, dict[str, Any]]:
    rows = []
    for case in cases:
        # Independent cases and rounds must not inherit previous intervention TTL or risk state.
        case_store = MemoryStore(":memory:")
        seed_demo_profile(case_store)
        agent = GuardianAgent(settings=settings, store=case_store, policy=policy,
                              patterns_path=patterns_path, llm=llm, config=config,
                              tool_runtime=ToolRuntime(store=case_store))
        try:
            a = agent.assess(case)
        finally:
            case_store.close()
        if "evidence_extraction" in a.degraded_dims:
            raise RuntimeError(f"{case.case_id}: 证据抽取失败，红队评测无效")
        tp = ""
        if trace_dir is not None:
            trace_dir.mkdir(parents=True, exist_ok=True)
            path = trace_dir / f"{config}__{case.case_id}.json"
            path.write_text(a.to_json(indent=2), encoding="utf-8")
            tp = str(path)
        rows.append(metrics_from_assessment(a, case, trace_path=tp))
    meta = {"llm_calls": llm.calls, "model_reported": llm.reported_model,
            "prompt_tokens": llm.prompt_tokens, "completion_tokens": llm.completion_tokens}
    return rows, meta


def mutate_case(client: LLMClient, case: CaseInput, mutation: str) -> dict[str, Any] | None:
    """让攻击方 LLM 生成一个语义等价、表层不同的变体。"""
    try:
        payload, resp = client.complete_json(
            system=REDTEAM_SYSTEM,
            user=redteam_user_prompt(case.turns, mutation, case.kind, case.transfer_turn),
            max_tokens=2200, temperature=0.9, use_cache=False,
        )
    except LLMError as exc:
        log.warning("变异失败（%s / %s）：%s", case.case_id, mutation, exc)
        return None
    if not isinstance(payload, dict):
        return None
    turns = [t for t in (payload.get("turns") or []) if isinstance(t, dict)
             and t.get("role") in ("fraud", "elder") and t.get("text")]
    if len(turns) < 2:
        return None
    tt = payload.get("transfer_turn")
    try:
        tt = int(tt) if tt is not None else None
    except (TypeError, ValueError):
        tt = None
    if tt is not None and not (1 <= tt <= len(turns)):
        tt = None
    se = bool(payload.get("se_attack"))
    se_type = payload.get("se_type") if se else None
    if se_type not in ("secrecy", "fake_authorization", "disable_guard", "elder_voice",
                       "privilege_lure"):
        se_type = "secrecy" if se else None
    return {
        "case_id": f"rt-{case.case_id}-{mutation[:4]}",
        "parent_case_id": case.case_id,
        "fraud_type": case.kind if case.kind in FRAUD_TYPES else "impersonate_official",
        "turns": turns,
        "transfer_turn": None,
        "suggested_transfer_turn": tt,
        "se_attack": se,
        "se_type": se_type,
        "gold_min_level": None,
        "gold_signals": [],
        "gold_tools": ["check_contact", "check_fraud_pattern"],
        "split": "dev",
        "source_note": f"自动红队变异（{MUTATION_CN.get(mutation, mutation)}），父样本 {case.case_id}",
        "manual_review": {"reviewed": False, "reviewer": "",
                          "date": time.strftime("%Y-%m-%d"),
                          "note": "待逐条审核话术合规、transfer_turn、gold_min_level；自动预筛不等于人工审核"},
        "mutation": mutation,
        "generation": {"model_requested": resp.model_requested,
                       "model_reported": resp.model_reported, "temperature": 0.9},
    }


def run_redteam(*, rounds: int = 2, settings: Settings | None = None,
                dataset_dir: Path | None = None, out_dir: Path | None = None,
                heldout_eval: bool = False, max_mutate: int | None = None,
                limit: int | None = None,
                seed: int = 20260927, reviewed_dir: Path | None = None) -> dict[str, Any]:
    if heldout_eval:
        raise ValueError("红队流程禁止评测 heldout；请在全部开发结束后用 runner 最终评测一次")
    settings = settings or get_settings(require_key=True)
    dataset_dir = dataset_dir or settings.dataset_dir
    out_dir = out_dir or (dataset_dir / "redteam")
    out_dir.mkdir(parents=True, exist_ok=True)
    policy = load_policy(settings.policy_path)
    patterns_path = settings.policy_path.parent / "fraud_patterns.yaml"
    rng = random.Random(seed)
    ds = load_dataset(dataset_dir, strict=True)

    dev_attack = [CaseInput.from_attack(r) for r in ds.attack if r.get("split") == "dev"]
    dev_benign = [CaseInput.from_benign(r) for r in ds.benign if r.get("split") == "dev"]
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit 必须大于 0")
        dev_attack, dev_benign = dev_attack[:limit], dev_benign[:limit]

    client = LLMClient(api_key=settings.api_key, model=settings.model, base_url=settings.base_url,
                       temperature=0.9, timeout=settings.request_timeout)
    store = MemoryStore(":memory:")
    seed_demo_profile(store)

    result: dict[str, Any] = {
        "policy_version": policy.version, "prompt_version": __import__(
            "silverguard.config", fromlist=["PROMPT_VERSION"]).PROMPT_VERSION,
        "dataset": {"dir": str(dataset_dir), "attack_sha256": ds.attack_sha256,
                    "benign_sha256": ds.benign_sha256, "counts": ds.counts()},
        "rounds": [], "heldout": {}, "notes": [
            "主结论用 heldout；dev 曲线只作为对抗迭代的过程证据。",
            "变异样本单独落盘，不回写 attack.jsonl（否则数据集 sha256 失去意义）。",
            (f"R0 使用开发子集 limit={limit}（attack {len(dev_attack)} / benign {len(dev_benign)}）；"
             "结果不外推到完整开发集。" if limit is not None else ""),
        ],
    }

    all_dev = dev_attack + dev_benign

    def snapshot(rows, meta, cases) -> dict[str, Any]:
        s = summarize("agent_memory", rows)
        attacks = [r for r in rows if r.is_attack]
        se = [r for r in rows if r.is_attack and r.se_attack]
        return {
            "n_attack": len(attacks), "n_benign": len(rows) - len(attacks),
            "IR": s.ir, "PIR": s.pir, "ASR": s.asr, "SE_ASR": s.se_asr, "n_se": len(se),
            "FPR_L3": s.fpr_l3, "over_intervention": s.over_intervention,
            "by_type_ir": s.per_type_ir, "meta": meta,
            "bypassed": sorted(r.case_id for r in attacks if not r.intercepted),
        }

    # ── R0 .. Rn ───────────────────────────────────────────────────
    current = list(all_dev)
    for rnd in range(rounds + 1):
        log.info("▶ 红队轮次 R%d（%d 条案例）", rnd, len(current))
        rows, meta = evaluate_cases(current, settings=settings, store=store, policy=policy,
                                    patterns_path=patterns_path, llm=client,
                                    # Keep red-team traces separate so this experiment never
                                    # overwrites replayable baseline evaluation traces.
                                    trace_dir=out_dir / "runs")
        snap = snapshot(rows, meta, current)
        snap["round"] = f"R{rnd}"
        snap["n_mutated_included"] = sum(1 for r in rows if r.is_attack and r.case_id.startswith("rt-"))
        result["rounds"].append(snap)
        if rnd == rounds:
            break

        # 只对"被正确拦下"的原始样本做变异
        intercepted = [c for c, r in zip(current, [rr for rr in rows])
                       if r.is_attack and r.intercepted and not c.case_id.startswith("rt-")]
        if not intercepted:
            result["notes"].append(f"R{rnd}: 没有被拦下的样本可供变异，共演进提前结束")
            break
        if max_mutate is not None:
            intercepted = intercepted[:max_mutate]
        new_rows: list[dict[str, Any]] = []
        for case in intercepted:
            mutation = rng.choice(MUTATIONS)
            row = mutate_case(client, case, mutation)
            if row is None:
                continue
            errs = [e for e in validate_case(row, is_attack=True)
                    if "审核痕迹" not in e and "gold_min_level" not in e]
            issues = compliance_scan([row])
            if errs or issues:
                log.warning("变异样本被拒（%s）：%s %s", row["case_id"], errs,
                            [i.reason for i in issues])
                continue
            new_rows.append(row)
        if not new_rows:
            result["notes"].append(f"R{rnd}: 变异样本全部未通过合规/结构预筛，共演进提前结束")
            break
        path = out_dir / f"attack_redteam_R{rnd + 1}.jsonl"
        write_jsonl(path, new_rows)
        result["rounds"][-1]["generated_variants"] = len(new_rows)
        result["rounds"][-1]["variant_file"] = str(path)
        result["rounds"][-1]["mutation_distribution"] = {
            m: sum(1 for r in new_rows if r.get("mutation") == m) for m in MUTATIONS if
            any(r.get("mutation") == m for r in new_rows)
        }
        reviewed_path = reviewed_dir / path.name if reviewed_dir else None
        if reviewed_path is None or not reviewed_path.exists():
            result["notes"].append(f"R{rnd + 1}: 候选已落盘，等待逐条审核和金标确认，未纳入指标")
            result["pending_review"] = str(path)
            break
        reviewed = [json.loads(line) for line in reviewed_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        allowed_parents = {c.case_id for c in intercepted}
        existing_ids = {c.case_id for c in current}
        for row in reviewed:
            errors = validate_case(row, is_attack=True)
            if (errors or compliance_scan([row]) or row.get("split") != "dev"
                    or row.get("parent_case_id") not in allowed_parents
                    or row.get("case_id") in existing_ids):
                raise ValueError(f"审核变体非法: {row.get('case_id')}: {errors}")
            existing_ids.add(row["case_id"])
        current = current + [CaseInput.from_attack(r) for r in reviewed]

    result["llm_totals"] = {"calls": client.calls, "model_reported": client.reported_model,
                            "prompt_tokens": client.prompt_tokens,
                            "completion_tokens": client.completion_tokens}
    client.close()
    store.close()
    return result


# ── 任务级指标：老人模拟器 + 劝说成功率 ─────────────────────────────
async def _elder_sim_async(settings: Settings, pressure: str, reminder: str) -> dict[str, Any]:
    from .prompts import ELDER_SIM_SYSTEM, ELDER_SIM_USER

    client = LLMClient(api_key=settings.api_key, model=settings.model, base_url=settings.base_url,
                       temperature=0.3)
    try:
        payload, resp = client.complete_json(
            system=ELDER_SIM_SYSTEM,
            user=ELDER_SIM_USER.format(pressure=pressure[:1200], reminder=reminder[:800]),
            max_tokens=300, temperature=0.3, use_cache=False)
    finally:
        client.close()
    if not isinstance(payload, dict):
        payload = {"will_transfer": True, "reason": "解析失败，保守视为仍会转账"}
    payload["model_reported"] = resp.model_reported
    return payload


def elder_simulator(settings: Settings, pressure: str, reminder: str) -> dict[str, Any]:
    import asyncio

    return asyncio.run(_elder_sim_async(settings, pressure, reminder))


def persuasion_experiment(*, settings: Settings | None = None, limit: int = 20,
                          dataset_dir: Path | None = None) -> dict[str, Any]:
    """任务级指标：在干预话术作用下，老人是否放弃资金操作（劝说成功率）。

    ⚠️ 边界如实声明：这是**LLM 模拟的老人**，不是真人；
    指标的可信度取决于模拟设定（见 prompts.ELDER_SIM_SYSTEM 的判定规则），
    所以报告里同时给出"无干预基线"作为对照——没有对照，
    "劝住了 80%"这种数字没有意义。
    """
    from .prompts import INTERVENTION_BY_LEVEL

    settings = settings or get_settings(require_key=True)
    ds = load_dataset(dataset_dir or settings.dataset_dir, strict=True)
    policy = load_policy(settings.policy_path)
    patterns_path = settings.policy_path.parent / "fraud_patterns.yaml"
    client = LLMClient(api_key=settings.api_key, model=settings.model, base_url=settings.base_url,
                       temperature=0.3)
    store = MemoryStore(":memory:")
    seed_demo_profile(store)
    agent = GuardianAgent(settings=settings, store=store, policy=policy,
                          patterns_path=patterns_path, llm=client, config="agent_memory",
                          tool_runtime=ToolRuntime(store=store))
    cases = [CaseInput.from_attack(r) for r in ds.attack if r.get("split") == "dev"][:limit]
    out = {"n": 0, "gave_up_with_intervention": 0, "gave_up_without_intervention": 0,
           "by_level": {}, "details": [], "model_reported": ""}
    for case in cases:
        a = agent.assess(case)
        pressure = "\n".join(t["text"] for t in case.turns if t.get("role") == "fraud")
        reminder = INTERVENTION_BY_LEVEL.get(a.max_level, INTERVENTION_BY_LEVEL["L2"])
        with_rem = elder_simulator(settings, pressure, reminder)
        without = elder_simulator(settings, pressure, "（没有收到任何提醒）")
        out["n"] += 1
        out["model_reported"] = with_rem.get("model_reported", out["model_reported"])
        if not with_rem.get("will_transfer", True):
            out["gave_up_with_intervention"] += 1
        if not without.get("will_transfer", True):
            out["gave_up_without_intervention"] += 1
        bucket = out["by_level"].setdefault(a.max_level, {"n": 0, "gave_up": 0})
        bucket["n"] += 1
        bucket["gave_up"] += 0 if with_rem.get("will_transfer", True) else 1
        out["details"].append({
            "case_id": case.case_id, "level": a.max_level, "reminder": reminder[:120],
            "with_intervention": with_rem.get("will_transfer"),
            "without_intervention": without.get("will_transfer"),
            "reason": with_rem.get("reason", "")[:120],
        })
        log.info("劝说实验 %d/%d（%s → %s）", out["n"], len(cases), case.case_id, a.max_level)
    out["persuasion_success_rate"] = (None if not out["n"] else
                                      round(out["gave_up_with_intervention"] / out["n"] * 100, 2))
    out["baseline_giveup_rate"] = (None if not out["n"] else
                                   round(out["gave_up_without_intervention"] / out["n"] * 100, 2))
    client.close()
    store.close()
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SilverGuard 自动红队共演进")
    p.add_argument("--rounds", type=int, default=2, help="共演进轮数（≥2 才有曲线）")
    p.add_argument("--dataset", default=None)
    p.add_argument("--traces-dir", default=str(REPO_ROOT / "data" / "runs"))
    p.add_argument("--max-mutate", type=int, default=None, help="每轮最多变异多少条（控成本）")
    p.add_argument("--limit", type=int, default=None, help="R0/Rn 每类最多评测条数（控成本）")
    p.add_argument("--no-heldout", action="store_true", help="兼容参数；红队始终封存 heldout")
    p.add_argument("--reviewed-dir", type=Path, default=None, help="逐条审核并确认金标后的变体目录")
    p.add_argument("--persuasion", action="store_true", help="额外跑老人模拟器（劝说成功率）")
    p.add_argument("--persuasion-limit", type=int, default=20)
    p.add_argument("--json-out", default=None)
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    settings = get_settings(require_key=True)
    result = run_redteam(rounds=args.rounds,
                         dataset_dir=Path(args.dataset) if args.dataset else None,
                         heldout_eval=False, max_mutate=args.max_mutate, limit=args.limit,
                         reviewed_dir=args.reviewed_dir)
    if args.persuasion:
        result["persuasion"] = persuasion_experiment(settings=settings, limit=args.persuasion_limit)
    print(json.dumps({k: v for k, v in result.items() if k != "rounds"}, ensure_ascii=False, indent=2))
    print("\n=== 绕过率曲线 ===")
    print("| 轮次 | 攻击集条数 | ASR | SE-ASR | 新增变异 |")
    print("| --- | --- | --- | --- | --- |")
    for r in result["rounds"]:
        print(f"| {r['round']} | {r['n_attack']} | {r['ASR']} | {r['SE_ASR']} | "
              f"{r.get('generated_variants', 0)} |")
    if result.get("heldout"):
        print("\n=== heldout 复测 ===")
        for k, v in result["heldout"].items():
            print(f"{k}: IR={v['IR']} PIR={v['PIR']} ASR={v['ASR']} SE-ASR={v['SE_ASR']}")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
        print(f"\nJSON → {args.json_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
