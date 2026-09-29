"""评测 runner：一条命令跑完全集出指标。

用法（详见 Makefile）：

    python -m silverguard.runner --configs rule,single_llm,agent,agent_memory --limit 5
    python -m silverguard.runner --configs agent_memory --split heldout --out docs/eval-report.md
    python -m silverguard.runner --thresholds          # 阈值扫描
    python -m silverguard.runner --matrix              # 策略引擎开 / 关对照（SE-ASR）

输出：控制台指标表 + JSON 结果 + Badcase 报表 + 逐案轨迹落库。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from .agent import CONFIG_LABELS, CONFIGS, CaseInput, GuardianAgent
from .config import PROMPT_VERSION, REPO_ROOT, get_settings
from .dataset import Dataset, load_dataset
from .llm import LLMClient
from .memory import MemoryStore, seed_demo_profile
from .metrics import (
    CaseMetrics,
    MetricSummary,
    badcase_table,
    deltas,
    metrics_from_assessment,
    summarize,
)
from .models import level_rank
from .policy import load_policy
from .tools import ToolRuntime

log = logging.getLogger("silverguard.runner")

GRADE_NAMES = {"L0": "L0 正常", "L1": "L1 可疑", "L2": "L2 高度可疑",
               "L3": "L3 极可能诈骗", "L4": "L4 确认诈骗特征"}


# ── 环境信息 ────────────────────────────────────────────────────────
def environment_note() -> dict[str, Any]:
    import platform
    import sys

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "Apple Silicon",
        "prompt_version": PROMPT_VERSION,
    }


# ── 主流程 ──────────────────────────────────────────────────────────
def build_cases(ds: Dataset, *, split: str, limit: int | None, kind_filter: str | None,
                attack_limit: int | None = None, benign_limit: int | None = None,
                elder_id: str = "elder-0001") -> list[CaseInput]:
    attacks = [r for r in ds.attack if split == "all" or r.get("split") == split]
    benign = [r for r in ds.benign if split == "all" or r.get("split") == split]
    if kind_filter:
        wanted = set(kind_filter.split(","))
        attacks = [r for r in attacks if r.get("fraud_type") in wanted]
        benign = [r for r in benign if r.get("kind") in wanted]
    if attack_limit is not None:
        attacks = attacks[:attack_limit]
    if benign_limit is not None:
        benign = benign[:benign_limit]
    if limit is not None:
        attacks, benign = attacks[:limit], benign[:limit]
    cases = [CaseInput.from_attack(r, elder_id) for r in attacks]
    cases += [CaseInput.from_benign(r, elder_id) for r in benign]
    return cases


#: CI 门禁子集规模：固定条数 + 完全确定性（按 case_id 排序取前 N）。
#: 为什么要有它：门禁要在每次 push 上跑，成本必须可控；
#: 而 A 组（规则基线）**不调用模型**，所以子集上的指标是确定性的、不会抖动。
CI_SUBSET_ATTACK = 40
CI_SUBSET_BENIGN = 25


def ci_subset(cases: list[CaseInput]) -> list[CaseInput]:
    attacks = sorted((c for c in cases if c.is_attack), key=lambda c: c.case_id)[:CI_SUBSET_ATTACK]
    benign = sorted((c for c in cases if not c.is_attack), key=lambda c: c.case_id)[:CI_SUBSET_BENIGN]
    return attacks + benign


def make_agent(*, settings, store: MemoryStore, policy, patterns_path: Path, config: str,
               llm: LLMClient | None, compaction: bool, cache: dict[str, Any],
               tool_fail: set[str] | None = None) -> GuardianAgent:
    rt = ToolRuntime(store=store, fail_forever=set(tool_fail or set()))
    return GuardianAgent(
        settings=settings, store=store, policy=policy, patterns_path=patterns_path,
        llm=llm, config=config, tool_runtime=rt, compaction=compaction, replay_cache=cache,
    )


def run_config(*, config: str, cases: list[CaseInput], settings, policy, patterns_path: Path,
               llm_factory, compaction: bool = False, trace_dir: Path | None = None,
               tool_fail: set[str] | None = None, seed_profile: bool = True) -> tuple[
        list[CaseMetrics], MetricSummary, list[dict[str, Any]], dict[str, Any]]:
    """跑一组配置的全部案例。返回 (逐案指标, 汇总, 轨迹, meta)。"""
    from .config import get_settings as _gs  # noqa: F401  （保持签名稳定）

    store = MemoryStore(":memory:")
    if seed_profile:
        seed_demo_profile(store)
    llm = llm_factory(config) if llm_factory else None
    cache: dict[str, Any] = {}
    agent = make_agent(settings=settings, store=store, policy=policy, patterns_path=patterns_path,
                       config=config, llm=llm, compaction=compaction, cache=cache,
                       tool_fail=tool_fail)
    rows: list[CaseMetrics] = []
    traces: list[dict[str, Any]] = []
    started = time.time()
    for i, case in enumerate(cases, start=1):
        if i > 1:
            store.close()
            store = MemoryStore(":memory:")
            if seed_profile:
                seed_demo_profile(store)
            agent = make_agent(settings=settings, store=store, policy=policy, patterns_path=patterns_path,
                               config=config, llm=llm, compaction=compaction, cache=cache,
                               tool_fail=tool_fail)
        assessment = agent.assess(case)
        tp = ""
        trace_payload = {
            **assessment.to_dict(),
            # 回放需要原始案例：轨迹文件必须自带 case，否则"回放"无从谈起
            "case": {"case_id": case.case_id, "turns": case.turns, "kind": case.kind,
                     "split": case.split, "is_attack": case.is_attack,
                     "transfer_turn": case.transfer_turn, "se_attack": case.se_attack,
                     "gold_min_level": case.gold_min_level, "gold_max_level": case.gold_max_level},
            "llm_cache": cache,
        }
        if trace_dir is not None:
            trace_dir.mkdir(parents=True, exist_ok=True)
            path = trace_dir / f"{config}__{case.case_id}.json"
            path.write_text(json.dumps(trace_payload, ensure_ascii=False, indent=2),
                            encoding="utf-8")
            tp = str(path.relative_to(REPO_ROOT)) if str(path).startswith(str(REPO_ROOT)) else str(path)
        rows.append(metrics_from_assessment(assessment, case, trace_path=tp))
        traces.append({"case_id": case.case_id, "config": config,
                       "assessment": assessment.to_dict(), "llm_cache": cache})
        if i % 25 == 0:
            log.info("[%s] %d/%d 完成（%.1fs）", config, i, len(cases), time.time() - started)
    summary = summarize(config, rows)
    meta = {
        "wall_clock_sec": round(time.time() - started, 1),
        "llm_calls": llm.calls if llm else 0,
        "model_requested": llm.model if llm else "",
        "model_reported": llm.reported_model if llm else "",
        "prompt_tokens": llm.prompt_tokens if llm else 0,
        "completion_tokens": llm.completion_tokens if llm else 0,
        "llm_cache_hits": llm.cache_hits if llm else 0,
        "tool_stats": {"scope": "last_case_only", **dict(agent.registry.stats)},
        "policy_version": policy.version,
        "policy_tier": policy.tier,
        "policy_mode": policy.mode,
        "compaction": compaction,
        "tool_fail_injected": sorted(tool_fail or set()),
        "llm_requests": llm.calls if llm else 0,
    }
    if llm:
        llm.close()
    store.close()
    return rows, summary, traces, meta


# ── 报告渲染 ────────────────────────────────────────────────────────
def fmt_pct(v: float | None) -> str:
    return "未测" if v is None else f"{v:.2f}%"


def metric_table(summaries: dict[str, MetricSummary]) -> str:
    head = ("| 配置 | n(attack) | n(benign) | `IR` | `PIR` | `FPR-L2` | `FPR-L3` | `FPR-L4` "
            "| `SE-ASR` | 越权率 | 工具正确率 | P95(ms) | 平均tokens |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    lines = [head]
    for cfg in CONFIGS:
        s = summaries.get(cfg)
        if s is None:
            continue
        lines.append(
            f"| {CONFIG_LABELS[cfg]} | {s.n_attack} | {s.n_benign} | {fmt_pct(s.ir)} | {fmt_pct(s.pir)} "
            f"| {fmt_pct(s.fpr_l2)} | {fmt_pct(s.fpr_l3)} | {fmt_pct(s.fpr_l4)} | {fmt_pct(s.se_asr)} "
            f"| {fmt_pct(s.unauthorized_rate)} | {fmt_pct(s.tool_accuracy)} | {s.p95_latency_ms} "
            f"| {s.mean_tokens} |"
        )
    return "\n".join(lines)


def console_report(summaries: dict[str, MetricSummary], metas: dict[str, dict[str, Any]],
                   *, title: str) -> str:
    out = [f"=== {title} ===", metric_table(summaries), ""]
    d = deltas(summaries)
    out.append("增益：")
    for k, v in d.items():
        out.append(f"  {k}: {'未测' if v is None else f'{v:+.2f} pt'}")
    out.append("")
    for cfg, m in metas.items():
        out.append(f"[{cfg}] wall={m['wall_clock_sec']}s llm_calls={m['llm_calls']} "
                   f"model_reported={m['model_reported']} tokens={m['prompt_tokens']}+{m['completion_tokens']}")
    return "\n".join(out)


def badcase_analysis(rows_by_config: dict[str, list[CaseMetrics]], *,
                     config: str = "agent_memory", per_group: int = 2) -> str:
    """失败案例分析：按 Badcase 类型给代表案例 + 逐轮轨迹。

    为什么这比指标表值钱：指标告诉你"掉了多少"，案例告诉你"为什么掉"。
    这里的逐轮轨迹是从落盘的 trace 里读的原始事实（信号、工具返回、等级迁移），
    不是事后复述。
    """
    rows = rows_by_config.get(config) or []
    table = badcase_table(rows)
    if not table:
        return "本次运行没有产生 Badcase。"
    lines = [f"配置：{CONFIG_LABELS.get(config, config)}（共 {len(rows)} 案）", ""]
    by_id = {r.case_id: r for r in rows}
    for kind, ids in table.items():
        lines.append(f"### {kind}（{len(ids)} 例）")
        lines.append("")
        lines.append("案例：" + ", ".join(ids[:12]) + (" …" if len(ids) > 12 else ""))
        lines.append("")
        for cid in ids[:per_group]:
            m = by_id[cid]
            gold = m.gold_min_level if m.is_attack else m.gold_max_level
            lines.append(f"**{cid}**（{m.kind}，split={m.split}，gold={gold}，"
                         f"max_level={m.max_level}，首次≥L2={m.first_l2_turn}，"
                         f"transfer_turn={m.transfer_turn}，动作={m.final_action}）")
            lines.append("")
            tp = m.trace_path
            if tp and Path(tp).is_file():
                try:
                    trace = json.loads(Path(tp).read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    trace = {}
                for turn in trace.get("turns", []):
                    sigs = ", ".join(s.get("type", "") for s in turn.get("signals", []))
                    tools = "; ".join(
                        f"{tc['name']}(ok={tc['ok']}"
                        + (",idem_skip" if tc.get("idempotent_skip") else "")
                        + (f",rejected={tc['rejected_by']}" if tc.get("rejected_by") else "")
                        + ")" for tc in turn.get("tool_calls", []))
                    lines.append(
                        f"  - 第 {turn.get('turn_index')} 轮 [{turn.get('speaker')}] "
                        f"建议={turn.get('suggested_level')} → 规则={turn.get('proposed_level')} → "
                        f"最终={turn.get('final_level')} | 信号: {sigs or '无'} "
                        f"| 工具: {tools or '无'}")
                    text = (turn.get("text") or "").strip()
                    if text:
                        lines.append(f"      原文：{text[:80]}")
            else:
                lines.append(f"  （轨迹文件未落盘：{tp or '无'}）")
            lines.append("")
    return "\n".join(lines)


def badcase_report(rows_by_config: dict[str, list[CaseMetrics]]) -> str:
    lines = ["### Badcase 分类报表", ""]
    for cfg, rows in rows_by_config.items():
        table = badcase_table(rows)
        lines.append(f"**{CONFIG_LABELS.get(cfg, cfg)}**（{len(rows)} 案）")
        if not table:
            lines.append("- 无 Badcase")
        for kind, ids in table.items():
            shown = ", ".join(ids[:12]) + (" …" if len(ids) > 12 else "")
            lines.append(f"- {kind}（{len(ids)}）：{shown}")
        lines.append("")
    return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SilverGuard 评测 runner")
    p.add_argument("--configs", default="rule,single_llm,agent,agent_memory",
                   help="逗号分隔的消融配置")
    p.add_argument("--split", default="dev", choices=["dev", "heldout", "all"])
    p.add_argument("--limit", type=int, default=None, help="attack / benign 各自取前 N 条")
    p.add_argument("--attack-limit", type=int, default=None)
    p.add_argument("--benign-limit", type=int, default=None)
    p.add_argument("--kind", default=None, help="只跑某些 fraud_type / kind（逗号分隔）")
    p.add_argument("--ci-subset", action="store_true",
                   help="CI 门禁用确定性子集：按 case_id 排序后取前 N 条（A 组与数据集完全确定）")
    p.add_argument("--dataset", default=None, help="数据集目录")
    p.add_argument("--out", default=None, help="把 Markdown 报告写到该路径")
    p.add_argument("--json-out", default=None, help="把完整 JSON 结果写到该路径")
    p.add_argument("--traces-dir", default="data/runs", help="逐案轨迹落盘目录")
    p.add_argument("--no-traces", action="store_true")
    p.add_argument("--thresholds", action="store_true", help="阈值扫描模式")
    p.add_argument("--matrix", action="store_true", help="策略引擎开/关对照（SE-ASR）")
    p.add_argument("--compaction", action="store_true", help="开启上下文压缩（对照实验）")
    p.add_argument("--tool-fail", default=None, help="注入工具永久失败（逗号分隔），容错实验用")
    p.add_argument("--tiers", default="strict,balanced,lenient,recall")
    p.add_argument("--extra-json", action="append", default=None,
                   help="把额外产物（红队/加固/限制）并入报告回填，形如 redteam=data/redteam.json")
    p.add_argument("--update-report", default=None,
                   help="把结果回填进该 Markdown 报告的锚点区块（如 docs/eval-report.md）")
    p.add_argument("--baseline", default=None, help="CI 门禁：基线 JSON 路径")
    p.add_argument("--gate", action="store_true", help="按基线做门禁判定（掉线即非零退出）")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def baseline_gate(summaries: dict[str, MetricSummary], baseline: dict[str, Any], *,
                  tol_pt: float = 2.0) -> tuple[bool, list[str]]:
    """CI 门禁：主指标（`PIR` / `IR` / `FPR-L3`）相对基线掉线超过容差即 fail。"""
    msgs: list[str] = []
    ok = True
    missing_configs = set(summaries) - set(baseline.get("summaries") or {})
    if missing_configs:
        ok = False
        msgs.append(f"❌ 当前配置缺少基线: {sorted(missing_configs)}；门禁未生效")
    if baseline.get("valid") is False:
        return False, ["❌ 基线标记为无效评测"]
    for cfg, bl in (baseline.get("summaries") or {}).items():
        cur = summaries.get(cfg)
        if cur is None:
            continue
        for key, direction in (("pir", "min"), ("ir", "min"), ("fpr_l3", "max")):
            b, c = bl.get(key), getattr(cur, key, None)
            if b is None or c is None:
                ok = False
                msgs.append(f"❌ [{cfg}] {key} 缺失，不能通过门禁")
                continue
            if direction == "min":
                if c < b - tol_pt:
                    ok = False
                    msgs.append(f"❌ [{cfg}] {key} 掉线：基线 {b} → 当前 {c}（容差 {tol_pt}pt）")
                else:
                    msgs.append(f"✅ [{cfg}] {key}: 基线 {b} → 当前 {c}")
            else:
                if c > b + tol_pt:
                    ok = False
                    msgs.append(f"❌ [{cfg}] {key} 越界：基线 {b} → 当前 {c}（容差 {tol_pt}pt）")
                else:
                    msgs.append(f"✅ [{cfg}] {key}: 基线 {b} → 当前 {c}")
    if not msgs:
        ok = False
        msgs.append("⚠️ 基线文件里没有可比对的指标，门禁未生效")
    return ok, msgs


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    dataset_dir = Path(args.dataset) if args.dataset else settings.dataset_dir
    ds = load_dataset(dataset_dir)
    patterns_path = settings.policy_path.parent / "fraud_patterns.yaml"
    policy = load_policy(settings.policy_path)
    trace_dir = None if args.no_traces else Path(args.traces_dir)

    counts = ds.counts()
    env = environment_note()
    started = time.time()

    def llm_factory(config: str) -> LLMClient | None:
        if config == "rule":
            return None
        return LLMClient(api_key=settings.api_key, model=settings.model, base_url=settings.base_url,
                         temperature=settings.temperature, timeout=settings.request_timeout)

    summaries: dict[str, MetricSummary] = {}
    metas: dict[str, dict[str, Any]] = {}
    rows_by_config: dict[str, list[CaseMetrics]] = {}
    traces_by_config: dict[str, list[dict[str, Any]]] = {}
    extra: dict[str, Any] = {}

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    cases = build_cases(ds, split=args.split, limit=args.limit, kind_filter=args.kind,
                        attack_limit=args.attack_limit, benign_limit=args.benign_limit)
    if args.ci_subset:
        cases = ci_subset(cases)
    if not cases:
        print("没有可用案例（检查 --split / --limit / --dataset）")
        return 2
    attack_n = sum(1 for c in cases if c.is_attack)
    benign_n = len(cases) - attack_n
    print(f"数据集：attack {counts['attack']} / benign {counts['benign']}；"
          f"本次跑 attack {attack_n} / benign {benign_n}（split={args.split}）")
    print(f"attack sha256={ds.attack_sha256}\nbenign sha256={ds.benign_sha256}")

    for cfg in configs:
        log.info("▶ 配置 %s", cfg)
        rows, summary, traces, meta = run_config(
            config=cfg, cases=cases, settings=settings, policy=policy,
            patterns_path=patterns_path, llm_factory=llm_factory,
            compaction=args.compaction, trace_dir=trace_dir,
            tool_fail=set((args.tool_fail or "").split(",")) - {""},
        )
        rows_by_config[cfg] = rows
        traces_by_config[cfg] = traces
        summaries[cfg] = summary
        metas[cfg] = meta

    # 阈值扫描
    if args.thresholds:
        extra["thresholds"] = {}
        for tier in [t.strip() for t in args.tiers.split(",") if t.strip()]:
            # Experiment conditions must not persist into the source policy file;
            # otherwise concurrent tests/runs observe a changing configuration.
            policy.set_tier(tier, persist=False)
            log.info("▶ 阈值档 %s", tier)
            rows, summary, _traces, meta = run_config(
                config="agent_memory", cases=cases, settings=settings, policy=policy,
                patterns_path=patterns_path, llm_factory=llm_factory,
                compaction=args.compaction, trace_dir=None,
            )
            extra["thresholds"][tier] = {
                "miss_rate": None if summary.ir is None else round(100 - summary.ir, 2),
                "fpr_any": None if summary.n_benign == 0 else round(
                    sum(1 for r in rows if not r.is_attack and level_rank(r.max_level) >= 2)
                    / summary.n_benign * 100, 2),
                "fpr_l3": summary.fpr_l3, "ir": summary.ir, "pir": summary.pir,
                "summary": summary.to_dict(), "meta": meta,
            }
        policy.set_tier("balanced")

    # 策略引擎开 / 关对照
    if args.matrix:
        extra["engine_matrix"] = {}
        for label, mode in (("engine_on", "policy"), ("engine_off", "llm_only")):
            policy.set_mode(mode, persist=False)
            log.info("▶ 策略引擎 %s", label)
            rows, summary, _traces, meta = run_config(
                config="agent_memory", cases=cases, settings=settings, policy=policy,
                patterns_path=patterns_path, llm_factory=llm_factory,
                compaction=args.compaction, trace_dir=None,
            )
            se = [r for r in rows if r.is_attack and r.se_attack]
            extra["engine_matrix"][label] = {
                "se_asr": summary.se_asr, "ir": summary.ir, "pir": summary.pir,
                "se_n": len(se), "se_missed": [r.case_id for r in se if not r.intercepted],
                "policy_mode": mode, "summary": summary.to_dict(),
            }
        policy.set_mode("policy")

    report = console_report(summaries, metas, title=f"SilverGuard 评测（split={args.split}）")
    if extra.get("thresholds"):
        report += "\n\n=== 阈值扫描（漏拦率 vs 误报率）===\n| 档位 | 漏拦率 | 误报率(≥L2) | FPR-L3 | IR | PIR |\n| --- | --- | --- | --- | --- | --- |\n"
        for tier, v in extra["thresholds"].items():
            report += (f"| {tier} | {v['miss_rate']} | {v['fpr_any']} | {v['fpr_l3']} | "
                       f"{v['ir']} | {v['pir']} |\n")
    if extra.get("engine_matrix"):
        report += "\n\n=== 策略引擎开/关对照 ===\n"
        for label, v in extra["engine_matrix"].items():
            report += (f"{label}: SE-ASR={v['se_asr']} IR={v['ir']} PIR={v['pir']} "
                       f"(社工样本 n={v['se_n']})\n")
    print("\n" + report)

    invalid_cases = [r.case_id for rows in rows_by_config.values() for r in rows
                     if "evidence_extraction" in r.degraded_dims]
    payload = {
        "valid": not invalid_cases,
        "invalid_evidence_cases": invalid_cases,
        "environment": env,
        "dataset": {
            "dir": str(dataset_dir), "attack_sha256": ds.attack_sha256,
            "benign_sha256": ds.benign_sha256, "counts": counts,
            "run_split": args.split, "run_attack": attack_n, "run_benign": benign_n,
        },
        "policy": {"version": policy.version, "tier": policy.tier, "mode": policy.mode,
                   "reload_count": policy.reload_count, "load_failures": policy.load_failures},
        "configs": configs,
        "summaries": {k: v.to_dict() for k, v in summaries.items()},
        "metas": metas,
        "deltas": deltas(summaries),
        "badcases": {k: badcase_table(v) for k, v in rows_by_config.items()},
        "extra": extra,
        "wall_clock_sec": round(time.time() - started, 1),
    }

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
        print(f"\nJSON 结果 → {args.json_out}")

    if invalid_cases:
        print("评测无效：模型证据抽取失败；诊断结果已保存，禁止回填报告或通过门禁", file=sys.stderr)
        return 2

    if args.extra_json:
        for item in args.extra_json:
            key, _, path = item.partition("=")
            key, path = key.strip(), path.strip()
            if not path or not Path(path).is_file():
                print(f"跳过 --extra-json {item}（文件不存在）", file=sys.stderr)
                continue
            payload["extra"][key] = json.loads(Path(path).read_text(encoding="utf-8"))
            print(f"并入 extra.{key} ← {path}")

    if args.update_report:
        body = render_markdown(payload, rows_by_config)
        target = Path(args.update_report)
        if not target.is_file():
            print(f"报告不存在：{target}", file=sys.stderr)
        else:
            blocks = report_blocks(payload, rows_by_config, body)
            target.write_text(fill_anchors(target.read_text(encoding="utf-8"), blocks),
                              encoding="utf-8")
            print(f"\n报告已回填 → {target}（锚点：{', '.join(sorted(blocks))}）")

    if args.out:
        body = render_markdown(payload, rows_by_config)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(body, encoding="utf-8")
        print(f"\nMarkdown 报告 → {args.out}")

    if args.gate:
        if not args.baseline:
            print("--gate 需要同时给 --baseline")
            return 2
        bl = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        ok, msgs = baseline_gate(summaries, bl)
        print("\n=== CI 门禁 ===")
        for m in msgs:
            print(m)
        if not ok:
            print("\n门禁判定：FAIL")
            return 1
        print("\n门禁判定：PASS")
    return 0


def fill_anchors(text: str, blocks: dict[str, str]) -> str:
    """把内容插进 `<!-- KEY-START --> ... <!-- KEY-END -->` 区块。

    为什么用锚点回填而不是整份重写报告：报告里有大量**人工写的解释文字**
    （口径、取舍、已知偏差），自动生成的数字不该覆盖它们。
    """
    import re

    for key, body in blocks.items():
        pattern = re.compile(rf"(<!-- {re.escape(key)}-START -->)(.*?)(<!-- {re.escape(key)}-END -->)",
                             re.DOTALL)
        if not pattern.search(text):
            continue
        text = pattern.sub(lambda m: f"{m.group(1)}\n{body.strip()}\n{m.group(3)}", text)
    return text


def report_blocks(payload: dict[str, Any], rows_by_config: dict[str, list[CaseMetrics]],
                  rendered: str) -> dict[str, str]:
    """把一次运行结果切成报告里的几个锚点区块。"""
    ds = payload["dataset"]
    blocks: dict[str, str] = {}

    rows_md = ["| 配置 | n(attack) | n(benign) | `IR` | `PIR` | `FPR-L2` | `FPR-L3` | `FPR-L4` "
               "| `SE-ASR` | 越权率 | 工具正确率 | P95(ms) | 平均 tokens | 平均工具调用 |",
               "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for cfg in payload["configs"]:
        s = payload["summaries"][cfg]
        rows_md.append(
            f"| {CONFIG_LABELS[cfg]} | {s['n_attack']} | {s['n_benign']} | {fmt_pct(s['ir'])} "
            f"| {fmt_pct(s['pir'])} | {fmt_pct(s['fpr_l2'])} | {fmt_pct(s['fpr_l3'])} "
            f"| {fmt_pct(s['fpr_l4'])} | {fmt_pct(s['se_asr'])} | {fmt_pct(s['unauthorized_rate'])} "
            f"| {fmt_pct(s['tool_accuracy'])} | {s['p95_latency_ms']} | {s['mean_tokens']} "
            f"| {s['mean_tool_calls']} |")
    rows_md.append("")
    rows_md.append("**增益（pt）**：" + "；".join(
        f"{k} = {'未测' if v is None else f'{v:+.2f}'}" for k, v in payload["deltas"].items()))
    rows_md.append("")
    rows_md.append(f"- 运行 split=`{ds['run_split']}`，attack {ds['run_attack']} / benign {ds['run_benign']}")
    rows_md.append(f"- `attack.jsonl` sha256：`{ds['attack_sha256']}`")
    rows_md.append(f"- `benign.jsonl` sha256：`{ds['benign_sha256']}`")
    blocks["EVAL-ABLATION"] = "\n".join(rows_md)

    if payload.get("extra", {}).get("thresholds"):
        lines = ["| 阈值档 | 漏拦率 `1-IR` | 误报率(≥L2) | `FPR-L3` | `IR` | `PIR` |",
                 "| --- | --- | --- | --- | --- | --- |"]
        for tier, v in payload["extra"]["thresholds"].items():
            lines.append(f"| {tier} | {v['miss_rate']} | {v['fpr_any']} | {v['fpr_l3']} | "
                         f"{v['ir']} | {v['pir']} |")
        lines.append("")
        lines.append("读法：档位越往 `recall` 走，漏拦越少、误报越多——**这对数字就是取舍点**。")
        blocks["EVAL-THRESHOLDS"] = "\n".join(lines)

    if payload.get("extra", {}).get("engine_matrix"):
        lines = ["| 条件 | 社工样本 n | `SE-ASR` | `IR` | `PIR` |", "| --- | --- | --- | --- | --- |"]
        for label, v in payload["extra"]["engine_matrix"].items():
            lines.append(f"| {label} | {v['se_n']} | {fmt_pct(v['se_asr'])} | {fmt_pct(v['ir'])} "
                         f"| {fmt_pct(v['pir'])} |")
        lines.append("")
        off = payload["extra"]["engine_matrix"].get("engine_off", {})
        on = payload["extra"]["engine_matrix"].get("engine_on", {})
        if off.get("se_asr") is not None and on.get("se_asr") is not None:
            diff = round(off["se_asr"] - on["se_asr"], 2)
            verdict = ("**策略引擎确实起作用**" if diff >= 5
                       else "差异 < 5pt：**这张牌不能打**，只能讲架构理由并如实说明实验未能区分两种方案")
            lines.append(f"`SE-ASR` 关 → 开：{fmt_pct(off['se_asr'])} → {fmt_pct(on['se_asr'])}"
                         f"（差 {diff:+.2f} pt）→ {verdict}")
            if off.get("se_missed"):
                lines.append(f"关掉策略引擎后成功绕过的样本：{', '.join(off['se_missed'][:15])}")
        blocks["EVAL-MATRIX"] = "\n".join(lines)

    if payload.get("extra", {}).get("redteam"):
        rt = payload["extra"]["redteam"]
        lines = ["| 轮次 | 攻击集条数 | `ASR` | `SE-ASR` | 本轮新增变异 |", "| --- | --- | --- | --- | --- |"]
        for r in rt.get("rounds", []):
            lines.append(f"| {r.get('round', '')} | {r.get('n_attack')} | {r.get('ASR')} "
                         f"| {r.get('SE_ASR')} | {r.get('generated_variants', 0)} |")
        if rt.get("heldout"):
            lines += ["", "**heldout 复测（封存集，最终只运行一次）**", "",
                      "| 时点 | `IR` | `PIR` | `ASR` | `SE-ASR` |", "| --- | --- | --- | --- | --- |"]
            for k, v in rt["heldout"].items():
                lines.append(f"| {k} | {v.get('IR')} | {v.get('PIR')} | {v.get('ASR')} "
                             f"| {v.get('SE_ASR')} |")
        if rt.get("persuasion"):
            p = rt["persuasion"]
            lines += ["", f"**老人模拟器（n={p.get('n')}）**：劝说成功率 "
                          f"{p.get('persuasion_success_rate')}%；无干预基线放弃率 "
                          f"{p.get('baseline_giveup_rate')}%（⚠️ LLM 模拟，非真人）"]
        blocks["EVAL-REDTEAM"] = "\n".join(lines)

    m = payload["metas"].get("agent_memory") or {}
    s = payload["summaries"].get("agent_memory") or {}
    if m:
        blocks["EVAL-SYSTEM"] = "\n".join([
            f"- 挂钟耗时：{m.get('wall_clock_sec')}s；LLM 调用 {m.get('llm_calls')} 次"
            f"（进程内缓存命中 {m.get('llm_cache_hits', 0)}）",
            f"- tokens（prompt + completion）：{m.get('prompt_tokens')} + {m.get('completion_tokens')}",
            f"- 单案平均：LLM {s.get('mean_llm_calls')} 次 / 工具 {s.get('mean_tool_calls')} 次 / "
            f"tokens {s.get('mean_tokens')}",
            f"- 延迟：P50 {s.get('p50_latency_ms')}ms，P95 {s.get('p95_latency_ms')}ms"
            f"（**含 LLM API 网络往返**）",
            f"- 工具层统计：`{json.dumps(m.get('tool_stats', {}), ensure_ascii=False)}`",
            f"- 模型（响应体真实名）：`{m.get('model_reported')}`",
        ])

    if payload.get("extra", {}).get("hardening"):
        hardening = payload["extra"]["hardening"]
        if isinstance(hardening, dict):
            blocks["EVAL-HARDENING"] = "```json\n" + json.dumps(
                hardening, ensure_ascii=False, indent=2) + "\n```"
        else:
            blocks["EVAL-HARDENING"] = str(hardening)

    if payload.get("extra", {}).get("badcase"):
        blocks["EVAL-BADCASE"] = payload["extra"]["badcase"]
    elif rows_by_config:
        blocks["EVAL-BADCASE"] = "\n".join([
            badcase_report(rows_by_config), "", badcase_analysis(rows_by_config),
        ])

    if payload.get("extra", {}).get("limits"):
        limits = payload["extra"]["limits"]
        blocks["EVAL-LIMITS"] = ("```json\n" + json.dumps(limits, ensure_ascii=False, indent=2)
                                  + "\n```" if isinstance(limits, dict) else str(limits))
    return blocks


def render_markdown(payload: dict[str, Any], rows_by_config: dict[str, list[CaseMetrics]]) -> str:
    """把一次运行渲染成 Markdown 报告片段（可嵌入 docs/eval-report.md）。"""
    ds = payload["dataset"]
    lines: list[str] = []
    lines.append("## 本次运行摘要\n")
    lines.append(f"- 运行时间（UTC）：{time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}")
    lines.append(f"- 数据集：`{ds['dir']}`；本次 split=`{ds['run_split']}`；"
                 f"attack {ds['run_attack']} / benign {ds['run_benign']}")
    lines.append(f"- `attack.jsonl` sha256：`{ds['attack_sha256']}`")
    lines.append(f"- `benign.jsonl` sha256：`{ds['benign_sha256']}`")
    lines.append(f"- prompt 版本：`{payload['environment']['prompt_version']}`；"
                 f"策略表版本：`{payload['policy']['version']}`（档位 `{payload['policy']['tier']}`）")
    models = {m.get("model_requested", "") + "→" + m.get("model_reported", "n/a")
              for m in payload["metas"].values()}
    lines.append(f"- 模型（请求名 → 响应返回的真实模型名字段）：{'；'.join(sorted(models))}")
    lines.append(f"- 全流程挂钟耗时：{payload['wall_clock_sec']}s\n")

    lines.append("### 指标汇总\n")
    lines.append("| 配置 | n(attack) | n(benign) | `IR` | `PIR` | `FPR-L2` | `FPR-L3` | `FPR-L4` "
                 "| `SE-ASR` | 越权率 | 工具调用正确率 | P95延迟(ms) | 平均tokens |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for cfg in payload["configs"]:
        s = payload["summaries"][cfg]
        lines.append(
            f"| {CONFIG_LABELS[cfg]} | {s['n_attack']} | {s['n_benign']} | {fmt_pct(s['ir'])} "
            f"| {fmt_pct(s['pir'])} | {fmt_pct(s['fpr_l2'])} | {fmt_pct(s['fpr_l3'])} "
            f"| {fmt_pct(s['fpr_l4'])} | {fmt_pct(s['se_asr'])} | {fmt_pct(s['unauthorized_rate'])} "
            f"| {fmt_pct(s['tool_accuracy'])} | {s['p95_latency_ms']} | {s['mean_tokens']} |")

    lines.append("\n### 分层拦截率（按手法类别，配置 D）\n")
    d = payload["summaries"].get("agent_memory") or {}
    if d.get("per_type_ir"):
        lines.append("| 手法 | n | `IR` |")
        lines.append("| --- | --- | --- |")
        from .dataset import FRAUD_TYPE_CN

        for kind, n in (d.get("per_type_n") or {}).items():
            lines.append(f"| {FRAUD_TYPE_CN.get(kind, kind)} | {n} | {fmt_pct(d['per_type_ir'].get(kind))} |")

    lines.append("\n### 四组消融增益\n")
    lines.append("| 对比 | 值（pt） |")
    lines.append("| --- | --- |")
    for k, v in payload["deltas"].items():
        lines.append(f"| {k} | {'未测' if v is None else f'{v:+.2f}'} |")

    lines.append("\n### 成本与延迟（配置 D）\n")
    m = payload["metas"].get("agent_memory") or {}
    if m:
        lines.append(f"- 挂钟耗时：{m['wall_clock_sec']}s；LLM 调用 {m['llm_calls']} 次"
                     f"（缓存命中 {m.get('llm_cache_hits', 0)}）")
        lines.append(f"- tokens（prompt + completion）：{m['prompt_tokens']} + {m['completion_tokens']}")
        s = payload["summaries"]["agent_memory"]
        lines.append(f"- 单案平均：LLM 调用 {s['mean_llm_calls']} 次 / 工具调用 {s['mean_tool_calls']} 次 "
                     f"/ tokens {s['mean_tokens']}；P50 {s['p50_latency_ms']}ms，P95 {s['p95_latency_ms']}ms")
        lines.append(f"- 工具层统计：`{json.dumps(m.get('tool_stats', {}), ensure_ascii=False)}`")

    if payload.get("extra", {}).get("thresholds"):
        lines.append("\n### 阈值扫描（漏拦率 vs 误报率）\n")
        lines.append("| 档位 | 漏拦率 `1-IR` | 误报率(≥L2) | `FPR-L3` | `IR` | `PIR` |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for tier, v in payload["extra"]["thresholds"].items():
            lines.append(f"| {tier} | {v['miss_rate']} | {v['fpr_any']} | {v['fpr_l3']} | "
                         f"{v['ir']} | {v['pir']} |")

    if payload.get("extra", {}).get("engine_matrix"):
        lines.append("\n### 策略引擎开 / 关对照（本项目的架构判断验证）\n")
        lines.append("| 条件 | 社工样本 n | `SE-ASR` | `IR` | `PIR` |")
        lines.append("| --- | --- | --- | --- | --- |")
        for label, v in payload["extra"]["engine_matrix"].items():
            lines.append(f"| {label} | {v['se_n']} | {fmt_pct(v['se_asr'])} | {fmt_pct(v['ir'])} "
                         f"| {fmt_pct(v['pir'])} |")

    lines.append("\n### 逐案细表（配置 D）\n")
    rows = rows_by_config.get("agent_memory") or []
    if rows:
        lines.append("| case_id | 类型 | split | max_level | gold | 首次≥L2 | transfer_turn | 动作 | 延迟ms |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in rows:
            gold = r.gold_min_level if r.is_attack else r.gold_max_level
            lines.append(f"| {r.case_id} | {r.kind} | {r.split} | {r.max_level} | {gold} | "
                         f"{r.first_l2_turn} | {r.transfer_turn} | {r.final_action} | {r.latency_ms} |")

    lines.append("\n### Badcase 分类报表\n")
    lines.append(badcase_report(rows_by_config))
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
