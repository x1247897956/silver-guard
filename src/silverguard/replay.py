"""轨迹录制与确定性回放。

问题（设计文档 §15.2 第 3 条）：LLM 不确定 → 传统单元测试失效 →
指标一波动，你分不清是"模型变了""prompt 变了"还是"策略变了"。

做法：

| 环节 | 机制 |
| --- | --- |
| 录制 | 每案落完整轨迹：逐轮输入 / 证据信号 / 工具调用（名 + 参数 + 返回）/ 建议等级 / 最终等级 / 动作 + **prompt / 模型 / 策略三版本号** |
| 回放 | 固定三版本号，用录制时的**模型输出**喂回（不再真调模型），工具不参与判定 |
| 判定 | **回放一致率** = 重放结果与录制结果一致的比例；不一致的案例单独列出 |
| 断点 | 可从第 N 轮起回放（调试"改一版策略看看第 3 轮会怎样"的基础） |

⚠️ 诚实的边界：本实现回放的是**证据抽取结果**（模型输出的结构化证据），
不是逐字节重放模型请求。因此它消除的是"工具与策略侧的不确定"，
**不消除模型本身的漂移**——所以报告里同时给出"不一致案例的归因"。
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent import CaseInput, GuardianAgent
from .config import REPO_ROOT, Settings, get_settings
from .memory import MemoryStore, seed_demo_profile
from .models import ToolCall
from .policy import load_policy
from .tools import ToolRuntime

log = logging.getLogger("silverguard.replay")


@dataclass
class ReplayOutcome:
    case_id: str
    recorded: dict[str, Any]
    replayed: dict[str, Any]

    @property
    def consistent(self) -> bool:
        return (self.replayed["max_level"] == self.recorded["max_level"]
                and self.replayed["first_l2_turn"] == self.recorded["first_l2_turn"]
                and self.replayed["action"] == self.recorded["action"])

    def to_dict(self) -> dict[str, Any]:
        return {"case_id": self.case_id, "recorded": self.recorded,
                "replayed": self.replayed, "consistent": self.consistent}


def load_trace(path: str | Path) -> dict[str, Any]:
    """轨迹文件 = Assessment.to_dict() + 评测 runner 附加的元数据。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def replay_trace(trace: dict[str, Any], *, settings: Settings, policy, patterns_path: Path,
                 config: str | None = None, upto_turn: int | None = None) -> dict[str, Any]:
    """回放单条轨迹。`upto_turn` 用于断点回放（只回放前 N 轮）。"""
    config = config or trace.get("config", "agent_memory")
    case_raw = trace.get("case") or {}
    turns = case_raw.get("turns") or [t.get("text") for t in trace.get("turns", [])]
    if not turns:
        raise ValueError(f"{trace.get('case_id')}: 轨迹里没有可回放的对话轮次")
    if upto_turn is not None:
        turns = turns[:upto_turn]
    case = CaseInput(
        case_id=trace.get("case_id", "replay"), turns=turns,
        elder_id=trace.get("elder_id", "elder-0001"), kind=case_raw.get("kind", "replay"),
        split=case_raw.get("split", "dev"), is_attack=case_raw.get("is_attack", True),
        transfer_turn=case_raw.get("transfer_turn"), se_attack=bool(case_raw.get("se_attack")),
        gold_min_level=case_raw.get("gold_min_level", "L2"),
    )
    store = MemoryStore(":memory:")
    seed_demo_profile(store)
    agent = GuardianAgent(settings=settings, store=store, policy=policy,
                          patterns_path=patterns_path, llm=None, config=config,
                          tool_runtime=ToolRuntime(store=store),
                          replay_cache=dict(trace.get("llm_cache") or {}))
    recorded_calls = [copy.deepcopy(call) for turn in trace.get("turns", [])
                      if upto_turn is None or turn.get("turn_index", 0) <= upto_turn
                      for call in turn.get("tool_calls", [])]
    pending = iter(recorded_calls)

    def recorded_call(name, args, **kwargs):
        raw = next(pending, None)
        if raw is None or raw["name"] != name or raw["args"] != args:
            raise ValueError(f"回放工具调用与录制不一致: {name}")
        return ToolCall(**raw)

    agent.registry.call = recorded_call
    agent.replaying = True
    try:
        a = agent.assess(case)
        if next(pending, None) is not None:
            raise ValueError("回放未消费全部录制工具调用")
    finally:
        store.close()
    return {"max_level": a.max_level, "first_l2_turn": a.first_l2_turn,
            "action": a.final_action, "timeline": a.level_timeline(),
            "policy_version": a.policy_version, "prompt_version": a.prompt_version}


def replay_dir(traces_dir: str | Path, *, settings: Settings | None = None,
               config: str = "agent_memory", limit: int | None = None,
               upto_turn: int | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    policy = load_policy(settings.policy_path)
    patterns_path = settings.policy_path.parent / "fraud_patterns.yaml"
    files = sorted(Path(traces_dir).glob(f"{config}__*.json"))
    # Legacy traces without cached model outputs cannot be replayed deterministically.
    # Exclude them from the denominator and report the count explicitly.
    stale_files: list[Path] = []
    usable_files: list[Path] = []
    for path in files:
        trace = load_trace(path)
        if not isinstance(trace.get("llm_cache"), dict) or not trace["llm_cache"]:
            stale_files.append(path)
        else:
            usable_files.append(path)
    files = usable_files
    if limit:
        files = files[:limit]
    outcomes: list[ReplayOutcome] = []
    mismatches: list[str] = []
    version_missing: list[str] = []
    for path in files:
        trace = load_trace(path)
        recorded = {
            "max_level": trace.get("max_level"), "first_l2_turn": trace.get("first_l2_turn"),
            "action": trace.get("final_action"),
        }
        if not all([trace.get("prompt_version"), trace.get("policy_version"), trace.get("model")]):
            version_missing.append(trace.get("case_id", path.name))
        try:
            replayed = replay_trace(trace, settings=settings, policy=policy,
                                    patterns_path=patterns_path, config=config,
                                    upto_turn=upto_turn)
        except Exception as exc:  # noqa: BLE001
            mismatches.append(f"{trace.get('case_id')}: 回放异常 {exc}")
            continue
        outcome = ReplayOutcome(case_id=trace.get("case_id", path.name), recorded=recorded,
                                replayed=replayed)
        outcomes.append(outcome)
        if not outcome.consistent:
            mismatches.append(
                f"{outcome.case_id}: 录制 {recorded['max_level']}/{recorded['first_l2_turn']}"
                f"/{recorded['action']} → 回放 {replayed['max_level']}/{replayed['first_l2_turn']}"
                f"/{replayed['action']}")
    n = len(files)
    rate = None if n == 0 else round(sum(1 for o in outcomes if o.consistent) / n * 100, 2)
    return {
        "replayed": n, "consistent": sum(1 for o in outcomes if o.consistent),
        "skipped_legacy_traces": len(stale_files),
        "replay_consistency_rate": rate, "mismatches": mismatches[:20],
        "trace_missing_versions": version_missing[:20],
        "policy_version": policy.version, "config": config, "upto_turn": upto_turn,
        "outcomes": [o.to_dict() for o in outcomes],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="轨迹确定性回放")
    p.add_argument("--traces", default=str(REPO_ROOT / "data" / "runs"))
    p.add_argument("--config", default="agent_memory")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--upto-turn", type=int, default=None, help="断点回放：只回放前 N 轮")
    p.add_argument("--json-out", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    result = replay_dir(args.traces, config=args.config, limit=args.limit, upto_turn=args.upto_turn)
    print(json.dumps({k: v for k, v in result.items() if k != "outcomes"},
                     ensure_ascii=False, indent=2))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
        print(f"JSON → {args.json_out}")
    return 0 if (result["replay_consistency_rate"] or 0) >= 95 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
