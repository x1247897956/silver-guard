"""评测指标：全部**确定性计算**，不经 LLM 判分。

口径（对应设计文档 §6.1 与采集表 §3）：

| 指标 | 公式 |
| --- | --- |
| `IR`  拦截率 | `#{a : max_level >= L2} / #Attack` |
| `PIR` 转账前拦截率 | `#{a : 首次 >=L2 的轮次 < transfer_turn} / #{a : transfer_turn != null}` |
| `FPR-Lx` | `#{b : max_level == Lx} / #Benign` |
| `ASR`  | `1 - IR` |
| `SE-ASR` | 只在 `se_attack == true` 子集上算 `1 - IR` |
| 越权率 | `#{未授权动作} / #{总动作}` |
| 过早打扰率 | `#{Benign 中 max_level >= L3 且不含资金动作} / #Benign` |
| 工具调用正确率 | 对 `gold_tools` 逐条核对"该调的调了且没报错" |

**为什么指标要分级而不是一个 F1**：漏拦的代价是钱，误报的代价是打扰，
两者不可通约；压成一个 F1 就把这个权衡藏起来了。
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any

from .models import Assessment, level_rank

#: 各消融配置"应当调用"的工具集（用于工具调用正确率的分母裁剪）
CONFIG_TOOL_SCOPE = {
    "rule": {"check_contact", "check_fraud_pattern"},
    "single_llm": set(),
    "agent": {"check_contact", "check_fraud_pattern", "notify_family", "record_case"},
    "agent_memory": {"check_contact", "check_fraud_pattern", "get_elder_profile",
                     "notify_family", "record_case"},
}


@dataclass
class CaseMetrics:
    case_id: str
    config: str
    split: str
    is_attack: bool
    kind: str
    max_level: str
    first_l2_turn: int | None
    transfer_turn: int | None
    se_attack: bool
    hard_negative: bool
    gold_min_level: str
    gold_max_level: str
    gold_signals: list[str]
    gold_tools: list[str]
    called_tools: list[str]
    tool_errors: int
    unauthorized_actions: int
    suppressed_actions: int
    latency_ms: int
    llm_calls: int
    tool_calls: int
    prompt_tokens: int
    completion_tokens: int
    final_action: str
    degraded_dims: list[str] = field(default_factory=list)
    tool_correct: bool = False
    tool_missing: list[str] = field(default_factory=list)
    tool_extra: list[str] = field(default_factory=list)
    tool_expected_scope: list[str] = field(default_factory=list)
    trace_path: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def intercepted(self) -> bool:
        return level_rank(self.max_level) >= 2

    @property
    def pir_hit(self) -> bool:
        return (self.first_l2_turn is not None and self.transfer_turn is not None
                and self.first_l2_turn < self.transfer_turn)

    @property
    def over_intervention(self) -> bool:
        """过早打扰：正常对话里触发 L3/L4（惊动家属 / 误阻断）。"""
        return (not self.is_attack) and level_rank(self.max_level) >= 3

    def badcase_type(self) -> str | None:
        if self.is_attack and not self.intercepted:
            return "漏拦"
        if self.is_attack and self.transfer_turn is not None and not self.pir_hit:
            return "拦截过晚（资金动作之后才拦）"
        if (not self.is_attack) and level_rank(self.max_level) > level_rank(self.gold_max_level):
            if level_rank(self.max_level) >= 4:
                return "误报-误阻断(L4)"
            if level_rank(self.max_level) >= 3:
                return "误报-惊动家属(L3)"
            return "误报-打扰老人(L2)"
        if self.unauthorized_actions:
            return "越权动作"
        if self.degraded_dims:
            return "工具失败降级"
        return None

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d.update({
            "intercepted": self.intercepted, "pir_hit": self.pir_hit,
            "over_intervention": self.over_intervention, "badcase_type": self.badcase_type(),
            "total_tokens": self.total_tokens,
        })
        return d


def metrics_from_assessment(a: Assessment, case: Any, *, trace_path: str = "") -> CaseMetrics:
    called = [tc.name for turn in a.turns for tc in turn.tool_calls]
    errors = sum(1 for turn in a.turns for tc in turn.tool_calls if not tc.ok)
    scope = CONFIG_TOOL_SCOPE.get(a.config, set())
    gold = [t for t in (case.gold_tools or []) if t in scope]
    missing = [t for t in gold if t not in called]
    extra = sorted({t for t in called} - scope)
    return CaseMetrics(
        case_id=case.case_id, config=a.config, split=case.split, is_attack=case.is_attack,
        kind=case.kind, max_level=a.max_level, first_l2_turn=a.first_l2_turn,
        transfer_turn=case.transfer_turn, se_attack=case.se_attack,
        hard_negative=case.hard_negative, gold_min_level=case.gold_min_level,
        gold_max_level=case.gold_max_level, gold_signals=list(case.gold_signals or []),
        gold_tools=list(case.gold_tools or []), called_tools=called, tool_errors=errors,
        unauthorized_actions=a.unauthorized_actions, suppressed_actions=a.suppressed_actions,
        latency_ms=a.latency_ms, llm_calls=a.llm_calls, tool_calls=a.tool_calls,
        prompt_tokens=a.prompt_tokens, completion_tokens=a.completion_tokens,
        final_action=a.final_action, degraded_dims=list(a.degraded_dims),
        tool_correct=(not missing and errors == 0),
        tool_missing=missing, tool_extra=extra, tool_expected_scope=sorted(scope),
        trace_path=trace_path,
    )


def pct(num: float, den: float) -> float | None:
    return None if den == 0 else round(num / den * 100.0, 2)


def percentile(values: list[int], q: float) -> int | None:
    """最近秩法（nearest-rank）：P95 = 排序后第 ceil(0.95·n) 个数。

    选它而不是线性插值，是因为"第 95 百分位延迟"在工程上就是"95% 的请求
    都快于这个值"，取真实样本更直观，也不引入插值出的假数。
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[idx]


@dataclass
class MetricSummary:
    config: str
    n_attack: int = 0
    n_benign: int = 0
    n_attack_all: int = 0
    n_benign_all: int = 0
    ir: float | None = None
    pir: float | None = None
    fpr_l2: float | None = None
    fpr_l3: float | None = None
    fpr_l4: float | None = None
    fpr_hard: float | None = None
    asr: float | None = None
    se_asr: float | None = None
    se_n: int = 0
    over_intervention: float | None = None
    unauthorized_rate: float | None = None
    unauthorized_count: int = 0
    actions_total: int = 0
    tool_accuracy: float | None = None
    per_type_ir: dict[str, float | None] = field(default_factory=dict)
    per_type_n: dict[str, int] = field(default_factory=dict)
    p50_latency_ms: int | None = None
    p95_latency_ms: int | None = None
    mean_llm_calls: float | None = None
    mean_tool_calls: float | None = None
    mean_prompt_tokens: float | None = None
    mean_completion_tokens: float | None = None
    mean_tokens: float | None = None
    failures: int = 0
    degraded_cases: int = 0
    suppressed_actions: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def summarize(config: str, rows: list[CaseMetrics]) -> MetricSummary:
    attacks = [r for r in rows if r.is_attack]
    benign = [r for r in rows if not r.is_attack]
    s = MetricSummary(config=config, n_attack=len(attacks), n_benign=len(benign))

    s.ir = pct(sum(1 for r in attacks if r.intercepted), len(attacks))
    pir_den = [r for r in attacks if r.transfer_turn is not None]
    s.pir = pct(sum(1 for r in pir_den if r.pir_hit), len(pir_den))
    s.asr = None if s.ir is None else round(100.0 - s.ir, 2)

    s.fpr_l2 = pct(sum(1 for r in benign if level_rank(r.max_level) == 2), len(benign))
    s.fpr_l3 = pct(sum(1 for r in benign if level_rank(r.max_level) == 3), len(benign))
    s.fpr_l4 = pct(sum(1 for r in benign if level_rank(r.max_level) >= 4), len(benign))
    hard = [r for r in benign if r.hard_negative]
    s.fpr_hard = pct(sum(1 for r in hard if level_rank(r.max_level) >= 2), len(hard))
    s.over_intervention = pct(sum(1 for r in benign if r.over_intervention), len(benign))

    se = [r for r in attacks if r.se_attack]
    s.se_n = len(se)
    s.se_asr = pct(sum(1 for r in se if not r.intercepted), len(se)) if se else None

    s.actions_total = sum(1 for r in rows if r.final_action not in ("none", ""))
    s.unauthorized_count = sum(r.unauthorized_actions for r in rows)
    s.unauthorized_rate = pct(s.unauthorized_count,
                              s.actions_total + s.unauthorized_count) if (s.actions_total + s.unauthorized_count) else 0.0
    s.suppressed_actions = sum(r.suppressed_actions for r in rows)

    scoped = [r for r in rows if r.tool_expected_scope]
    s.tool_accuracy = pct(sum(1 for r in scoped if r.tool_correct), len(scoped))

    for r in attacks:
        s.per_type_n[r.kind] = s.per_type_n.get(r.kind, 0) + 1
    for kind in s.per_type_n:
        subset = [r for r in attacks if r.kind == kind]
        s.per_type_ir[kind] = pct(sum(1 for r in subset if r.intercepted), len(subset))

    lat = [r.latency_ms for r in rows if r.latency_ms]
    s.p50_latency_ms = percentile(lat, 0.50)
    s.p95_latency_ms = percentile(lat, 0.95)
    if rows:
        s.mean_llm_calls = round(statistics.fmean(r.llm_calls for r in rows), 2)
        s.mean_tool_calls = round(statistics.fmean(r.tool_calls for r in rows), 2)
        s.mean_prompt_tokens = round(statistics.fmean(r.prompt_tokens for r in rows), 1)
        s.mean_completion_tokens = round(statistics.fmean(r.completion_tokens for r in rows), 1)
        s.mean_tokens = round(statistics.fmean(r.total_tokens for r in rows), 1)
    s.failures = sum(1 for r in rows if r.badcase_type() == "漏拦")
    s.degraded_cases = sum(1 for r in rows if r.degraded_dims)
    return s


def badcase_table(rows: list[CaseMetrics]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for r in rows:
        t = r.badcase_type()
        if t:
            out.setdefault(t, []).append(r.case_id)
    return {k: sorted(v) for k, v in sorted(out.items())}


def compare(summaries: dict[str, MetricSummary]) -> list[dict[str, Any]]:
    """四组消融对照（含单调性与增益）。"""
    order = ["rule", "single_llm", "agent", "agent_memory"]
    rows = []
    for cfg in order:
        if cfg not in summaries:
            continue
        s = summaries[cfg]
        rows.append({
            "config": cfg, "ir": s.ir, "pir": s.pir, "fpr_l2": s.fpr_l2, "fpr_l3": s.fpr_l3,
            "fpr_l4": s.fpr_l4, "se_asr": s.se_asr, "tool_accuracy": s.tool_accuracy,
            "p95_latency_ms": s.p95_latency_ms, "mean_tokens": s.mean_tokens,
            "mean_tool_calls": s.mean_tool_calls, "mean_llm_calls": s.mean_llm_calls,
            "unauthorized_rate": s.unauthorized_rate,
        })
    return rows


def deltas(summaries: dict[str, MetricSummary]) -> dict[str, float | None]:
    def g(cfg: str, key: str) -> float | None:
        s = summaries.get(cfg)
        return None if s is None else getattr(s, key)

    def d(a: str, b: str, key: str) -> float | None:
        x, y = g(a, key), g(b, key)
        if x is None or y is None:
            return None
        return round(y - x, 2)

    return {
        "B_minus_A_ir": d("rule", "single_llm", "ir"),
        "C_minus_B_ir": d("single_llm", "agent", "ir"),
        "D_minus_C_ir": d("agent", "agent_memory", "ir"),
        "B_minus_A_pir": d("rule", "single_llm", "pir"),
        "C_minus_B_pir": d("single_llm", "agent", "pir"),
        "D_minus_C_pir": d("agent", "agent_memory", "pir"),
        "A_to_D_pir": d("rule", "agent_memory", "pir"),
        "A_to_D_ir": d("rule", "agent_memory", "ir"),
        "D_minus_C_fpr_l3": d("agent", "agent_memory", "fpr_l3"),
    }
