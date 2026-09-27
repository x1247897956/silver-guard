"""核心数据模型：信号、证据、决策、轨迹。

全部用 dataclass 而不是 pydantic，是为了让核心链路（策略引擎 / 状态机 /
工具层）零第三方依赖——这样单元测试与确定性回放不依赖网络和重依赖。
FastAPI 层再用 pydantic 做请求校验。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

LEVELS = ("L0", "L1", "L2", "L3", "L4")
LEVEL_RANK = {lv: i for i, lv in enumerate(LEVELS)}

#: 五类风险信号（设计文档 §5.1）
SIGNAL_TYPES = (
    "identity_doubt",     # 身份可疑
    "urgency",            # 紧迫性施压
    "money_action",       # 资金动作
    "secrecy",            # 保密要求
    "channel_anomaly",    # 渠道异常
)
#: 仅作上下文/触发器的附加信号（不计入信号数阈值）
CONTEXT_SIGNAL_TYPES = ("victim_compromise",)  # 老人已被说服

ALL_SIGNAL_TYPES = SIGNAL_TYPES + CONTEXT_SIGNAL_TYPES

#: 兼容 LLM 常见的同义输出
SIGNAL_ALIASES = {
    "authority_claim": "identity_doubt",
    "identity": "identity_doubt",
    "impersonation": "identity_doubt",
    "pressure": "urgency",
    "time_pressure": "urgency",
    "urgency_pressure": "urgency",
    "money": "money_action",
    "fund_action": "money_action",
    "transfer": "money_action",
    "secrecy_request": "secrecy",
    "confidentiality": "secrecy",
    "channel": "channel_anomaly",
    "unknown_channel": "channel_anomaly",
    "compromise": "victim_compromise",
    "elder_compliance": "victim_compromise",
}


def level_rank(level: str) -> int:
    return LEVEL_RANK.get(level, 0)


def level_at_least(level: str, floor: str) -> bool:
    return level_rank(level) >= level_rank(floor)


@dataclass
class Signal:
    """一条风险信号，必须带原文片段（可归因是硬要求）。"""

    type: str
    quote: str = ""
    speaker: str = ""
    turn_index: int | None = None
    confidence: float = 0.0
    note: str = ""

    @staticmethod
    def from_raw(raw: dict[str, Any]) -> "Signal | None":
        t = str(raw.get("type", raw.get("signal", ""))).strip()
        t = SIGNAL_ALIASES.get(t, t)
        if t not in ALL_SIGNAL_TYPES:
            return None
        conf = raw.get("confidence", 0.0)
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.0
        ti = raw.get("turn_index", raw.get("turn"))
        try:
            ti = int(ti) if ti is not None else None
        except (TypeError, ValueError):
            ti = None
        return Signal(
            type=t,
            quote=str(raw.get("quote", raw.get("evidence", "")))[:400],
            speaker=str(raw.get("speaker", "")),
            turn_index=ti,
            confidence=max(0.0, min(1.0, conf)),
            note=str(raw.get("note", ""))[:200],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolCall:
    """一次工具调用。参数、返回值、耗时、是否降级全部落库。

    ``return_raw`` 用于确定性回放：回放时不再真调工具，而是喂回当时的值。
    """

    name: str
    args: dict[str, Any]
    ok: bool
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    latency_ms: int = 0
    attempts: int = 1
    degraded: bool = False
    degraded_reason: str = ""
    idempotent_skip: bool = False
    rejected_by: str = ""       # schema / whitelist / privilege
    turn_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TurnRecord:
    """逐轮轨迹：这是"确定性回放"的最小单元。"""

    turn_index: int
    speaker: str
    text: str
    signals: list[Signal] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    suggested_level: str = "L0"
    proposed_level: str = "L0"
    final_level: str = "L0"
    action: str = "none"
    suppressed_action: str = ""
    degraded_dims: list[str] = field(default_factory=list)
    latency_ms: int = 0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    context_tokens: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["signals"] = [s.to_dict() for s in self.signals]
        d["tool_calls"] = [t.to_dict() for t in self.tool_calls]
        return d


@dataclass
class Assessment:
    """一次完整评估（一条轨迹的最终结果 + 全轨迹）。"""

    case_id: str
    elder_id: str
    config: str
    max_level: str = "L0"
    first_l2_turn: int | None = None
    final_action: str = "none"
    proposed_max_level: str = "L0"
    suggested_max_level: str = "L0"
    turns: list[TurnRecord] = field(default_factory=list)
    signals: list[Signal] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    policy_version: str = ""
    prompt_version: str = ""
    model: str = ""
    report_model: str = ""
    latency_ms: int = 0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: int = 0
    degraded_dims: list[str] = field(default_factory=list)
    action_log: list[dict[str, Any]] = field(default_factory=list)
    unauthorized_actions: int = 0
    suppressed_actions: int = 0
    memory_context_used: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["turns"] = [t.to_dict() for t in self.turns]
        d["signals"] = [s.to_dict() for s in self.signals]
        d["total_tokens"] = self.total_tokens
        return d

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def level_timeline(self) -> list[str]:
        return [t.final_level for t in self.turns]
