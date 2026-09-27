"""上下文预算管理：长轨迹分段摘要 + 状态外置 + 工具结果裁剪。

对应设计文档 §15.2 第 5 条。口径纪律：**必须同时给出省下的 tokens 与指标代价**，
只写"省了多少"不写"掉了多少指标"会被当成作弊，所以这里把三件事都算出来：

- ``compact_turns``：把已决策过的历史轮次压成摘要（原始对话不进上下文）；
- ``signal_state_block``：结构化状态外置（信号是结论，**不参与压缩**，逐条保留）；
- ``trim_tool_result``：工具返回只留结构化字段 + 一句话摘要，原文进 runs 不进口径。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .models import Signal


def rough_tokens(text: str) -> int:
    """粗估 token 数。

    刻意不引入 tiktoken：评测报告里凡是用到它的地方都注明"粗估"，
    真实 token 数一律以 API 返回的 usage 为准（那才是计费口径）。
    中文 ≈ 1 字 1 token，英文 ≈ 4 字符 1 token，取折中。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return int(cjk + other / 3.2) + 1


def trim_tool_result(result: dict[str, Any], *, keep: tuple[str, ...] | None = None) -> dict[str, Any]:
    """工具结果裁剪：只保留结构化字段 + 一句摘要。"""
    if not isinstance(result, dict):
        return {"summary": str(result)[:160]}
    keep_keys = keep or (
        "summary", "pattern_hit", "score", "is_whitelist", "first_seen", "in_memory",
        "sent", "idempotent_skip", "recorded", "rule_version", "targets",
    )
    trimmed = {k: v for k, v in result.items() if k in keep_keys}
    if "hits" in result and isinstance(result["hits"], list):
        trimmed["hit_ids"] = [h.get("id") for h in result["hits"][:6]]
    return trimmed


@dataclass
class ContextBudget:
    """会话级上下文预算与压缩统计。"""

    summarize_every: int = 6          # 每 N 轮压一次
    max_history_turns: int = 4        # 未压缩时最多回放的原始轮数
    enabled: bool = False             # 默认关闭；开关对照实验时打开
    raw_tokens_seen: int = 0
    compacted_tokens: int = 0
    summaries: list[str] = field(default_factory=list)
    tool_raw_tokens: int = 0
    tool_trimmed_tokens: int = 0
    #: 不压缩时会进入上下文的对话块 tokens（与 compacted_tokens 构成"省了多少"的分母）
    full_block_tokens: int = 0
    #: 压缩后实际进入上下文的对话块 tokens
    sent_block_tokens: int = 0

    def observe_raw(self, text: str) -> None:
        """记录"不压缩时会进入上下文的原始 tokens"。Agent 每轮调用一次。"""
        self.raw_tokens_seen += rough_tokens(text)

    def measure(self, turns: list[dict[str, Any]]) -> tuple[int, int]:
        """对照实验用：返回 (不压缩的 tokens, 压缩后的 tokens)。

        走的是同一条 ``compact_turns`` 代码路径，避免"对照跑的是另一套逻辑"。
        """
        block = self.compact_turns(turns, len(turns))
        compressed = sum(rough_tokens(t.get("text", "")) for t in block)
        raw = sum(rough_tokens(t.get("text", "")) for t in turns)
        return raw, compressed

    def observe_tool(self, raw: dict[str, Any], trimmed: dict[str, Any]) -> None:
        self.tool_raw_tokens += rough_tokens(json.dumps(raw, ensure_ascii=False))
        self.tool_trimmed_tokens += rough_tokens(json.dumps(trimmed, ensure_ascii=False))

    @property
    def token_saving_pct(self) -> float:
        """对话块 + 工具结果的综合节省比例（口径：只算真正进入上下文的那些 token）。"""
        total = self.full_block_tokens + self.tool_raw_tokens
        if total <= 0:
            return 0.0
        kept = self.sent_block_tokens + self.tool_trimmed_tokens
        return max(0.0, (total - kept) / total * 100.0)

    def compact_turns(self, turns: list[dict[str, Any]], decided_upto: int) -> list[dict[str, Any]]:
        """返回"这一轮要喂给模型的对话块"，并同时记账。

        两种模式的**记账口径完全相同**（分母都是"若什么都不压会进去多少"），
        所以开/关两次运行的差异只来自压缩本身，而不是来自两套统计代码。
        """
        if not self.enabled:
            block = list(turns[:decided_upto])
        else:
            older = turns[:max(0, decided_upto - self.max_history_turns)]
            recent = turns[max(0, decided_upto - self.max_history_turns):decided_upto]
            block = []
            if older and len(older) >= self.summarize_every:
                # 摘要只保留"说话人 + 该轮主干"，且**风险信号另有结构化外置**
                # （见 signal_state_block），所以摘要不承担保真责任——
                # 这是它敢压这么狠的前提。
                summary = self._summarize(older)
                self.summaries.append(summary)
                block.append({"role": "history_summary", "text": summary})
            block.extend(recent)

        full = sum(rough_tokens(t.get("text", "")) for t in turns[:decided_upto])
        sent = sum(rough_tokens(t.get("text", "")) for t in block)
        self.full_block_tokens += full
        self.sent_block_tokens += sent
        self.compacted_tokens += sent
        return block

    @staticmethod
    def _summarize(turns: list[dict[str, Any]]) -> str:
        """确定性摘要（抽取式）：保留说话人 + 该轮主干，不引入第二次 LLM 调用。

        为什么用抽取式而不是 LLM 摘要：压缩本身不该再引入一次非确定性；
        风险信号另有结构化外置（见 ``signal_state_block``），不依赖摘要保真。
        """
        bits = []
        for i, t in enumerate(turns, start=1):
            text = (t.get("text") or "").strip().replace("\n", " ")
            bits.append(f"{t.get('role', '?')}#{i}:{text[:48]}")
        return "前序轮次摘要 → " + " | ".join(bits)

    def reset(self) -> None:
        self.raw_tokens_seen = 0
        self.compacted_tokens = 0
        self.tool_raw_tokens = 0
        self.tool_trimmed_tokens = 0
        self.summaries.clear()
        self.full_block_tokens = 0
        self.sent_block_tokens = 0


def signal_state_block(signals: list[Signal], *, current_level: str, actions: list[str],
                       confirmed: list[str]) -> str:
    """结构化状态外置：不靠"把整段对话塞回 prompt"来记住已发生的事。"""
    by_type: dict[str, Signal] = {}
    for s in signals:
        if s.type not in by_type or s.confidence > by_type[s.type].confidence:
            by_type[s.type] = s
    lines = [f"当前等级：{current_level}"]
    if actions:
        lines.append("已执行动作：" + ", ".join(actions))
    if confirmed:
        lines.append("已确认事实：" + ", ".join(confirmed))
    if by_type:
        lines.append("已确认信号（逐条保留，不经摘要）：")
        for s in by_type.values():
            lines.append(f"  - {s.type}（conf={s.confidence:.2f}）原文：{s.quote[:80]}")
    else:
        lines.append("已确认信号：无")
    return "\n".join(lines)
