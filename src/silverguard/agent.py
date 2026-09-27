"""决策链路：证据抽取 → 工具调用 → 风险建议 → 策略引擎 → 分级干预。

四段式（设计文档 §5）：
  ① 证据抽取：LLM 从多轮对话抽 5 类信号（+ 老人配合度），每条带原文片段
  ② 工具调用：check_contact / check_fraud_pattern / get_elder_profile /
              notify_family / record_case —— 由**确定性触发条件**驱动（见 _plan_tools）
  ③ 风险建议：LLM 给建议等级（只是建议）
  ④ 策略引擎：确定性规则决定最终等级与动作；LLM 建议不可越过

消融四组（设计文档 §6.2）：
  A rule          规则基线：正则打分 → 策略表，零 LLM
  B single_llm    单次 LLM：整段轨迹一次调用，无工具、无记忆、无逐轮
  C agent         Agent：逐轮 LLM + 工具（无长期记忆）
  D agent_memory  C + 长期记忆（档案 / 白名单 / 历史事件 / 近期支出）

⚠️ 工具触发为什么是确定性的而不是"让 LLM 自己决定调哪个工具"：
   评测要能归因。让模型自由选工具会把"工具该不该被调用"和"模型今天心情如何"
   混在一起，指标一波动就没法定位。这里改成：**信号 → 工具**的确定性映射
   （表见 _plan_tools），模型只负责抽信号。这也让"工具调用正确率"可计算。
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import prompts
from .config import PROMPT_VERSION, Settings
from .context import ContextBudget, trim_tool_result
from .llm import LLMClient, LLMError
from .memory import MemoryStore, SessionState
from .models import (
    Assessment,
    Signal,
    ToolCall,
    TurnRecord,
    level_rank,
)
from .policy import PolicyContext, PolicyEngine
from .tools import (
    ToolRegistry,
    ToolRuntime,
    load_pattern_rules,
    pattern_score,
)

log = logging.getLogger("silverguard.agent")

CONFIGS = ("rule", "single_llm", "agent", "agent_memory")
CONFIG_LABELS = {
    "rule": "A 规则基线",
    "single_llm": "B 单次 LLM",
    "agent": "C Agent + 工具",
    "agent_memory": "D Agent + 工具 + 长期记忆",
}

_IDENT_RE = re.compile(
    r"(?:\+?\d[\d\-\s]{6,15}\d)"                 # 电话
    r"|(?:[\w.\-]{2,24}@[\w.\-]{2,24})"          # 邮箱
    r"|(?:(?:微信|QQ|钉钉|某平台|短视频平台|社交账号)\s*(?:号|账号)?[:：]?\s*[\w\-]{2,24})"
)
_GENERIC_IDENT = {"某平台", "社交账号", "短视频平台"}

#: 老人"已被说服"的确定性语言特征。
#: 为什么不用 LLM 判：这是策略表的输入之一，必须可复现、可审计；
#: LLM 的 victim_compromise 结论作为**补充**（两者取或）。
_COMPROMISE_PATTERNS = (
    re.compile(r"(是我自己要|我自己要|我愿意转|我这就转|我马上去转|我现在就转|马上打过去)"),
    re.compile(r"(你别管|不用你管|不要你管|别拦着我|不要拦我)"),
    re.compile(r"(我听他的|人家说得对|人家是正规的|不会有问题)"),
    re.compile(r"(别再提醒|不要提醒|不用提醒|别再烦我|取消提醒|关掉提醒)"),
)

#: 规则 ID → 信号类型。A 组（规则基线）靠它把字面命中转成信号。
RULE_SIGNAL_MAP = {
    "fp_identity_unverifiable": "identity_doubt",
    "fp_secrecy_request": "secrecy",
    "fp_urgency_pressure": "urgency",
    "fp_money_action": "money_action",
    "fp_channel_anomaly": "channel_anomaly",
    "fp_high_return_lure": "urgency",
    "fp_authorized_claim": "secrecy",
    "fp_privilege_lure": "channel_anomaly",
}


@dataclass
class CaseInput:
    """一条评测案例（attack 或 benign 的规范化形态）。"""

    case_id: str
    turns: list[dict[str, Any]]
    split: str = "dev"
    kind: str = ""                 # attack 的 fraud_type 或 benign 的 kind
    is_attack: bool = True
    transfer_turn: int | None = None
    se_attack: bool = False
    se_type: str | None = None
    gold_min_level: str = "L2"
    gold_max_level: str = "L1"
    gold_signals: list[str] = field(default_factory=list)
    gold_tools: list[str] = field(default_factory=list)
    elder_id: str = "elder-0001"
    hard_negative: bool = False
    source_note: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_attack(row: dict[str, Any], elder_id: str = "elder-0001") -> "CaseInput":
        return CaseInput(
            case_id=row["case_id"], turns=list(row.get("turns", [])), split=row.get("split", "dev"),
            kind=row.get("fraud_type", ""), is_attack=True,
            transfer_turn=row.get("transfer_turn"), se_attack=bool(row.get("se_attack")),
            se_type=row.get("se_type"), gold_min_level=row.get("gold_min_level", "L2"),
            gold_signals=list(row.get("gold_signals", []) or []),
            gold_tools=list(row.get("gold_tools", []) or []),
            elder_id=row.get("elder_id") or elder_id, source_note=row.get("source_note", ""), raw=row,
        )

    @staticmethod
    def from_benign(row: dict[str, Any], elder_id: str = "elder-0001") -> "CaseInput":
        return CaseInput(
            case_id=row["case_id"], turns=list(row.get("turns", [])), split=row.get("split", "dev"),
            kind=row.get("kind", ""), is_attack=False, transfer_turn=None,
            gold_max_level=row.get("gold_max_level", "L0"),
            gold_signals=list(row.get("gold_signals", []) or []),
            gold_tools=list(row.get("gold_tools", []) or []),
            elder_id=row.get("elder_id") or elder_id,
            hard_negative=bool(row.get("hard_negative")), source_note=row.get("source_note", ""),
            raw=row,
        )


def detect_compromise(turns: list[dict[str, Any]], upto: int | None = None) -> tuple[bool, str]:
    """确定性识别"老人已被说服"（返回是否 + 原文片段）。

    它是策略表 L4 / L3 的触发字段之一，所以不能只依赖模型判断。
    """
    chunk = turns if upto is None else turns[:upto]
    for t in reversed(chunk):
        if t.get("role") != "elder":
            continue
        text = t.get("text", "")
        for rx in _COMPROMISE_PATTERNS:
            m = rx.search(text)
            if m:
                return True, m.group(0)
    return False, ""


def extract_identifiers(turns: list[dict[str, Any]]) -> list[str]:
    """从轨迹里确定性抽取"真实出现过的标识符"。

    这是工具权限最小化的基础：模型不能自造标识，只能引用这里出现过的值。
    """
    found: list[str] = []
    for t in turns:
        for m in _IDENT_RE.finditer(t.get("text", "")):
            val = m.group(0).strip()
            val = re.sub(r"\s+", "", val)
            if not val or val in _GENERIC_IDENT:
                continue
            if val not in found:
                found.append(val)
    return found


@dataclass
class Trace:
    """一次评估的完整轨迹（可落盘、可回放）。"""

    case_id: str
    config: str
    assessment: Assessment
    llm_requests: list[dict[str, Any]] = field(default_factory=list)
    tool_returns: dict[str, dict[str, Any]] = field(default_factory=dict)


class GuardianAgent:
    def __init__(self, *, settings: Settings, store: MemoryStore, policy: PolicyEngine,
                 patterns_path: str | Path | None = None, llm: LLMClient | None = None,
                 config: str = "agent_memory", use_memory: bool | None = None,
                 tool_runtime: ToolRuntime | None = None,
                 compaction: bool = False,
                 replay_cache: dict[str, Any] | None = None,
                 policy_tier: str | None = None) -> None:
        if config not in CONFIGS:
            raise ValueError(f"未知配置 `{config}`，可选 {CONFIGS}")
        self.settings = settings
        self.store = store
        self.policy = policy
        self.config = config
        self.use_memory = (config == "agent_memory") if use_memory is None else use_memory
        self.replay_cache = replay_cache or {}
        self.replaying = bool(replay_cache)
        self.llm = llm
        patterns_path = Path(patterns_path or (settings.policy_path.parent / "fraud_patterns.yaml"))
        self.rules, self.rule_version = load_pattern_rules(patterns_path)
        self.rt = tool_runtime or ToolRuntime(
            store=store, pattern_rules=self.rules, pattern_version=self.rule_version
        )
        self.rt.pattern_rules = self.rules
        self.rt.pattern_version = self.rule_version
        self.registry = ToolRegistry(self.rt, policy=policy)
        self.budget = ContextBudget(enabled=compaction)
        self.policy_tier = policy_tier
        self.tool_failures = 0
        self.compensation_count = 0
        self._profile_call: ToolCall | None = None

    # ── 对外入口 ────────────────────────────────────────────────────
    def assess(self, case: CaseInput, *, elder_id: str | None = None) -> Assessment:
        started = time.perf_counter()
        elder_id = elder_id or case.elder_id
        self.rt.known_elder_ids.add(elder_id)
        self.rt.known_identifiers.update(extract_identifiers(case.turns))
        session_id = f"{case.case_id}::{self.config}"
        state = self.store.ensure_session(session_id, elder_id, case.case_id, self.policy.version)
        llm_before = (self.llm.calls, self.llm.prompt_tokens, self.llm.completion_tokens) if self.llm else (0, 0, 0)
        assessment = Assessment(
            case_id=case.case_id, elder_id=elder_id, config=self.config,
            policy_version=self.policy.version, prompt_version=PROMPT_VERSION,
            model=self.llm.model if self.llm else "", report_model="",
        )
        try:
            if self.config == "rule":
                self._run_rule(case, assessment, state, session_id)
            elif self.config == "single_llm":
                self._run_single_llm(case, assessment, state, session_id)
            else:
                self._run_agent(case, assessment, state, session_id)
        except LLMError as exc:  # 模型侧彻底失败 → 显式降级，不猜
            assessment.reasons.append(f"LLM 不可用，该案降级为未知维度：{exc}")
            assessment.degraded_dims.append("evidence_extraction")
            self.store.log_action(case_id=case.case_id, session_id=session_id, elder_id=elder_id,
                                  turn_index=None, kind="degraded", action="llm_unavailable",
                                  payload={"error": str(exc)})
        finally:
            self.store.save_session(state)
        assessment.max_level = state.current_level
        assessment.latency_ms = int((time.perf_counter() - started) * 1000)
        if self.llm:
            assessment.llm_calls = self.llm.calls - llm_before[0]
            assessment.prompt_tokens = self.llm.prompt_tokens - llm_before[1]
            assessment.completion_tokens = self.llm.completion_tokens - llm_before[2]
            assessment.report_model = self.llm.reported_model
        self.store.add_event(elder_id, assessment.max_level,
                             f"{case.case_id} 判定 {assessment.max_level} / {assessment.final_action}")
        self._save_run(case, assessment)
        return assessment

    # ── 确定性规则信号（A 组与 C/D 组共用同一份实现）──────────────────
    def _rule_signals(self, text: str, *, speaker: str, turn_index: int,
                      floor: float = 0.6) -> list[Signal]:
        """把已知话术规则的命中转成信号。

        A 组（规则基线）只用它；C/D 组把它与 LLM 抽出的信号合并——
        因为 ``check_fraud_pattern`` 本来就是被 Agent 调用的工具之一，
        "规则结论进入决策"这件事必须对两组都成立，否则工具调用正确率没有意义。
        """
        hit, _score, hits = pattern_score(self.rules, text)
        if not hit:
            return []
        out: list[Signal] = []
        for h in hits:
            sig_type = RULE_SIGNAL_MAP.get(h["id"], "channel_anomaly")
            if any(s.type == sig_type for s in out):
                continue
            frag = (h.get("matched") or [""])[0]
            out.append(Signal(type=sig_type, quote=frag, speaker=speaker, turn_index=turn_index,
                              confidence=floor, note=f"规则 {h['id']}"))
        return out

    # ── A 组：规则基线（零 LLM）────────────────────────────────────
    def _run_rule(self, case: CaseInput, a: Assessment, state: SessionState, session_id: str) -> None:
        for i, turn in enumerate(case.turns, start=1):
            signals = self._rule_signals(turn.get("text", ""), speaker=turn.get("role", ""),
                                         turn_index=i)
            record = TurnRecord(turn_index=i, speaker=turn.get("role", ""), text=turn.get("text", ""))
            record.signals = signals
            a.signals.extend(signals)
            ctx = self._policy_ctx(a, signals, state, suggested="L0")
            decision = self.policy.decide(ctx)
            self._apply(record, decision, state, a, session_id, case, i, use_tools=False)
            a.turns.append(record)
        self._finalize(a)

    # ── B 组：单次 LLM（无工具 / 无记忆 / 无逐轮）────────────────────
    def _run_single_llm(self, case: CaseInput, a: Assessment, state: SessionState,
                        session_id: str) -> None:
        payload = self._call_evidence(case.turns, upto=None, memory_note="", already=[])
        signals = self._parse_signals(payload)
        a.signals.extend(signals)
        record = TurnRecord(turn_index=0, speaker="__trajectory__", text="(整段轨迹单次判定)")
        record.signals = signals
        record.suggested_level = self._norm_level(payload.get("suggested_level"))
        record.llm_calls = 1
        ctx = self._policy_ctx(a, signals, state, suggested=record.suggested_level,
                               compromise=bool(payload.get("victim_compromise")))
        decision = self.policy.decide(ctx)
        self._apply(record, decision, state, a, session_id, case, 0, use_tools=False)
        a.turns.append(record)
        self._finalize(a)

    # ── C / D 组：逐轮 Agent ───────────────────────────────────────
    def _run_agent(self, case: CaseInput, a: Assessment, state: SessionState, session_id: str) -> None:
        profile_block = ""
        if self.use_memory:
            resident_id = case.elder_id
            self.rt.known_elder_ids.add(resident_id)
            profile = self.registry.call("get_elder_profile", {"elder_id": resident_id},
                                         level="L0", turn_index=0, session_id=session_id)
            a.tool_calls += 1
            self._profile_call = profile
            if profile.ok:
                a.memory_context_used = True
                p = profile.result
                lines = [f"老人档案：{p.get('name', '')} {p.get('age', '')}岁；{p.get('note', '')}"]
                if p.get("recent_events"):
                    lines.append("历史被诱导事件：" + "；".join(
                        e["summary"][:40] for e in p["recent_events"][:2]))
                if p.get("recent_transactions"):
                    lines.append("近期大额支出：" + "；".join(
                        f"{t['amount']:.0f}元({t['counterparty']})" for t in p["recent_transactions"][:2]))
                if p.get("whitelist"):
                    lines.append("白名单家属：" + "；".join(c["label"] for c in p["whitelist"][:3]))
                profile_block = "\n".join(lines)
                prior_high = any(level_rank(e["level"]) >= 3 for e in p.get("recent_events", []))
                if prior_high:
                    state.confirmed.append("历史上有过 L3 被诱导事件")
            else:
                a.degraded_dims.append("elder_profile")
                profile_block = "（老人档案维度未知：工具失败，不做任何假设）"

        pre_records: list[TurnRecord] = []
        if self._profile_call is not None:
            # 记忆读取发生在第一轮之前；单独记一条 turn_index=0 的轨迹，
            # 这样"长期记忆真的被读到"在 runs / trace 里可核对。
            record0 = TurnRecord(turn_index=0, speaker="__memory__", text="(读取长期记忆)")
            record0.tool_calls.append(self._profile_call)
            pre_records.append(record0)
            self._profile_call = None

        for i, turn in enumerate(case.turns, start=1):
            text = turn.get("text", "")
            self.budget.observe_raw(text)
            payload = self._call_evidence(case.turns, upto=i,
                                          memory_note=profile_block if self.use_memory else "",
                                          already=[s.type for s in a.signals])
            signals = self._parse_signals(payload)
            # 确定性规则信号（与 check_fraud_pattern 同一份规则表）：
            # 它让"工具结论真的进入了决策"可被验证，也让 C/D 组在
            # 证据抽取不可用（如无 LLM 的离线冒烟）时仍能做出等级判断。
            so_far = "\n".join(t2.get("text", "") for t2 in case.turns[:i])
            rule_sigs = [s for s in self._rule_signals(so_far, speaker=turn.get("role", ""),
                                                       turn_index=i)
                         if not any(x.type == s.type for x in signals)]
            signals = signals + rule_sigs
            a.signals.extend(signals)
            record = TurnRecord(turn_index=i, speaker=turn.get("role", ""), text=text)
            record.signals = signals
            record.suggested_level = self._norm_level(payload.get("suggested_level"))
            record.llm_calls = 1
            record.notes.append(str(payload.get("reason", ""))[:160])

            # ② 工具调用（确定性触发）
            tool_signals = list(signals)
            self._plan_tools(case, a, record, state, session_id, i, tool_signals)
            pattern_hit = any(tc.name == "check_fraud_pattern" and tc.result.get("pattern_hit")
                              for tc in record.tool_calls)
            comp_text, comp_quote = detect_compromise(case.turns, upto=i)
            compromise = (bool(payload.get("victim_compromise"))
                          or any(s.type == "victim_compromise" for s in signals)
                          or comp_text)
            if comp_text and not any(s.type == "victim_compromise" for s in signals):
                sig = Signal(type="victim_compromise", quote=comp_quote,
                             speaker="elder", turn_index=i, confidence=0.7,
                             note="确定性判定：老人已呈配合倾向")
                signals = signals + [sig]
                a.signals.append(sig)
                record.signals = signals
            # 长期记忆：历史 L3 事件 → 抬升最低可疑度（这是 C/D 的唯一预期差异来源）
            memory_floor = "L0"
            if self.use_memory and any("历史上有过" in c for c in state.confirmed):
                memory_floor = "L1"
            record.notes.append(f"memory_floor={memory_floor}")
            ctx = self._policy_ctx(a, signals, state, suggested=record.suggested_level,
                                   pattern_hit=pattern_hit, compromise=compromise)
            if level_rank(memory_floor) > 0:
                ctx.signal_types.add("channel_anomaly")
                ctx.signal_confidence["channel_anomaly"] = max(
                    ctx.signal_confidence.get("channel_anomaly", 0.0), 0.5)
                record.notes.append("长期记忆：存在历史 L3 事件 → 计为弱渠道异常信号")
            decision = self.policy.decide(ctx)
            self._apply(record, decision, state, a, session_id, case, i, use_tools=True)
            # 不可逆动作必须在**等级已落到会话状态之后**评估：
            # 用 level_before 判断会整体晚一轮，且拿不到"本会话是否已通知过"的正确视角。
            if level_rank(state.current_level) >= 3:
                self._maybe_notify(case, a, record, state, session_id, i)
            a.turns.append(record)
        a.turns = pre_records + a.turns
        self._finalize(a)

    # ── 工具规划（确定性映射）───────────────────────────────────────
    def _plan_tools(self, case: CaseInput, a: Assessment, record: TurnRecord, state: SessionState,
                    session_id: str, turn_index: int, signals: list[Signal]) -> None:
        types = {s.type for s in signals}
        all_text = "\n".join(t.get("text", "") for t in case.turns[:turn_index])
        level_now = state.current_level
        ctx_kw = {"case_id": case.case_id, "elder_id": case.elder_id, "session_id": session_id}

        if True:
            call = self.registry.call("check_fraud_pattern", {"text": all_text},
                                      level=level_now, turn_index=turn_index, session_id=session_id)
            self._record_tool(a, record, call, raw_text=all_text, **ctx_kw)
            pattern_hit = bool(call.result.get("pattern_hit"))

            # 只有当轨迹里真的出现了标识符才查联系人（权限最小化 + 避免无意义调用）
            idents = extract_identifiers(case.turns[:turn_index])
            if idents and (types & {"channel_anomaly", "identity_doubt", "money_action"} or pattern_hit):
                for ident in idents[:2]:
                    c2 = self.registry.call("check_contact", {"identifier": ident, "elder_id": case.elder_id},
                                            level=level_now, turn_index=turn_index, session_id=session_id)
                    self._record_tool(a, record, c2, raw_text=ident, **ctx_kw)

    def _maybe_notify(self, case: CaseInput, a: Assessment, record: TurnRecord,
                      state: SessionState, session_id: str, turn_index: int) -> None:
        """不可逆动作（通知家属）：授权等级由策略表决定，幂等由工具层保证。"""
        summary = f"{case.case_id}（{case.kind}）达到 {state.current_level}"
        call = self.registry.call(
            "notify_family",
            {"elder_id": case.elder_id, "summary": summary, "level": state.current_level},
            level=state.current_level, turn_index=turn_index, session_id=session_id)
        self._record_tool(a, record, call, raw_text=summary, case_id=case.case_id,
                          elder_id=case.elder_id, session_id=session_id)
        if call.rejected_by == "privilege":
            a.unauthorized_actions += 1
        elif call.idempotent_skip:
            a.suppressed_actions += 1
        elif call.ok and not call.idempotent_skip:
            if "notify_family" not in state.actions_taken:
                state.actions_taken.append("notify_family")

    def _record_tool(self, a: Assessment, record: TurnRecord, call: ToolCall, *,
                     raw_text: str, case_id: str = "", elder_id: str = "",
                     session_id: str = "") -> None:
        raw_result = dict(call.result or {})
        trimmed = trim_tool_result(raw_result)
        trimmed["_raw_keys"] = sorted(raw_result.keys())
        self.budget.observe_tool(raw_result, trimmed)
        call.result = trimmed
        record.tool_calls.append(call)
        a.tool_calls += 1
        if call.degraded:
            self.tool_failures += 1
            record.degraded_dims.append(call.name)
            a.degraded_dims.append(call.name)
        self.store.log_action(
            case_id=case_id, session_id=session_id, elder_id=elder_id,
            turn_index=record.turn_index, kind="tool_call", action=call.name,
            payload={"args": call.args, "ok": call.ok, "error": call.error,
                     "degraded": call.degraded, "idempotent_skip": call.idempotent_skip,
                     "rejected_by": call.rejected_by},
        )

    # ── 模型调用 ────────────────────────────────────────────────────
    def _call_evidence(self, turns: list[dict[str, Any]], *, upto: int | None,
                       memory_note: str, already: list[str]) -> dict[str, Any]:
        limit = upto if upto is not None else len(turns)
        # 上下文组装**总是**走 compact_turns（即使没有模型、即使压缩关闭）：
        # 这样"压缩省了多少 token"在离线通道下也能被测到，
        # 而且开关两侧用的是同一段记账代码，差异只来自压缩本身。
        history_turns = self.budget.compact_turns(turns, limit)
        user = prompts.evidence_user_prompt(history_turns, upto=None, memory_note=memory_note,
                                            already=already)
        if self.llm is None:
            return {"signals": [], "suggested_level": "L0", "victim_compromise": False}
        key = f"evidence|{self.llm.model}|{PROMPT_VERSION}|{hash(user)}"
        from_cache = self.replaying and key in self.replay_cache
        payload: dict[str, Any]
        if from_cache:
            payload = self.replay_cache[key]
        else:
            payload, _resp = self.llm.complete_json(system=prompts.EVIDENCE_SYSTEM, user=user,
                                                    max_tokens=1200)
            if not isinstance(payload, dict):
                payload = {"signals": [], "suggested_level": "L0"}
            if self.replaying:
                raise LLMError("回放模式缺少该轮录制结果（轨迹不完整）")
            self.replay_cache[key] = payload
        return payload

    # ── 解析与工具函数 ──────────────────────────────────────────────
    @staticmethod
    def _parse_signals(payload: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        raw = payload.get("signals") or []
        if isinstance(raw, dict):
            raw = [raw]
        for item in raw:
            if not isinstance(item, dict):
                continue
            sig = Signal.from_raw(item)
            if sig is not None:
                out.append(sig)
        # 去重：同一类型只留置信度最高的一条
        best: dict[str, Signal] = {}
        for s in out:
            if s.type not in best or s.confidence > best[s.type].confidence:
                best[s.type] = s
        if payload.get("victim_compromise") and "victim_compromise" not in best:
            best["victim_compromise"] = Signal(type="victim_compromise", quote="", confidence=0.6,
                                               note="模型判定老人已被说服")
        return list(best.values())

    @staticmethod
    def _norm_level(value: Any) -> str:
        text = str(value or "").strip().upper()
        return text if text in ("L0", "L1", "L2", "L3", "L4") else "L0"

    def _policy_ctx(self, a: Assessment, signals: list[Signal], state: SessionState, *,
                    suggested: str = "L0", pattern_hit: bool = False,
                    compromise: bool = False) -> PolicyContext:
        """构造策略上下文。

        ``extra_signals`` 是**本轮**刚抽出的信号：策略必须在"证据到达的那一刻"
        就生效，不能等下一轮——否则拦截会整体推迟一轮，``PIR`` 直接失真。
        """
        pool = list(a.signals) + list(signals)
        types = {s.type for s in pool}
        conf: dict[str, float] = {}
        for s in pool:
            conf[s.type] = max(conf.get(s.type, 0.0), s.confidence)
        return PolicyContext(
            signal_types=types, signal_confidence=conf, pattern_hit=pattern_hit,
            victim_compromise=compromise or ("victim_compromise" in types),
            money_action="money_action" in types, suggested_level=suggested,
            current_level=state.current_level,
        )

    def _apply(self, record: TurnRecord, decision, state: SessionState, a: Assessment,
               session_id: str, case: CaseInput, turn_index: int, *, use_tools: bool) -> None:
        before = state.current_level
        record.proposed_level = decision.proposed_level
        record.final_level = decision.level
        record.action = decision.action
        record.suppressed_action = decision.suppressed_action
        record.notes.extend(decision.reasons)
        if level_rank(decision.level) > level_rank(state.current_level):
            state.current_level = decision.level
        state.policy_version = self.policy.version
        if decision.action != "none" and decision.action not in state.actions_taken:
            state.actions_taken.append(decision.action)
        a.final_action = decision.action
        a.reasons.extend(r for r in decision.reasons if r not in a.reasons)
        if decision.suppressed_action:
            a.suppressed_actions += 1
        self.store.log_action(
            case_id=case.case_id, session_id=session_id, elder_id=case.elder_id, turn_index=turn_index,
            kind="state_transition", level_from=before, level_to=state.current_level,
            action=decision.action,
            payload={"proposed": decision.proposed_level, "suggested": decision.suggested_level,
                     "policy_version": decision.policy_version, "tier": decision.tier,
                     "reasons": decision.reasons, "suppressed": decision.suppressed_action},
        )

    def _finalize(self, a: Assessment) -> None:
        a.max_level = max((t.final_level for t in a.turns), key=level_rank, default="L0")
        a.proposed_max_level = max((t.proposed_level for t in a.turns), key=level_rank, default="L0")
        a.suggested_max_level = max((t.suggested_level for t in a.turns), key=level_rank, default="L0")
        first = next((t.turn_index for t in a.turns
                      if t.turn_index >= 1 and level_rank(t.final_level) >= 2), None)
        a.first_l2_turn = first
        a.signals = list({s.type: s for s in a.signals}.values())
        a.prompt_tokens = sum(t.prompt_tokens for t in a.turns)
        a.completion_tokens = sum(t.completion_tokens for t in a.turns)

    def _save_run(self, case: CaseInput, a: Assessment) -> None:
        self.store.save_run({
            "case_id": case.case_id, "split": case.split, "config": a.config, "elder_id": a.elder_id,
            "model": self.llm.model if self.llm else "rule-based",
            "report_model": a.report_model or (self.llm.reported_model if self.llm else "n/a"),
            "prompt_version": PROMPT_VERSION, "policy_version": a.policy_version,
            "policy_tier": self.policy.tier, "dataset_sha": case.raw.get("_dataset_sha", ""),
            "max_level": a.max_level, "proposed_level": a.proposed_max_level,
            "suggested_level": a.suggested_max_level, "first_l2_turn": a.first_l2_turn,
            "final_action": a.final_action, "unauthorized": a.unauthorized_actions,
            "suppressed": a.suppressed_actions, "llm_calls": a.llm_calls,
            "tool_calls": a.tool_calls, "prompt_tokens": a.prompt_tokens,
            "completion_tokens": a.completion_tokens, "latency_ms": a.latency_ms,
            "degraded_dims": a.degraded_dims,
        })

    # ── 确定性回放 ──────────────────────────────────────────────────
    def replay(self, case: CaseInput, trace: dict[str, Any]) -> Assessment:
        """用录制时的模型输出 + 工具返回重放，不再真调模型/工具。

        一致率 = 重放得到的 (max_level, first_l2_turn, final_action) 与录制一致的比例。
        """
        cache = dict(trace.get("llm_cache", {}))
        self.replay_cache = cache
        self.replaying = True
        self.llm = None
        return self.assess(case)


def action_timeline(a: Assessment) -> str:
    return " → ".join(t.final_level for t in a.turns) if a.turns else "(空)"
