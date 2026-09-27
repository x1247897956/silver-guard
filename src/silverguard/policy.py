"""策略引擎：把"证据"确定性映射为"风险等级 + 干预动作"。

设计要点（对应设计文档 §5.3 / §15.2 第 1 条）：

1. **独立成型**：规则全部在 ``config/policy.yaml``，代码里没有任何等级判断的硬编码；
2. **热加载**：按文件 mtime+size 做变更探测，改配置无需重启进程；
3. **版本化**：``version`` 字段随每次决策回传，落进 ``runs`` 表与评测报告；
4. **失败降级**：解析失败 / 校验失败 → **保留旧策略并告警**，绝不进入"无策略"状态；
5. **LLM 不可越过**：不可逆动作（通知家属 / 阻断）需要策略表声明的授权等级，
   LLM 的"建议等级"只在 ``mode: llm_only``（关掉策略引擎的对照实验）里才被采纳。
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .models import (
    SIGNAL_TYPES,
    level_rank,
)

log = logging.getLogger("silverguard.policy")

LEVELS = ("L0", "L1", "L2", "L3", "L4")


class PolicyLoadError(RuntimeError):
    """策略表不可用（解析失败 / 结构不合法）。"""


@dataclass
class PolicyContext:
    """决策所需的事实。全部来自确定性来源（记忆层 / 工具层 / 证据层）。"""

    #: 本轮为止累计的信号类型 → 最高置信度
    signal_types: set[str] = field(default_factory=set)
    signal_confidence: dict[str, float] = field(default_factory=dict)
    #: 命中了已知话术规则（check_fraud_pattern 的确定性结论）
    pattern_hit: bool = False
    #: 老人已呈现配合倾向（转述"是我自己要转的"/答应操作）
    victim_compromise: bool = False
    #: 是否已出现资金动作信号
    money_action: bool = False
    #: LLM 的建议等级（仅作建议；是否采纳取决于 mode）
    suggested_level: str = "L0"
    #: 当前已累计的等级（用于单调不降）
    current_level: str = "L0"

    @property
    def counting_signals(self) -> set[str]:
        return self.signal_types & set(SIGNAL_TYPES)

    @property
    def max_confidence(self) -> float:
        return max(self.signal_confidence.values(), default=0.0)


@dataclass
class Decision:
    level: str
    action: str
    reasons: list[str] = field(default_factory=list)
    proposed_level: str = ""
    suggested_level: str = ""
    policy_version: str = ""
    tier: str = ""
    authorized: bool = True
    authorization_note: str = ""
    suppressed_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "action": self.action,
            "reasons": list(self.reasons),
            "proposed_level": self.proposed_level,
            "suggested_level": self.suggested_level,
            "policy_version": self.policy_version,
            "tier": self.tier,
            "authorized": self.authorized,
            "suppressed_action": self.suppressed_action,
        }


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise PolicyLoadError(msg)


class PolicyEngine:
    """线程安全的策略表加载器 + 决策器。"""

    def __init__(self, path: str | Path, *, autoload: bool = True) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._data: dict[str, Any] = {}
        self._version = "unloaded"
        self._tier = "balanced"
        self._mode = "policy"
        self._sig: tuple[float, int] | None = None
        self.reload_count = 0
        self.load_failures: list[str] = []
        self.last_error: str = ""
        self.decisions_made = 0
        if autoload:
            self.reload(force=True)

    # ── 加载与热加载 ────────────────────────────────────────────────
    def _file_sig(self) -> tuple[float, int]:
        st = self.path.stat()
        return (st.st_mtime, st.st_size)

    def maybe_reload(self) -> bool:
        """探测文件变更并按需重载。返回是否真的重载了。"""
        try:
            sig = self._file_sig()
        except OSError:
            return False
        with self._lock:
            if self._sig == sig:
                return False
        return self.reload(force=True)

    def reload(self, *, force: bool = False) -> bool:
        """重载策略表。成功返回 True；失败**保留旧策略**并记录告警。"""
        with self._lock:
            try:
                sig = self._file_sig()
            except OSError as exc:  # 文件被删/不可读
                return self._fail(f"策略表不可读：{exc}")
            if not force and self._sig == sig:
                return False
            try:
                raw = self.path.read_text(encoding="utf-8")
                data = yaml.safe_load(raw)
                self._validate(data)
            except Exception as exc:  # noqa: BLE001 —— 任何解析/校验异常都必须被挡住
                return self._fail(f"策略表加载失败（保留旧策略 v{self._version}）：{exc}")
            self._data = data
            self._version = str(data.get("version", "unversioned"))
            self._tier = str(data.get("policy", {}).get("active_tier", "balanced"))
            self._mode = str(data.get("policy", {}).get("mode", "policy"))
            self._sig = sig
            self.reload_count += 1
            log.info("策略表已加载 version=%s tier=%s mode=%s", self._version, self._tier, self._mode)
            return True

    def _fail(self, msg: str) -> bool:
        self.last_error = msg
        self.load_failures.append(msg)
        log.warning("%s", msg)
        return False

    @staticmethod
    def _validate(data: Any) -> None:
        _require(isinstance(data, dict), "顶层必须是映射")
        for key in ("version", "levels", "rules", "actions", "level_actions", "policy"):
            _require(key in data, f"缺少必需字段 `{key}`")
        levels = data["levels"]
        for lv in LEVELS:
            _require(lv in levels, f"levels 缺少 {lv}")
        rules = data["rules"]
        _require(isinstance(rules, dict) and rules, "rules 不能为空")
        for lv, rule in rules.items():
            # 同一等级允许多条规则：键名形如 `L3` / `L3_pattern`（`_` 后为通道名）。
            _require(str(lv).split("_", 1)[0] in LEVELS, f"rules 出现未知等级 {lv}")
            _require(isinstance(rule, dict), f"rules.{lv} 必须是映射")
            _require("when" in rule, f"rules.{lv} 缺少 when")
        for lv, act in data["level_actions"].items():
            _require(lv in LEVELS, f"level_actions 出现未知等级 {lv}")
            _require(act in data["actions"], f"level_actions.{lv} 指向未定义动作 {act}")
        for name, spec in data["actions"].items():
            _require(isinstance(spec, dict), f"actions.{name} 必须是映射")
            _require("level" in spec, f"actions.{name} 缺少 level")
            # 不可逆动作必须显式声明授权所需等级：这是"LLM 不能越过策略表"的
            # 机械保证——没有这一条，越权保护就退化成代码里的隐式约定。
            if spec.get("reversible") is False:
                _require("requires_authorized_level" in spec,
                         f"不可逆动作 actions.{name} 必须声明 requires_authorized_level")
        pol = data["policy"]
        _require(isinstance(pol.get("tiers"), dict) and pol["tiers"], "policy.tiers 不能为空")
        tier = pol.get("active_tier", "balanced")
        _require(tier in pol["tiers"], f"active_tier `{tier}` 不在 policy.tiers 中")

    # ── 属性 ────────────────────────────────────────────────────────
    @property
    def version(self) -> str:
        return self._version

    @property
    def tier(self) -> str:
        return self._tier

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def monotonic(self) -> bool:
        with self._lock:
            return bool(self._data.get("policy", {}).get("monotonic", True))

    @property
    def idempotency_ttl(self) -> int:
        with self._lock:
            return int(self._data.get("policy", {}).get("idempotency_ttl_sec", 86400))

    @property
    def loaded(self) -> bool:
        return bool(self._data)

    def data(self) -> dict[str, Any]:
        with self._lock:
            return self._data

    def tier_params(self, tier: str | None = None) -> dict[str, Any]:
        with self._lock:
            pol = self._data.get("policy", {})
            tiers = pol.get("tiers", {})
            return dict(tiers.get(tier or self._tier, {}))

    def set_tier(self, tier: str, *, persist: bool = True) -> None:
        """阈值扫描用：切换阈值档位（可选写回 YAML，从而也走一次热加载路径）。

        ⚠️ 这是**实验开关**，不是运行时行为。写回失败（只读文件系统 / CI 里 YAML 被锁）
        不算错误：退回内存切换并把事实说清楚，绝不静默改错配置。
        """
        with self._lock:
            tiers = self._data.get("policy", {}).get("tiers", {})
            _require(tier in tiers, f"未知阈值档 `{tier}`，可选：{sorted(tiers)}")
            if persist:
                try:
                    text = self.path.read_text(encoding="utf-8")
                    new_text, n = _replace_scalar(text, "  active_tier:", f"  active_tier: {tier}")
                    if n == 1:
                        self.path.write_text(new_text, encoding="utf-8")
                except OSError as exc:
                    log.warning("阈值档写回失败（改为内存切换）：%s", exc)
                else:
                    self.maybe_reload()
            self._tier = tier

    def set_mode(self, mode: str, *, persist: bool = True) -> None:
        """控制对照实验用：``llm_only`` 表示关掉策略引擎，直接采纳 LLM 建议等级。"""
        _require(mode in ("policy", "llm_only"), "mode 只能是 policy / llm_only")
        with self._lock:
            if persist:
                try:
                    text = self.path.read_text(encoding="utf-8")
                    new_text, n = _replace_scalar(text, "  mode:", f"  mode: {mode}")
                    if n == 1:
                        self.path.write_text(new_text, encoding="utf-8")
                except OSError as exc:
                    log.warning("策略模式写回失败（改为内存切换）：%s", exc)
                else:
                    self.maybe_reload()
            self._mode = mode

    # ── 决策 ────────────────────────────────────────────────────────
    def _rule_hits(self, rule: dict[str, Any], ctx: PolicyContext) -> tuple[bool, str]:
        """返回 (是否命中, 命中说明)。"""
        counts = ctx.counting_signals
        when = rule.get("when", {})
        need_all = list(when.get("require_all", []))
        need_any = list(when.get("require_any", []))
        fields = rule.get("when_fields", {})

        missing = [s for s in need_all if s not in ctx.signal_types]
        if missing:
            return False, f"缺 {','.join(missing)}"
        if need_any and not (set(need_any) & ctx.signal_types):
            return False, f"未命中任一 {','.join(need_any)}"
        for key, want in fields.items():
            got = bool(getattr(ctx, key, False))
            if bool(want) is not got:
                return False, f"字段 {key}={got} 不满足"

        params = self.tier_params()
        min_signals = int(rule.get("min_signals", 0))
        if min_signals:
            # tiers 覆盖"信号数阈值"：只有显式声明 signal_floor: L2 的规则才被覆盖，
            # 其余规则的 min_signals 是固定语义（阈值扫描不该改变它们）。
            if str(rule.get("signal_floor", "")) == "L2" and "min_signals_l2" in params:
                min_signals = int(params["min_signals_l2"])
            if len(counts) < min_signals:
                return False, f"信号数 {len(counts)} < {min_signals}"
        min_conf = float(rule.get("min_confidence", 0.0))
        if min_conf:
            if str(rule.get("signal_floor", "")) == "L2" and "min_confidence" in params:
                min_conf = float(params["min_confidence"])
            if ctx.max_confidence < min_conf:
                return False, f"最高置信度 {ctx.max_confidence:.2f} < {min_conf}"
        joined = ",".join(sorted(counts)) or "无"
        return True, f"信号[{joined}]"

    def evaluate(self, ctx: PolicyContext) -> tuple[str, list[str]]:
        """按规则表求出"证据所能支撑的最高等级"。

        同一等级可以有多条规则（例如 ``L3`` 与 ``L3_pattern``）：按 YAML 中的
        声明顺序取第一条命中的，全部未命中才落到下一等级。这样"主通道 + 补充通道"
        能表达在配置里，而不需要把 OR 逻辑硬编码进代码。
        """
        with self._lock:
            rules = list(self._data.get("rules", {}).items())
        by_level: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for key, rule in rules:
            lv = str(key).split("_", 1)[0]
            if lv in LEVELS and isinstance(rule, dict):
                by_level.setdefault(lv, []).append((str(key), rule))

        best = "L0"
        reasons: list[str] = []
        for lv in ("L4", "L3", "L2", "L1"):
            for key, rule in by_level.get(lv, []):
                hit, why = self._rule_hits(rule, ctx)
                if hit:
                    best = lv
                    tag = lv if key == lv else f"{lv}({key})"
                    reasons.append(f"{tag} 命中：{why}")
                    break
            if best != "L0":
                break
        if not reasons:
            reasons.append("L0：无规则命中")
        return best, reasons

    def action_for(self, level: str) -> str:
        with self._lock:
            return str(self._data.get("level_actions", {}).get(level, "none"))

    def action_spec(self, action: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._data.get("actions", {}).get(action, {}))

    def decide(self, ctx: PolicyContext) -> Decision:
        """最终决策：确定性规则 → 等级 → 动作，并施加单调不降与授权检查。"""
        self.maybe_reload()
        self.decisions_made += 1
        proposed, reasons = self.evaluate(ctx)

        if self.mode == "llm_only":
            # 对照实验：关掉策略引擎，直接采纳 LLM 的建议等级。
            suggested = ctx.suggested_level if ctx.suggested_level in LEVELS else "L0"
            level = suggested
            reasons.insert(0, f"[llm_only] 策略引擎已关闭，采纳 LLM 建议等级 {suggested}")
        else:
            level = proposed
            suggested = ctx.suggested_level if ctx.suggested_level in LEVELS else "L0"
            if level_rank(suggested) > level_rank(level):
                reasons.append(
                    f"LLM 建议 {suggested} 高于规则可支撑的 {level}，不予升级（LLM 不可越过策略表）"
                )

        # 单调不降：等级只能升，要降必须改策略表
        if self.monotonic and level_rank(level) < level_rank(ctx.current_level):
            reasons.append(
                f"单调不降：本轮规则给出 {level}，但会话当前等级为 {ctx.current_level}，维持 {ctx.current_level}"
            )
            level = ctx.current_level

        # 降级诱导审计：规则给出的等级低于会话当前等级即记录（用于实验统计）
        suppressed = ""
        if level_rank(proposed) < level_rank(ctx.current_level) and self.mode != "llm_only":
            suppressed = self.action_for(proposed)

        action = self.action_for(level)
        spec = self.action_spec(action)
        authorized = True
        note = ""
        required = spec.get("requires_authorized_level")
        if required and spec.get("reversible") is False:
            if level_rank(level) < level_rank(str(required)):
                authorized = False
                note = f"动作 {action} 需要策略表授权等级 {required}，当前等级 {level} → 动作被拒"
                reasons.append(note)
                fallback = self.action_for("L2" if level_rank(level) >= 2 else "L1")
                action = fallback

        return Decision(
            level=level,
            action=action,
            reasons=reasons,
            proposed_level=proposed,
            suggested_level=ctx.suggested_level,
            policy_version=self.version,
            tier=self.tier,
            authorized=authorized,
            authorization_note=note,
            suppressed_action=suppressed,
        )


def _replace_scalar(text: str, prefix: str, replacement: str) -> tuple[str, int]:
    out, n = [], 0
    for line in text.splitlines():
        if line.startswith(prefix) and n == 0:
            out.append(replacement)
            n = 1
        else:
            out.append(line)
    return "\n".join(out) + ("\n" if text.endswith("\n") else ""), n


def load_policy(path: str | Path | None = None) -> PolicyEngine:
    from .config import get_settings

    env_path = os.environ.get("SILVERGUARD_POLICY")
    target = Path(env_path) if env_path else (Path(path) if path else get_settings().policy_path)
    return PolicyEngine(target)
