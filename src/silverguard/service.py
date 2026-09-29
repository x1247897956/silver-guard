"""FastAPI 服务层。

- ``POST /assess``    对话轨迹 → 风险等级 + 干预动作 + 理由
- ``POST /intervene`` 生成对老人的分级劝说话术
- ``POST /replay``    用录制轨迹做确定性回放（工具走 mock）
- ``GET  /policy``    当前策略表版本 / 档位 / 模式（热加载状态）
- ``GET  /health``    存活探针；``GET /metrics`` 运行计数
- ``WS   /ws/assess`` 逐轮流式推送决策（顺带项，不作为复杂度主打）

服务是**旁路监督**形态：它读完整轨迹并给出等级与动作，不代替老人说话。
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from . import PROMPT_VERSION
from .agent import CONFIGS, CaseInput, GuardianAgent
from .config import get_settings
from .llm import LLMClient, LLMError
from .memory import MemoryStore, seed_demo_profile
from .policy import PolicyEngine, load_policy
from .prompts import INTERVENTION_BY_LEVEL, INTERVENTION_SYSTEM
from .replay import replay_trace
from .tools import ToolRuntime

log = logging.getLogger("silverguard.service")

ALLOWED_ROLES = {"fraud", "elder", "family", "caller"}


class Turn(BaseModel):
    role: str
    text: str

    @field_validator("role")
    @classmethod
    def _role(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ALLOWED_ROLES:
            raise ValueError(f"role 只能是 {sorted(ALLOWED_ROLES)}")
        return v

    @field_validator("text")
    @classmethod
    def _text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text 不能为空")
        if len(v) > 2000:
            raise ValueError("单轮 text 过长（>2000 字）")
        return v


class AssessRequest(BaseModel):
    case_id: str = Field(default="adhoc-0001", max_length=64)
    elder_id: str = Field(default="elder-0001", max_length=64)
    turns: list[Turn] = Field(min_length=1, max_length=60)
    config: str = "agent_memory"
    allow_llm: bool = True

    @field_validator("config")
    @classmethod
    def _config(cls, v: str) -> str:
        if v not in CONFIGS:
            raise ValueError(f"config 只能是 {list(CONFIGS)}")
        return v


class InterveneRequest(BaseModel):
    level: str = "L2"
    summary: str = ""

    @field_validator("level")
    @classmethod
    def _level(cls, v: str) -> str:
        v = v.strip().upper()
        if v not in ("L1", "L2", "L3", "L4"):
            raise ValueError("level 只能是 L1/L2/L3/L4")
        return v


class ReplayRequest(BaseModel):
    trace: dict[str, Any]
    config: str = "agent_memory"


def build_app(*, settings=None, store: MemoryStore | None = None,
              policy: PolicyEngine | None = None) -> FastAPI:
    settings = settings or get_settings()
    store = store or MemoryStore(settings.db_path)
    seed_demo_profile(store)
    policy = policy or load_policy(settings.policy_path)
    patterns_path = settings.policy_path.parent / "fraud_patterns.yaml"

    app = FastAPI(
        title="SilverGuard",
        description="银发反诈守护 Agent：多轮风险决策 + 分级干预（旁路监督形态）",
        version="0.1.0",
    )
    state: dict[str, Any] = {"assess_calls": 0, "llm_errors": 0, "last_level": "L0"}

    def make_agent(config: str, *, allow_llm: bool = True) -> GuardianAgent:
        llm = None
        if config != "rule" and allow_llm:
            if not settings.has_api_key:
                raise HTTPException(status_code=503, detail="服务未配置 DEEPSEEK_API_KEY")
            llm = LLMClient(api_key=settings.api_key, model=settings.model,
                            base_url=settings.base_url, temperature=settings.temperature,
                            timeout=settings.request_timeout)
        rt = ToolRuntime(store=store)
        return GuardianAgent(settings=settings, store=store, policy=policy,
                             patterns_path=patterns_path, llm=llm, config=config,
                             tool_runtime=rt)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "policy_version": policy.version,
                "policy_loaded": policy.loaded, "prompt_version": PROMPT_VERSION,
                "model": settings.model, "llm_enabled": settings.has_api_key}

    @app.get("/policy")
    def policy_info() -> dict[str, Any]:
        policy.maybe_reload()
        return {"version": policy.version, "path": str(policy.path), "loaded": policy.loaded,
                "tier": policy.tier, "mode": policy.mode,
                "monotonic": policy.monotonic, "idempotency_ttl_sec": policy.idempotency_ttl,
                "reload_count": policy.reload_count, "load_failures": policy.load_failures,
                "decisions_made": policy.decisions_made,
                "tiers": sorted(policy.data().get("policy", {}).get("tiers", {}))}

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        return dict(state)

    @app.post("/assess")
    def assess(req: AssessRequest) -> JSONResponse:
        config = "rule" if not req.allow_llm else req.config
        try:
            agent = make_agent(config, allow_llm=req.allow_llm)
        except HTTPException:
            raise
        turns = [t.model_dump() for t in req.turns]
        case = CaseInput(case_id=req.case_id, turns=turns, elder_id=req.elder_id,
                         kind="adhoc", is_attack=True, gold_min_level="L2")
        try:
            assessment = agent.assess(case)
        except LLMError as exc:
            state["llm_errors"] += 1
            raise HTTPException(status_code=502, detail=f"模型不可用：{exc}") from exc
        finally:
            if agent.llm:
                agent.llm.close()
        state["assess_calls"] += 1
        state["last_level"] = assessment.max_level
        if "evidence_extraction" in assessment.degraded_dims:
            state["llm_errors"] += 1
            return JSONResponse({"detail": "模型不可用，评估未完成",
                                 "partial_assessment": assessment.to_dict()}, status_code=502)
        return JSONResponse({
            "case_id": assessment.case_id,
            "config": config,
            "max_level": assessment.max_level,
            "action": assessment.final_action,
            "first_l2_turn": assessment.first_l2_turn,
            "reasons": assessment.reasons,
            "signals": [s.to_dict() for s in assessment.signals],
            "timeline": assessment.level_timeline(),
            "policy_version": assessment.policy_version,
            "prompt_version": assessment.prompt_version,
            "model": assessment.report_model or assessment.model,
            "latency_ms": assessment.latency_ms,
            "llm_calls": assessment.llm_calls,
            "tokens": {"prompt": assessment.prompt_tokens,
                       "completion": assessment.completion_tokens},
            "degraded_dims": assessment.degraded_dims,
            "trace": assessment.to_dict(),
        })

    @app.post("/intervene")
    def intervene(req: InterveneRequest) -> JSONResponse:
        """分级劝说话术。没有 key 时退化为**确定性模板**，不假装调用过模型。"""
        guide = INTERVENTION_BY_LEVEL[req.level]
        if not settings.has_api_key:
            return JSONResponse({"level": req.level, "text": guide, "source": "template",
                                 "note": "未配置模型，返回策略表内的确定性话术模板"})
        try:
            with LLMClient(api_key=settings.api_key, model=settings.model,
                           base_url=settings.base_url, temperature=0.3) as client:
                resp = client.complete(
                    system=INTERVENTION_SYSTEM,
                    user=f"风险等级：{req.level}\n要求：{guide}\n已知情况：{req.summary or '（无）'}\n"
                         "请输出 3 句以内的口头提醒。",
                    max_tokens=300, temperature=0.3, use_cache=False,
                )
            return JSONResponse({"level": req.level, "text": resp.text.strip(),
                                 "source": "llm", "model": resp.model_reported})
        except LLMError as exc:
            return JSONResponse({"level": req.level, "text": guide, "source": "template",
                                 "note": f"模型调用失败，降级为模板：{exc}"}, status_code=200)

    @app.post("/replay")
    def replay(req: ReplayRequest) -> JSONResponse:
        """确定性回放：用录制轨迹里的工具返回与模型输出重放，不再真调外部依赖。"""
        trace = req.trace
        case_raw = trace.get("case") or {}
        if not case_raw.get("turns"):
            raise HTTPException(status_code=400, detail="trace.case.turns 缺失，无法回放")
        try:
            replayed = replay_trace(trace, settings=settings, policy=policy,
                                    patterns_path=patterns_path, config=req.config)
        except (ValueError, LLMError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        recorded = trace.get("assessment") or trace
        same = (replayed["max_level"] == recorded.get("max_level")
                and replayed["first_l2_turn"] == recorded.get("first_l2_turn")
                and replayed["action"] == recorded.get("final_action"))
        return JSONResponse({
            "case_id": trace.get("case_id", case_raw.get("case_id", "replay")),
            "replayed": replayed,
            "recorded": {"max_level": recorded.get("max_level"),
                         "first_l2_turn": recorded.get("first_l2_turn"),
                         "action": recorded.get("final_action")},
            "consistent": bool(same),
        })

    @app.websocket("/ws/assess")
    async def ws_assess(ws: WebSocket) -> None:
        """逐轮推送决策（旁路监督的流式形态）。

        客户端每发一条消息 = 追加一轮对话；服务端回推当前等级与动作。
        """
        await ws.accept()
        turns: list[dict[str, str]] = []
        case_id = "ws-session"
        try:
            while True:
                payload = await ws.receive_json()
                if payload.get("type") == "reset":
                    turns = []
                    await ws.send_json({"type": "reset_ok"})
                    continue
                if payload.get("type") == "turn":
                    role = str(payload.get("role", "fraud")).lower()
                    text = str(payload.get("text", "")).strip()
                    if role not in ALLOWED_ROLES or not text:
                        await ws.send_json({"type": "error", "detail": "role/text 不合法"})
                        continue
                    turns.append({"role": role, "text": text})
                    case_id = str(payload.get("case_id", case_id))
                    agent = make_agent("rule", allow_llm=False)   # 流式用规则通道，零成本、零延迟
                    assessment = agent.assess(CaseInput(
                        case_id=case_id, turns=turns, kind="ws", is_attack=True, gold_min_level="L2"))
                    await ws.send_json({
                        "type": "decision", "turn": len(turns),
                        "max_level": assessment.max_level, "action": assessment.final_action,
                        "signals": [s.type for s in assessment.signals],
                        "policy_version": assessment.policy_version,
                    })
                else:
                    await ws.send_json({"type": "error", "detail": "未知消息类型"})
        except WebSocketDisconnect:
            return
        except Exception as exc:  # noqa: BLE001
            log.warning("WebSocket 异常：%s", exc)
            try:
                await ws.close()
            except Exception:  # noqa: BLE001
                pass

    return app


#: ``uvicorn silverguard.service:app`` 直接可用；测试请用 build_app() 注入内存库。
app = build_app()
