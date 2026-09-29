"""确定性回放 + FastAPI 服务 + MCP 工具暴露的测试（全部离线）。"""

from __future__ import annotations

import json
from pathlib import Path

from silverguard.agent import CaseInput, GuardianAgent
from silverguard.memory import MemoryStore
from silverguard.tools import ToolRuntime

from .conftest import PATTERNS_PATH

PATHS_POLICY = PATTERNS_PATH.parent / "policy.yaml"


def _record_trace(store, policy, llm, case: CaseInput) -> dict:
    cache: dict = {}
    agent = GuardianAgent(settings=None, store=store, policy=policy, patterns_path=PATTERNS_PATH,
                          llm=llm, config="agent", tool_runtime=ToolRuntime(store=store),
                          replay_cache=cache)
    a = agent.assess(case)
    return {**a.to_dict(),
            "case": {"case_id": case.case_id, "turns": case.turns, "kind": case.kind,
                     "split": case.split, "is_attack": True,
                     "transfer_turn": case.transfer_turn, "se_attack": case.se_attack},
            "llm_cache": cache}


CASE = CaseInput(case_id="rp-1", turns=[
    {"role": "fraud", "text": "您好，我是银行的工作人员，您账户异常，需要核验。"},
    {"role": "elder", "text": "怎么核验？"},
    {"role": "fraud", "text": "您先把资金转到我们的安全账户，别告诉子女，否则账户会被冻结。"},
    {"role": "elder", "text": "好，我这就转。"},
], kind="impersonate_official", transfer_turn=3, se_attack=True)


def test_replay_dir_reports_consistency(store, policy, tmp_path: Path, fake_llm):
    from silverguard.config import Settings
    from silverguard.replay import replay_dir

    trace = _record_trace(store, policy, fake_llm, CASE)
    trace_dir = tmp_path / "runs"
    trace_dir.mkdir()
    (trace_dir / "agent__rp-1.json").write_text(json.dumps(trace, ensure_ascii=False),
                                                encoding="utf-8")
    settings = Settings(api_key="x", model="deepseek-chat", policy_path=PATHS_POLICY)
    result = replay_dir(trace_dir, settings=settings, config="agent")
    assert result["replayed"] == 1
    assert result["replay_consistency_rate"] == 100.0
    assert result["mismatches"] == []
    assert result["trace_missing_versions"] == []


def test_replay_skips_legacy_trace_without_model_cache(tmp_path: Path):
    """旧轨迹缺少模型输出，必须单独计数，不能混入回放分母。"""
    from silverguard.replay import replay_dir

    trace_dir = tmp_path / "runs"
    trace_dir.mkdir()
    (trace_dir / "agent__legacy.json").write_text(
        json.dumps({"case_id": "legacy", "turns": []}), encoding="utf-8")
    result = replay_dir(trace_dir, config="agent")
    assert result["replayed"] == 0
    assert result["skipped_legacy_traces"] == 1
    assert result["replay_consistency_rate"] is None


def test_replay_detects_policy_version_drift(store, policy, tmp_path: Path, fake_llm):
    """改一版策略后回放 → 一致率下降，且能定位到"是策略版本引起的"。"""
    from silverguard.config import Settings
    from silverguard.replay import replay_dir

    trace = _record_trace(store, policy, fake_llm, CASE)
    trace["max_level"] = "L1"          # 假装录制时是 L1（即当时策略更保守）
    trace["final_action"] = "soft_reminder"
    trace_dir = tmp_path / "runs"
    trace_dir.mkdir()
    (trace_dir / "agent__rp-1.json").write_text(json.dumps(trace, ensure_ascii=False),
                                                encoding="utf-8")
    settings = Settings(api_key="x", model="deepseek-chat", policy_path=PATHS_POLICY)
    result = replay_dir(trace_dir, settings=settings, config="agent")
    assert result["replay_consistency_rate"] == 0.0
    assert result["mismatches"]


def test_breakpoint_replay_only_uses_first_n_turns(store, policy):
    from silverguard.config import Settings
    from silverguard.replay import replay_trace

    from .conftest import FakeLLM

    trace = _record_trace(store, policy, FakeLLM(), CASE)
    settings = Settings(api_key="x", model="deepseek-chat", policy_path=PATHS_POLICY)
    full = replay_trace(trace, settings=settings, policy=policy, patterns_path=PATTERNS_PATH)
    early = replay_trace(trace, settings=settings, policy=policy, patterns_path=PATTERNS_PATH,
                         upto_turn=1)
    assert full["max_level"] in ("L2", "L3", "L4")
    assert len(early["timeline"]) == 1


def test_service_assess_endpoint_offline(tmp_path: Path):
    from fastapi.testclient import TestClient

    from silverguard.config import Settings
    from silverguard.service import build_app

    settings = Settings(api_key="", model="deepseek-chat", policy_path=PATHS_POLICY,
                        db_path=tmp_path / "svc.db")
    app = build_app(settings=settings, store=MemoryStore(":memory:"))
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200 and health.json()["status"] == "ok"

    resp = client.post("/assess", json={
        "case_id": "svc-1", "elder_id": "elder-0001", "allow_llm": False,
        "turns": [{"role": "fraud", "text": "我是检察院的，您把钱转到核查账户，别告诉子女。"},
                  {"role": "elder", "text": "好。"}],
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["max_level"] in ("L2", "L3", "L4")
    assert body["action"] in ("require_confirm", "notify_family", "block_assist")
    assert body["policy_version"]
    assert body["reasons"]


def test_service_rejects_malformed_request(tmp_path: Path):
    from fastapi.testclient import TestClient

    from silverguard.config import Settings
    from silverguard.service import build_app

    settings = Settings(api_key="", model="deepseek-chat", policy_path=PATHS_POLICY,
                        db_path=tmp_path / "svc.db")
    client = TestClient(build_app(settings=settings, store=MemoryStore(":memory:")))
    assert client.post("/assess", json={"turns": []}).status_code == 422
    assert client.post("/assess", json={"turns": [{"role": "hacker", "text": "x"}]}).status_code == 422
    assert client.post("/assess", json={"turns": [{"role": "fraud", "text": "  "}]}).status_code == 422


def test_service_intervene_degrades_to_template_without_key(tmp_path: Path):
    from fastapi.testclient import TestClient

    from silverguard.config import Settings
    from silverguard.service import build_app

    settings = Settings(api_key="", model="deepseek-chat", policy_path=PATHS_POLICY,
                        db_path=tmp_path / "svc.db")
    client = TestClient(build_app(settings=settings, store=MemoryStore(":memory:")))
    resp = client.post("/intervene", json={"level": "L3", "summary": "老人被要求转账"})
    assert resp.status_code == 200
    assert resp.json()["source"] == "template", "没有 key 时必须如实降级，不许假装调用过模型"


def test_service_policy_endpoint_exposes_version(tmp_path: Path):
    from fastapi.testclient import TestClient

    from silverguard.config import Settings
    from silverguard.service import build_app

    settings = Settings(api_key="", model="deepseek-chat", policy_path=PATHS_POLICY,
                        db_path=tmp_path / "svc.db")
    client = TestClient(build_app(settings=settings, store=MemoryStore(":memory:")))
    body = client.get("/policy").json()
    assert body["version"] and body["monotonic"] is True and body["loaded"] is True


def test_service_websocket_streams_decisions(tmp_path: Path, monkeypatch):
    from fastapi.testclient import TestClient

    from silverguard.config import Settings
    from silverguard.service import build_app

    settings = Settings(api_key="", model="deepseek-chat", policy_path=PATHS_POLICY,
                        db_path=tmp_path / "svc.db")
    client = TestClient(build_app(settings=settings, store=MemoryStore(":memory:")))
    with client.websocket_connect("/ws/assess") as ws:
        ws.send_json({"type": "turn", "role": "fraud",
                      "text": "我是市局的，请您把资金转到核查账户，别告诉子女。"})
        msg = ws.receive_json()
        assert msg["type"] == "decision" and msg["turn"] == 1
        assert msg["max_level"] in ("L2", "L3", "L4")


def test_mcp_exposes_at_least_three_tools():
    """MCP 暴露的工具必须与内部工具层同源（同一份 SCHEMAS / 同一份实现）。"""
    from silverguard.mcp_server import TOOL_DESCRIPTIONS, _schema_for
    from silverguard.tools import ToolRegistry

    exposed = [n for n in TOOL_DESCRIPTIONS if n != "record_case"]
    assert len(exposed) >= 3
    for name in exposed:
        schema = _schema_for(name)
        assert schema["type"] == "object"
        assert schema["required"], f"{name} 必须声明必需参数"
    assert hasattr(ToolRegistry, "call")


def test_mcp_roundtrip_through_stdio():
    """端到端：把 MCP server 当外部进程拉起来，走标准握手逐个调用工具。

    这是"MCP 真的通了"的唯一硬证据——只有代码不算。
    """
    import asyncio
    import os
    import sys as _sys

    from silverguard.mcp_client import run, summarize

    os.environ["SILVERGUARD_DB"] = ":memory:"   # 每次自检都用干净库（幂等键不跨次命中）
    out = asyncio.run(run(python=_sys.executable))
    checks = summarize(out)["checks"]
    assert checks["tools_exposed"] >= 3
    assert checks["whitelist_lookup_ok"]
    assert checks["pattern_matched"]
    assert checks["profile_read_ok"]
    assert checks["invented_identifier_rejected"], "模型自造标识必须被拒"
    assert checks["low_privilege_rejected"], "不可逆动作必须受策略表授权等级约束"
    assert checks["forged_privilege_rejected"]
    assert "notify_family" not in {tool["name"] for tool in out["tools"]}


def test_service_reports_provider_failure_instead_of_success(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from silverguard.config import Settings
    from silverguard.llm import LLMError
    from silverguard.service import build_app

    from .conftest import FakeLLM

    class UnavailableLLM(FakeLLM):
        def complete_json(self, **kwargs):
            raise LLMError('HTTP 402')

    monkeypatch.setattr('silverguard.service.LLMClient', lambda **kw: UnavailableLLM())
    settings = Settings(api_key='test', policy_path=PATHS_POLICY, db_path=tmp_path / 'svc.db')
    client = TestClient(build_app(settings=settings, store=MemoryStore(':memory:')))
    response = client.post('/assess', json={
        'case_id': 'provider-failure', 'turns': [{'role': 'caller', 'text': '您好'}],
    })
    assert response.status_code == 502
    assert 'evidence_extraction' in response.json()['partial_assessment']['degraded_dims']
    assert client.get('/metrics').json()['llm_errors'] == 1


def test_service_replay_accepts_runner_trace_without_touching_live_store(store, policy, fake_llm):
    from fastapi.testclient import TestClient

    from silverguard.config import Settings
    from silverguard.service import build_app

    trace = _record_trace(store, policy, fake_llm, CASE)
    live_store = MemoryStore(':memory:')
    settings = Settings(api_key='', policy_path=PATHS_POLICY)
    client = TestClient(build_app(settings=settings, store=live_store, policy=policy))
    response = client.post('/replay', json={'trace': trace, 'config': 'agent'})
    assert response.status_code == 200
    assert response.json()['consistent'] is True
    assert live_store.fetch_runs() == []
    live_store.close()
