"""pytest 共享夹具：全部离线，不访问网络。

测试的纪律：**不 mock 被测逻辑本身**。策略引擎、状态机、工具层、指标计算
都是纯确定性代码，直接跑真的；只有"模型调用"用假客户端替换。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from silverguard.config import Settings
from silverguard.llm import LLMResponse
from silverguard.memory import MemoryStore, seed_demo_profile
from silverguard.policy import PolicyEngine
from silverguard.tools import ToolRuntime

REPO = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO / "config" / "policy.yaml"
PATTERNS_PATH = REPO / "config" / "fraud_patterns.yaml"


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(api_key="test-key", model="deepseek-chat",
                    base_url="https://api.deepseek.com",
                    db_path=tmp_path / "test.db", policy_path=POLICY_PATH)


@pytest.fixture()
def store() -> MemoryStore:
    s = MemoryStore(":memory:")
    seed_demo_profile(s)
    yield s
    s.close()


@pytest.fixture()
def policy(tmp_path: Path) -> PolicyEngine:
    """复制一份策略表到临时目录，避免测试改到仓库里的真配置。"""
    target = tmp_path / "policy.yaml"
    target.write_text(POLICY_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return PolicyEngine(target)


@pytest.fixture()
def tool_runtime(store: MemoryStore) -> ToolRuntime:
    return ToolRuntime.from_files(store, PATTERNS_PATH,
                                  known_identifiers={"+86-138-0000-0001", "wechat:daughter-01"},
                                  known_elder_ids={"elder-0001"})


class FakeLLM:
    """假模型：按"用户提示里出现的标记"返回预设证据，完全确定。

    它只替代**模型调用**这一层；Agent 的解析、策略、工具、状态机全部跑真代码。
    """

    def __init__(self, script: list[dict] | None = None, *, model: str = "deepseek-chat",
                 reported: str = "deepseek-flash") -> None:
        self.model = model
        self.reported_model = reported
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cache_hits = 0
        self.script = list(script or [])
        self.seen_prompts: list[str] = []

    def complete(self, *, system: str, user: str, max_tokens: int = 1400,
                 temperature: float | None = None, use_cache: bool = True) -> LLMResponse:
        self.calls += 1
        self.prompt_tokens += 100
        self.completion_tokens += 50
        self.seen_prompts.append(user)
        return LLMResponse(text=json.dumps({"signals": [], "suggested_level": "L0"}),
                           model_requested=self.model, model_reported=self.reported_model,
                           prompt_tokens=100, completion_tokens=50)

    def complete_json(self, *, system: str, user: str, max_tokens: int = 1400,
                      temperature: float | None = None,
                      use_cache: bool = True) -> tuple[dict, LLMResponse]:
        self.calls += 1
        self.prompt_tokens += 100
        self.completion_tokens += 50
        self.seen_prompts.append(user)
        payload = self.script.pop(0) if self.script else {"signals": [], "suggested_level": "L0"}
        resp = LLMResponse(text=json.dumps(payload, ensure_ascii=False),
                           model_requested=self.model, model_reported=self.reported_model,
                           prompt_tokens=100, completion_tokens=50)
        return payload, resp

    def close(self) -> None:
        return None

    def price_note(self) -> dict:
        return {"model_requested": self.model, "model_reported": self.reported_model,
                "calls": self.calls}


@pytest.fixture()
def fake_llm() -> FakeLLM:
    return FakeLLM()
