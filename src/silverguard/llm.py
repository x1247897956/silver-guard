"""DeepSeek 客户端（OpenAI 兼容 /chat/completions）。

设计取舍：**刻意不引入 LangChain**。
本项目的链路是"抽证据 → 调工具 → 规则定动作"，每一步都要能被确定性回放，
用一层薄薄的 HTTP 客户端把"发了什么、收到什么、花了多少 token、耗时多少"
全部落进轨迹，比套一个编排框架更容易做归因。见 docs/design-notes.md。

⚠️ 实测事实：请求 `deepseek-chat` 时，响应体里的 `model` 字段返回 `deepseek-flash`。
评测报告里写的是**响应里的真实模型名**，不是请求名。
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

log = logging.getLogger("silverguard.llm")


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    model_requested: str
    model_reported: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    latency_ms: int = 0
    attempts: int = 1
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _extract_json(text: str) -> Any:
    """从模型输出里抠出 JSON。容忍 ```json 围栏与前后废话。"""
    if not text:
        raise LLMError("空响应")
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = cleaned.find(opener), cleaned.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError(f"无法从响应中解析 JSON：{text[:200]}")


class LLMClient:
    """带重试的同步客户端。评测是离线批处理，同步足够，也不引入并发复杂度。"""

    def __init__(self, *, api_key: str, model: str, base_url: str, temperature: float = 0.0,
                 timeout: float = 90.0, max_retries: int = 3, transport: httpx.BaseTransport | None = None,
                 response_cache: dict[str, LLMResponse] | None = None) -> None:
        if not api_key:
            raise LLMError("缺少 API key")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_retries = max_retries
        self.calls = 0
        self.failures = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reported_model = ""
        self._cache = response_cache if response_cache is not None else {}
        self.cache_hits = 0
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self._client = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout,
                                    transport=transport)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ── 核心调用 ────────────────────────────────────────────────────
    def complete(self, *, system: str, user: str, max_tokens: int = 1400,
                 temperature: float | None = None, use_cache: bool = True) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        key = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if use_cache and key in self._cache:
            self.cache_hits += 1
            cached = self._cache[key]
            self.reported_model = cached.model_reported or self.reported_model
            return cached

        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            started = time.perf_counter()
            try:
                resp = self._client.post("/chat/completions", json=payload)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise LLMError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
                data = resp.json()
                choice = (data.get("choices") or [{}])[0]
                text = (choice.get("message") or {}).get("content") or ""
                usage = data.get("usage") or {}
                out = LLMResponse(
                    text=text,
                    model_requested=self.model,
                    model_reported=str(data.get("model", "")),
                    prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                    completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                    cached_tokens=int(usage.get("prompt_cache_hit_tokens", 0) or 0),
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    attempts=attempt,
                    raw=data,
                )
                self.calls += 1
                self.prompt_tokens += out.prompt_tokens
                self.completion_tokens += out.completion_tokens
                self.reported_model = out.model_reported or self.reported_model
                if use_cache:
                    self._cache[key] = out
                return out
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                # Authentication, billing and request errors need external changes;
                # retrying them only delays an explicit degraded result.
                if isinstance(exc, httpx.HTTPStatusError):
                    status = exc.response.status_code
                    last_error = f"HTTP {status}: {exc.response.reason_phrase}"
                    if 400 <= status < 500 and status not in (408, 429):
                        break
                if attempt < self.max_retries:
                    time.sleep(0.6 * (2 ** (attempt - 1)))
                    continue
        self.failures += 1
        raise LLMError(f"模型调用失败（{attempt} 次尝试后）：{last_error}")

    def complete_json(self, *, system: str, user: str, max_tokens: int = 1400,
                      temperature: float | None = None, use_cache: bool = True) -> tuple[Any, LLMResponse]:
        resp = self.complete(system=system, user=user, max_tokens=max_tokens,
                             temperature=temperature, use_cache=use_cache)
        return _extract_json(resp.text), resp

    def price_note(self) -> dict[str, Any]:
        return {
            "model_requested": self.model,
            "model_reported": self.reported_model,
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cache_hits": self.cache_hits,
        }
