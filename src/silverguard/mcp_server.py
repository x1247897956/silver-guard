"""MCP server：把工具集暴露给外部 Agent 调用。

用官方 Python SDK（``mcp``）的 stdio 传输。启动：

    python -m silverguard.mcp_server

客户端配置示例（Claude Desktop / DSH / 任意 MCP host）：

    {
      "mcpServers": {
        "silverguard": {
          "command": "/abs/path/.venv/bin/python",
          "args": ["-m", "silverguard.mcp_server"]
        }
      }
    }

暴露的四个 tool 与内部工具层**是同一份实现**（`tools.py`），因此
schema 校验、权限最小化、幂等、降级行为在 MCP 侧完全一致——不存在
"内部一套、外部一套"的偏差。

⚠️ 边界（如实声明）：工具接口是真的，**数据是本地 mock**（SQLite 档案库）。
不接任何真实账户、通讯录或短信通道。
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .config import get_settings
from .memory import MemoryStore, seed_demo_profile
from .policy import load_policy
from .tools import SCHEMAS, ToolRegistry, ToolRuntime

log = logging.getLogger("silverguard.mcp")

TOOL_DESCRIPTIONS: dict[str, str] = {
    "check_contact": "核验一个联系人在老人记忆库中的状态：是否白名单、是否首次出现、历史被举报次数。"
                     "identifier 必须是对话轨迹里真实出现过的值，模型自造会被拒绝。",
    "check_fraud_pattern": "对给定文本做已知诈骗话术的确定性规则匹配，返回命中规则、权重和与版本号。",
    "get_elder_profile": "读取老人长期记忆：档案、白名单家属、历史被诱导事件、近期大额支出。",
    "notify_family": "（mock）向白名单家属发出风险提醒。不可逆动作：需要策略表授权等级；"
                     "TTL 内对同一对象只会真正执行一次（干预幂等）；非白名单 member_id 直接拒绝。",
}


def _schema_for(name: str) -> dict[str, Any]:
    spec = SCHEMAS[name]
    props: dict[str, Any] = {}
    required: list[str] = []
    for key, typ in spec["required"].items():
        props[key] = {"type": "string" if typ is str else "number"}
        required.append(key)
    for key, typ in spec.get("optional", {}).items():
        props[key] = {"type": "string" if typ is str else "number"}
    return {"type": "object", "properties": props, "required": required,
            "additionalProperties": False, "$schema": "http://json-schema.org/draft-07/schema#"}


def build_server(*, store: MemoryStore | None = None,
                 settings: Any = None) -> tuple[Server, ToolRegistry]:
    """构造 MCP server（MCP Python SDK 2.x 的构造器回调风格）。

    关键点：工具层与内部评测**共用同一份实现与同一份规则表**——
    否则"MCP 暴露的工具"和"Agent 真正调用的工具"会悄悄分叉，
    外部调用看到的就只是一层漂亮的假接口。
    """
    settings = settings or get_settings()
    store = store or MemoryStore(settings.db_path)
    seed_demo_profile(store)
    policy = load_policy(settings.policy_path)
    patterns_path = Path(settings.policy_path).parent / "fraud_patterns.yaml"
    rt = ToolRuntime.from_files(store, patterns_path)
    # 白名单与标识符来自记忆层：外部调用同样受"禁止自造标识"约束
    for elder in ("elder-0001",):
        rt.known_elder_ids.add(elder)
        for contact in store.list_contacts(elder):
            rt.known_identifiers.add(contact["identifier"])
    registry = ToolRegistry(rt, policy=policy)
    async def list_tools(_ctx: Any,
                         _params: types.PaginatedRequestParams | None) -> types.ListToolsResult:
        tools = []
        for name in registry.names():
            if name == "record_case":
                continue  # record_case 是内部落库动作，不对外开放
            tools.append(types.Tool(
                name=name,
                description=TOOL_DESCRIPTIONS.get(name, name),
                inputSchema=_schema_for(name),
            ))
        return types.ListToolsResult(tools=tools)

    async def call_tool(_ctx: Any,
                        params: types.CallToolRequestParams) -> types.CallToolResult:
        args = dict(params.arguments or {})
        # 外部调用方必须显式声明自己的风险等级；默认 L0 → 不可逆动作会被权限层拒绝
        level = str(args.pop("__authorized_level", "L0"))
        if params.name == "record_case":
            payload: dict[str, Any] = {"ok": False, "error": "record_case 不对外开放"}
        else:
            call = registry.call(params.name, args, level=level)
            payload = {
                "ok": call.ok,
                "result": call.result,
                "error": call.error,
                "rejected_by": call.rejected_by,
                "idempotent_skip": call.idempotent_skip,
                "degraded": call.degraded,
                "policy_version": policy.version,
            }
        return types.CallToolResult(
            content=[types.TextContent(type="text",
                                       text=json.dumps(payload, ensure_ascii=False, indent=2))],
            isError=not payload.get("ok", False),
        )

    # mcp>=2 用构造函数回调注册 handler（1.x 的装饰器风格已移除）
    server: Server = Server(
        "silverguard",
        version="0.1.0",
        instructions="银发反诈守护 Agent 的工具集（只读核验 + mock 通知）。数据为本地 mock，不接真实账户。",
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )
    return server, registry


async def _run() -> None:
    server, _registry = build_server()
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(_run())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
