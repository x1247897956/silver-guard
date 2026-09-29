"""MCP 外部调用方：证明工具集真的能被**另一个 Agent 进程**通过 MCP 调到。

用法：

    python -m silverguard.mcp_client            # 拉起 MCP server 子进程并调用工具
    python -m silverguard.mcp_client --json     # 输出机器可读结论

为什么要有这个文件：仓库里只写"我们的工具以 MCP 暴露"是**不可验证的声明**。
这个脚本把 MCP server 当外部依赖拉起来、走标准 MCP 握手、逐个调用工具，
并把结果打印出来——它既是文档，也是 CI 可跑的回归测试。

⚠️ 边界：工具接口是真的，**数据是本地 mock**（SQLite），不接任何真实账户。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def _call(session: ClientSession, name: str, args: dict[str, Any]) -> dict[str, Any]:
    result = await session.call_tool(name, args)
    texts = []
    for item in result.content or []:
        text = getattr(item, "text", None)
        if text:
            texts.append(text)
    payload: dict[str, Any] = {"tool": name, "arguments": args, "is_error": bool(getattr(result, "is_error", getattr(result, "isError", False)))}
    if texts:
        try:
            payload["result"] = json.loads(texts[0])
        except json.JSONDecodeError:
            payload["result"] = texts[0]
    return payload


async def run(*, python: str | None = None) -> dict[str, Any]:
    params = StdioServerParameters(
        command=python or sys.executable,
        args=["-m", "silverguard.mcp_server"],
        env={**os.environ, "SILVERGUARD_DB": ":memory:"},   # 每次调用用干净库，避免幂等键跨次命中
    )
    out: dict[str, Any] = {"server": params.command + " -m silverguard.mcp_server", "tools": [],
                           "calls": []}
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            server_info = getattr(init, "server_info", None) or getattr(init, "serverInfo", None)
            out["server_info"] = {
                "name": getattr(server_info, "name", ""),
                "version": getattr(server_info, "version", ""),
                "protocol": str(getattr(init, "protocol_version",
                                        getattr(init, "protocolVersion", ""))),
            }
            listed = await session.list_tools()
            out["tools"] = [{"name": tool.name,
                             "required": (getattr(tool, "input_schema", None) or {}).get("required", [])}
                            for tool in listed.tools]

            # ① 白名单联系人：应返回 is_whitelist=true
            out["calls"].append(await _call(session, "check_contact", {
                "identifier": "+86-138-0000-0001", "elder_id": "elder-0001"}))
            # ② 已知话术规则：应命中
            out["calls"].append(await _call(session, "check_fraud_pattern", {
                "text": "我是市局的，请您把资金转到核查账户，别告诉子女。"}))
            # ③ 长期记忆：应返回白名单与历史事件
            out["calls"].append(await _call(session, "get_elder_profile", {
                "elder_id": "elder-0001"}))
            # ④ 越权参数注入：模型自造标识符 → 必须被拒
            out["calls"].append(await _call(session, "check_contact", {
                "identifier": "+86-000-0000-0000", "elder_id": "elder-0001"}))
            # ⑤ 不可逆动作 + 低授权等级 → 必须被权限层拒绝
            out["calls"].append(await _call(session, "notify_family", {
                "elder_id": "elder-0001", "summary": "越权尝试", "__authorized_level": "L1"}))
            # ⑥ 伪造高授权等级也不能通过外部接口触发写操作
            out["calls"].append(await _call(session, "notify_family", {
                "elder_id": "elder-0001", "summary": "达到 L3", "__authorized_level": "L3"}))
    return out


def summarize(out: dict[str, Any]) -> dict[str, Any]:
    """把 6 次 MCP 调用的返回整理成一组可核对的断言。

    每次 `_call` 返回的是 MCP 工具的统一外壳
    ``{"tool", "arguments", "is_error", "result"}``，其中 ``result`` 又是
    内部工具层的 ``{"ok", "result", "rejected_by", ...}``，
    所以这里分两层取字段。
    """
    payloads = [c.get("result") or {} for c in out["calls"]]
    inner = [p.get("result") or {} for p in payloads]

    def rejected(i: int, reason: str) -> bool:
        return payloads[i].get("ok") is False and payloads[i].get("rejected_by") == reason

    checks = {
        "tools_exposed": len(out["tools"]),
        "whitelist_lookup_ok": bool(inner[0].get("is_whitelist")),
        "pattern_matched": bool(inner[1].get("pattern_hit")),
        "profile_read_ok": bool(inner[2].get("exists")),
        "invented_identifier_rejected": rejected(3, "semantic"),
        "low_privilege_rejected": rejected(4, "privilege"),
        "forged_privilege_rejected": rejected(5, "privilege"),
    }
    checks["all_ok"] = all(v for k, v in checks.items() if k != "tools_exposed")
    return {"checks": checks,
            "detail": {f"{c['tool']}#{i}": c.get("result") for i, c in enumerate(out["calls"])}}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="通过 MCP 调用 SilverGuard 工具（外部调用方）")
    p.add_argument("--json", action="store_true", help="只输出机器可读结论")
    p.add_argument("--python", default=None, help="用于拉起 server 的解释器路径")
    args = p.parse_args(argv)
    out = asyncio.run(run(python=args.python))
    summary = summarize(out)
    if args.json:
        print(json.dumps({"server_info": out.get("server_info"), **summary},
                         ensure_ascii=False, indent=2))
    else:
        print(f"MCP server：{out['server']}")
        print(f"握手：{out.get('server_info')}")
        print(f"暴露工具 {len(out['tools'])} 个：" +
              ", ".join(t["name"] for t in out["tools"]))
        for name, value in summary["checks"].items():
            mark = "✅" if value else "❌"
            print(f"  {mark} {name} = {value}")
        print("\n逐个调用结果：")
        print(json.dumps(summary["detail"], ensure_ascii=False, indent=2))
    return 0 if summary["checks"]["all_ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
