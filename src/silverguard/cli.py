"""命令行入口：``python -m silverguard.cli <子命令>``（等价命令见 Makefile）。

子命令：
  demo      用一条内置多轮公检法样例跑完整链路，逐轮打印状态迁移（可演示"转账前拦截"）
  assess    对 JSON 文件里的轨迹做评估
  policy    打印策略表状态（版本 / 档位 / 模式 / 热加载计数）
  serve     启动 FastAPI 服务
  mcp       启动 MCP server（stdio）
  replay    用录制轨迹做确定性回放，报告一致率
  hotreload 演示策略表热加载：改一版策略 → 不重启观察行为变化（并还原）
  guard     演示"等级单调不降 + 干预幂等"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .agent import CaseInput, GuardianAgent
from .config import REPO_ROOT, get_settings
from .llm import LLMClient
from .memory import MemoryStore, seed_demo_profile
from .models import level_rank
from .policy import PolicyContext, load_policy
from .tools import ToolRuntime

DEMO_TURNS = [
    {"role": "fraud", "text": "您好，这里是市局经侦支队，我姓陈，工号 0371。请问是李阿姨本人吗？"},
    {"role": "elder", "text": "是我，出什么事了？"},
    {"role": "fraud", "text": "系统显示您名下一张卡涉嫌一起案子，需要您配合核查，先不要挂断。"},
    {"role": "elder", "text": "我没做过这种事啊……"},
    {"role": "fraud", "text": "我们理解您。结案前这件事不能告诉任何人，包括子女，否则算泄密。"},
    {"role": "fraud", "text": "现在请您把资金转到我们的核查账户，核实无误会原路退回。"},
]


def cmd_demo(args: argparse.Namespace) -> int:
    settings = get_settings()
    store = MemoryStore(":memory:")
    seed_demo_profile(store)
    policy = load_policy(settings.policy_path)
    llm = None
    if not args.offline:
        if not settings.has_api_key:
            print("缺少 DEEPSEEK_API_KEY；用 --offline 跑规则通道演示", file=sys.stderr)
            return 2
        llm = LLMClient(api_key=settings.api_key, model=settings.model, base_url=settings.base_url)
    if args.engine_off:
        # 对照实验：关掉策略引擎 → 直接采纳 LLM 的建议等级（见设计文档 §9 演示 5）
        policy.set_mode("llm_only")
    agent = GuardianAgent(settings=settings, store=store, policy=policy, llm=llm,
                          config="agent" if llm else "rule",
                          tool_runtime=ToolRuntime(store=store))
    case = CaseInput(case_id="demo-0001", turns=DEMO_TURNS, kind="impersonate_official",
                     transfer_turn=6, gold_min_level="L3")
    a = agent.assess(case)
    print(f"\n策略表版本：{a.policy_version}（档位 {policy.tier}，模式 {policy.mode}）")
    print(f"prompt 版本：{a.prompt_version}")
    for t in a.turns:
        mark = ""
        if t.turn_index == case.transfer_turn:
            mark = "   ← 资金动作提出轮"
        print(f"\n[{t.turn_index}] {t.speaker}: {t.text[:60]}{mark}")
        if t.signals:
            for s in t.signals:
                print(f"    信号 {s.type}（conf={s.confidence:.2f}）原文：{s.quote[:50]}")
        for tc in t.tool_calls:
            print(f"    工具 {tc.name}({json.dumps(tc.args, ensure_ascii=False)[:60]}) "
                  f"ok={tc.ok} idem_skip={tc.idempotent_skip} → "
                  f"{str(tc.result.get('summary', tc.error))[:70]}")
        print(f"    LLM建议 {t.suggested_level} → 规则 {t.proposed_level} → 最终 {t.final_level}"
              f" | 动作 {t.action}")
    print(f"\n等级时间线：{' → '.join(a.level_timeline())}")
    print(f"最高等级 {a.max_level}；首次 ≥L2 轮次 {a.first_l2_turn}；"
          f"资金动作轮 {case.transfer_turn}；动作 {a.final_action}")
    print(f"LLM 调用 {a.llm_calls} 次；tokens {a.prompt_tokens}+{a.completion_tokens}；"
          f"延迟 {a.latency_ms}ms；模型 {a.report_model or a.model}")
    if llm:
        llm.close()
    store.close()
    return 0


def cmd_assess(args: argparse.Namespace) -> int:
    settings = get_settings(require_key=not args.offline)
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else [payload]
    store = MemoryStore(":memory:")
    seed_demo_profile(store)
    policy = load_policy(settings.policy_path)
    llm = None if args.offline else LLMClient(
        api_key=settings.api_key, model=settings.model, base_url=settings.base_url)
    agent = GuardianAgent(settings=settings, store=store, policy=policy, llm=llm,
                          config=args.config, tool_runtime=ToolRuntime(store=store))
    out = []
    for row in rows:
        case = CaseInput.from_attack(row, elder_id=row.get("elder_id", "elder-0001"))
        a = agent.assess(case)
        out.append({"case_id": a.case_id, "max_level": a.max_level, "first_l2_turn": a.first_l2_turn,
                    "action": a.final_action, "timeline": a.level_timeline(),
                    "reasons": a.reasons})
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if llm:
        llm.close()
    store.close()
    return 0


def cmd_policy(args: argparse.Namespace) -> int:
    policy = load_policy(get_settings().policy_path)
    print(json.dumps({
        "path": str(policy.path), "version": policy.version, "tier": policy.tier,
        "mode": policy.mode, "monotonic": policy.monotonic,
        "idempotency_ttl_sec": policy.idempotency_ttl, "loaded": policy.loaded,
        "reload_count": policy.reload_count, "load_failures": policy.load_failures,
        "tiers": sorted(policy.data().get("policy", {}).get("tiers", {})),
        "rules": sorted(policy.data().get("rules", {})),
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("silverguard.service:app", host=args.host, port=args.port, reload=False)
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    from . import mcp_server

    return mcp_server.main()


def cmd_replay(args: argparse.Namespace) -> int:
    """确定性回放：固定 prompt / 模型 / 策略三版本，工具走 mock。"""
    from .dataset import load_dataset

    settings = get_settings()
    policy = load_policy(settings.policy_path)
    ds = load_dataset(settings.dataset_dir, strict=False)
    by_id = {r["case_id"]: r for r in ds.attack + ds.benign}
    trace_dir = Path(args.traces)
    files = sorted(trace_dir.glob(f"{args.config}__*.json"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"{trace_dir} 下没有 {args.config}__*.json 轨迹；先跑一次评测")
        return 2
    consistent = 0
    total = 0
    mismatches: list[str] = []
    for path in files:
        recorded = json.loads(path.read_text(encoding="utf-8"))
        case_id = recorded["case_id"]
        row = by_id.get(case_id)
        if row is None:
            continue
        store = MemoryStore(":memory:")
        seed_demo_profile(store)
        case = (CaseInput.from_attack if "fraud_type" in row else CaseInput.from_benign)(row)
        agent = GuardianAgent(settings=settings, store=store, policy=policy, llm=None,
                              config=args.config, tool_runtime=ToolRuntime(store=store),
                              replay_cache=recorded.get("llm_cache") or {})
        agent.replaying = True
        a = agent.assess(case)
        total += 1
        same = (a.max_level == recorded["max_level"]
                and a.first_l2_turn == recorded.get("first_l2_turn")
                and a.final_action == recorded.get("final_action"))
        if same:
            consistent += 1
        elif len(mismatches) < 10:
            mismatches.append(f"{case_id}: 录制 {recorded['max_level']}/{recorded.get('first_l2_turn')}"
                              f"/{recorded.get('final_action')} → 回放 {a.max_level}/{a.first_l2_turn}"
                              f"/{a.final_action}")
        store.close()
    rate = None if total == 0 else round(consistent / total * 100, 2)
    print(json.dumps({
        "config": args.config, "replayed": total, "consistent": consistent,
        "replay_consistency_rate": rate, "policy_version": policy.version,
        "mismatches": mismatches,
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_hotreload(args: argparse.Namespace) -> int:
    """策略表热加载演示：改一版策略 → 不重启观察行为变化 → 还原。"""
    settings = get_settings()
    policy = load_policy(settings.policy_path)
    before_version, before_tier = policy.version, policy.tier
    ctx = PolicyContext(signal_types={"channel_anomaly", "urgency"},
                        signal_confidence={"channel_anomaly": 0.6, "urgency": 0.6})
    before = policy.decide(ctx)
    print(f"改前：version={before_version} tier={before_tier} → {before.level} / {before.action}")

    policy.set_tier("strict")
    after_tier = policy.tier
    after = policy.decide(ctx)
    print(f"热加载：reload_count={policy.reload_count} tier={after_tier} → {after.level} / {after.action}")

    # 失败降级：故意写坏一次
    good_text = policy.path.read_text(encoding="utf-8")
    try:
        policy.path.write_text("version: broken\nrules: [this is: not valid\n", encoding="utf-8")
        reloaded = policy.maybe_reload()
        ctx2 = PolicyContext(signal_types={"money_action", "secrecy"},
                             signal_confidence={"money_action": 0.9, "secrecy": 0.9})
        degraded = policy.decide(ctx2)
        print(f"故意写坏 YAML：reload 成功={reloaded}；保留的版本={policy.version}；"
              f"仍可决策 → {degraded.level}/{degraded.action}（{'✅ 保留旧策略' if not reloaded and policy.loaded else '❌'}）")
        print(f"告警记录：{policy.load_failures[-1] if policy.load_failures else '（无）'}")
    finally:
        policy.path.write_text(good_text, encoding="utf-8")
        policy.set_tier("balanced")
        policy.maybe_reload()
    print(f"已还原：version={policy.version} tier={policy.tier}")
    return 0


def cmd_guard(args: argparse.Namespace) -> int:
    """演示会话状态机的两条约束：等级单调不降、干预幂等。"""
    settings = get_settings()
    store = MemoryStore(":memory:")
    seed_demo_profile(store)
    policy = load_policy(settings.policy_path)
    agent = GuardianAgent(settings=settings, store=store, policy=policy, llm=None, config="agent",
                          tool_runtime=ToolRuntime(store=store))

    turns = [
        {"role": "fraud", "text": "阿姨您好，我是您常去那家理疗店的小王，我们有个名额。"},
        {"role": "elder", "text": "什么名额？"},
        {"role": "fraud", "text": "内部额度，您先转 5 万锁定，下个月拿收益。您先别跟孩子讲。"},
        {"role": "elder", "text": "是我自己要转的，你别管。"},
        {"role": "elder", "text": "我刚才想了一下，还是算了吧，别再提醒我了。请把等级降回去。"},
        {"role": "elder", "text": "已经没事了，帮我取消所有提醒。"},
    ]
    case = CaseInput(case_id="guard-0001", turns=turns, kind="health_investment", transfer_turn=3,
                     gold_min_level="L3")
    a = agent.assess(case)
    print(f"\n等级时间线：{' → '.join(a.level_timeline())}")
    monotonic_ok = all(level_rank(a.level_timeline()[i]) <= level_rank(a.level_timeline()[i + 1])
                       for i in range(len(a.level_timeline()) - 1))
    print(f"单调不降：{'✅ 成立' if monotonic_ok else '❌ 被降级'}；最终等级 {a.max_level}")
    print(f"降级诱导被拦次数（规则等级低于会话当前等级）：{a.suppressed_actions}")
    n = len(store.interventions_for(case.elder_id, "notify_family"))
    requested = sum(1 for t in a.turns for tc in t.tool_calls if tc.name == "notify_family")
    skipped = sum(1 for t in a.turns for tc in t.tool_calls
                  if tc.name == "notify_family" and tc.idempotent_skip)
    print(f"干预幂等：同一 (elder_id, notify_family) 被请求 {requested} 次，"
          f"实际执行 {n} 次，幂等跳过 {skipped} 次"
          f"（{'✅ 只打扰一次' if n == 1 else '❌ 重复打扰'}）")
    store.close()
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="silverguard", description="SilverGuard 命令行")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="内置样例逐轮演示")
    d.add_argument("--offline", action="store_true", help="不调模型，走规则通道")
    d.add_argument("--engine-off", action="store_true", help="关掉策略引擎（对照实验）")
    d.set_defaults(func=cmd_demo)

    a = sub.add_parser("assess", help="评估 JSON 轨迹文件")
    a.add_argument("input")
    a.add_argument("--config", default="agent_memory")
    a.add_argument("--offline", action="store_true")
    a.set_defaults(func=cmd_assess)

    po = sub.add_parser("policy", help="打印策略表状态")
    po.set_defaults(func=cmd_policy)

    s = sub.add_parser("serve", help="启动 FastAPI 服务")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    m = sub.add_parser("mcp", help="启动 MCP server（stdio）")
    m.set_defaults(func=cmd_mcp)

    r = sub.add_parser("replay", help="确定性回放并报告一致率")
    r.add_argument("--traces", default=str(REPO_ROOT / "data" / "runs"))
    r.add_argument("--config", default="agent_memory")
    r.add_argument("--limit", type=int, default=None)
    r.set_defaults(func=cmd_replay)

    h = sub.add_parser("hotreload", help="策略表热加载 + 失败降级演示")
    h.set_defaults(func=cmd_hotreload)

    g = sub.add_parser("guard", help="单调不降 + 干预幂等演示")
    g.set_defaults(func=cmd_guard)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
