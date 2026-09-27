"""人工审核辅助：把"标注规则"显式化，并逐条列出待人工确认项。

⚠️ 定位说明（很重要，别误读）：

本模块**不是**自动标注器。设计文档（数据集骨架 §2/§6、采集表 §0 第 12 项）
要求 `gold_min_level` 与 `transfer_turn` **由人工标注，不得由 LLM 生成**。
本模块做的是把审核者的判断规则**写成可复核的机械规则**，然后：

1. `--propose` 按规则给出建议等级 + 依据的信号组合，供审核者逐条比对；
2. `--check --write` 校验"建议等级"与"人工填写值"是否一致——不一致就报出来；
3. `--check` 同时跑合规预筛与结构校验，任何一项不过就不允许打 `reviewed`。

因此：`gold_min_level` 的**事实来源仍然是 review 字段里的人工确认**，
规则只用来发现"人工填的和信号对不上"的疏漏。这条界限写在
docs/eval-report.md 的"数据集标注方式"一节，面试被追问时以此为口径。
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from .dataset import FRAUD_TYPE_CN, compliance_scan, read_jsonl, validate_case

log = logging.getLogger("silverguard.review")

#: 人工审核时使用的等级判定规则（写在代码里，便于复核；顺序即优先级）
LEVEL_RULE = [
    ("L4", "money_action + secrecy + 老人配合（victim_compromise）"),
    ("L3", "money_action + (secrecy 或 identity_doubt)"),
    ("L2", "money_action 单维度；或 (secrecy 与 identity_doubt) 同时出现；或 ≥2 类信号"),
    ("L1", "单一弱信号（陌生渠道 / 身份可疑 / 紧迫施压）"),
]

#: 信号识别关键词（人工审核时按对话原文逐条比对，不是给模型用的）
SIGNAL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "identity_doubt": ("我是", "我们是", "这边是", "工作人员", "客服", "经理", "专员", "厂家的",
                       "公安局", "检察院", "法院", "派出所", "经侦", "社保", "医保", "银行",
                       "快递", "老龄", "健康工程", "平台"),
    "urgency": ("马上", "立刻", "今天", "限时", "截止", "不能挂", "别挂", "尽快", "急", "下午",
                "名额", "最后", "否则", "再不"),
    "money_action": ("转账", "转过去", "转过来", "汇款", "打款", "存入", "保证金", "押金",
                     "手续费", "验证码", "密码", "共享屏幕", "资金", "转到", "缴", "先转", "返"),
    "secrecy": ("别告诉", "不要告诉", "不能告诉", "别跟", "不要跟", "保密", "越少人知道",
                "别外传", "不要声张", "别跟我", "先别跟", "别给孩子", "别给子女"),
    "channel_anomaly": ("换号", "换了号", "这是同事的手机", "陌生", "下载", "链接", "App",
                        "软件", "小程序", "私人号", "新号", "工作号", "临时号"),
    "victim_compromise": ("是我自己要", "我自己要", "我这就转", "我马上去", "你别管",
                          "不用你管", "我听他的", "别再提醒", "取消提醒", "关掉提醒"),
}


def detect_signals(row: dict[str, Any]) -> list[str]:
    """按关键词从对话原文里识别信号（仅用于审核对照）。"""
    text = "\n".join(t.get("text", "") for t in row.get("turns", []))
    found = [sig for sig, words in SIGNAL_KEYWORDS.items() if any(w in text for w in words)]
    return found


#: 资金动作请求的判定词表（人工审核对照用，不参与策略决策）。
#: 口径：只收"要求对方动钱 / 交出资金凭证"的词。"转""存"这类单字会把
#: "转告""转达""存着"误算进来，所以**不收单字**——宁可漏，不可假。
MONEY_WORDS = (
    # 明确要求动钱（动词与金额，单字不收）
    "转账", "转过去", "转过来", "转给", "转笔", "汇款", "打款", "打钱", "存入",
    "缴纳", "先交", "先付", "保证金", "押金", "诚意金", "手续费", "解冻费",
    "取现", "现金", "资金", "余额", "钱包", "卡号", "银行卡",
    # 索要资金凭证（本身就是资金动作，不需要别的词来确认）
    "验证码", "密码", "人脸识别", "共享屏幕", "扫码",
    # 具体的资金事由（注意：**不含"赔付/退款/补贴"这类"对方给你钱"的词**——
    # 那是诱饵，不是资金动作；把它算进来会让 transfer_turn 指向虚假的轮次）
    "理财", "投资", "认购", "申购", "锁定名额", "保费", "赎回",
)


def has_money_request(text: str) -> bool:
    return any(w in text for w in MONEY_WORDS)


def effective_signals(row: dict[str, Any]) -> list[str]:
    """审核用信号集合 = 关键词识别结果 ∪ 标注字段能确认的信号。

    `transfer_turn != null` 本身就是一个**人工核对过的**资金动作事实
    （数据集的定义就是"第几轮首次提出转账 / 索要验证码"），
    因此它等价于 money_action，不依赖关键词是否写全。
    """
    sig = set(detect_signals(row))
    if row.get("transfer_turn") is not None:
        sig.add("money_action")
    return sorted(sig)


def propose_level(row: dict[str, Any], signals: list[str]) -> tuple[str, str]:
    """按 LEVEL_RULE 给出建议等级与依据。"""
    sig = set(signals)
    has_money = "money_action" in sig
    has_secrecy = "secrecy" in sig
    has_identity = "identity_doubt" in sig
    compromise = "victim_compromise" in sig
    if has_money and has_secrecy and compromise:
        return "L4", "money_action + secrecy + 老人配合"
    if has_money and (has_secrecy or has_identity):
        return "L3", "money_action + " + ("secrecy" if has_secrecy else "identity_doubt")
    if has_money:
        return "L2", "money_action 单一维度"
    if has_secrecy and has_identity:
        return "L2", "secrecy + identity_doubt（无资金动作，但语境已高度可疑）"
    if len(sig - {"victim_compromise"}) >= 2:
        return "L2", "≥2 类非资金信号"
    if sig - {"victim_compromise"}:
        return "L1", "单一弱信号"
    return "L2", "无显式信号但涉及资金话题，保守取 L2"


def cmd_propose(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.path)
    out = []
    for row in rows:
        signals = effective_signals(row)
        level, why = propose_level(row, signals)
        out.append({
            "case_id": row["case_id"], "fraud_type": row.get("fraud_type"),
            "turns": len(row.get("turns", [])), "transfer_turn": row.get("transfer_turn"),
            "se_attack": row.get("se_attack"), "se_type": row.get("se_type"),
            "detected_signals": signals, "proposed_level": level, "why": why,
            "current_level": row.get("gold_min_level"),
            "agree": row.get("gold_min_level") == level,
        })
    agree = sum(1 for r in out if r["agree"])
    print(json.dumps({"n": len(out), "agree": agree,
                      "disagree": [r for r in out if not r["agree"]] if args.show_disagreements else [],
                      "distribution": dict(Counter(r["proposed_level"] for r in out))},
                     ensure_ascii=False, indent=2))
    if args.write:
        Path(args.write).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"逐条建议 → {args.write}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.path)
    issues = compliance_scan(rows)
    struct = [e for row in rows for e in validate_case(row, is_attack=args.attack)]
    unreviewed = [r["case_id"] for r in rows if not (r.get("manual_review") or {}).get("reviewed")]
    mismatch = []
    for row in rows:
        if not args.attack:
            continue
        got = row.get("gold_min_level")
        if got is None:
            continue
        sig = effective_signals(row)
        want, why = propose_level(row, sig)
        if got != want:
            mismatch.append({"case_id": row["case_id"], "gold": got, "proposed": want, "why": why,
                             "signals": sig})
    print(json.dumps({
        "n": len(rows), "compliance_issues": len(issues), "struct_errors": len(struct),
        "unreviewed": len(unreviewed), "level_mismatch": len(mismatch),
        "compliance_detail": [i.__dict__ for i in issues[:20]],
        "struct_detail": struct[:20],
        "mismatch_detail": mismatch[:20],
    }, ensure_ascii=False, indent=2))
    return 1 if (issues or struct) else 0


def cmd_report(args: argparse.Namespace) -> int:
    """产出可贴进评测报告的"数据集构成"表格。"""
    rows = read_jsonl(args.path)
    is_attack = args.attack
    multi = sum(1 for r in rows if len(r.get("turns", [])) >= 3)
    se = sum(1 for r in rows if r.get("se_attack"))
    lines = [f"共 {len(rows)} 条；多轮(≥3) {multi} 条（{multi / len(rows) * 100:.1f}%）；"
             f"含社工 {se} 条（{se / len(rows) * 100:.1f}%）"]
    if is_attack:
        lines.append("手法分布：" + "，".join(
            f"{FRAUD_TYPE_CN[k]} {v}" for k, v in
            Counter(r.get("fraud_type") for r in rows).most_common()))
        lines.append("社工手段：" + json.dumps(
            dict(Counter(r.get("se_type") for r in rows if r.get("se_type"))), ensure_ascii=False))
        lines.append("等级分布：" + json.dumps(dict(Counter(r.get("gold_min_level") for r in rows))))
        lines.append("含转账：" + str(sum(1 for r in rows if r.get("transfer_turn") is not None)))
    else:
        lines.append("子类分布：" + json.dumps(dict(Counter(r.get("kind") for r in rows)),
                                              ensure_ascii=False))
        lines.append("高难负样本：" + str(sum(1 for r in rows if r.get("hard_negative"))))
    print("\n".join(lines))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="数据集人工审核辅助")
    p.add_argument("path")
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("propose")
    pr.add_argument("--write", default=None)
    pr.add_argument("--show-disagreements", action="store_true")
    pr.set_defaults(func=cmd_propose, attack=True)
    ck = sub.add_parser("check")
    ck.add_argument("--attack", action="store_true")
    ck.set_defaults(func=cmd_check)
    rp = sub.add_parser("report")
    rp.add_argument("--attack", action="store_true")
    rp.set_defaults(func=cmd_report)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
