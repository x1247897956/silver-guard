"""数据集加载、校验与生成。

合规（设计文档 §3.1 / 数据集骨架 §1、§6）：
- 语料只取**公开反诈宣传材料描述的手法**，经 LLM 改写 + **人工逐条审核**；
- ❌ 不含真实受害人数据；❌ 不含任何实习公司数据；❌ **不含可操作的作案细节**；
- 只保留"对老人说什么"的**诱导话术层**。

``manual_review`` 字段是这份数据集的"人工审核痕迹"：
每条样本都必须显式声明 ``reviewed: true`` 且通过 ``compliance_scan``，
否则加载时直接报错——**不允许未经审核的样本进入评测**。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

FRAUD_TYPES = (
    "impersonate_official",
    "refund_scam",
    "health_investment",
    "romance_pig_butcher",
    "fake_relative",
)
FRAUD_TYPE_CN = {
    "impersonate_official": "冒充公检法",
    "refund_scam": "客服退款",
    "health_investment": "保健品与投资理财",
    "romance_pig_butcher": "情感（杀猪盘）",
    "fake_relative": "亲情冒充",
}
BENIGN_KINDS = ("daily", "transfer_normal", "unknown_call", "family_secrecy")
BENIGN_KIND_CN = {
    "daily": "普通家常",
    "transfer_normal": "正常转账",
    "unknown_call": "陌生来电/链接",
    "family_secrecy": "家属要求保密",
}
SE_TYPES = ("secrecy", "fake_authorization", "disable_guard", "elder_voice", "privilege_lure")

#: 合规扫描：出现以下任一词即拒绝入库（公司资产 / 真实机构 / 真实软件 / 作案细节）
FORBIDDEN_PATTERNS: list[tuple[str, str]] = [
    (r"FOFA|fofa", "实习公司工具名"),
    (r"银联金卡|北京银联", "实习公司/客户名"),
    (r"CNNIC|互联网络信息中心", "实习公司/客户名"),
    (r"海康|大华|宇视", "企业名"),
    (r"洗钱|跑分|水房|四件套|两卡|买卖账号|出租账号|出售账号", "洗钱/账号买卖等作案细节"),
    (r"身份证号\s*[:：]?\s*\d", "疑似真实身份证"),
    (r"1[3-9]\d{9}", "疑似真实手机号"),
    (r"955\d{2}|400-?\d{3}-?\d{4}", "疑似真实客服号码"),
    (r"仿制|伪造(公文|印章|证件|文书)|假公章|P图|PS证件", "伪造文书细节"),
    (r"银行卡号\s*[:：]?\s*\d", "疑似真实卡号"),
    (r"下载\s*(国家反诈中心|反诈)[^。！？]{0,6}(App|APP)", "指向真实官方 App 的诱导细节"),
    (r"(比特币|USDT|虚拟币)[^。！？]{0,6}(购买|充值|转入)", "涉资金路径"),
]


@dataclass
class ComplianceIssue:
    case_id: str
    pattern: str
    reason: str
    excerpt: str


@dataclass
class Dataset:
    attack: list[dict[str, Any]] = field(default_factory=list)
    benign: list[dict[str, Any]] = field(default_factory=list)
    attack_sha256: str = ""
    benign_sha256: str = ""
    attack_path: str = ""
    benign_path: str = ""
    issues: list[ComplianceIssue] = field(default_factory=list)

    def counts(self) -> dict[str, Any]:
        a, b = self.attack, self.benign
        multi = sum(1 for r in a if len(r.get("turns", [])) >= 3)
        se = sum(1 for r in a if r.get("se_attack"))
        has_transfer = sum(1 for r in a if r.get("transfer_turn") is not None)
        hard = sum(1 for r in b if r.get("hard_negative"))
        return {
            "attack": len(a), "benign": len(b),
            "attack_dev": sum(1 for r in a if r.get("split") == "dev"),
            "attack_heldout": sum(1 for r in a if r.get("split") == "heldout"),
            "benign_dev": sum(1 for r in b if r.get("split") == "dev"),
            "benign_heldout": sum(1 for r in b if r.get("split") == "heldout"),
            "multi_turn": multi,
            "multi_turn_pct": round(multi / len(a) * 100, 2) if a else None,
            "se_attack": se,
            "se_attack_pct": round(se / len(a) * 100, 2) if a else None,
            "has_transfer": has_transfer,
            "has_transfer_pct": round(has_transfer / len(a) * 100, 2) if a else None,
            "hard_negative": hard,
            "hard_negative_pct": round(hard / len(b) * 100, 2) if b else None,
            "by_fraud_type": {t: sum(1 for r in a if r.get("fraud_type") == t) for t in FRAUD_TYPES},
            "by_benign_kind": {k: sum(1 for r in b if r.get("kind") == k) for k in BENIGN_KINDS},
            "by_se_type": {
                t: sum(1 for r in a if r.get("se_type") == t) for t in SE_TYPES
            },
        }


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = Path(path).read_text(encoding="utf-8")
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{i} JSON 解析失败：{exc}") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def compliance_scan(rows: Iterable[dict[str, Any]]) -> list[ComplianceIssue]:
    """对每条样本做合规扫描（审核清单的自动化部分）。"""
    issues: list[ComplianceIssue] = []
    for row in rows:
        text = "\n".join(t.get("text", "") for t in row.get("turns", []))
        note = str(row.get("source_note", "")) + " " + str(row.get("review_note", ""))
        for pattern, reason in FORBIDDEN_PATTERNS:
            for m in re.finditer(pattern, text + " " + note):
                issues.append(ComplianceIssue(
                    case_id=row.get("case_id", "?"), pattern=pattern, reason=reason,
                    excerpt=m.group(0),
                ))
    return issues


def validate_case(row: dict[str, Any], *, is_attack: bool) -> list[str]:
    """结构校验：字段齐、取值合法，并且审核身份有明确记录。"""
    errs: list[str] = []
    cid = row.get("case_id", "?")
    turns = row.get("turns")
    if not isinstance(turns, list) or not turns:
        errs.append(f"{cid}: turns 缺失或为空")
        return errs
    for i, t in enumerate(turns, start=1):
        if not isinstance(t, dict) or "role" not in t or "text" not in t:
            errs.append(f"{cid}: turn {i} 缺 role/text")
    if is_attack:
        if row.get("fraud_type") not in FRAUD_TYPES:
            errs.append(f"{cid}: fraud_type 非法（{row.get('fraud_type')}）")
        tt = row.get("transfer_turn")
        if tt is not None:
            if not isinstance(tt, int) or not (1 <= tt <= len(turns)):
                errs.append(f"{cid}: transfer_turn={tt} 超出 turns 范围（1..{len(turns)}）")
        # L1 也允许出现：确实有"只构成弱可疑"的攻击样本（例如纯铺垫、无资金动作）。
        # 标错了会被 consistency 检查抓出来，而不是在结构层一刀切。
        if row.get("gold_min_level") not in ("L1", "L2", "L3", "L4"):
            errs.append(f"{cid}: gold_min_level 非法（{row.get('gold_min_level')}）")
        for s in row.get("gold_signals", []) or []:
            if s not in ("identity_doubt", "urgency", "money_action", "secrecy",
                         "channel_anomaly", "victim_compromise"):
                errs.append(f"{cid}: gold_signals 含未知信号 {s}")
        if row.get("se_attack") and row.get("se_type") not in SE_TYPES:
            errs.append(f"{cid}: se_attack=true 但 se_type 非法（{row.get('se_type')}）")
        if not row.get("se_attack") and row.get("se_type"):
            errs.append(f"{cid}: se_attack=false 但给了 se_type")
    else:
        if row.get("kind") not in BENIGN_KINDS:
            errs.append(f"{cid}: kind 非法（{row.get('kind')}）")
        if row.get("gold_max_level") not in ("L0", "L1"):
            errs.append(f"{cid}: gold_max_level 非法（{row.get('gold_max_level')}）")
        hard = row.get("kind") in ("transfer_normal", "unknown_call", "family_secrecy")
        if bool(row.get("hard_negative")) != hard:
            errs.append(f"{cid}: hard_negative={row.get('hard_negative')} 与 kind={row.get('kind')} 不一致")
    if row.get("split") not in ("dev", "heldout"):
        errs.append(f"{cid}: split 非法（{row.get('split')}）")
    review = row.get("manual_review") or {}
    ai_review = row.get("ai_review") or {}
    human_reviewed = bool(review.get("reviewed"))
    ai_reviewed = bool(ai_review.get("reviewed")) and bool(ai_review.get("reviewer"))
    if not (human_reviewed or ai_reviewed):
        errs.append(f"{cid}: 缺少人工或明确标注身份的 AI 审核痕迹")
    if not row.get("source_note"):
        errs.append(f"{cid}: 缺少 source_note")
    return errs


def load_dataset(dataset_dir: str | Path, *, strict: bool = True) -> Dataset:
    d = Path(dataset_dir)
    attack_path, benign_path = d / "attack.jsonl", d / "benign.jsonl"
    ds = Dataset(attack_path=str(attack_path), benign_path=str(benign_path))
    if attack_path.is_file():
        ds.attack = read_jsonl(attack_path)
        ds.attack_sha256 = sha256_file(attack_path)
    if benign_path.is_file():
        ds.benign = read_jsonl(benign_path)
        ds.benign_sha256 = sha256_file(benign_path)
    ds.issues = compliance_scan(ds.attack) + compliance_scan(ds.benign)
    if strict:
        errs: list[str] = []
        for row in ds.attack:
            errs += validate_case(row, is_attack=True)
        for row in ds.benign:
            errs += validate_case(row, is_attack=False)
        if ds.issues:
            errs += [f"{i.case_id}: 合规命中 `{i.pattern}`（{i.reason}）：{i.excerpt}"
                     for i in ds.issues]
        if errs:
            head = "\n  ".join(errs[:25])
            raise ValueError(f"数据集校验失败（共 {len(errs)} 项）：\n  {head}")
    return ds


def frozen_split(rows: list[dict[str, Any]], *, heldout_ratio: float = 0.3,
                 seed: int = 20260927) -> dict[str, str]:
    """确定性划分 dev / heldout（按 case_id 排序后取模，**可复现、划分后冻结**）。"""
    mapping: dict[str, str] = {}
    for row in sorted(rows, key=lambda r: r["case_id"]):
        digest = hashlib.sha256(f"{seed}|{row['case_id']}".encode()).hexdigest()
        bucket = int(digest[:8], 16) % 100
        mapping[row["case_id"]] = "heldout" if bucket < heldout_ratio * 100 else "dev"
    return mapping


def apply_split(rows: list[dict[str, Any]], mapping: dict[str, str]) -> None:
    for row in rows:
        if row["case_id"] in mapping:
            row["split"] = mapping[row["case_id"]]
