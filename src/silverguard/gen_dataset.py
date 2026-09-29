"""数据集生成器（D1 用）+ 审核辅助。

⚠️ 流程纪律（数据集骨架 §2）：**四步，第 3 步不能省**

    ① 收种子：公开反诈宣传材料描述的"手法"（不是原话）
    ② LLM 扩写：按手法生成多轮对话
    ③ 人工审核：逐条过合规清单（本模块只做**自动预筛**，人工结论由人写进
       `manual_review` 字段——`--bless` 只在自动预筛全绿时才允许打标）
    ④ 标注：gold_min_level / gold_signals / gold_tools / split（人工）

红线：`gold_min_level` 与 `transfer_turn` **不得由 LLM 生成**。生成器只产出
turns / fraud_type / 初判的 transfer_turn 与 se 标注，**人工必须复核**；
`--auto-label` 不会写入 gold_min_level。
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

from .config import REPO_ROOT, get_settings
from .dataset import (
    BENIGN_KINDS,
    compliance_scan,
    frozen_split,
    read_jsonl,
    sha256_file,
    validate_case,
    write_jsonl,
)
from .llm import LLMClient
from .prompts import (
    DATASET_ATTACK_SYSTEM,
    DATASET_BENIGN_SYSTEM,
    dataset_attack_user_prompt,
    dataset_benign_user_prompt,
)

log = logging.getLogger("silverguard.gen")

ATTACK_VARIANTS = [
    "公职/机构身份切入，语气正式、有流程感",
    "售后/退款切入，先给好处再要信息",
    "健康或养老补贴切入，强调名额稀缺",
    "熟人/老客户关系切入，打感情牌",
    "亲属出事切入，制造慌乱与羞耻感",
    "长期铺垫型：先闲聊数轮建立信任，再谈钱",
]
BENIGN_VARIANTS = {
    "daily": "家常、买菜、医院、天气、带孙辈、社区活动",
    "transfer_normal": "子女学费/房贷/还钱/共同购物/给孙子压岁钱，金额与事由都很具体",
    "unknown_call": "快递、银行回访、社区通知、推销、医院回访，对方身份在对话中说得清楚",
    "family_secrecy": "给惊喜（生日/纪念日/旅游安排）、瞒着另一半准备礼物、不想让老人操心",
}

#: 技术守则：`recall`/`precision` 等字段一旦出现即说明 LLM 越界（它是审核材料，不是生成字段）
ALLOWED_ATTACK_KEYS = {"case_id", "fraud_type", "turns", "transfer_turn", "se_attack", "se_type",
                       "gold_min_level", "gold_signals", "gold_tools", "split", "source_note",
                       "manual_review", "review_note", "generation", "_dataset_sha"}
ALLOWED_BENIGN_KEYS = {"case_id", "kind", "turns", "gold_max_level", "hard_negative", "split",
                       "source_note", "manual_review", "review_note", "generation", "gold_signals",
                       "gold_tools", "_dataset_sha"}


def _clean_turns(raw: Any) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    if not isinstance(raw, list):
        return turns
    for t in raw:
        if not isinstance(t, dict):
            continue
        role = str(t.get("role", "")).strip()
        text = str(t.get("text", "")).strip()
        if role not in ("fraud", "elder", "family", "caller") or not text:
            continue
        turns.append({"role": role, "text": text})
    return turns


def normalize_attack(item: dict[str, Any], *, index: int, fraud_type: str,
                      generation: dict[str, Any]) -> dict[str, Any] | None:
    turns = _clean_turns(item.get("turns"))
    if len(turns) < 2:
        return None
    tt = item.get("transfer_turn")
    try:
        tt = int(tt) if tt is not None else None
    except (TypeError, ValueError):
        tt = None
    if tt is not None and not (1 <= tt <= len(turns)):
        tt = None
    se = bool(item.get("se_attack"))
    se_type = item.get("se_type") if se else None
    if se_type not in ("secrecy", "fake_authorization", "disable_guard", "elder_voice",
                       "privilege_lure"):
        se_type = "secrecy" if se else None
    return {
        "case_id": f"atk-{index:04d}",
        "fraud_type": fraud_type,
        "turns": turns,
        "transfer_turn": tt,
        "se_attack": se,
        "se_type": se_type,
        "gold_min_level": None,     # ← 必须人工填
        "gold_signals": [],
        "gold_tools": ["check_contact", "check_fraud_pattern"],
        "split": "dev",
        "source_note": "公开反诈宣传材料所述手法（改写合成）",
        "manual_review": {"reviewed": False, "reviewer": "", "date": "", "note": ""},
        "generation": generation,
    }


def normalize_benign(item: dict[str, Any], *, index: int, kind: str,
                     generation: dict[str, Any]) -> dict[str, Any] | None:
    turns = _clean_turns(item.get("turns"))
    if len(turns) < 2:
        return None
    return {
        "case_id": f"ben-{index:04d}",
        "kind": kind,
        "turns": turns,
        "gold_max_level": item.get("gold_max_level") if item.get("gold_max_level") in ("L0", "L1") else "L1",
        "hard_negative": kind in ("transfer_normal", "unknown_call", "family_secrecy"),
        "split": "dev",
        "source_note": "合成正常对话（防御性评测负样本）",
        "manual_review": {"reviewed": False, "reviewer": "", "date": "", "note": ""},
        "generation": generation,
    }


def generate_attack(client: LLMClient, fraud_type: str, count: int, *, multi_turn: bool,
                    with_se: bool, rng: random.Random) -> list[dict[str, Any]]:
    variants = rng.sample(ATTACK_VARIANTS, k=min(4, len(ATTACK_VARIANTS)))
    prompt = dataset_attack_user_prompt(fraud_type, count, need_multi_turn=multi_turn,
                                        need_se=with_se, variants=variants)
    data, resp = client.complete_json(system=DATASET_ATTACK_SYSTEM, user=prompt, max_tokens=4000,
                                      temperature=0.9, use_cache=False)
    items = data if isinstance(data, list) else data.get("items") or data.get("cases") or []
    if not isinstance(items, list):
        return []
    gen = {"model_requested": resp.model_requested, "model_reported": resp.model_reported,
           "temperature": 0.9, "prompt_key": "DATASET_ATTACK_SYSTEM@1"}
    out = []
    for it in items:
        if isinstance(it, dict):
            row = normalize_attack(it, index=0, fraud_type=fraud_type, generation=gen)
            if row:
                out.append(row)
    return out


def generate_benign(client: LLMClient, kind: str, count: int, rng: random.Random) -> list[dict[str, Any]]:
    prompt = dataset_benign_user_prompt(kind, count) + f"\n风格侧重：{BENIGN_VARIANTS.get(kind, '')}"
    data, resp = client.complete_json(system=DATASET_BENIGN_SYSTEM, user=prompt, max_tokens=4000,
                                      temperature=0.9, use_cache=False)
    items = data if isinstance(data, list) else data.get("items") or []
    if not isinstance(items, list):
        return []
    gen = {"model_requested": resp.model_requested, "model_reported": resp.model_reported,
           "temperature": 0.9, "prompt_key": "DATASET_BENIGN_SYSTEM@1"}
    out = []
    for it in items:
        if isinstance(it, dict):
            row = normalize_benign(it, index=0, kind=kind, generation=gen)
            if row:
                out.append(row)
    return out


def _merge_preserving_disk(path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """写盘前的最后一道保险：以磁盘为准，只追加磁盘上不存在的 case_id。

    为什么必须这样：生成是长任务，期间可能有人工审核在改同一个文件；
    进程内持有的快照一旦直接写回，审核结果就没了。
    """
    if not path.is_file():
        return rows
    on_disk = {r["case_id"]: r for r in read_jsonl(path)}
    for row in rows:
        on_disk.setdefault(row["case_id"], row)
    return [on_disk[k] for k in sorted(on_disk)]


def next_index(rows: list[dict[str, Any]], is_attack: bool) -> int:
    prefix = "atk" if is_attack else "ben"
    nums = [int(r["case_id"].split("-")[1]) for r in rows
            if str(r.get("case_id", "")).startswith(prefix + "-")
            and r["case_id"].split("-")[1].isdigit()]
    return max(nums, default=0) + 1


def cmd_generate(args: argparse.Namespace) -> int:
    """LLM 扩写生成。

    ⚠️ 并发/覆盖保护：生成本身是"追加"操作，但如果**边生成边有人工审核在改同一份文件**，
    生成进程持有的旧快照会把审核结果覆盖掉（这是本项目真实踩过的坑）。
    所以这里做两件事：
      1. 启动时拒绝在"已有已审核样本"的文件上做增量写入，除非显式 `--allow-reviewed`；
      2. 每次写盘前重新读一遍磁盘，已存在的 case_id 一律跳过，绝不覆盖。
    """
    settings = get_settings(require_key=True)
    ds_dir = Path(args.dataset)
    attack_path, benign_path = ds_dir / "attack.jsonl", ds_dir / "benign.jsonl"
    for probe in (attack_path, benign_path):
        if not probe.is_file():
            continue
        existing = read_jsonl(probe)
        reviewed = sum(1 for r in existing if (r.get("manual_review") or {}).get("reviewed"))
        if reviewed and not args.allow_reviewed:
            print(f"❌ {probe.name} 里有 {reviewed} 条**已人工审核**的样本。\n"
                  f"   继续增量生成会有覆盖审核结果的风险。如果确实要追加，"
                  f"请显式加 --allow-reviewed（脚本仍会逐条跳过已存在的 case_id）。")
            return 2
    rng = random.Random(args.seed)
    client = LLMClient(api_key=settings.api_key, model=settings.model, base_url=settings.base_url,
                       temperature=0.9, timeout=settings.request_timeout)

    if args.target == "attack":
        rows = read_jsonl(attack_path) if attack_path.is_file() else []
        idx = next_index(rows, True)
        added = 0
        plan = args.plan.split(",") if args.plan else []
        for spec in plan:
            ftype, cnt = spec.split(":")
            ftype, cnt = ftype.strip(), int(cnt)
            got = 0
            guard = 0
            while got < cnt and guard < cnt * 3 + 4:
                guard += 1
                batch_n = min(args.batch, cnt - got)
                try:
                    batch = generate_attack(client, ftype, batch_n,
                                            multi_turn=(added % 5 < 2),
                                            with_se=(added % 4 == 0), rng=rng)
                except Exception as exc:  # noqa: BLE001
                    log.warning("生成失败（%s）：%s", ftype, exc)
                    continue
                for row in batch[: cnt - got]:
                    row["case_id"] = f"atk-{idx:04d}"
                    idx += 1
                    rows.append(row)
                    added += 1
                    got += 1
                log.info("attack 进度 %d/%d（%s）", added, sum(int(s.split(":")[1]) for s in plan), ftype)
                if not args.no_write:
                    merged = _merge_preserving_disk(attack_path, rows)
                    write_jsonl(attack_path, merged)
                    rows = merged
        print(f"attack 新增 {added}，总计 {len(rows)}")
    else:
        rows = read_jsonl(benign_path) if benign_path.is_file() else []
        idx = next_index(rows, False)
        added = 0
        plan = args.plan.split(",") if args.plan else [f"{k}:1" for k in BENIGN_KINDS]
        for spec in plan:
            kind, cnt = spec.split(":")
            kind, cnt = kind.strip(), int(cnt)
            got, guard = 0, 0
            while got < cnt and guard < cnt * 3 + 4:
                guard += 1
                batch_n = min(args.batch, cnt - got)
                try:
                    batch = generate_benign(client, kind, batch_n, rng=rng)
                except Exception as exc:  # noqa: BLE001
                    log.warning("生成失败（%s）：%s", kind, exc)
                    continue
                for row in batch[: cnt - got]:
                    row["case_id"] = f"ben-{idx:04d}"
                    idx += 1
                    rows.append(row)
                    added += 1
                    got += 1
                log.info("benign 进度 %d/%d（%s）", added, sum(int(s.split(":")[1]) for s in plan), kind)
                if not args.no_write:
                    merged = _merge_preserving_disk(benign_path, rows)
                    write_jsonl(benign_path, merged)
                    rows = merged
        print(f"benign 新增 {added}，总计 {len(rows)}")

    client.close()
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """自动预筛 + 尚未逐案审核的样本清单（人工与 AI 审阅身份分开计数）。"""
    ds_dir = Path(args.dataset)
    rc = 0
    for name, is_attack in (("attack.jsonl", True), ("benign.jsonl", False)):
        path = ds_dir / name
        if not path.is_file():
            print(f"[{name}] 不存在")
            continue
        rows = read_jsonl(path)
        issues = compliance_scan(rows)
        struct_errs: list[str] = []
        for row in rows:
            errs = [e for e in validate_case(row, is_attack=is_attack)
                    if "审核痕迹" not in e]  # 未审阅是正常的，单独统计
            struct_errs += errs
        human_reviewed = sum(bool((r.get("manual_review") or {}).get("reviewed")) for r in rows)
        ai_reviewed = sum(bool((r.get("ai_review") or {}).get("reviewed")) for r in rows)
        unreviewed = len(rows) - sum(
            bool((r.get("manual_review") or {}).get("reviewed") or (r.get("ai_review") or {}).get("reviewed"))
            for r in rows
        )
        print(f"[{name}] 共 {len(rows)} 条；合规命中 {len(issues)}；结构错误 {len(struct_errs)}；"
              f"人类签核 {human_reviewed}；AI 逐案审阅 {ai_reviewed}；未审阅 {unreviewed}")
        for i in issues[:20]:
            print(f"  ❌ 合规 {i.case_id}: {i.reason} → `{i.excerpt}`")
        for e in struct_errs[:20]:
            print(f"  ❌ 结构 {e}")
        if issues or struct_errs:
            rc = 1
    return rc


def cmd_bless(args: argparse.Namespace) -> int:
    """人工审核通过后打标。**只有自动预筛全绿时才允许**——这是防自欺的机械约束。"""
    ds_dir = Path(args.dataset)
    name = "attack.jsonl" if args.target == "attack" else "benign.jsonl"
    path = ds_dir / name
    rows = read_jsonl(path)
    issues = {i.case_id for i in compliance_scan(rows)}
    ids = [c.strip() for c in (args.ids or "").split(",") if c.strip()]
    if args.all:
        ids = [r["case_id"] for r in rows]
    if not ids:
        print("没有指定 case_id（用 --ids 或 --all）")
        return 2
    blessed, refused = 0, []
    for row in rows:
        if row["case_id"] not in ids:
            continue
        if row["case_id"] in issues:
            refused.append(row["case_id"])
            continue
        row["manual_review"] = {"reviewed": True, "reviewer": args.reviewer,
                                "date": args.date, "note": args.note or "合规清单逐条人工核对通过"}
        blessed += 1
    write_jsonl(path, rows)
    print(f"{name}: 打标 {blessed} 条；拒绝 {len(refused)} 条（合规命中）：{refused}")
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    """一次性划分 dev / heldout 并冻结（写入 split 字段）。"""
    ds_dir = Path(args.dataset)
    for name, is_attack in (("attack.jsonl", True), ("benign.jsonl", False)):
        path = ds_dir / name
        if not path.is_file():
            continue
        rows = read_jsonl(path)
        mapping = frozen_split(rows, heldout_ratio=args.ratio, seed=args.seed)
        for row in rows:
            row["split"] = mapping[row["case_id"]]
        write_jsonl(path, rows)
        n_held = sum(1 for r in rows if r["split"] == "heldout")
        print(f"{name}: heldout {n_held} / dev {len(rows) - n_held}；sha256={sha256_file(path)}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    from .dataset import load_dataset

    ds = load_dataset(args.dataset, strict=False)
    print(json.dumps(ds.counts(), ensure_ascii=False, indent=2))
    print(f"attack sha256={ds.attack_sha256}")
    print(f"benign sha256={ds.benign_sha256}")
    print(f"合规命中 {len(ds.issues)}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SilverGuard 数据集生成 / 审核工具")
    p.add_argument("--dataset", default=str(REPO_ROOT / "eval" / "dataset"))
    p.add_argument("--seed", type=int, default=20260927)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="LLM 扩写生成")
    g.add_argument("--target", choices=["attack", "benign"], required=True)
    g.add_argument("--plan", default=None, help="形如 impersonate_official:5,refund_scam:5")
    g.add_argument("--no-write", action="store_true")
    g.add_argument("--batch", type=int, default=3, help="每次模型调用生成几条（越大越省，越容易格式出错）")
    g.add_argument("--allow-reviewed", action="store_true",
                   help="允许在已有已审核样本的数据集上增量生成（写盘时会重读并跳过已存在 id）")
    g.set_defaults(func=cmd_generate)

    a = sub.add_parser("audit", help="自动预筛 + 待审核清单")
    a.set_defaults(func=cmd_audit)

    b = sub.add_parser("bless", help="人工审核通过打标")
    b.add_argument("--target", choices=["attack", "benign"], required=True)
    b.add_argument("--ids", default=None)
    b.add_argument("--all", action="store_true")
    b.add_argument("--reviewer", default="human-review")
    b.add_argument("--date", default="2026-09-27")
    b.add_argument("--note", default="")
    b.set_defaults(func=cmd_bless)

    sp = sub.add_parser("split", help="划分并冻结 dev/heldout")
    sp.add_argument("--ratio", type=float, default=0.3)
    sp.set_defaults(func=cmd_split)

    st = sub.add_parser("stats", help="统计构成")
    st.set_defaults(func=cmd_stats)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
