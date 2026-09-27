# eval/ —— 评测目录

| 文件 | 作用 |
| --- | --- |
| `dataset/attack.jsonl` | 红队话术集（LLM 依公开反诈宣传材料所述**手法**改写 + 人工逐条审核） |
| `dataset/benign.jsonl` | 正常对话集（含高难负样本：真提到转账 / 陌生来电 / 家属要求保密） |
| `dataset/redteam/` | 自动红队共演进产生的变异样本（**独立落盘**，不回写原始数据集，保证 sha256 有意义） |
| `runner.py` | 评测入口（薄封装，实现见 `src/silverguard/runner.py`） |
| `redteam.py` | 自动红队共演进入口（实现见 `src/silverguard/redteam.py`） |
| `replay.py` | 轨迹确定性回放入口（实现见 `src/silverguard/replay.py`） |
| `generate_dataset.py` | 数据集生成/审核入口（实现见 `src/silverguard/gen_dataset.py`） |
| `baseline.json` | **CI 离线门禁基线**：A 组在 `--ci-subset` 确定性子集上的指标 |
| `baseline-ci.json` | **CI 全链路门禁基线**：D 组在 `--ci-subset` 上的指标（需要 API key） |

## 为什么实现放在 `src/` 而这里只有薄封装

实现要能被单元测试、被服务层复用、被 `pip install`；
而仓库读者的第一直觉是"评测在 `eval/` 下"。两边都要满足，所以这里只放入口，
并在文件头写明真实位置——避免同一份逻辑存在两个副本。

## 数据集纪律（进 git，改一次就要更新报告里的 sha256）

- `transfer_turn` / `gold_min_level` **人工核对**，不由 LLM 生成；
- 审核规则公开在 `src/silverguard/review.py::LEVEL_RULE`，可复算；
- `dev` / `heldout` 划分由 `frozen_split()` 按 `sha256(seed|case_id)` 确定性生成，**划分后冻结**；
- 入库前过合规预筛（`make dataset-audit`），命中公司资产词 / 真实号码 / 作案细节即拒绝。
