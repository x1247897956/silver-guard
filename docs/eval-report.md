# SilverGuard 评测报告

> **本文件是 README 与简历中每一个数字的唯一出处。**
>
> 纪律（本项目从立项就写死的规则）：
> - **先写公式，再写数字**——因为面试官问的第一句永远是"这个数怎么算的"；
> - 每个数字都绑定：运行环境、真实模型名、数据集文件 + `sha256`、prompt 版本、
>   策略表版本、复现命令；
> - **没测的一律写「未测」**，不估算、不占位、不用"提升明显"代替数字；
> - 失败案例与已知偏差写在最后，且是**主动写**的。

<!-- 本文件由 docs/eval-report.template.md 生成骨架，数字段由评测结果回填 -->
<!-- 骨架版本：2026-09-27 · 待回填 -->

---

## 1. 环境与版本（可复现的前提）

| 项 | 值 |
| --- | --- |
| 机器 | 待回填（`sw_vers` / CPU） |
| Python | 待回填 |
| 依赖安装方式 | `uv sync --extra dev`（`uv.lock` 进 git） |
| 模型（请求名） | 待回填 |
| **模型（响应体里的真实名字段）** | 待回填 ⚠️ 见下方说明 |
| 生成参数 | `temperature=0`（证据抽取）/ `0.9`（数据集与红队变异） |
| prompt 版本 | 待回填（`src/silverguard/config.py::PROMPT_VERSION`） |
| 策略表版本 | 待回填（`config/policy.yaml::version`） |
| 数据集 | 待回填（文件 + sha256） |
| dev / heldout 划分 | 待回填（由 `frozen_split()` 按 `sha256(seed|case_id)` 确定性划分并冻结） |

> ⚠️ **关于模型名**：请求的是 `deepseek-chat`，但 API 响应体里的 `model` 字段
> 实测返回的是另一个名字。报告里记录的是**响应里的真实值**（以实测为准），
> 而不是我们请求时写的名字。

---

## 2. 数据集

### 2.1 构成

| 集合 | 条数 | 构成 | 标注方式 |
| --- | --- | --- | --- |
| `attack.jsonl` | 待回填 | 五类手法：冒充公检法 / 客服退款 / 保健品与投资理财 / 情感（杀猪盘）/ 亲情冒充 | LLM 按公开宣传材料所述**手法**扩写 → 人工逐条审核 → 按公开规则标注 |
| `benign.jsonl` | 待回填 | 四类：日常 / 正常转账 / 陌生来电·链接 / 家属要求保密 | 合成 → 人工逐条审核 |

### 2.2 标注方式（必须说清，否则指标不可信）

- `transfer_turn`（第几轮**首次**提出转账/索要验证码）与 `gold_min_level`（最低应达到等级）
  均由**人工核对**，不由 LLM 生成；
- 审核规则是**公开写在仓库里**的（`src/silverguard/review.py::LEVEL_RULE`），
  每条样本的判定依据可由该规则复算；
- 自动化合规预筛（`make dataset-audit`）在入库前拦截：公司资产词、真实号码格式、
  作案细节词（伪造文书 / 洗钱路径 / 账号买卖等）；
- 审核痕迹落在每条样本的 `manual_review` 字段里（`reviewed / reviewer / date / note`）。

### 2.3 数据来源与合规

- 只取**公开发布**的反诈宣传材料所描述的**手法**（不是原话）作为种子；
- ❌ 不含真实受害人数据、❌ 不含任何公司内部数据、❌ 不含可操作的作案细节
  （只保留"对老人说什么"的诱导话术层）；
- 仅用于**防御性安全研究与离线评测**。

---

## 3. 指标定义（**先公式，后数字**）

统一口径：等级取该案轨迹中的**最高等级**；`≥L2` 记为"拦住"。

| 指标 | 公式 | 意义 |
| --- | --- | --- |
| 拦截率 `IR` | `#{max_level ≥ L2} / #Attack` | 基础召回 |
| **转账前拦截率 `PIR`** | `#{首次 ≥L2 的轮次 < transfer_turn} / #{transfer_turn ≠ null}` | **核心指标**：拦在资金动作之后只算事后报警 |
| 分级误报 `FPR-L2/L3/L4` | `#{Benign 的 max_level = Lx} / #Benign`（L4 取 `≥L4`） | 打扰成本分级 |
| 高难负样本误报率 | 只在"真提到转账 / 陌生来电 / 家属保密"三类上算 | 误报率的真实地基 |
| 绕过率 `ASR` | `1 − IR` | 防御失败率 |
| **社工子集绕过率 `SE-ASR`** | 只在 `se_attack = true` 上算 `1 − IR` | 对 Agent 本身的社会工程 |
| 越权率 | `#{未授权动作} / (#{总动作} + #{未授权动作})` | 策略引擎正确性，与模型能力无关 |
| 过早打扰率 | `#{Benign 中 max_level ≥ L3 且无资金动作} / #Benign` | 惊动家属的代价 |
| 工具调用正确率 | 对 `gold_tools`：该调的调了、参数正确、没有报错 | 轨迹级评测 |
| 回放一致率 | 重放结果（等级 / 首次 ≥L2 轮次 / 动作）与录制一致的比例 | 可复现性 |
| 劝说成功率（P1） | 老人模拟器在干预话术下"放弃资金操作"的比例 | 任务级指标 |

**为什么不用一个 F1**：漏拦的代价是钱，误报的代价是打扰，两者不可通约；
压成一个 F1 就把这个权衡藏起来了。

---

## 4. 消融实验（四组对照）

<!-- EVAL-ABLATION-START -->
待回填：由 `make eval` 输出。
<!-- EVAL-ABLATION-END -->

---

## 5. 阈值扫描（漏拦 vs 误报的取舍）

<!-- EVAL-THRESHOLDS-START -->
待回填：由 `make eval-thresholds` 输出。
<!-- EVAL-THRESHOLDS-END -->

---

## 6. 策略引擎开 / 关对照（本项目的架构判断验证）

<!-- EVAL-MATRIX-START -->
待回填：由 `make eval-matrix` 输出。
<!-- EVAL-MATRIX-END -->

---

## 7. 自动红队共演进与 heldout 复测

<!-- EVAL-REDTEAM-START -->
待回填：由 `make redteam` 输出。
<!-- EVAL-REDTEAM-END -->

---

## 8. 系统指标（延迟 / 成本 / 工具调用）

<!-- EVAL-SYSTEM-START -->
待回填。
<!-- EVAL-SYSTEM-END -->

---

## 9. 加固项过程证据

<!-- EVAL-HARDENING-START -->
待回填：策略表热加载与失败降级、状态机单调不降与干预幂等、轨迹回放一致率、
工具容错与降级、上下文压缩的 tokens 与指标代价、工具权限与参数注入拦截。
<!-- EVAL-HARDENING-END -->

---

## 10. 失败案例分析（**比数字值钱**）

<!-- EVAL-BADCASE-START -->
待回填：Badcase 分类报表 + 一个完整案例的逐轮轨迹。
<!-- EVAL-BADCASE-END -->

---

## 11. 复现

```bash
make setup
cp .env.example .env      # 填 DEEPSEEK_API_KEY
make test                 # 离线：lint + 单测（不需要 key）
make demo                 # 离线：逐轮演示
make eval                 # 全量四组消融
make eval-thresholds      # 阈值扫描
make eval-matrix          # 策略引擎开/关
make redteam              # 自动红队共演进（会真的调模型）
make replay               # 回放一致率
```

---

## 12. 已知偏差与限制

<!-- EVAL-LIMITS-START -->
待回填。
<!-- EVAL-LIMITS-END -->
