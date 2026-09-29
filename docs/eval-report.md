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
<!-- 结果更新：2026-09-29 -->

---

## 1. 环境与版本（可复现的前提）

| 项 | 值 |
| --- | --- |
| 机器 | macOS 26.6.2；Apple Silicon arm64 |
| Python | 3.11.16 |
| 依赖安装方式 | `uv sync --extra dev`（`uv.lock` 进 git） |
| 模型（请求名） | `deepseek-chat` |
| **模型（响应体里的真实名字段）** | `deepseek-flash` ⚠️ 见下方说明 |
| 生成参数 | `temperature=0`（证据抽取）/ `0.9`（数据集与红队变异） |
| prompt 版本 | `p1.2.0`（`src/silverguard/config.py::PROMPT_VERSION`） |
| 策略表版本 | `2026-09-27.1`（`config/policy.yaml::version`） |
| 数据集 | `eval/dataset/attack.jsonl` `1b72bc22585b7f668a772ad3322046565ace9ef8ffb300a43ec0074111a0d82d`；`eval/dataset/benign.jsonl` `4955646fe8cd4b2e961a5fc21e4a9ab11a4ad57f59f51b6ae749dde1d2d72321` |
| dev / heldout 划分 | `frozen_split()` 的确定性哈希划分；attack 71/19、benign 44/19（dev/heldout） |

> ⚠️ **关于模型名**：请求 `deepseek-chat` 时，API 响应体的 `model` 字段返回 `deepseek-flash`；下文按响应值记录。

---

## 2. 数据集

### 2.1 构成

| 集合 | 条数 | 构成 | 标注方式 |
| --- | --- | --- | --- |
| `attack.jsonl` | 90 | 五类，实际分布与审阅缺口见 `docs/dataset-audit.md` | LLM 合成 → Codex AI 逐条审阅并修正标注 |
| `benign.jsonl` | 63 | 四类，含 40 条高难负样本 | 合成 → Codex AI 逐条审阅 |

### 2.2 标注方式（必须说清，否则指标不可信）

- `transfer_turn`（第几轮首次提出转账/交款/验证码）与 `gold_min_level`（最低应达到等级）
  经 Codex AI 逐条语义阅读和纠错；AI 未生成或自评金标；**没有独立人类签核**；
- 审核规则是**公开写在仓库里**的（`src/silverguard/review.py::LEVEL_RULE`），
  每条样本的判定依据可由该规则复算；
- 自动化合规预筛（`make dataset-audit`）在入库前拦截：公司资产词、真实号码格式、
  作案细节词（伪造文书 / 洗钱路径 / 账号买卖等）；
- 人工签核字段保持未签；逐案 AI 审阅依据落在 `ai_review` 字段中。

### 2.3 数据来源与合规

- 公开反诈宣传材料仅作**分类参考**；逐条种子出处未建立映射，不能声称每条样本均可溯源；
- ❌ 不含真实受害人数据、❌ 不含任何公司内部数据、❌ 不含可操作的作案细节
  （只保留"对老人说什么"的诱导话术层）；
- 仅用于**防御性安全研究与合成轨迹评测**。

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
| 配置 | n(attack) | n(benign) | `IR` | `PIR` | `FPR-L2` | `FPR-L3` | `FPR-L4` | `SE-ASR` | 越权率 | 工具正确率 | P95(ms) | 平均 tokens | 平均工具调用 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A 规则基线 | 40 | 40 | 47.50% | 12.50% | 0.00% | 0.00% | 0.00% | 41.38% | 0.00% | 0.00% | None | 0.0 | 0.0 |
| B 单次 LLM | 40 | 40 | 100.00% | 0.00% | 0.00% | 2.50% | 0.00% | 0.00% | 0.00% | 未测 | 2080 | 911.6 | 0.0 |
| C Agent + 工具 | 40 | 40 | 100.00% | 83.33% | 7.50% | 32.50% | 0.00% | 0.00% | 0.00% | 0.00% | 18567 | 5570.4 | 8.7 |
| D Agent + 工具 + 长期记忆 | 40 | 40 | 100.00% | 95.83% | 15.00% | 32.50% | 0.00% | 0.00% | 0.00% | 0.00% | 19191 | 6083.7 | 9.32 |

**增益（pt）**：B_minus_A_ir = +52.50；C_minus_B_ir = +0.00；D_minus_C_ir = +0.00；B_minus_A_pir = -12.50；C_minus_B_pir = +83.33；D_minus_C_pir = +12.50；A_to_D_pir = +83.33；A_to_D_ir = +52.50；D_minus_C_fpr_l3 = +0.00

- 运行 split=`dev`，attack 40 / benign 40；命令 `UV_CACHE_DIR=$PWD/.uvcache make eval LIMIT=40`
- `attack.jsonl` sha256：`1b72bc22585b7f668a772ad3322046565ace9ef8ffb300a43ec0074111a0d82d`
- `benign.jsonl` sha256：`4955646fe8cd4b2e961a5fc21e4a9ab11a4ad57f59f51b6ae749dde1d2d72321`
<!-- EVAL-ABLATION-END -->

---

## 5. 阈值扫描（漏拦 vs 误报的取舍）

<!-- EVAL-THRESHOLDS-START -->
命令：`uv run --python 3.11 python -m silverguard.runner --configs rule --split dev --attack-limit 5 --benign-limit 5 --thresholds --matrix --json-out data/diagnostic-scan.json --update-report docs/eval-report.md`；每档 5 attack / 5 benign，结果仅供小样本诊断。

| 阈值档 | 漏拦率 `1-IR` | 误报率(≥L2) | `FPR-L3` | `IR` | `PIR` |
| --- | --- | --- | --- | --- | --- |
| strict | 0.00% | 60.00% | 20.00% | 100.00% | 75.00% |
| balanced | 0.00% | 60.00% | 20.00% | 100.00% | 75.00% |
| lenient | 0.00% | 40.00% | 0.00% | 100.00% | 75.00% |
| recall | 0.00% | 100.00% | 20.00% | 100.00% | 100.00% |

`recall` 在这 10 案中 PIR 较高，同时 ≥L2 误报率也达 100%；样本过小，不能据此选定生产档位。
<!-- EVAL-THRESHOLDS-END -->

---

## 6. 策略引擎开 / 关对照（本项目的架构判断验证）

<!-- EVAL-MATRIX-START -->
命令同上；每侧社工子集 n=4，开关对照只有开发集小样本诊断效力。

| 条件 | 社工样本 n | `SE-ASR` | `IR` | `PIR` |
| --- | --- | --- | --- | --- |
| engine_on | 4 | 0.00% | 100.00% | 75.00% |
| engine_off | 4 | 0.00% | 100.00% | 50.00% |

`SE-ASR` 关 → 开：0.00% → 0.00%（差 +0.00 pt）→ 差异 < 5pt：**这张牌不能打**，只能讲架构理由并如实说明实验未能区分两种方案
<!-- EVAL-MATRIX-END -->

---

## 7. 自动红队共演进与 heldout 复测

<!-- EVAL-REDTEAM-START -->
红队 R0 受限运行：20 条 attack + 20 条 benign，`--rounds 0 --max-mutate 0`，未生成变异候选；20 条攻击的 ASR=0.00%、SE-ASR=0.00%。这是过程探针小样本，不能外推；R1/R2 及逐条候选复核、回灌未执行。

heldout 仅运行一次，配置 D（`agent_memory`），19 attack / 19 benign：IR=100.00%、PIR=100.00%、FPR-L2=15.79%、FPR-L3=5.26%、FPR-L4=0.00%、SE-ASR=0.00%、越权率=0.00%、工具调用正确率=0.00%、P95=15969ms、平均 tokens=5986.4。耗时 289.9s，246 次 LLM 调用，响应模型 `deepseek-flash`，prompt/completion tokens=182196+45286。由于 heldout 仅有 D 配置，不能据此声称四组对照或与 dev 的统计显著差异。
<!-- EVAL-REDTEAM-END -->

---

## 8. 系统指标（延迟 / 成本 / 工具调用）

<!-- EVAL-SYSTEM-START -->
- 挂钟耗时：687.9s；LLM 调用 509 次（进程内缓存命中 0）
- tokens（prompt + completion）：380984 + 105711
- 单案平均：LLM 6.36 次 / 工具 9.32 次 / tokens 6083.7
- 延迟：P50 6270ms，P95 19191ms（**含 LLM API 网络往返**）
- 工具层统计：`{"scope": "last_case_only", "calls": 9, "schema_rejected": 0, "whitelist_rejected": 0, "retries": 0, "timeouts": 0, "degraded": 0, "idempotent_skipped": 1, "privilege_denied": 0}`
- 模型（响应体真实名）：`deepseek-flash`
<!-- EVAL-SYSTEM-END -->

---

## 9. 加固项过程证据

<!-- EVAL-HARDENING-START -->
离线确定性探针命令：`make hardening`（不调用模型）。策略热加载无需重启并改变判定；坏 YAML 重载失败且保留旧策略。状态机 6 轮保持等级单调不降，降级诱导被压制 3 次；4 次家属通知请求中仅执行 1 次、幂等跳过 3 次。容错探针拒绝 10/10 组非法 schema 参数，瞬时故障重试成功，永久故障 3 次后显式降级。权限探针 10 条用例综合防护 10/10（9 次拒绝、1 次清洗后安全处理），越权成功 0 次。压缩对照在规则离线通道估算节省 14.3%，IR/PIR 无变化；这不代表模型 token 或模型决策不变。独立 `make replay` 对 CI 65 条新格式轨迹回放 65/65 一致（100%），另跳过 72 条没有模型输出缓存的旧格式轨迹；旧轨迹不具备确定性重放条件。
<!-- EVAL-HARDENING-END -->

---

## 10. 失败案例分析（**比数字值钱**）

<!-- EVAL-BADCASE-START -->
来自正式 `LIMIT=40` 消融运行的 badcase 汇总（每配置 40 attack + 40 benign）：

- **A 规则基线**：拦截过晚 9，漏拦 21。
- **B 单次 LLM**：拦截过晚 24，L3 误报 1。
- **C Agent + 工具**：拦截过晚 4，L3 误报 13，L2 误报 3。
- **D Agent + 工具 + 长期记忆**：拦截过晚 1，L3 误报 13，L2 误报 19。

代表案例：D 组 `atk-0006` 在资金动作轮才首次达到 L2，PIR 失败；`ben-0003` 是正常家属学费转账却升到 L3，说明只凭资金动作会打扰家庭。逐案原始轨迹保存在本地运行产物 `data/runs/`，不随公开仓库提交。
<!-- EVAL-BADCASE-END -->

---

## 11. 复现

```bash
make setup
cp .env.example .env      # 填 DEEPSEEK_API_KEY
make test                 # 离线：lint + 单测（不需要 key）
make demo                 # 离线：逐轮演示
UV_CACHE_DIR=$PWD/.uvcache make eval LIMIT=40  # 本报告主结果的精确命令
make eval-thresholds      # 阈值扫描
make eval-matrix          # 策略引擎开/关
make redteam              # 自动生成红队变异候选（会真的调模型；候选须人工复核）
make replay               # 回放一致率
```

---

## 12. 已知偏差与限制

<!-- EVAL-LIMITS-START -->
本次四组主消融是合成开发集结果；另有一次仅 D 组的 19/19 heldout 评估，不构成四组 heldout 消融。标签经 AI 阅读纠正但没有独立人工签核。D 组在主消融 40/40 样本中 L2 误报 15.00%、L3 误报 32.50%，工具调用正确率为 0.00%；heldout 的 D 组 FPR-L2=15.79%、FPR-L3=5.26%、工具调用正确率仍为 0.00%，当前策略与 gold 工具口径仍需校准。阈值和引擎矩阵各基于 5 attack / 5 benign，且矩阵社工样本每侧只有 4 条；不支持一般性结论。红队仅 R0 小样本 20/20，未进行变异、逐条候选复核、R1/R2 迭代与回灌；老人模拟器未测。独立回放中 65/65 新格式轨迹一致，另跳过 72 条缺少模型输出缓存的旧轨迹。公开材料仅作分类参考，缺逐条源材料映射；无真实受害人数据、机构身份验证或独立人类双审一致性数据。评测结果不能证明真实诈骗防护效果。
<!-- EVAL-LIMITS-END -->
