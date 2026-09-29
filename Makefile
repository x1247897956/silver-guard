# SilverGuard —— 一键复现入口
#
# 为什么用 uv 而不是裸 python3：
#   本机系统 python3 是 3.9，本项目要求 >=3.10；uv 还能把依赖与解释器都固定在
#   仓库内的缓存目录里，避免污染用户 HOME（也便于在沙箱 / CI 中运行）。
#
# 环境变量说明（必须显式指定，否则 uv 会去写 ~/.cache 与 ~/.local）：
#   UV_CACHE_DIR          —— 包缓存目录（工作区内）
#   UV_PYTHON_INSTALL_DIR —— uv 自己下载的解释器目录（工作区内）

SHELL := /bin/bash
ROOT  := $(shell pwd)
PY    ?= 3.11

export UV_CACHE_DIR          := $(ROOT)/.uvcache
export UV_PYTHON_INSTALL_DIR := $(ROOT)/.uvpython
export PYTHONPATH            := $(ROOT)/src

UV     := uv run --python $(PY)
UVDEV  := uv run --python $(PY) --extra dev

DATASET_DIR ?= eval/dataset
RUNS_DIR    ?= data/runs
CONFIGS     ?= rule,single_llm,agent,agent_memory
SPLIT       ?= dev
LIMIT       ?=

.DEFAULT_GOAL := help
.PHONY: help setup check-key lint fmt test report \
        run serve demo policy hotreload guard mcp mcp-client \
        eval eval-quick eval-heldout eval-thresholds eval-matrix eval-gate replay redteam \
        dataset-stats dataset-audit dataset-gen dataset-audit-strict \
        fault-tolerance compaction replay-demo clean

help:
	@echo "SilverGuard make 目标："
	@echo "  make setup             安装依赖（uv sync，含 dev）并自检"
	@echo "  make check-key         检查 .env 里的 DEEPSEEK_API_KEY（不打印密钥）"
	@echo "  make test              lint + 单元测试（不需要 API key）"
	@echo "  make run / serve       启动 FastAPI 服务（127.0.0.1:8000）"
	@echo "  make demo              内置多轮样例逐轮演示（离线，走规则通道）"
	@echo "  make policy            打印策略表状态（版本/档位/模式/热加载计数）"
	@echo "  make hotreload         策略表热加载 + 加载失败降级演示"
	@echo "  make guard             会话状态机演示：等级单调不降 + 干预幂等"
	@echo "  make mcp               以 stdio 启动 MCP server"
	@echo "  make mcp-client        以外部调用方身份走 MCP 协议调工具并自检"
	@echo "  make eval              全量评测（四组消融，split=$(SPLIT)）+ Markdown 报告"
	@echo "  make eval-quick        小样本冒烟（--limit 3）"
	@echo "  make eval-heldout      封存的 heldout 集复测"
	@echo "  make eval-thresholds   阈值扫描（漏拦率 vs 误报率）"
	@echo "  make eval-matrix       策略引擎开/关对照（SE-ASR，本项目的架构判断验证）"
	@echo "  make eval-gate         CI 门禁：与 eval/baseline.json 比对，掉线即非零退出"
	@echo "  make report            把 data/*.json 的全部产物回填进 $(REPORT)"
	@echo "  make hardening         加固项过程证据（热加载/状态机/回放/容错/权限）"
	@echo "  make replay            轨迹确定性回放一致率"
	@echo "  make redteam           生成红队候选；逐条审核后才能进入下一轮，heldout 始终封存"
	@echo "  make fault-tolerance   工具容错与降级实验（注入工具故障）"
	@echo "  make compaction        上下文压缩对照（tokens 省多少 + 指标代价）"
	@echo "  make dataset-stats     数据集构成统计 + sha256"
	@echo "  make dataset-audit     数据集合规/结构自动预筛"
	@echo "  make clean             清理运行产物（保留数据集）"

# ── 环境 ────────────────────────────────────────────────────────────
setup:
	@command -v uv >/dev/null || { echo "❌ 需要先安装 uv：https://docs.astral.sh/uv/"; exit 2; }
	uv sync --python $(PY) --extra dev
	$(UV) python -c "import sys, silverguard, yaml, httpx, fastapi, mcp; print('python', sys.version.split()[0]); print('silverguard', silverguard.__version__, 'import ok')"
	$(UVDEV) python -m pytest -q --collect-only >/dev/null && echo "✅ pytest 可用"
	@test -f .env && echo "✅ .env 存在（已 gitignore）" || echo "⚠️  缺少 .env：cp .env.example .env 并填 DEEPSEEK_API_KEY"

check-key:
	@test -f .env || { echo "❌ 缺少 .env：cp .env.example .env 并填写 DEEPSEEK_API_KEY"; exit 2; }
	@grep -q '^DEEPSEEK_API_KEY=.\+' .env || { echo "❌ .env 里 DEEPSEEK_API_KEY 为空"; exit 2; }
	@echo "✅ .env 已就绪（密钥不会被打印）"

lint:
	$(UVDEV) ruff check src eval tests

fmt:
	$(UVDEV) ruff check --fix src eval tests

test:
	$(UVDEV) ruff check src eval tests
	$(UVDEV) python -m pytest -q

# ── 服务与演示 ──────────────────────────────────────────────────────
run serve:
	$(UV) python -m silverguard.cli serve --host 127.0.0.1 --port 8000

demo:
	$(UV) python -m silverguard.cli demo --offline

policy:
	$(UV) python -m silverguard.cli policy

hotreload:
	$(UV) python -m silverguard.cli hotreload

guard:
	$(UV) python -m silverguard.cli guard

mcp:
	$(UV) python -m silverguard.cli mcp

mcp-client:
	$(UV) python -m silverguard.mcp_client

# ── 评测 ────────────────────────────────────────────────────────────
REPORT ?= docs/eval-report.md

eval: check-key
	$(UV) python -m silverguard.runner --configs $(CONFIGS) --split $(SPLIT) \
		$(if $(LIMIT),--limit $(LIMIT),) \
		--traces-dir $(RUNS_DIR) \
		--json-out data/eval-$(SPLIT).json \
		--out data/eval-$(SPLIT).md \
		--update-report $(REPORT)

# 把全部产物一次性回填进评测报告（会在报告锚点区块里覆盖上一次的内容）
report: check-key
	$(UV) python -m silverguard.runner --configs $(CONFIGS) --split $(SPLIT) \
		--json-out data/eval-$(SPLIT).json --out data/eval-$(SPLIT).md \
		--update-report $(REPORT) \
		$(foreach f,thresholds matrix redteam hardening limits,$$(test -f data/$(f).json && echo --extra-json $(f)=data/$(f).json))

eval-quick: check-key
	$(UV) python -m silverguard.runner --configs $(CONFIGS) --split $(SPLIT) --limit 3 --no-traces

eval-heldout: check-key
	$(UV) python -m silverguard.runner --configs $(CONFIGS) --split heldout \
		--json-out data/eval-heldout.json --out data/eval-heldout.md

# 阈值扫描与引擎开关对照跑在**确定性子集**上：成本可控，且给出的是取舍曲线的形状。
# 全量版把 --ci-subset 去掉即可（时间成本约为 4 倍）。
eval-thresholds: check-key
	$(UV) python -m silverguard.runner --configs rule --split dev --ci-subset --thresholds \
		--json-out data/thresholds.json --out data/eval-thresholds.md \
		--update-report $(REPORT)

eval-matrix: check-key
	$(UV) python -m silverguard.runner --configs agent_memory --split dev --ci-subset --matrix \
		--json-out data/matrix.json --out data/eval-matrix.md \
		--update-report $(REPORT)

eval-gate: check-key
	$(UV) python -m silverguard.runner --configs rule --split dev --ci-subset --no-traces \
		--json-out data/eval-ci.json --baseline eval/baseline.json --gate

compaction: check-key
	$(UV) python -m silverguard.runner --configs agent_memory --split dev --compaction \
		--json-out data/eval-compaction.json --out data/eval-compaction.md

fault-tolerance: check-key
	$(UV) python -m silverguard.runner --configs agent_memory --split dev \
		--tool-fail check_contact,notify_family --no-traces \
		--json-out data/eval-fault.json --out data/eval-fault.md

redteam: check-key
	$(UV) python -m silverguard.redteam --rounds 2 --traces-dir $(RUNS_DIR) \
		--max-mutate 20 --persuasion \
		--json-out data/redteam.json --out data/redteam.md

hardening: check-key
	$(UV) python -m silverguard.hardening --json-out data/hardening.json

limits:
	$(UV) python -m silverguard.hardening --limits-only --json-out data/limits.json

replay:
	$(UV) python -m silverguard.replay --traces $(RUNS_DIR) --config agent_memory

replay-demo: replay

# ── 数据集 ──────────────────────────────────────────────────────────
dataset-stats:
	$(UV) python -m silverguard.gen_dataset stats

dataset-audit:
	$(UV) python -m silverguard.gen_dataset audit

dataset-gen:
	$(UV) python -m silverguard.gen_dataset generate --target attack \
		--plan impersonate_official:8,refund_scam:8,health_investment:8,romance_pig_butcher:8,fake_relative:8 --batch 2

clean:
	rm -rf data/runs data/*.db data/*.db-wal data/*.db-shm data/*.json data/*.md
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
