#!/usr/bin/env bash
# 干净环境一键跑通自检（不需要 API key，不访问网络）。
#
# 用途：验证"面试官 clone 下来能不能跑起来"这条叙事。
#   bash scripts/smoke.sh
#
# 它会做四件事：
#   1. 用 uv sync 在一个干净的解释器里装依赖（不动你的全局环境）
#   2. 跑离线单元测试
#   3. 跑三条真正体现项目主张的演示：策略热加载/失败降级、状态机、MCP 外部调用
#   4. 跑一遍 A 组规则基线评测（零模型调用）并校验 CI 门禁
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.uvcache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$ROOT/.uvpython}"
PY="${PY:-3.11}"

step() { printf '\n\033[1;36m▶ %s\033[0m\n' "$1"; }

step "1/5 安装依赖（uv sync --extra dev）"
uv sync --python "$PY" --extra dev

step "2/5 离线单元测试（ruff + pytest）"
uv run --python "$PY" --extra dev ruff check src eval tests
uv run --python "$PY" --extra dev python -m pytest -q

step "3/5 数据集合规与结构预筛（零写入）"
uv run --python "$PY" python -m silverguard.gen_dataset audit

step "4/5 机制演示：策略热加载 / 失败降级 / 状态机与幂等 / MCP 外部调用"
uv run --python "$PY" python -m silverguard.cli hotreload
uv run --python "$PY" python -m silverguard.cli guard
uv run --python "$PY" python -m silverguard.mcp_client --json

step "5/5 A 组规则基线评测 + CI 门禁（零模型调用，完全确定性）"
uv run --python "$PY" python -m silverguard.runner \
  --configs rule --split dev --ci-subset --no-traces \
  --json-out data/smoke-eval.json \
  --baseline eval/baseline.json --gate

printf '\n\033[1;32m✅ smoke 通过：干净环境可复现；以上全部离线，未使用任何 API key。\033[0m\n'
