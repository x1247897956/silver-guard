"""运行配置：环境变量读取 + 路径解析。

刻意不引入 dotenv 依赖：`.env` 解析器只有十几行，避免为一个字段拉一个包。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: prompt 版本号：任何对证据抽取提示词的语义改动都必须同时提升它，
#: 否则评测报告里的"prompt 版本 ↔ 指标"对应关系会失真。
PROMPT_VERSION = "p1.2.0"

DEFAULT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com"


def load_dotenv(path: Path | None = None, *, override: bool = False) -> dict[str, str]:
    """极简 .env 解析：`KEY=VALUE`，忽略注释与空行，不去引号以外的东西。"""
    path = path or REPO_ROOT / ".env"
    loaded: dict[str, str] = {}
    if not path.is_file():
        return loaded
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


@dataclass
class Settings:
    api_key: str = ""
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    db_path: Path = field(default_factory=lambda: REPO_ROOT / "data" / "silverguard.db")
    policy_path: Path = field(default_factory=lambda: REPO_ROOT / "config" / "policy.yaml")
    dataset_dir: Path = field(default_factory=lambda: REPO_ROOT / "eval" / "dataset")
    runs_dir: Path = field(default_factory=lambda: REPO_ROOT / "data" / "runs")
    temperature: float = 0.0
    request_timeout: float = 90.0
    max_retries: int = 3

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)


def get_settings(*, require_key: bool = False) -> Settings:
    load_dotenv()
    db_raw = os.environ.get("SILVERGUARD_DB", "data/silverguard.db")
    if db_raw == ":memory:":
        # 特殊值：调用方明确要求"用完即弃"的内存库（外部 MCP 调用自检就是这么用的）
        db_path = Path(":memory:")
    else:
        db_path = Path(db_raw)
        if not db_path.is_absolute():
            db_path = REPO_ROOT / db_path
    settings = Settings(
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        model=os.environ.get("SILVERGUARD_MODEL", DEFAULT_MODEL),
        base_url=os.environ.get("SILVERGUARD_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
        db_path=db_path,
    )
    if require_key and not settings.has_api_key:
        raise RuntimeError(
            "缺少 DEEPSEEK_API_KEY：请复制 .env.example 为 .env 并填写（.env 已被 .gitignore 忽略）"
        )
    return settings
