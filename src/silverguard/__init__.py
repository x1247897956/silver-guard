"""SilverGuard —— 银发反诈守护 Agent。

读取多轮对话轨迹 → 抽取风险证据 → 调用工具 → 定风险等级 → 执行分级干预。
核心设计：**LLM 只给风险建议，动作由确定性策略引擎映射**（见 policy.py）。
"""

from .config import PROMPT_VERSION, Settings, get_settings
from .models import Assessment, Signal, ToolCall, TurnRecord
from .policy import PolicyContext, PolicyEngine, load_policy

__all__ = [
    "PROMPT_VERSION",
    "Settings",
    "get_settings",
    "Assessment",
    "Signal",
    "ToolCall",
    "TurnRecord",
    "PolicyContext",
    "PolicyEngine",
    "load_policy",
]

__version__ = "0.1.0"
