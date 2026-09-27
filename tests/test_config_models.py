"""配置、数据模型归一化、版本号纪律测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from silverguard.config import PROMPT_VERSION, Settings, load_dotenv
from silverguard.models import (
    ALL_SIGNAL_TYPES,
    CONTEXT_SIGNAL_TYPES,
    LEVELS,
    SIGNAL_TYPES,
    Signal,
    level_at_least,
    level_rank,
)


def test_signal_vocabulary_is_the_documented_five_plus_context():
    assert SIGNAL_TYPES == ("identity_doubt", "urgency", "money_action", "secrecy", "channel_anomaly")
    assert CONTEXT_SIGNAL_TYPES == ("victim_compromise",)
    assert set(ALL_SIGNAL_TYPES) == set(SIGNAL_TYPES) | set(CONTEXT_SIGNAL_TYPES)


def test_level_helpers():
    assert [level_rank(lv) for lv in LEVELS] == [0, 1, 2, 3, 4]
    assert level_at_least("L3", "L2") and not level_at_least("L1", "L2")
    assert level_rank("bogus") == 0


def test_signal_alias_normalization():
    """LLM 常输出 authority_claim / pressure 这类名字，必须归一化而不是丢弃。"""
    assert Signal.from_raw({"type": "authority_claim", "quote": "x"}).type == "identity_doubt"
    assert Signal.from_raw({"type": "pressure"}).type == "urgency"
    assert Signal.from_raw({"type": "transfer"}).type == "money_action"
    assert Signal.from_raw({"type": "不存在的信号"}) is None


def test_signal_confidence_clamped_and_quote_truncated():
    s = Signal.from_raw({"type": "urgency", "quote": "字" * 1000, "confidence": 7})
    assert s.confidence == 1.0
    assert len(s.quote) == 400
    s2 = Signal.from_raw({"type": "urgency", "confidence": -3})
    assert s2.confidence == 0.0


def test_signal_turn_index_coercion():
    assert Signal.from_raw({"type": "urgency", "turn_index": "3"}).turn_index == 3
    assert Signal.from_raw({"type": "urgency", "turn_index": "x"}).turn_index is None


def test_prompt_version_is_declared():
    assert PROMPT_VERSION and PROMPT_VERSION.startswith("p")


def test_load_dotenv_does_not_require_file(tmp_path: Path):
    assert load_dotenv(tmp_path / "missing.env") == {}


def test_load_dotenv_parses_and_strips_quotes(tmp_path: Path):
    p = tmp_path / ".env"
    p.write_text('# comment\nFOO=bar\nQUOTED="a b"\nEMPTY=\n', encoding="utf-8")
    loaded = load_dotenv(p)
    assert loaded["FOO"] == "bar"
    assert loaded["QUOTED"] == "a b"
    assert loaded["EMPTY"] == ""


def test_settings_require_key_raises(tmp_path: Path, monkeypatch):
    from silverguard import config as cfg

    monkeypatch.setattr(cfg, "load_dotenv", lambda *a, **k: {})
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        cfg.get_settings(require_key=True)
    s = cfg.get_settings()
    assert isinstance(s, Settings) and s.has_api_key is False
