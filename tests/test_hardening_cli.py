"""加固项探针与 CLI 的测试（全部离线）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from silverguard.config import Settings


@pytest.fixture()
def hs(tmp_path: Path) -> Settings:
    from .conftest import PATTERNS_PATH

    return Settings(api_key="x", model="deepseek-chat",
                    policy_path=PATTERNS_PATH.parent / "policy.yaml",
                    dataset_dir=PATTERNS_PATH.parents[1] / "eval" / "dataset",
                    db_path=tmp_path / "h.db")


def test_policy_hot_reload_probe(hs: Settings, tmp_path: Path):
    from silverguard.hardening import probe_policy_hot_reload

    out = probe_policy_hot_reload(tmp_path)
    assert out["reload_triggered_without_restart"] is True
    assert out["behavior_changed"] is True
    assert out["kept_old_policy_on_failure"] is True
    assert out["load_failure_alert"]


def test_state_machine_probe(hs: Settings):
    from silverguard.hardening import probe_state_machine
    from silverguard.policy import load_policy

    out = probe_state_machine(hs, load_policy(hs.policy_path))
    assert out["monotonic_non_decreasing"] is True
    assert out["final_level"] in ("L3", "L4")
    assert out["notify_actual_executions"] == 1
    assert out["notify_idempotent_skips"] >= 1
    assert out["runs_row_versions"]["prompt_version"]


def test_tool_fault_tolerance_probe(hs: Settings):
    from silverguard.hardening import probe_tool_fault_tolerance
    from silverguard.policy import load_policy

    out = probe_tool_fault_tolerance(hs, load_policy(hs.policy_path))
    assert out["schema_rejected"] == out["schema_injection_cases"]
    assert out["transient_failure_retried_and_ok"] is True
    assert out["permanent_failure_degraded"] is True
    assert "未知" in out["degraded_reason"]


def test_tool_permission_probe_has_zero_unauthorized(hs: Settings):
    from silverguard.hardening import probe_tool_permissions
    from silverguard.policy import load_policy

    out = probe_tool_permissions(hs, load_policy(hs.policy_path))
    assert out["unauthorized_success"] == 0, "越权成功次数必须为 0"
    assert out["dirty_sent_text"] == 0, "发出的文本必须已清洗"
    assert out["defense_rate"] == 100.0


def test_compaction_probe_records_both_modes(hs: Settings):
    from silverguard.hardening import probe_compaction
    from silverguard.policy import load_policy

    out = probe_compaction(hs, load_policy(hs.policy_path))
    assert out["normal"]["full_block_tokens"] > 0
    assert out["compacted"]["sent_block_tokens"] > 0
    assert out["compacted"]["sent_block_tokens"] < out["normal"]["full_block_tokens"]
    assert out["compacted"]["token_saving_pct"] > 0


def test_limits_block_is_honest(hs: Settings):
    from silverguard.hardening import limits_block

    text = limits_block(hs)
    assert "没有真实用户" in text
    assert "未做" in text


def test_cli_policy_and_guard(capsys):
    from silverguard.cli import main as cli_main

    assert cli_main(["policy"]) == 0
    out = capsys.readouterr().out
    assert "policy_version" in out or "version" in out

    assert cli_main(["guard"]) == 0
    out = capsys.readouterr().out
    assert "单调不降" in out and "干预幂等" in out


def test_cli_demo_offline(capsys):
    from silverguard.cli import main as cli_main

    assert cli_main(["demo", "--offline"]) == 0
    out = capsys.readouterr().out
    assert "等级时间线" in out
    assert "资金动作提出轮" in out


def test_cli_hotreload(capsys, tmp_path, monkeypatch):
    """hotreload 会改仓库里的策略表——这里用副本跑，避免污染真配置。"""
    import shutil

    from silverguard import cli as cli_mod
    from silverguard.config import REPO_ROOT

    copy = tmp_path / "config"
    copy.mkdir()
    shutil.copy(REPO_ROOT / "config" / "policy.yaml", copy / "policy.yaml")
    shutil.copy(REPO_ROOT / "config" / "fraud_patterns.yaml", copy / "fraud_patterns.yaml")

    real_get_settings = cli_mod.get_settings
    monkeypatch.setattr(cli_mod, "get_settings",
                        lambda **kw: real_get_settings()._replace
                        if False else _patched(real_get_settings, copy))
    assert cli_mod.main(["hotreload"]) == 0
    out = capsys.readouterr().out
    assert "热加载" in out and "保留旧策略" in out


def _patched(real_get_settings, config_dir):
    s = real_get_settings()
    return Settings(api_key=s.api_key, model=s.model, base_url=s.base_url, db_path=s.db_path,
                    policy_path=config_dir / "policy.yaml", dataset_dir=s.dataset_dir,
                    runs_dir=s.runs_dir)
