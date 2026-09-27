#!/usr/bin/env python
"""评测 runner 入口（薄封装，实现见 src/silverguard/runner.py）。

    python eval/runner.py --configs rule,single_llm,agent,agent_memory --split dev
    python eval/runner.py --thresholds --json-out data/eval-thresholds.json
    python eval/runner.py --matrix
    python eval/runner.py --gate --baseline eval/baseline.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from silverguard.runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
