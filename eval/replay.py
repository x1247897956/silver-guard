#!/usr/bin/env python
"""轨迹确定性回放入口（薄封装，实现见 src/silverguard/replay.py）。

    python eval/replay.py --traces data/runs --config agent_memory
    python eval/replay.py --upto-turn 3      # 断点回放
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from silverguard.replay import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
