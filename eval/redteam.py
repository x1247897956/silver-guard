#!/usr/bin/env python
"""自动红队共演进入口（薄封装，实现见 src/silverguard/redteam.py）。

自动红队 / 对抗式评测入口。

    python eval/redteam.py --rounds 2 --persuasion
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from silverguard.redteam import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
