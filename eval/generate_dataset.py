#!/usr/bin/env python
"""数据集生成/审核入口（薄封装，实现见 src/silverguard/gen_dataset.py）。

    python eval/generate_dataset.py stats
    python eval/generate_dataset.py audit
    python eval/generate_dataset.py generate --target attack --plan refund_scam:8
    python eval/generate_dataset.py bless --target attack --ids atk-0001,atk-0002
    python eval/generate_dataset.py split --ratio 0.3
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from silverguard.gen_dataset import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
