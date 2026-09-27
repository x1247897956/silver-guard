"""评测包入口。

实现全部在 `src/silverguard/` 里（可安装、可复用、可单测）；
这里只放**面向仓库读者的薄入口**，让"评测在哪个文件"一眼可见：

    python eval/runner.py --configs rule --split dev
    python eval/redteam.py --rounds 2
    python eval/replay.py

等价于 `python -m silverguard.runner` / `-m silverguard.redteam` / `-m silverguard.replay`。
"""
