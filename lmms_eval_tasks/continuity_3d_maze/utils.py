import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import make_hooks  # noqa: E402
from topobench_eval.datasets import data_root  # noqa: E402

ENV_DATA_ROOT = data_root('continuity_3d_maze')

globals().update(make_hooks(ENV_DATA_ROOT))
