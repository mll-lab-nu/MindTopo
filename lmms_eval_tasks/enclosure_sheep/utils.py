import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import make_hooks  # noqa: E402

ENV_DATA_ROOT = (
    Path(__file__).resolve().parents[2]
    / "environments/enclosure_sheep/output"
)

globals().update(make_hooks(ENV_DATA_ROOT))
