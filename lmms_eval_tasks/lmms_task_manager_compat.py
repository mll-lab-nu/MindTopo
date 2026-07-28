"""Runtime compatibility for lmms-eval task discovery in TopoBench runs."""

from __future__ import annotations

import os
from functools import wraps
from typing import Any, Callable


_PATCH_ATTR = "_topobench_lmms_task_manager_compat"
_FALSE_VALUES = {"0", "false", "no", "off"}


def should_include_default_tasks(include_path: Any, include_defaults: bool) -> bool:
    if include_path is None:
        return include_defaults

    override = os.environ.get("TOPOBENCH_LMMS_INCLUDE_DEFAULT_TASKS", "").strip().lower()
    if override in _FALSE_VALUES:
        return False

    return include_defaults


def _wrap_task_manager_init(original: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(original)
    def __init__(
        self: Any,
        verbosity: str = "INFO",
        include_path: Any = None,
        include_defaults: bool = True,
        model_name: str | None = None,
    ) -> Any:
        return original(
            self,
            verbosity=verbosity,
            include_path=include_path,
            include_defaults=should_include_default_tasks(include_path, include_defaults),
            model_name=model_name,
        )

    setattr(__init__, _PATCH_ATTR, True)
    return __init__


def install() -> None:
    try:
        from lmms_eval.tasks import TaskManager
    except Exception:
        return

    if not getattr(TaskManager.__init__, _PATCH_ATTR, False):
        TaskManager.__init__ = _wrap_task_manager_init(TaskManager.__init__)  # type: ignore[method-assign]
