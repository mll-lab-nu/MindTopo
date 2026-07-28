"""TopoBench runtime hooks for Python subprocesses."""

from __future__ import annotations

try:
    from lmms_task_manager_compat import install as install_lmms_task_manager_compat

    install_lmms_task_manager_compat()
except Exception:
    pass

try:
    from openai_chat_compat import install as install_openai_chat_compat

    install_openai_chat_compat()
except Exception:
    pass

try:
    from lmms_response_cache_compat import install as install_lmms_response_cache_compat

    install_lmms_response_cache_compat()
except Exception:
    pass
