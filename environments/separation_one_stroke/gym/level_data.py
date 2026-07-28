from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional


BUILTIN_LEVELS: List[Dict[str, Any]] = [
    {
        "W": 3,
        "H": 3,
        "cells": [
            ["red", None],
            [None, "blue"],
        ],
        "solution": ["U", "R", "R", "U"],
    },
    {
        "W": 3,
        "H": 3,
        "cells": [
            ["red", None],
            ["red", "blue"],
        ],
        "solution": ["R", "U", "U", "R"],
    },
    {
        "W": 3,
        "H": 3,
        "cells": [
            [None, "blue"],
            ["red", "blue"],
        ],
        "solution": ["U", "R", "U", "R"],
    },
    {
        "W": 4,
        "H": 4,
        "cells": [
            ["red", None, "red"],
            [None, None, None],
            [None, "blue", "blue"],
        ],
        "solution": ["U", "U", "R", "R", "R", "U"],
    },
    {
        "W": 4,
        "H": 4,
        "cells": [
            [None, None, "red"],
            ["blue", "blue", None],
            ["blue", None, "red"],
        ],
        "solution": ["U", "R", "R", "U", "U", "R"],
    },
    {
        "W": 4,
        "H": 4,
        "cells": [
            ["red", "blue", "blue"],
            ["red", "blue", None],
            ["red", "red", "blue"],
        ],
        "solution": ["R", "U", "U", "R", "U", "R"],
    },
    {
        "W": 4,
        "H": 4,
        "cells": [
            ["red", "green", None],
            [None, "green", None],
            ["blue", None, "blue"],
        ],
        "solution": ["R", "U", "L", "U", "R", "R", "R", "U"],
    },
    {
        "W": 4,
        "H": 4,
        "cells": [
            ["blue", None, "red"],
            [None, "red", None],
            ["green", "green", "red"],
        ],
        "solution": ["R", "U", "L", "U", "R", "R", "U", "R"],
    },
    {
        "W": 4,
        "H": 4,
        "cells": [
            ["green", None, "red"],
            ["green", None, "green"],
            [None, "green", "blue"],
        ],
        "solution": ["R", "U", "R", "R", "U", "L", "U", "R"],
    },
    {
        "W": 5,
        "H": 5,
        "cells": [
            ["blue", "blue", None, "blue"],
            [None, None, None, None],
            ["red", "red", None, None],
            [None, None, None, "blue"],
        ],
        "solution": ["U", "U", "R", "R", "U", "U", "R", "R"],
    },
    {
        "W": 5,
        "H": 5,
        "cells": [
            ["red", None, "blue", None],
            [None, "red", "blue", "blue"],
            [None, "red", None, "blue"],
            ["red", None, "blue", None],
        ],
        "solution": ["R", "U", "R", "U", "U", "U", "R", "R"],
    },
    {
        "W": 5,
        "H": 5,
        "cells": [
            ["red", "red", "red", "red"],
            ["blue", "blue", "blue", "blue"],
            ["blue", None, "blue", None],
            ["blue", None, "blue", None],
        ],
        "solution": ["U", "R", "R", "R", "R", "U", "U", "U"],
    },
    {
        "W": 5,
        "H": 5,
        "cells": [
            ["red", "green", None, "green"],
            [None, "green", None, "green"],
            ["blue", None, "green", None],
            ["blue", None, None, None],
        ],
        "solution": ["R", "U", "L", "U", "R", "U", "U", "R", "R", "R"],
    },
    {
        "W": 5,
        "H": 5,
        "cells": [
            ["blue", None, "red", None],
            ["red", None, "red", None],
            ["green", "red", None, "red"],
            ["green", "green", None, "red"],
        ],
        "solution": ["R", "U", "L", "U", "R", "U", "R", "U", "R", "R"],
    },
    {
        "W": 5,
        "H": 5,
        "cells": [
            ["green", "blue", "blue", "blue"],
            ["blue", None, None, None],
            ["red", "red", None, "blue"],
            ["red", "red", "blue", "blue"],
        ],
        "solution": ["R", "U", "L", "U", "R", "R", "U", "U", "R", "R"],
    },
    {
        "W": 6,
        "H": 6,
        "cells": [
            ["red", None, "red", None, "red"],
            [None, "red", None, "red", None],
            ["red", None, "red", None, "red"],
            [None, None, None, None, None],
            [None, "blue", "blue", "blue", None],
        ],
        "solution": ["U", "U", "U", "U", "R", "R", "R", "R", "U", "R"],
    },
    {
        "W": 6,
        "H": 6,
        "cells": [
            [None, "blue", None, "blue", None],
            ["red", "blue", None, "blue", None],
            ["red", "blue", None, "blue", None],
            ["red", "blue", None, "blue", None],
            [None, "red", "blue", None, "blue"],
        ],
        "solution": ["U", "R", "U", "U", "U", "R", "U", "R", "R", "R"],
    },
    {
        "W": 6,
        "H": 6,
        "cells": [
            [None, None, None, None, "blue"],
            [None, "red", "red", "blue", "blue"],
            ["red", "red", "red", "red", "blue"],
            ["red", "red", "red", "red", "blue"],
            [None, "red", "red", None, None],
        ],
        "solution": ["U", "R", "R", "R", "U", "R", "U", "U", "U", "R"],
    },
    {
        "W": 6,
        "H": 6,
        "cells": [
            ["blue", "blue", "red", "red", "red"],
            ["blue", "blue", "blue", "blue", "red"],
            ["blue", "blue", "blue", None, "red"],
            ["blue", "blue", "blue", None, "red"],
            ["blue", None, "blue", None, "red"],
        ],
        "solution": ["R", "R", "U", "R", "R", "U", "U", "U", "U", "R"],
    },
    {
        "W": 6,
        "H": 6,
        "cells": [
            [None, "blue", "blue", "blue", "blue"],
            ["red", None, "red", None, "blue"],
            ["red", None, "red", "red", "red"],
            ["red", "red", "red", "red", "red"],
            ["red", "red", "red", "red", "red"],
        ],
        "solution": ["U", "R", "R", "R", "U", "R", "R", "U", "U", "U"],
    },
]


def get_builtin_level(index: int) -> Optional[Dict[str, Any]]:
    if index < 0 or index >= len(BUILTIN_LEVELS):
        return None
    return deepcopy(BUILTIN_LEVELS[index])


def get_builtin_level_count() -> int:
    return len(BUILTIN_LEVELS)


def count_level_colors(level: Dict[str, Any]) -> int:
    colors = set()
    for row in level.get("cells", []):
        for cell in row:
            if cell:
                colors.add(str(cell))
    return len(colors)
