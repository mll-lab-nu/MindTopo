"""
Canonical question phrasing for the enclosure_sheep benchmark.

The benchmark exposes four question classes, and each class now uses exactly
one phrasing to keep evaluation stable and prompt content easy to audit.
"""

from __future__ import annotations


PHRASINGS: dict[str, list[str]] = {
    "Q1_count_inside": [
        "How many sheep cannot escape to the outside? "
        "Count sheep that are trapped inside fences. "
        "Do not count sheep that are already outside all fences or sheep that can escape through a gap.",
    ],
    "Q2_escape_possibility": [
        "Which sheep can escape to the outside through gaps in the fence? List their IDs.",
    ],
    "Q3_max_cell_count": [
        "This scene shows a partitioned fenced area with labeled regions (A, B, C, ...).\n"
        "Which region contains the most sheep?",
    ],
    "Q4_fence_repair": [
        "What is the minimum number of fence gaps that must be repaired "
        "to fully close the outermost fence?",
    ],
}
