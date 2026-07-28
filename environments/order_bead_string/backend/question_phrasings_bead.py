"""
Canonical question phrasing for the order_bead_string benchmark.

The benchmark exposes two question classes, and each class now uses exactly
one phrasing to keep prompt wording stable.
"""

PHRASINGS = {
    "T_BS01_describe_sequence": [
        "Look at this 3D image of a curved string with colored beads threaded on it.\n"
        "Trace the string from one end to the other and list all bead colors in the order you encounter them.",
    ],
    "T_BS02_pair_relationship": [
        "You are shown two images of bead strings. Each string has colored beads on a curved rope.\n"
        "Compare the bead color sequences on both strings and determine their relationship.\n"
        "Ignore the shape of the string. Focus only on the sequence of bead colors.\n"
        "REVERSED means one sequence is the exact opposite order of the other; "
        "for example, RED, GREEN, BLUE, WHITE and WHITE, BLUE, GREEN, RED are REVERSED.\n"
        "CYCLIC_ROTATION means the same circular order with a different starting bead, without reversing; "
        "for example, RED, GREEN, BLUE, WHITE and BLUE, WHITE, RED, GREEN are CYCLIC_ROTATION.",
    ],
}
