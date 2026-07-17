"""Compatibility wrapper for the null-space QP self-motion test.

Run:
    python null_qp_self_motion_test.py
"""

from __future__ import annotations

try:
    from .null_main import main
except ImportError:
    from null_main import main


if __name__ == "__main__":
    main()
