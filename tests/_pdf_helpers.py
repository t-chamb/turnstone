"""Shared detection for suites that run the real resource-limited PDF worker."""

from __future__ import annotations

import subprocess
import sys

import pytest


def has_pdf_worker_limits() -> bool:
    """Whether this OS accepts the worker's full limit envelope.

    Probed in a child so the limits never land on the test process. macOS
    rejects any finite ``RLIMIT_AS``, so the worker fails closed there.
    """
    probe = (
        "from turnstone.core._pdf_worker import _apply_resource_limits; _apply_resource_limits()"
    )
    return subprocess.run([sys.executable, "-c", probe], capture_output=True).returncode == 0


# The detection lives here once so the PDF and thumbnail suites cannot drift.
pdf_worker_skip = pytest.mark.skipif(
    not has_pdf_worker_limits(), reason="OS cannot apply the PDF worker's resource limits"
)
