"""Lightweight timing and memory measurement for scripts."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import psutil

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RunStats:
    elapsed_s: float
    peak_rss_bytes: int

    @property
    def peak_rss_gb(self) -> float:
        return self.peak_rss_bytes / 1e9


def process_tree_rss_bytes(root_pid: int | None = None) -> int:
    """Sum RSS for a process and all its descendants."""
    try:
        root = psutil.Process(root_pid or os.getpid())
    except psutil.NoSuchProcess:
        return 0

    total = root.memory_info().rss
    for child in root.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.NoSuchProcess:
            continue
    return total


def run_with_stats(fn: Callable[[], T], *, poll_interval_s: float = 0.05) -> tuple[T, RunStats]:
    """Run `fn` and return its result plus wall time and peak process-tree RSS."""
    peak_rss_bytes = 0
    stop = threading.Event()

    def monitor() -> None:
        nonlocal peak_rss_bytes
        while not stop.is_set():
            peak_rss_bytes = max(peak_rss_bytes, process_tree_rss_bytes())
            stop.wait(poll_interval_s)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    t0 = time.perf_counter()
    monitor_thread.start()
    try:
        result = fn()
    finally:
        stop.set()
        monitor_thread.join()
        peak_rss_bytes = max(peak_rss_bytes, process_tree_rss_bytes())

    return result, RunStats(elapsed_s=time.perf_counter() - t0, peak_rss_bytes=peak_rss_bytes)
