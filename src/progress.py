"""无额外依赖的终端进度条。"""

from __future__ import annotations

import sys
import time
from typing import Optional, TextIO


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "--:--"
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return "%02d:%02d:%02d" % (hours, minutes, seconds)
    return "%02d:%02d" % (minutes, seconds)


class ProgressBar:
    """在交互式终端显示进度；重定向输出时不污染 JSON 日志。"""

    def __init__(
        self,
        total: Optional[int],
        label: str,
        *,
        stream: Optional[TextIO] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        if total is not None and total < 0:
            raise ValueError("进度总数不能小于 0")
        self.total = total
        self.label = label
        self.stream = stream or sys.stderr
        self.enabled = self.stream.isatty() if enabled is None else enabled
        self.current = 0
        self.started = time.monotonic()
        self.closed = False

    def set_status(self, status: str) -> None:
        if self.enabled:
            self._render(status)

    def update(self, amount: int = 1, *, status: str = "") -> None:
        if amount < 0:
            raise ValueError("进度增量不能小于 0")
        self.current += amount
        if self.total is not None:
            self.current = min(self.current, self.total)
        if self.enabled:
            self._render(status)

    def _render(self, status: str) -> None:
        elapsed = max(time.monotonic() - self.started, 0.001)
        rate = self.current / elapsed
        if self.total is None:
            progress = "%s" % self.current
            bar = ""
            eta = "--:--"
        else:
            ratio = self.current / self.total if self.total else 1.0
            filled = int(24 * ratio)
            bar = "[%s%s]" % ("#" * filled, "." * (24 - filled))
            progress = "%s/%s" % (self.current, self.total)
            eta = _format_duration((self.total - self.current) / rate if rate > 0 else None)
        suffix = " %s" % status if status else ""
        line = "%s %s %s %6.2f%% %8.1f/s ETA %s%s" % (
            self.label,
            bar,
            progress,
            (self.current / self.total * 100) if self.total else 0.0,
            rate,
            eta,
            suffix,
        )
        self.stream.write("\r%-160s" % line)
        self.stream.flush()

    def close(self) -> None:
        if not self.closed and self.enabled:
            self.stream.write("\n")
            self.stream.flush()
        self.closed = True
