"""Safe start/stop lifecycle for on-demand PyTorch traces."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from threading import RLock
from typing import Any, Callable, Dict, Optional, Sequence

import torch


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")


class RuntimeProfiler:
    """Own at most one profiler and export traces below one configured directory."""

    def __init__(
        self,
        output_dir: Path,
        *,
        include_cuda: bool = True,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
        profiler_factory: Optional[Callable[[Sequence[Any]], Any]] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.include_cuda = bool(include_cuda)
        self.record_shapes = bool(record_shapes)
        self.profile_memory = bool(profile_memory)
        self.with_stack = bool(with_stack)
        self._profiler_factory = profiler_factory
        self._lock = RLock()
        self._profile: Optional[Any] = None
        self._path: Optional[Path] = None
        self._started_at: Optional[str] = None
        self._completed = 0

    @staticmethod
    def _default_name() -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        return f"feno-profile-{stamp}"

    @classmethod
    def _validate_name(cls, name: Optional[str]) -> str:
        resolved = cls._default_name() if name is None else str(name).strip()
        if not _SAFE_NAME.fullmatch(resolved):
            raise ValueError(
                "profile name must use 1-96 ASCII letters, digits, '.', '_' or '-'"
            )
        return resolved

    def _make_profiler(self, activities: Sequence[Any]) -> Any:
        if self._profiler_factory is not None:
            return self._profiler_factory(activities)
        return torch.profiler.profile(
            activities=list(activities),
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
        )

    def start(self, *, name: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            if self._profile is not None:
                raise RuntimeError("a profiler session is already active")
            stem = self._validate_name(name)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = (self.output_dir / f"{stem}.json").resolve()
            if path.parent != self.output_dir.resolve():
                raise ValueError("profile path escaped the configured output directory")
            if path.exists():
                raise FileExistsError(f"profile already exists: {path.name}")
            activities = [torch.profiler.ProfilerActivity.CPU]
            if self.include_cuda and torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            profile = self._make_profiler(activities)
            profile.start()
            self._profile = profile
            self._path = path
            self._started_at = datetime.now(timezone.utc).isoformat()
            return self.status()

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            if self._profile is None or self._path is None:
                raise RuntimeError("no profiler session is active")
            profile = self._profile
            path = self._path
            started_at = self._started_at
            try:
                profile.stop()
                profile.export_chrome_trace(str(path))
            finally:
                self._profile = None
                self._path = None
                self._started_at = None
            self._completed += 1
            return {
                "active": False,
                "file": path.name,
                "started_at": started_at,
                "stopped_at": datetime.now(timezone.utc).isoformat(),
                "completed_sessions": self._completed,
            }

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "active": self._profile is not None,
                "file": None if self._path is None else self._path.name,
                "started_at": self._started_at,
                "completed_sessions": self._completed,
                "include_cuda": self.include_cuda,
            }
