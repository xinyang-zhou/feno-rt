"""Safe JSONL request recording and deterministic trace replay."""

from __future__ import annotations

import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from threading import RLock
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union
from uuid import uuid4

import numpy as np
import torch


TRACE_SCHEMA_VERSION = 1
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_jsonable(value: Any) -> Any:
    """Convert request inputs to finite, portable JSON values."""
    if torch.is_tensor(value):
        return to_jsonable(value.detach().to(device="cpu").tolist())
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("trace values must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"trace value is not JSON serializable: {type(value).__name__}")


def output_summary(output: Any) -> Dict[str, Any]:
    tensor = torch.as_tensor(output).detach().to(device="cpu").contiguous()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "checksum": float(tensor.to(dtype=torch.float64).sum().item()),
        "finite": bool(torch.isfinite(tensor).all().item()),
    }


class RequestTraceRecorder:
    """Append trace records below one server-configured directory."""

    def __init__(
        self,
        output_dir: Union[str, Path],
        *,
        max_bytes: int = 128 * 1024 * 1024,
        max_records: int = 100_000,
    ) -> None:
        if max_bytes < 1 or max_records < 1:
            raise ValueError("trace capacity limits must be positive")
        self.output_dir = Path(output_dir)
        self.max_bytes = int(max_bytes)
        self.max_records = int(max_records)
        self._lock = RLock()
        self._handle: Optional[Any] = None
        self._path: Optional[Path] = None
        self._started_at: Optional[str] = None
        self._records = 0
        self._bytes = 0
        self._completed_files = 0

    @staticmethod
    def _default_name() -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        return f"feno-trace-{stamp}"

    @classmethod
    def _validate_name(cls, name: Optional[str]) -> str:
        resolved = cls._default_name() if name is None else str(name).strip()
        if not _SAFE_NAME.fullmatch(resolved):
            raise ValueError(
                "trace name must use 1-96 ASCII letters, digits, '.', '_' or '-'"
            )
        return resolved

    def _write(self, payload: Mapping[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("request recording is not active")
        line = json.dumps(to_jsonable(payload), ensure_ascii=False, allow_nan=False)
        encoded_bytes = len((line + "\n").encode("utf-8"))
        if self._bytes + encoded_bytes > self.max_bytes:
            raise RuntimeError(f"trace reached its {self.max_bytes}-byte limit")
        self._handle.write(line + "\n")
        self._handle.flush()
        self._bytes += encoded_bytes

    def start(
        self,
        *,
        name: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            if self._handle is not None:
                raise RuntimeError("request recording is already active")
            stem = self._validate_name(name)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = (self.output_dir / f"{stem}.jsonl").resolve()
            if path.parent != self.output_dir.resolve():
                raise ValueError("trace path escaped the configured output directory")
            self._handle = path.open("x", encoding="utf-8", buffering=1)
            self._path = path
            self._started_at = _utc_now()
            self._records = 0
            self._bytes = 0
            try:
                self._write(
                    {
                        "schema_version": TRACE_SCHEMA_VERSION,
                        "kind": "metadata",
                        "created_at": self._started_at,
                        "metadata": dict(metadata or {}),
                    }
                )
            except BaseException:
                self._handle.close()
                path.unlink(missing_ok=True)
                self._handle = None
                self._path = None
                self._started_at = None
                self._bytes = 0
                raise
            return self.status()

    def record(
        self,
        request: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> bool:
        with self._lock:
            if self._handle is None:
                return False
            if self._records >= self.max_records:
                raise RuntimeError(f"trace reached its {self.max_records}-record limit")
            self._write(
                {
                    "schema_version": TRACE_SCHEMA_VERSION,
                    "kind": "request",
                    "sequence": self._records,
                    "recorded_at": _utc_now(),
                    "request": dict(request),
                    "result": dict(result),
                }
            )
            self._records += 1
            return True

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            if self._handle is None or self._path is None:
                raise RuntimeError("request recording is not active")
            path = self._path
            started_at = self._started_at
            records = self._records
            bytes_written = self._bytes
            self._handle.close()
            self._handle = None
            self._path = None
            self._started_at = None
            self._records = 0
            self._bytes = 0
            self._completed_files += 1
            return {
                "active": False,
                "file": path.name,
                "records": records,
                "bytes": bytes_written,
                "started_at": started_at,
                "stopped_at": _utc_now(),
                "completed_files": self._completed_files,
            }

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "active": self._handle is not None,
                "file": None if self._path is None else self._path.name,
                "records": self._records,
                "bytes": self._bytes,
                "started_at": self._started_at,
                "completed_files": self._completed_files,
                "max_bytes": self.max_bytes,
                "max_records": self.max_records,
            }


def load_trace(
    path: Union[str, Path],
    *,
    max_bytes: int = 128 * 1024 * 1024,
) -> List[Dict[str, Any]]:
    """Load and validate a bounded Stage 6 JSONL trace."""
    resolved = Path(path)
    size = resolved.stat().st_size
    if size > max_bytes:
        raise ValueError(f"trace exceeds the {max_bytes}-byte safety limit")
    records: List[Dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on trace line {line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"trace line {line_number} must be a JSON object")
            if record.get("schema_version") != TRACE_SCHEMA_VERSION:
                raise ValueError(f"unsupported trace schema on line {line_number}")
            if record.get("kind") not in {"metadata", "request"}:
                raise ValueError(f"invalid trace record kind on line {line_number}")
            if record["kind"] == "request":
                if not isinstance(record.get("request"), dict) or not isinstance(
                    record.get("result"), dict
                ):
                    raise ValueError(f"invalid request record on line {line_number}")
            records.append(record)
    if not any(record.get("kind") == "metadata" for record in records):
        raise ValueError("trace does not contain metadata")
    return records


def _percentile(values: Sequence[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


async def replay_trace(
    engine: Any,
    contexts: Mapping[str, Any],
    trace: Union[str, Path, Sequence[Mapping[str, Any]]],
    *,
    wave_size: int = 32,
    verify_checksums: bool = True,
    request_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """Replay request records against an engine and return correctness metrics."""
    if wave_size < 1:
        raise ValueError("wave_size must be positive")
    records = load_trace(trace) if isinstance(trace, (str, Path)) else list(trace)
    requests = [record for record in records if record.get("kind") == "request"]
    if not requests:
        raise ValueError("trace does not contain request records")
    prefix = request_prefix or f"replay-{uuid4().hex}"
    latencies_ms: List[float] = []
    checksums: List[float] = []
    checksum_errors: List[float] = []
    statuses: Counter[str] = Counter()
    workers: Counter[str] = Counter()
    started = time.perf_counter()

    async def run_one(index: int, record: Mapping[str, Any]) -> None:
        request = record["request"]
        context_id = str(request.get("context_id", ""))
        if context_id not in contexts:
            statuses["missing_context"] += 1
            return
        request_started = time.perf_counter()
        try:
            handle = await engine.submit(
                contexts[context_id],
                request["source_position"],
                request["frequency"],
                receiver_positions=request.get("receiver_positions"),
                positions_are_normalized=bool(
                    request.get("positions_are_normalized", False)
                ),
                denormalize=bool(request.get("denormalize", False)),
                cache_level=str(request.get("cache_level", "all")),
                timeout_s=request.get("timeout_s"),
                priority=int(request.get("priority", 0)),
                request_id=f"{prefix}-{index}",
            )
            output = await handle
        except Exception:
            statuses["failed"] += 1
            return
        summary = output_summary(output)
        statuses["succeeded"] += 1
        latencies_ms.append((time.perf_counter() - request_started) * 1_000.0)
        checksums.append(summary["checksum"])
        worker_id = str(getattr(handle, "worker_id", "local"))
        workers[worker_id] += 1
        expected = record.get("result", {})
        if verify_checksums and expected.get("status") == "succeeded":
            expected_checksum = expected.get("checksum")
            if expected_checksum is not None:
                scale = max(abs(float(expected_checksum)), 1.0)
                checksum_errors.append(
                    abs(summary["checksum"] - float(expected_checksum)) / scale
                )

    for offset in range(0, len(requests), wave_size):
        wave = requests[offset : offset + wave_size]
        await asyncio.gather(
            *(run_one(offset + index, record) for index, record in enumerate(wave))
        )
    elapsed_s = time.perf_counter() - started
    return {
        "trace_requests": len(requests),
        "elapsed_s": elapsed_s,
        "throughput_requests_per_second": len(requests) / max(elapsed_s, 1e-12),
        "status_counts": dict(statuses),
        "worker_counts": dict(workers),
        "latency": {
            "mean_ms": sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0,
            "p50_ms": _percentile(latencies_ms, 50) if latencies_ms else 0.0,
            "p95_ms": _percentile(latencies_ms, 95) if latencies_ms else 0.0,
            "p99_ms": _percentile(latencies_ms, 99) if latencies_ms else 0.0,
            "max_ms": max(latencies_ms) if latencies_ms else 0.0,
        },
        "aggregate_checksum": sum(checksums),
        "max_relative_checksum_error": max(checksum_errors, default=0.0),
        "verified_checksums": len(checksum_errors),
    }
