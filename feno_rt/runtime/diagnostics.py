"""Opt-in host timeline and NVTX instrumentation; never formal timing evidence."""

from contextlib import contextmanager, nullcontext
from functools import wraps
import json
import os
from pathlib import Path
from threading import RLock, get_native_id
from time import perf_counter_ns


class EngineTrace:
    """Bounded Chrome trace events plus optional NVTX process ranges.

    NVTX start/end handles allow overlapping async requests without corrupting
    a thread-local push/pop stack. JSON timestamps use the engine's clock.
    Export only after the engine has drained. No tensor or output is retained.
    """

    def __init__(self, *, nvtx=False, max_events=100_000):
        if max_events < 1:
            raise ValueError("max_events must be positive")
        self.max_events = max_events
        self.origin_ns = perf_counter_ns()
        self.events = []
        self.dropped_events = 0
        self._requests = {}
        self._lock = RLock()
        self._nvtx = None
        if nvtx:
            import torch
            if not torch.cuda.is_available():
                raise ValueError("NVTX diagnostics require a CUDA runtime")
            self._nvtx = torch.cuda.nvtx

    def _begin(self, name, **args):
        label = f"{name}:{args['request_id']}" if "request_id" in args else name
        handle = self._nvtx.range_start(label) if self._nvtx is not None else None
        return (name, perf_counter_ns(), get_native_id(), args, handle)

    def _end(self, token, *, end_ns=None, start_ns=None, tid=None):
        name, start, thread, args, handle = token
        end = perf_counter_ns() if end_ns is None else end_ns
        if handle is not None:
            self._nvtx.range_end(handle)
        start = start if start_ns is None else start_ns
        event = dict(name=name, cat="feno.host", ph="X", pid=os.getpid(),
                     tid=thread if tid is None else tid,
                     ts=(start - self.origin_ns) / 1_000,
                     dur=max(0, end - start) / 1_000, args=args)
        with self._lock:
            if len(self.events) < self.max_events:
                self.events.append(event)
            else:
                self.dropped_events += 1

    @contextmanager
    def span(self, name, **args):
        token = self._begin(name, **args)
        try:
            yield
        finally:
            self._end(token)

    def request_enqueued(self, request):
        with self._lock:
            self._requests[request.request_id] = (
                self._begin("feno.request_e2e", request_id=request.request_id),
                self._begin("feno.queue_wait", request_id=request.request_id),
            )

    def request_started(self, request):
        with self._lock:
            entry = self._requests.get(request.request_id)
            if entry is not None and entry[1] is not None:
                self._end(entry[1], start_ns=request.enqueued_ns,
                          end_ns=request.started_ns, tid=-(request.sequence_id + 1))
                self._requests[request.request_id] = (entry[0], None)

    def request_completed(self, request, status):
        with self._lock:
            entry = self._requests.pop(request.request_id, None)
            if entry is None:
                return
            end = request.completed_ns or perf_counter_ns()
            if entry[1] is not None:
                self._end(entry[1], start_ns=request.enqueued_ns, end_ns=end,
                          tid=-(request.sequence_id + 1))
            entry[0][3]["status"] = status
            self._end(entry[0], start_ns=request.submitted_ns, end_ns=end,
                      tid=-(request.sequence_id + 1))

    def snapshot(self):
        with self._lock:
            return dict(traceEvents=list(self.events), displayTimeUnit="ms",
                        metadata=dict(result_class="diagnostic", clock="perf_counter_ns",
                                      nvtx_enabled=self._nvtx is not None,
                                      dropped_events=self.dropped_events,
                                      active_requests=len(self._requests)))

    def write(self, path):
        Path(path).write_text(json.dumps(self.snapshot(), indent=2) + "\n", encoding="utf-8")


def diagnostic_range(name):
    """Instrument synchronous methods only; disabled tracing avoids a context."""
    def decorate(fn):
        @wraps(fn)
        def wrapped(self, *args, **kwargs):
            trace = self.trace
            if trace is None:
                return fn(self, *args, **kwargs)
            with trace.span(name):
                return fn(self, *args, **kwargs)
        return wrapped
    return decorate


def span(trace, name, **args):
    return trace.span(name, **args) if trace is not None else nullcontext()
