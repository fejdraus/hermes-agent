"""Корневой спан обхода не становится текущим в контексте OpenTelemetry.

Ход начинается в одном потоке и завершается в другом (шлюз исполняет агента в
executor). Привязка контекста при старте и отвязка при завершении дают токен,
созданный в уже несуществующем контексте: каждый ход писал в лог
``Failed to detach context`` со стеком. Родитель дочерним спанам задаётся явно,
поэтому слот «текущего спана» не нужен вовсе.
"""
from __future__ import annotations

import importlib
import inspect

lf = importlib.import_module("plugins.observability.langfuse")


def _start_trace_source() -> str:
    for name in ("_start_root_trace", "_start_trace", "_open_trace"):
        fn = getattr(lf, name, None)
        if fn is not None:
            return inspect.getsource(fn)
    raise AssertionError("trace-opening function not found in the plugin")


def test_root_span_is_created_without_attaching_context():
    source = _start_trace_source()
    assert "client.start_observation(" in source
    assert "start_as_current_observation" not in source


def test_root_context_is_never_entered():
    source = _start_trace_source()
    assert "__enter__" not in source


def test_children_are_parented_explicitly():
    """Nothing relies on the current-span slot the root no longer occupies."""
    source = inspect.getsource(lf._start_child_observation)
    assert "root_span.start_observation(" in source
