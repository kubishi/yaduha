"""Tests for yaduha.logger module: loggers, inject_logs, global logger."""

import json
import threading

import pytest

from yaduha.logger import (
    JsonLogger,
    NoLogger,
    PrintLogger,
    get_global_logger,
    get_log_context,
    inject_logs,
    set_global_logger,
)

# -- PrintLogger --


def test_print_logger_outputs_to_stdout(capsys):
    logger = PrintLogger()
    logger.log({"event": "test", "value": 42})
    captured = capsys.readouterr()
    assert "event" in captured.out
    assert "test" in captured.out
    assert "timestamp" in captured.out


def test_print_logger_includes_metadata(capsys):
    logger = PrintLogger(metadata={"run_id": "abc"})
    logger.log({"event": "test"})
    captured = capsys.readouterr()
    assert "run_id" in captured.out
    assert "abc" in captured.out


# -- JsonLogger --


def test_json_logger_writes_jsonl(tmp_jsonl):
    logger = JsonLogger(file_path=tmp_jsonl)
    logger.log({"event": "hello"})
    logger.log({"event": "world"})

    lines = tmp_jsonl.read_text().strip().split("\n")
    assert len(lines) == 2

    data = json.loads(lines[0])
    assert data["event"] == "hello"
    assert "timestamp" in data


def test_json_logger_rejects_non_jsonl():
    with pytest.raises(ValueError, match="jsonl"):
        JsonLogger(file_path="/tmp/bad.txt")


def test_json_logger_thread_safety(tmp_jsonl):
    logger = JsonLogger(file_path=tmp_jsonl)
    n_threads = 10
    n_writes = 50

    def writer(thread_id):
        for i in range(n_writes):
            logger.log({"thread": thread_id, "i": i})

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = tmp_jsonl.read_text().strip().split("\n")
    assert len(lines) == n_threads * n_writes

    # Every line should be valid JSON
    for line in lines:
        json.loads(line)


def test_json_logger_includes_metadata(tmp_jsonl):
    logger = JsonLogger(file_path=tmp_jsonl, metadata={"experiment": "test1"})
    logger.log({"value": 1})

    data = json.loads(tmp_jsonl.read_text().strip())
    assert data["experiment"] == "test1"
    assert data["value"] == 1


# -- NoLogger --


def test_no_logger_is_noop(capsys, tmp_jsonl):
    logger = NoLogger()
    logger.log({"event": "should_not_appear"})
    assert capsys.readouterr().out == ""


# -- inject_logs / get_log_context --


def test_inject_logs_adds_context():
    assert get_log_context() == {}  # clean state

    with inject_logs(model="gpt-4o", run=1):
        ctx = get_log_context()
        assert ctx["MODEL"] == "gpt-4o"
        assert ctx["RUN"] == 1

    assert get_log_context() == {}  # cleaned up


def test_inject_logs_nesting():
    with inject_logs(a="outer"):
        assert get_log_context()["A"] == "outer"

        with inject_logs(b="inner", a="overridden"):
            ctx = get_log_context()
            assert ctx["A"] == "overridden"
            assert ctx["B"] == "inner"

        # After inner exits, only outer remains
        ctx = get_log_context()
        assert ctx["A"] == "outer"
        assert "B" not in ctx


def test_inject_logs_appears_in_log_output(tmp_jsonl):
    logger = JsonLogger(file_path=tmp_jsonl)

    with inject_logs(strategy="pipeline"):
        logger.log({"event": "translate"})

    data = json.loads(tmp_jsonl.read_text().strip())
    assert data["STRATEGY"] == "pipeline"
    assert data["event"] == "translate"


# -- Global logger --


def test_get_global_logger_default_is_nologger():
    import yaduha.logger as mod

    old = mod.global_logger
    try:
        mod.global_logger = None
        result = get_global_logger()
        assert isinstance(result, NoLogger)
    finally:
        mod.global_logger = old


def test_set_and_get_global_logger():
    import yaduha.logger as mod

    old = mod.global_logger
    try:
        custom = PrintLogger()
        set_global_logger(custom)
        assert get_global_logger() is custom
    finally:
        mod.global_logger = old


# -- Logger.get_sublogger --


def test_get_sublogger_merges_metadata():
    parent = PrintLogger(metadata={"experiment": "exp1"})
    child = parent.get_sublogger(run=42)

    assert child.metadata == {"experiment": "exp1", "run": 42}
    # Parent unchanged
    assert parent.metadata == {"experiment": "exp1"}


def test_get_sublogger_overrides_metadata():
    parent = PrintLogger(metadata={"level": "info"})
    child = parent.get_sublogger(level="debug")
    assert child.metadata["level"] == "debug"
    assert parent.metadata["level"] == "info"


# -- Logger base class --


def test_logger_log_adds_timestamp(capsys):
    logger = PrintLogger()
    logger.log({"event": "test"})
    captured = capsys.readouterr()
    assert "timestamp" in captured.out
