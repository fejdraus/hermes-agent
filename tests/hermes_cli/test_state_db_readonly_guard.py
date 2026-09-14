"""Читающие команды не открывают state.db работающего агента.

Второй процесс, открывший базу, при закрытии чекпойнтит и УДАЛЯЕТ `state.db-wal`/`-shm`.
Живой gateway остаётся с дескрипторами на удалённые иноды, и его следующая запись падает
с DeletedWalGenerationError — пользователь видит «unexpected error» на своём следующем
сообщении, далеко от команды, которая всё сломала. Поэтому проверка держателей идёт
по таблице процессов и при любом сомнении отвечает «занято».
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli.state_db_readonly import live_process_holds_state_db


@pytest.fixture
def db(tmp_path) -> Path:
    """A real (empty) SQLite file: the guard checks holders, the counter parses the file."""
    import sqlite3

    path = tmp_path / "state.db"
    sqlite3.connect(str(path)).close()
    return path


def test_reports_busy_when_a_process_holds_the_db(db, monkeypatch):
    monkeypatch.setattr("hermes_state_holders.foreign_state_db_holders", lambda _p: [(4242, str(db))])
    assert live_process_holds_state_db(db) is True


def test_reports_free_when_nobody_holds_it(db, monkeypatch):
    monkeypatch.setattr("hermes_state_holders.foreign_state_db_holders", lambda _p: [])
    assert live_process_holds_state_db(db) is False


def test_unknown_holders_count_as_busy(db, monkeypatch):
    """Fail closed: a failed scan must not be read as quiet."""
    def boom(_p):
        raise OSError("cannot read /proc")

    monkeypatch.setattr("hermes_state_holders.foreign_state_db_holders", boom)
    assert live_process_holds_state_db(db) is True


def test_absent_database_is_not_busy(tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_state_holders.foreign_state_db_holders", lambda _p: [])
    assert live_process_holds_state_db(tmp_path / "nothing.db") is False


def test_session_count_declines_while_the_db_is_held(db, monkeypatch):
    from hermes_cli import doctor_state

    monkeypatch.setattr("hermes_cli.state_db_readonly.live_process_holds_state_db", lambda _p=None: True)

    def fail(*_a, **_k):
        raise AssertionError("doctor opened a database another process holds")

    monkeypatch.setattr("sqlite3.connect", fail)
    assert doctor_state._session_count(db) is None


def test_session_count_reads_when_the_db_is_free(db, monkeypatch):
    import sqlite3

    from hermes_cli import doctor_state

    real = sqlite3.connect(str(db))
    real.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT)")
    real.execute("INSERT INTO sessions VALUES ('a')")
    real.commit()
    real.close()

    monkeypatch.setattr("hermes_cli.state_db_readonly.live_process_holds_state_db", lambda _p=None: False)
    assert doctor_state._session_count(db) == 1


def test_stats_snapshot_declines_while_the_db_is_held(db, monkeypatch):
    """Even a mode=ro connection creates and then unlinks the sidecars."""
    from hermes_state_dbfile import collect_state_db_stats

    monkeypatch.setattr("hermes_state_holders.foreign_state_db_holders", lambda _p: [(7, "gateway")])

    def fail(*_a, **_k):
        raise AssertionError("stats opened a database another process holds")

    monkeypatch.setattr("hermes_state._connect_tracked_db", fail)
    stats = collect_state_db_stats(db)
    assert stats["holders_prevented_read"] is True
    assert stats["messages"] is None


def test_stats_snapshot_reads_when_nobody_holds_it(db, monkeypatch):
    from hermes_state_dbfile import collect_state_db_stats

    monkeypatch.setattr("hermes_state_holders.foreign_state_db_holders", lambda _p: [])
    stats = collect_state_db_stats(db)
    assert stats.get("holders_prevented_read") is None
    assert stats["page_size"] is not None

