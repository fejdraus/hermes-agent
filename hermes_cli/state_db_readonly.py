"""Guard for read-only callers that would otherwise open a live agent's state.db.

Opening the database from a second process is not harmless: when SQLite closes the last
connection it checkpoints and *unlinks* ``state.db-wal`` and ``-shm``. A gateway already
running keeps descriptors on those now-deleted inodes, and its next write refuses with
``DeletedWalGenerationError`` — the agent then answers "unexpected error" to whatever the
user sent next, far from the command that actually caused it.

So a diagnostic must ask first, and ask without touching the database: holders are read
from the process table, not by probing SQLite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def live_process_holds_state_db(db_path: Optional[Path] = None) -> bool:
    """True when another process currently has the state database open.

    Fails closed: if holders cannot be determined, the answer is "yes, someone holds it",
    because guessing "free" is what costs a running agent its WAL.
    """
    try:
        from hermes_state_holders import foreign_state_db_holders
    except Exception:
        return True
    try:
        path = Path(db_path) if db_path is not None else _default_state_db_path()
    except Exception:
        return True
    if path is None or not path.exists():
        return False
    try:
        return bool(foreign_state_db_holders(path))
    except Exception:
        return True


def _default_state_db_path() -> Optional[Path]:
    from hermes_cli.config import get_hermes_home

    return Path(get_hermes_home()) / "state.db"
