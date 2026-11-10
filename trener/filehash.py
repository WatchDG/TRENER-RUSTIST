"""SHA-256 of a file, computed once per process for a file that did not change.

A corpus is read several times in a run (the early check, the plan, the test split, the seals of the manifest, the key
of the speech code cache): without the memo every pass hashes every recording again. The memo key is the path with
the size and the modification time, so a rewritten file is hashed again.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_MEMO: dict[tuple[str, int, int], str] = {}


def file_sha256(path: str | Path) -> str:
    """Hex SHA-256 of the file's contents (memoized while the file's size and modification time stay the same)."""
    resolved = Path(path)
    info = resolved.stat()
    key = (str(resolved.resolve()), info.st_size, info.st_mtime_ns)
    digest = _MEMO.get(key)
    if digest is None:
        hasher = hashlib.sha256()
        with resolved.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                hasher.update(block)
        digest = _MEMO[key] = hasher.hexdigest()
    return digest


__all__ = ["file_sha256"]
