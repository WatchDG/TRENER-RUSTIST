"""Training speedups: the memoized file hash and the new configuration fields."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from trener import filehash
from trener.config import ConfigError, JoltConfig, ModelConfig, OptimConfig, validate_config


def test_file_hash_is_computed_once_per_unchanged_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "clip.bin"
    path.write_bytes(b"first")
    reads: list[int] = []
    original = Path.open

    def spy(self: Path, *args: object, **kwargs: object) -> object:
        if self == path and args and args[0] == "rb":
            reads.append(1)
        return original(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "open", spy)
    assert filehash.file_sha256(path) == hashlib.sha256(b"first").hexdigest()
    assert filehash.file_sha256(path) == hashlib.sha256(b"first").hexdigest()
    assert len(reads) == 1
    path.write_bytes(b"second, longer")  # a changed file is hashed again
    assert filehash.file_sha256(path) == hashlib.sha256(b"second, longer").hexdigest()
    assert len(reads) == 2


@pytest.mark.parametrize(
    ("model", "match"),
    [
        (ModelConfig(checkpoint_free_layers=-1), "checkpoint_free_layers"),
        (ModelConfig(gradient_checkpointing=False, checkpoint_free_layers=4), "checkpoint_free_layers"),
        (ModelConfig(media_cache_gb=0.0), "media_cache_gb"),
    ],
)
def test_invalid_speed_settings_are_rejected(model: ModelConfig, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        validate_config(JoltConfig(model=model))


def test_speed_settings_default_to_the_reference_behaviour() -> None:
    cfg = JoltConfig(model=ModelConfig(checkpoint_free_layers=8), optim=OptimConfig(fused_step=True))
    validate_config(cfg)
    assert JoltConfig().model.checkpoint_free_layers == 0 and JoltConfig().optim.fused_step is False
    assert cfg.to_dict()["model"]["checkpoint_free_layers"] == 8 and cfg.to_dict()["optim"]["fused_step"] is True
