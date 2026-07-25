"""Tests for shared utilities."""

from pathlib import Path

from reversescore.utils import ensure_dirs, find_audio_files, safe_stem_name


def test_ensure_dirs(tmp_path: Path) -> None:
    dirs = [tmp_path / "a", tmp_path / "b" / "c"]
    ensure_dirs(*dirs)
    for d in dirs:
        assert d.is_dir()


def test_find_audio_files(tmp_path: Path) -> None:
    (tmp_path / "song.wav").touch()
    (tmp_path / "song.mid").touch()
    (tmp_path / "notes.txt").touch()
    files = find_audio_files(tmp_path)
    assert len(files) == 1
    assert files[0].name == "song.wav"


def test_safe_stem_name() -> None:
    assert safe_stem_name("La Cumparsita (1935)") == "la_cumparsita__1935_"
