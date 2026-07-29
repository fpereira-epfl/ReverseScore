"""Tests for bulk filename renaming utilities and CLI."""

from pathlib import Path

import pytest
from asp.cli import app
from asp.renaming import normalize_special_chars, rename_remove_pattern
from typer.testing import CliRunner

runner = CliRunner()


def test_rename_remove_pattern(tmp_path: Path) -> None:
    (tmp_path / "Song-y-su-orquesta-típica.aifc").touch()
    (tmp_path / "Another-y-su-orquesta-típica.m4a").touch()
    (tmp_path / "Keep-this-name.wav").touch()

    renames = rename_remove_pattern(tmp_path, "-y-su-orquesta-típica")

    assert len(renames) == 2
    assert (tmp_path / "Song.aifc").is_file()
    assert (tmp_path / "Another.m4a").is_file()
    assert (tmp_path / "Keep-this-name.wav").is_file()
    assert not (tmp_path / "Song-y-su-orquesta-típica.aifc").exists()


def test_rename_remove_pattern_dry_run(tmp_path: Path) -> None:
    original = tmp_path / "Song-y-su-orquesta-típica.aifc"
    original.touch()

    renames = rename_remove_pattern(tmp_path, "-y-su-orquesta-típica", dry_run=True)

    assert len(renames) == 1
    assert original.is_file()
    assert not (tmp_path / "Song.aifc").exists()


def test_rename_remove_pattern_collision(tmp_path: Path) -> None:
    (tmp_path / "Song-y-su-orquesta-típica.aifc").touch()
    (tmp_path / "Song.aifc").touch()

    renames = rename_remove_pattern(tmp_path, "-y-su-orquesta-típica")

    assert len(renames) == 1
    assert (tmp_path / "Song_1.aifc").is_file()
    assert not (tmp_path / "Song-y-su-orquesta-típica.aifc").exists()


def test_rename_remove_pattern_recursive(tmp_path: Path) -> None:
    subdir = tmp_path / "sub"
    subdir.mkdir()
    (subdir / "Song-y-su-orquesta-típica.aifc").touch()

    renames = rename_remove_pattern(tmp_path, "-y-su-orquesta-típica", recursive=True)

    assert len(renames) == 1
    assert (subdir / "Song.aifc").is_file()


def test_rename_remove_pattern_no_matches(tmp_path: Path) -> None:
    (tmp_path / "Song.aifc").touch()

    renames = rename_remove_pattern(tmp_path, "-missing")

    assert renames == []


def test_rename_remove_pattern_invalid_directory(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(NotADirectoryError):
        rename_remove_pattern(missing, "-pattern")


def test_rename_remove_pattern_empty_pattern(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Specify --remove-pattern"):
        rename_remove_pattern(tmp_path, "")


def test_rename_normalize_only(tmp_path: Path) -> None:
    (tmp_path / "Canción.aifc").touch()
    (tmp_path / "Año_más.aifc").touch()
    (tmp_path / "Plain.aifc").touch()

    renames = rename_remove_pattern(tmp_path, normalize=True)

    assert len(renames) == 2
    assert (tmp_path / "Cancion.aifc").is_file()
    assert (tmp_path / "Ano_mas.aifc").is_file()
    assert (tmp_path / "Plain.aifc").is_file()


def test_rename_normalize_only_dry_run(tmp_path: Path) -> None:
    original = tmp_path / "Canción.aifc"
    original.touch()

    renames = rename_remove_pattern(tmp_path, normalize=True, dry_run=True)

    assert len(renames) == 1
    assert original.is_file()
    assert not (tmp_path / "Cancion.aifc").exists()


def test_cli_rename(tmp_path: Path) -> None:
    (tmp_path / "Song-y-su-orquesta-típica.aifc").touch()

    result = runner.invoke(
        app,
        ["rename", "--folder", str(tmp_path), "--remove-pattern", "-y-su-orquesta-típica"],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "Song.aifc").is_file()
    assert "Song-y-su-orquesta-típica.aifc" in result.output
    assert "Song.aifc" in result.output


def test_cli_rename_dry_run(tmp_path: Path) -> None:
    original = tmp_path / "Song-y-su-orquesta-típica.aifc"
    original.touch()

    result = runner.invoke(
        app,
        [
            "rename",
            "--folder",
            str(tmp_path),
            "--remove-pattern",
            "-y-su-orquesta-típica",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert original.is_file()
    assert "Would rename" in result.output


def test_normalize_special_chars() -> None:
    assert normalize_special_chars("Canción") == "Cancion"
    assert normalize_special_chars("orquesta típica") == "orquesta tipica"
    assert normalize_special_chars("año") == "ano"
    assert normalize_special_chars("ÁéÍóÚ") == "AeIoU"


def test_rename_remove_pattern_normalize(tmp_path: Path) -> None:
    (tmp_path / "Canción-y-su-orquesta-típica.aifc").touch()

    renames = rename_remove_pattern(
        tmp_path,
        "-y-su-orquesta-típica",
        normalize=True,
    )

    assert len(renames) == 1
    assert (tmp_path / "Cancion.aifc").is_file()


def test_rename_remove_pattern_normalize_ascii_pattern(tmp_path: Path) -> None:
    (tmp_path / "Canción-y-su-orquesta-típica.aifc").touch()

    renames = rename_remove_pattern(
        tmp_path,
        "-y-su-orquesta-tipica",
        normalize=True,
    )

    assert len(renames) == 1
    assert (tmp_path / "Cancion.aifc").is_file()


def test_cli_rename_normalize_special_chars(tmp_path: Path) -> None:
    (tmp_path / "Canción-y-su-orquesta-típica.aifc").touch()

    result = runner.invoke(
        app,
        [
            "rename",
            "--folder",
            str(tmp_path),
            "--remove-pattern",
            "-y-su-orquesta-típica",
            "--normalize-special-chars",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "Cancion.aifc").is_file()


def test_cli_rename_normalize_short_flag(tmp_path: Path) -> None:
    (tmp_path / "Canción-y-su-orquesta-típica.aifc").touch()

    result = runner.invoke(
        app,
        [
            "rename",
            "--folder",
            str(tmp_path),
            "--remove-pattern",
            "-y-su-orquesta-típica",
            "-nsc",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "Cancion.aifc").is_file()


def test_cli_rename_normalize_only(tmp_path: Path) -> None:
    (tmp_path / "Canción.aifc").touch()

    result = runner.invoke(
        app,
        ["rename", "--folder", str(tmp_path), "-nsc"],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "Cancion.aifc").is_file()
    assert "Canción.aifc" in result.output
    assert "Cancion.aifc" in result.output


def test_cli_rename_normalize_only_dry_run(tmp_path: Path) -> None:
    original = tmp_path / "Canción.aifc"
    original.touch()

    result = runner.invoke(
        app,
        ["rename", "--folder", str(tmp_path), "-nsc", "-n"],
    )

    assert result.exit_code == 0, result.output
    assert original.is_file()
    assert "Would rename" in result.output


def test_cli_rename_no_pattern_no_normalize(tmp_path: Path) -> None:
    (tmp_path / "Song.aifc").touch()

    result = runner.invoke(
        app,
        ["rename", "--folder", str(tmp_path)],
    )

    assert result.exit_code == 1, result.output
    assert "Specify --remove-pattern" in result.output
