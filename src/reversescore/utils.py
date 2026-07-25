"""Shared utilities for the ReverseScore pipeline."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path


def ensure_dirs(*paths: Path) -> None:
    """Create one or more directories if they do not exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def find_audio_files(directory: Path, extensions: Iterable[str] | None = None) -> list[Path]:
    """Return sorted list of audio files in ``directory``.

    Args:
        directory: Directory to scan.
        extensions: Audio extensions to include. Defaults to common formats.

    Returns:
        Sorted list of matching file paths.
    """
    if extensions is None:
        extensions = {".wav",".flac",".mp3",".ogg",".m4a",".aac"}
    ext_set = {e.lower().lstrip(".") for e in extensions}
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower().lstrip(".") in ext_set]
    return sorted(files)


def run_command(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a shell command and return the completed process.

    Args:
        cmd: Command and arguments.
        cwd: Working directory.
        check: Raise ``CalledProcessError`` on non-zero exit.

    Returns:
        Completed process with captured stdout/stderr.
    """
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
    )


def find_program(name: str) -> Path:
    """Locate an executable in PATH.

    Also searches the directory of the current Python interpreter so that
    console scripts installed in the same virtual environment are found even
    when the environment is not activated.

    Args:
        name: Program name.

    Returns:
        Path to executable.

    Raises:
        FileNotFoundError: If the program cannot be found.
    """
    python_bin = Path(sys.executable).parent
    search_path = os.environ.get("PATH", "")
    if str(python_bin) not in search_path.split(os.pathsep):
        search_path = f"{python_bin}{os.pathsep}{search_path}"
    path = shutil.which(name, path=search_path)
    if path is None:
        raise FileNotFoundError(f"Required program not found in PATH: {name}")
    return Path(path)


def safe_stem_name(name: str) -> str:
    """Return a filesystem-safe stem name."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name).lower()
