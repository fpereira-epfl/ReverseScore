"""Bulk filename renaming utilities."""

from __future__ import annotations

import logging
import unicodedata
from pathlib import Path

logger = logging.getLogger(__name__)


def normalize_special_chars(text: str) -> str:
    """Replace accented and other Latin characters with ASCII equivalents.

    Uses NFKD decomposition and drops non-ASCII combining marks, so ``ñ``
    becomes ``n``, ``é`` becomes ``e``, ``ó`` becomes ``o``, etc.

    Args:
        text: Input string.

    Returns:
        ASCII-normalized string.
    """
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii")


def rename_remove_pattern(
    directory: Path,
    pattern: str = "",
    *,
    dry_run: bool = False,
    recursive: bool = False,
    normalize: bool = False,
) -> list[tuple[Path, Path]]:
    """Rename files in ``directory`` by removing ``pattern`` and/or normalizing.

    The pattern is removed from the full filename (including the extension).
    Files whose new name would be empty or invalid are skipped with a warning.
    If the target name already exists, a numeric suffix ``_1``, ``_2``, etc. is
    appended before the extension.

    When ``normalize`` is True, both the filename and the pattern are
    ASCII-normalized first, making the match accent-insensitive and producing
    ASCII-only output names. If ``pattern`` is empty and ``normalize`` is True,
    only normalization is performed.

    Args:
        directory: Directory to scan.
        pattern: Substring to remove from filenames. Optional when normalizing.
        dry_run: If True, compute but do not apply renames.
        recursive: If True, scan subdirectories as well.
        normalize: If True, strip accents and Latin special characters.

    Returns:
        List of ``(old_path, new_path)`` tuples for files that changed.

    Raises:
        NotADirectoryError: If ``directory`` is not a directory.
        ValueError: If ``pattern`` is empty and ``normalize`` is False.
    """
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    if not pattern and not normalize:
        raise ValueError("Specify --remove-pattern, --normalize-special-chars, or both")

    if normalize:
        pattern = normalize_special_chars(pattern)

    results: list[tuple[Path, Path]] = []
    iterator = directory.rglob("*") if recursive else directory.iterdir()

    for path in iterator:
        if not path.is_file():
            continue

        working_name = normalize_special_chars(path.name) if normalize else path.name
        new_name = working_name.replace(pattern, "") if pattern else working_name
        if new_name == path.name:
            continue

        if not new_name or new_name in {".", ".."}:
            logger.warning(
                "Skipping %s: new name would be invalid %r",
                path,
                new_name,
            )
            continue

        target = _unique_target(path.parent / new_name)
        results.append((path, target))

        if dry_run:
            logger.info("Would rename %s -> %s", path.name, target.name)
        else:
            path.rename(target)
            logger.info("Renamed %s -> %s", path.name, target.name)

    return results


def _unique_target(target: Path) -> Path:
    """Return ``target`` with a numeric suffix if it already exists."""
    if not target.exists():
        return target

    stem = target.stem
    suffix = target.suffix
    parent = target.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1
