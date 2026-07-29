"""Shared Rich console and emoji-feedback helpers for the ASP CLI."""

from __future__ import annotations

import os
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table


def make_console(no_color: bool = False) -> Console:
    """Return a Rich console respecting ``NO_COLOR`` and ``--no-color``."""
    force_terminal = None
    if no_color or os.environ.get("NO_COLOR"):
        force_terminal = True
    return Console(
        force_terminal=force_terminal, no_color=no_color or bool(os.environ.get("NO_COLOR"))
    )


def make_handler(console: Console) -> RichHandler:
    """Return a Rich logging handler wired to ``console``."""
    return RichHandler(console=console, rich_tracebacks=True, show_time=False, show_path=False)


def section(console: Console, title: str, emoji: str = "🎵") -> None:
    """Print a bold, coloured section header."""
    console.print(f"\n[bold magenta]{emoji} {title}[/bold magenta]")


def info(console: Console, message: str, emoji: str = "ℹ️ ") -> None:
    """Print an informational line."""
    console.print(f"{emoji} {message}")


def ok(console: Console, message: str, emoji: str = "✅") -> None:
    """Print a success line."""
    console.print(f"{emoji} [green]{message}[/green]")


def warn(console: Console, message: str, emoji: str = "⚠️ ") -> None:
    """Print a warning line."""
    console.print(f"{emoji} [yellow]{message}[/yellow]")


def error(console: Console, message: str, emoji: str = "❌") -> None:
    """Print an error line."""
    console.print(f"{emoji} [red]{message}[/red]")


def kv_table(
    console: Console,
    title: str,
    rows: list[tuple[str, Any]],
    key_style: str = "cyan",
    value_style: str = "magenta",
) -> None:
    """Print a key/value table."""
    table = Table(title=title)
    table.add_column("Key", style=key_style)
    table.add_column("Value", style=value_style)
    for key, value in rows:
        table.add_row(str(key), str(value))
    console.print(table)


def format_duration(seconds: float) -> str:
    """Return a human-readable MM:SS.mmm or HH:MM:SS.mmm duration."""
    if seconds < 0 or not seconds:
        return "0:00.000"
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    if hours:
        return f"{hours:d}:{minutes:02d}:{whole_seconds:02d}.{millis:03d}"
    return f"{minutes:d}:{whole_seconds:02d}.{millis:03d}"


def human_size(bytes_value: int) -> str:
    """Return a human-readable byte size."""
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(bytes_value)
    unit = 0
    while value >= 1024 and unit < len(units) - 1:
        value /= 1024
        unit += 1
    if unit == 0:
        return f"{int(value)} {units[unit]}"
    return f"{value:.2f} {units[unit]}"
