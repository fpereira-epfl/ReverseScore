"""Command-line interface for the ReverseScore pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .config import PipelineConfig
from .pipeline import run_pipeline
from .transcription import transcribe_stems
from .utils import ensure_dirs, find_audio_files

app = typer.Typer(
    name="reversescore",
    help="Transcribe tango audio into MusicXML scores.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def main(
    verbose: Annotated[int, typer.Option("--verbose", "-v", count=True, help="Increase verbosity.")] = 0,
) -> None:
    """Configure global logging for all subcommands."""
    level = logging.DEBUG if verbose > 0 else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@app.command()
def transcribe(
    audio: Annotated[Path, typer.Argument(help="Path to the input audio file.")],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o", help="Output directory.")] = Path("./out"),
    time_signature: Annotated[str, typer.Option("--time-signature", "-t", help="Expected time signature.")] = "4/4",
    grid: Annotated[int, typer.Option("--grid", "-g", help="Quantization grid denominator (4 = 16ths).")] = 4,
    onset: Annotated[float, typer.Option("--onset", help="basic-pitch onset threshold.")] = 0.5,
    frame: Annotated[float, typer.Option("--frame", help="basic-pitch frame threshold.")] = 0.3,
    model: Annotated[str, typer.Option("--demucs-model", help="demucs model name.")] = "htdemucs",
    split_bandoneon: Annotated[bool, typer.Option("--split-bandoneon", "--no-split-bandoneon", help="Split bandoneon into treble/bass staves.")] = True,
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Overwrite existing intermediate files.")] = False,
) -> None:
    """Run the full audio-to-score pipeline on a single audio file."""
    audio = audio.expanduser().resolve()
    if not audio.is_file():
        console.print(f"[red]Audio file not found:[/red] {audio}")
        raise typer.Exit(code=1)

    config = PipelineConfig(
        output_dir=output_dir,
        time_signature=time_signature,
        quantization_grid=grid,
        onset_threshold=onset,
        frame_threshold=frame,
        demucs_model=model,
        split_bandoneon=split_bandoneon,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=False,
    ) as progress:
        progress.add_task("Running pipeline...", total=None)
        result = run_pipeline(
            audio,
            config,
            overwrite_separation=overwrite,
            overwrite_transcription=overwrite,
        )

    _print_stems(result.stems)
    _print_midis(result.midi_paths)
    _print_results(result.musicxml_path, result.midi_score_path)


@app.command()
def separate(
    audio: Annotated[Path, typer.Argument(help="Path to the input audio file.")],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o", help="Output directory.")] = Path("./out"),
    model: Annotated[str, typer.Option("--demucs-model", help="demucs model name.")] = "htdemucs",
) -> None:
    """Run only the demucs source-separation step."""
    from .separation import separate_stems as run_separation

    audio = audio.expanduser().resolve()
    if not audio.is_file():
        console.print(f"[red]Audio file not found:[/red] {audio}")
        raise typer.Exit(code=1)

    config = PipelineConfig(output_dir=output_dir, demucs_model=model)
    ensure_dirs(config.separation_dir)

    stems = run_separation(audio, config)
    _print_stems(stems)


@app.command()
def stems_to_midi(
    stems_dir: Annotated[Path, typer.Argument(help="Directory containing stem WAV files.")],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o", help="Output directory.")] = Path("./out"),
    onset: Annotated[float, typer.Option("--onset", help="basic-pitch onset threshold.")] = 0.5,
    frame: Annotated[float, typer.Option("--frame", help="basic-pitch frame threshold.")] = 0.3,
) -> None:
    """Transcribe existing separated stems into per-stem MIDI files."""
    stems_dir = stems_dir.expanduser().resolve()
    if not stems_dir.is_dir():
        console.print(f"[red]Directory not found:[/red] {stems_dir}")
        raise typer.Exit(code=1)

    files = find_audio_files(stems_dir)
    if not files:
        console.print("[red]No audio files found in[/red]", stems_dir)
        raise typer.Exit(code=1)

    stems = {f.stem: f for f in files}
    config = PipelineConfig(
        output_dir=output_dir,
        onset_threshold=onset,
        frame_threshold=frame,
    )
    midi_paths = transcribe_stems(stems, config)
    _print_midis(midi_paths)


def _print_stems(stems: dict[str, Path]) -> None:
    table = Table(title="Separated Stems")
    table.add_column("Stem", style="cyan")
    table.add_column("Path", style="magenta")
    for label, path in stems.items():
        table.add_row(label, str(path))
    console.print(table)


def _print_midis(midi_paths: dict[str, Path]) -> None:
    table = Table(title="Transcribed MIDI")
    table.add_column("Stem", style="cyan")
    table.add_column("MIDI Path", style="magenta")
    for label, path in midi_paths.items():
        table.add_row(label, str(path))
    console.print(table)


def _print_results(musicxml_path: Path, midi_path: Path) -> None:
    table = Table(title="Exported Scores")
    table.add_column("Format", style="cyan")
    table.add_column("Path", style="magenta")
    table.add_row("MusicXML", str(musicxml_path))
    table.add_row("MIDI", str(midi_path))
    console.print(table)
    console.print("[green]Done![/green] Open the MusicXML file in MuseScore to review and edit.")


if __name__ == "__main__":
    app()
