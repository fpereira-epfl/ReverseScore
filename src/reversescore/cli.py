"""Command-line interface for the ReverseScore pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any, Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .config import PipelineConfig
from .conversion import convert_directory_to_wav
from .conversion import convert_to_wav as _convert_to_wav
from .pipeline import run_pipeline
from .transcription import transcribe_stems
from .utils import ensure_dirs, find_audio_files, safe_stem_name

app = typer.Typer(
    name="reversescore",
    help="Transcribe tango audio into MusicXML scores.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["--help", "-h"]},
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
    output_dir: Annotated[Optional[Path], typer.Option("--output-dir", "-o", help="Output directory. Defaults to ./data/trans/<song>/.")] = None,
    time_signature: Annotated[str, typer.Option("--time-signature", "-t", help="Expected time signature.")] = "4/4",
    grid: Annotated[int, typer.Option("--grid", "-g", help="Quantization grid denominator (4 = 16ths).")] = 4,
    onset: Annotated[float, typer.Option("--onset", help="basic-pitch onset threshold.")] = 0.5,
    frame: Annotated[float, typer.Option("--frame", help="basic-pitch frame threshold.")] = 0.3,
    model: Annotated[str, typer.Option("--demucs-model", help="demucs model name.")] = "htdemucs",
    split_bandoneon: Annotated[bool, typer.Option("--split-bandoneon", "--no-split-bandoneon", help="Split bandoneon into treble/bass staves.")] = True,
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Overwrite existing intermediate files.")] = False,
    bandoneon: Annotated[Optional[bool], typer.Option("--bandoneon", "--no-bandoneon", help="Hint that bandoneon is present/absent.")] = None,
    violin: Annotated[Optional[bool], typer.Option("--violin", "--no-violin", help="Hint that violin is present/absent.")] = None,
    piano: Annotated[Optional[bool], typer.Option("--piano", "--no-piano", help="Hint that piano is present/absent.")] = None,
    voice: Annotated[Optional[bool], typer.Option("--voice", "--no-voice", help="Hint that voice is present/absent.")] = None,
    bass: Annotated[Optional[bool], typer.Option("--bass", "--no-bass", help="Hint that bass is present/absent.")] = None,
    drums: Annotated[Optional[bool], typer.Option("--drums", "--no-drums", help="Hint that drums are present/absent.")] = None,
    guitar: Annotated[Optional[bool], typer.Option("--guitar", "--no-guitar", help="Hint that guitar is present/absent.")] = None,
) -> None:
    """Run the full audio-to-score pipeline on a single audio file.

    By default outputs are organized under ./data/trans/<song>/.
    Instrument hints help label staves correctly and skip excluded stems,
    but they do not improve source separation quality.
    """
    audio = audio.expanduser().resolve()
    if not audio.is_file():
        console.print(f"[red]Audio file not found:[/red] {audio}")
        raise typer.Exit(code=1)

    if output_dir is None:
        output_dir = Path("./data/trans") / safe_stem_name(audio.stem)

    hints: list[str] = []
    exclusions: list[str] = []
    for name, value in (
        ("bandoneon", bandoneon),
        ("violin", violin),
        ("piano", piano),
        ("voice", voice),
        ("bass", bass),
        ("drums", drums),
        ("guitar", guitar),
    ):
        if value is True:
            hints.append(name)
        elif value is False:
            exclusions.append(name)

    config = PipelineConfig(
        output_dir=output_dir,
        time_signature=time_signature,
        quantization_grid=grid,
        onset_threshold=onset,
        frame_threshold=frame,
        demucs_model=model,
        split_bandoneon=split_bandoneon,
        instrument_hints=hints,  # type: ignore[arg-type]
        instrument_exclusions=exclusions,  # type: ignore[arg-type]
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


@app.command()
def convert_to_wav(
    audio: Annotated[Optional[Path], typer.Argument(help="Path to a single audio file. Converts directory from config if omitted.")] = None,
    input_dir: Annotated[Optional[Path], typer.Option("--input-dir", "-i", help="Directory containing source audio files.")] = None,
    wav_output_dir: Annotated[Optional[Path], typer.Option("--wav-output-dir", "-w", help="Directory for converted WAV files.")] = None,
    ffmpeg_path: Annotated[Optional[str], typer.Option("--ffmpeg", help="Path or name of the ffmpeg executable.")] = None,
    sample_rate: Annotated[int, typer.Option("--sample-rate", "-r", help="Target sample rate in Hz.")] = 44100,
    channels: Annotated[int, typer.Option("--channels", "-c", help="Number of audio channels.")] = 2,
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Overwrite existing WAV files.")] = False,
) -> None:
    """Convert audio files to WAV using ffmpeg.

    If a single file is provided, convert just that file. Otherwise all
    supported files in ``input_dir`` (default: ``./data/m4a`` from
    ``config.yaml``) are converted to ``wav_output_dir`` (default
    ``./data/wav``).
    """
    config_kwargs: dict[str, Any] = {}
    if wav_output_dir is not None:
        config_kwargs["wav_output_dir"] = wav_output_dir
    if ffmpeg_path is not None:
        config_kwargs["ffmpeg_path"] = ffmpeg_path
    if input_dir is not None:
        config_kwargs["input_dir"] = input_dir

    config = PipelineConfig(**config_kwargs)

    if audio is not None:
        audio = audio.expanduser().resolve()
        if not audio.is_file():
            console.print(f"[red]Audio file not found:[/red] {audio}")
            raise typer.Exit(code=1)
        out_path = config.wav_dir / f"{safe_stem_name(audio.stem)}.wav"
        _convert_to_wav(audio, out_path, config, sample_rate=sample_rate, channels=channels, overwrite=overwrite)
        _print_wavs({audio.stem: out_path})
        return

    input_dir = config.input_dir
    if not input_dir.is_dir():
        console.print(f"[red]Input directory not found:[/red] {input_dir}")
        raise typer.Exit(code=1)

    results = convert_directory_to_wav(
        input_dir,
        config.wav_dir,
        config,
        sample_rate=sample_rate,
        channels=channels,
        overwrite=overwrite,
    )
    _print_wavs(results)


def _print_stems(stems: dict[str, Path]) -> None:
    table = Table(title="Separated Stems")
    table.add_column("Stem", style="cyan")
    table.add_column("Path", style="magenta")
    for label, path in stems.items():
        table.add_row(label, str(path))
    console.print(table)


def _print_wavs(wavs: dict[str, Path]) -> None:
    table = Table(title="Converted WAVs")
    table.add_column("Source", style="cyan")
    table.add_column("WAV Path", style="magenta")
    for label, path in wavs.items():
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
