"""Command-line interface for the ReverseScore pipeline."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Annotated, Any, Optional, cast

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .config import PipelineConfig, TranscriptionBackend
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

_VALID_BACKENDS = frozenset({"basic-pitch", "yourmt3"})


def _validate_backend(value: str) -> str:
    """Validate the transcription backend name."""
    if value.lower() not in _VALID_BACKENDS:
        raise typer.BadParameter(
            f"backend must be one of {sorted(_VALID_BACKENDS)}, got {value!r}"
        )
    return value.lower()


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
    backend: Annotated[str, typer.Option("--backend", "-b", help="Transcription backend: basic-pitch or yourmt3.")] = "basic-pitch",
    split_bandoneon: Annotated[bool, typer.Option("--split-bandoneon", "--no-split-bandoneon", help="Split bandoneon into treble/bass staves.")] = True,
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Overwrite existing intermediate files.")] = False,
    bandoneon: Annotated[bool, typer.Option("--bandoneon/--no-bandoneon", help="Hint that bandoneon is present/absent.")] = False,
    violin: Annotated[bool, typer.Option("--violin/--no-violin", help="Hint that violin is present/absent.")] = False,
    piano: Annotated[bool, typer.Option("--piano/--no-piano", help="Hint that piano is present/absent.")] = False,
    voice: Annotated[bool, typer.Option("--voice/--no-voice", help="Hint that voice is present/absent.")] = False,
    bass: Annotated[bool, typer.Option("--bass/--no-bass", help="Hint that bass is present/absent.")] = False,
    drums: Annotated[bool, typer.Option("--drums/--no-drums", help="Hint that drums are present/absent.")] = False,
    guitar: Annotated[bool, typer.Option("--guitar/--no-guitar", help="Hint that guitar is present/absent.")] = False,
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
        transcription_backend=cast(TranscriptionBackend, backend),
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
            transcription_backend=backend,
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
    backend: Annotated[str, typer.Option("--backend", "-b", help="Transcription backend: basic-pitch or yourmt3.")] = "basic-pitch",
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
        transcription_backend=cast(TranscriptionBackend, backend),
    )
    midi_paths = transcribe_stems(stems, config, backend=backend)
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


@app.command()
def identify(
    audio: Annotated[Optional[Path], typer.Argument(help="Path to a single input audio file (aifc, m4a, etc.).")] = None,
    folder: Annotated[Optional[Path], typer.Option("--folder", "-d", help="Directory containing audio files to identify.")] = None,
    pattern: Annotated[str, typer.Option("--pattern", "-p", help="Pattern to match inside --folder. Use %% as a digit wildcard.")] = "*.aifc",
    json_output: Annotated[bool, typer.Option("--json", help="Print the raw ShazamIO response(s) as JSON.")] = False,
    rename: Annotated[bool, typer.Option("--rename", "-r", help="Rename matched files to 'artist-name_song-name.ext'.")] = False,
    delay: Annotated[float, typer.Option("--delay", help="Seconds to wait between batch requests to avoid rate limiting.")] = 3.0,
) -> None:
    """Identify the artist and song title using ShazamIO.

    Provide a single ``audio`` file or a ``--folder`` with a ``--pattern``.
    ``.aifc`` files are converted to WAV with ffmpeg before recognition if
    ShazamIO cannot read them directly. With ``--rename``, each matched file is
    renamed to ``artist-name_song-name.<ext>`` after a successful match. A
    ``--delay`` (default 3 s) is applied between batch requests.
    """
    from .recognition import find_matching_files, format_track_filename, recognize_song

    # Resolve inputs and build the list of files to process.
    if audio is not None and folder is not None:
        console.print("[red]Specify either an audio file or --folder, not both.[/red]")
        raise typer.Exit(code=1)

    files: list[Path] = []
    if folder is not None:
        folder = folder.expanduser().resolve()
        if not folder.is_dir():
            console.print(f"[red]Folder not found:[/red] {folder}")
            raise typer.Exit(code=1)
        files = find_matching_files(folder, pattern)
        if not files:
            console.print(f"[red]No files matching '{pattern}' found in[/red] {folder}")
            raise typer.Exit(code=1)
    elif audio is not None:
        audio = audio.expanduser().resolve()
        if not audio.is_file():
            console.print(f"[red]Audio file not found:[/red] {audio}")
            raise typer.Exit(code=1)
        files = [audio]
    else:
        console.print("[red]Specify an audio file or --folder.[/red]")
        raise typer.Exit(code=1)

    results: list[dict[str, Any]] = []
    for index, file_path in enumerate(files):
        if index > 0 and delay > 0:
            time.sleep(delay)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=False,
        ) as progress:
            progress.add_task(f"Identifying {file_path.name}...", total=None)
            try:
                result = recognize_song(file_path)
            except Exception as exc:
                console.print(f"[red]Recognition failed for {file_path.name}:[/red] {exc}")
                result = {"input_path": str(file_path), "error": str(exc)}

        if rename and "error" not in result:
            try:
                new_stem = format_track_filename(result.get("artist"), result.get("title"))
            except ValueError as exc:
                console.print(f"[red]Cannot rename {file_path.name}:[/red] {exc}")
            else:
                ext = file_path.suffix.lower()
                target = file_path.with_name(f"{new_stem}{ext}")
                original_target = target
                counter = 1
                while target.exists():
                    target = original_target.with_name(f"{new_stem}_{counter}{ext}")
                    counter += 1

                file_path.rename(target)
                result["renamed_to"] = str(target)
                console.print(f"[green]Renamed[/green] {file_path.name} -> {target.name}")

        results.append(result)

    # Output.
    if json_output:
        if len(results) == 1:
            payload = results[0].get("raw", results[0])
        else:
            payload = [r.get("raw", r) for r in results]
        console.print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    for result in results:
        table = Table(title=f"Recognized Track: {Path(result['input_path']).name}")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Title", result.get("title") or "-")
        table.add_row("Artist", result.get("artist") or "-")
        table.add_row("Album", result.get("album") or "-")
        table.add_row("Genre", result.get("genre") or "-")
        if "renamed_to" in result:
            table.add_row("Renamed to", str(Path(result["renamed_to"]).name))
        if "error" in result:
            table.add_row("Error", result["error"])
        console.print(table)


@app.command()
def rename(
    folder: Annotated[Path, typer.Option("--folder", "-d", help="Directory containing files to rename.")],
    remove_pattern: Annotated[Optional[str], typer.Option("--remove-pattern", "-p", help="Substring to remove from each filename. Optional if --normalize-special-chars is used.")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", "-n", help="Preview renames without applying them.")] = False,
    recursive: Annotated[bool, typer.Option("--recursive", "-R", help="Rename files in subdirectories too.")] = False,
    normalize_special_chars: Annotated[bool, typer.Option("--normalize-special-chars", "-nsc", help="Remove accents and Latin special characters from filenames.")] = False,
) -> None:
    """Remove a substring from and/or normalize all filenames in a folder.

    Examples:
        reversescore rename -d ./tracks -p "-y-su-orquesta-típica"
        reversescore rename -d ./tracks -p "-y-su-orquesta-típica" -nsc
        reversescore rename -d ./tracks -nsc
    """
    from .renaming import rename_remove_pattern

    folder = folder.expanduser().resolve()
    try:
        renames = rename_remove_pattern(
            folder,
            remove_pattern or "",
            dry_run=dry_run,
            recursive=recursive,
            normalize=normalize_special_chars,
        )
    except Exception as exc:
        console.print(f"[red]Rename failed:[/red] {exc}")
        raise typer.Exit(code=1) from None

    if not renames:
        console.print("[yellow]No files changed.[/yellow]")
        return

    action = "Would rename" if dry_run else "Renamed"
    table = Table(title=f"{action} Files")
    table.add_column("Original", style="cyan")
    table.add_column("New", style="magenta")
    for old_path, new_path in renames:
        table.add_row(old_path.name, new_path.name)
    console.print(table)


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
