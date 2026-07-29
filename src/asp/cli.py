"""Command-line interface for ASP: Audio Score Processor."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated, Any, Optional, cast

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from . import __version__
from ._console import error, format_duration, human_size, info, kv_table, ok, section, warn
from .audio_convert import _format_from_path, convert_audio_files, summarize_conversion
from .audio_scan import ScanResult, scan_audio
from .audio_split import split_recording
from .audio_trim import trim_silence
from .config import PipelineConfig, TranscriptionBackend
from .pipeline import run_pipeline
from .transcription import transcribe_stems
from .utils import ensure_dirs, find_audio_files, safe_stem_name

app = typer.Typer(
    name="asp",
    help="ASP: Audio Score Processor — convert, analyse, split, trim and transcribe music files.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["--help", "-h"]},
)
console = Console()

_VALID_BACKENDS = frozenset({"basic-pitch", "yourmt3"})


def _validate_backend(value: str) -> str:
    """Validate the transcription backend name."""
    if value.lower() not in _VALID_BACKENDS:
        raise typer.BadParameter(f"backend must be one of {sorted(_VALID_BACKENDS)}, got {value!r}")
    return value.lower()


def _validate_format(value: str | None) -> str | None:
    if value is None:
        return None
    allowed = {"m4a", "aac", "mp3", "flac", "wav", "aiff", "aifc"}
    if value.lower() not in allowed:
        raise typer.BadParameter(f"format must be one of {sorted(allowed)}, got {value!r}")
    return value.lower()


def _validate_denoise(value: str) -> str:
    allowed = {"none", "soft", "medium", "hard"}
    if value.lower() not in allowed:
        raise typer.BadParameter(f"denoise preset must be one of {sorted(allowed)}, got {value!r}")
    return value.lower()


def _validate_threshold(value: str) -> str:
    if not re.fullmatch(r"-?\d+(?:\.\d+)?dB?", value, re.IGNORECASE):
        raise typer.BadParameter(f"threshold must look like -40dB, got {value!r}")
    return value


def _exit(message: str, code: int = 1) -> None:
    error(console, message)
    raise typer.Exit(code=code)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"ASP {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Increase verbosity (use multiple times for more detail).",
        ),
    ] = 0,
    no_color: Annotated[bool, typer.Option("--no-color", help="Disable coloured output.")] = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Configure global logging and console colours for all subcommands."""
    global console
    console = Console(no_color=no_color or bool(os.environ.get("NO_COLOR")))
    level = logging.DEBUG if verbose >= 2 else (logging.INFO if verbose == 1 else logging.WARNING)
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[
            RichHandler(console=console, rich_tracebacks=True, show_time=False, show_path=False)
        ],
    )


@app.command()
def transcribe(
    audio: Annotated[Path, typer.Argument(help="Path to the input audio file.")],
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--output-dir", "-o", help="Output directory. Defaults to ./data/trans/<song>/."
        ),
    ] = None,
    time_signature: Annotated[
        str, typer.Option("--time-signature", "-t", help="Expected time signature.")
    ] = "4/4",
    grid: Annotated[
        int, typer.Option("--grid", "-g", help="Quantization grid denominator (4 = 16ths).")
    ] = 4,
    onset: Annotated[float, typer.Option("--onset", help="basic-pitch onset threshold.")] = 0.5,
    frame: Annotated[float, typer.Option("--frame", help="basic-pitch frame threshold.")] = 0.3,
    model: Annotated[str, typer.Option("--demucs-model", help="demucs model name.")] = "htdemucs",
    backend: Annotated[
        str,
        typer.Option(
            "--backend",
            "-b",
            help="Transcription backend: basic-pitch or yourmt3.",
            callback=_validate_backend,
        ),
    ] = "basic-pitch",
    split_bandoneon: Annotated[
        bool,
        typer.Option(
            "--split-bandoneon/--no-split-bandoneon",
            help="Split bandoneon into treble/bass staves.",
        ),
    ] = True,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Overwrite existing intermediate files.")
    ] = False,
    bandoneon: Annotated[
        bool,
        typer.Option("--bandoneon/--no-bandoneon", help="Hint that bandoneon is present/absent."),
    ] = False,
    violin: Annotated[
        bool, typer.Option("--violin/--no-violin", help="Hint that violin is present/absent.")
    ] = False,
    piano: Annotated[
        bool, typer.Option("--piano/--no-piano", help="Hint that piano is present/absent.")
    ] = False,
    voice: Annotated[
        bool, typer.Option("--voice/--no-voice", help="Hint that voice is present/absent.")
    ] = False,
    bass: Annotated[
        bool, typer.Option("--bass/--no-bass", help="Hint that bass is present/absent.")
    ] = False,
    drums: Annotated[
        bool, typer.Option("--drums/--no-drums", help="Hint that drums are present/absent.")
    ] = False,
    guitar: Annotated[
        bool, typer.Option("--guitar/--no-guitar", help="Hint that guitar is present/absent.")
    ] = False,
) -> None:
    """Run the full audio-to-score pipeline on a single audio file."""
    audio = audio.expanduser().resolve()
    if not audio.is_file():
        _exit(f"Audio file not found: {audio}")

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

    section(console, "Transcribing audio to score", "🎼")
    info(console, f"Input: {audio}")
    info(console, f"Output: {output_dir}")
    info(console, f"Backend: {backend}")

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
    ok(console, "Done! Open the MusicXML file in MuseScore to review and edit.")


@app.command()
def separate(
    audio: Annotated[Path, typer.Argument(help="Path to the input audio file.")],
    output_dir: Annotated[
        Path, typer.Option("--output-dir", "-o", help="Output directory.")
    ] = Path("./out"),
    model: Annotated[str, typer.Option("--demucs-model", help="demucs model name.")] = "htdemucs",
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Re-run separation even if stems exist.")
    ] = False,
) -> None:
    """Run only the demucs source-separation step."""
    from .separation import separate_stems as run_separation

    audio = audio.expanduser().resolve()
    if not audio.is_file():
        _exit(f"Audio file not found: {audio}")

    config = PipelineConfig(output_dir=output_dir, demucs_model=model)
    ensure_dirs(config.separation_dir)

    section(console, "Separating audio into stems", "🎚️")
    info(console, f"Input: {audio}")
    info(console, f"Model: {model}")
    info(console, f"Output: {config.separation_dir}")

    stems = run_separation(audio, config, overwrite=overwrite)
    _print_stems(stems)
    ok(console, f"Separated {len(stems)} stems")


@app.command()
def stems_to_midi(
    stems_dir: Annotated[Path, typer.Argument(help="Directory containing stem WAV files.")],
    output_dir: Annotated[
        Path, typer.Option("--output-dir", "-o", help="Output directory.")
    ] = Path("./out"),
    onset: Annotated[float, typer.Option("--onset", help="basic-pitch onset threshold.")] = 0.5,
    frame: Annotated[float, typer.Option("--frame", help="basic-pitch frame threshold.")] = 0.3,
    backend: Annotated[
        str,
        typer.Option(
            "--backend",
            "-b",
            help="Transcription backend: basic-pitch or yourmt3.",
            callback=_validate_backend,
        ),
    ] = "basic-pitch",
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Re-transcribe existing MIDI files.")
    ] = False,
) -> None:
    """Transcribe existing separated stems into per-stem MIDI files."""
    stems_dir = stems_dir.expanduser().resolve()
    if not stems_dir.is_dir():
        _exit(f"Directory not found: {stems_dir}")

    files = find_audio_files(stems_dir)
    if not files:
        _exit(f"No audio files found in {stems_dir}")

    stems = {f.stem: f for f in files}
    config = PipelineConfig(
        output_dir=output_dir,
        onset_threshold=onset,
        frame_threshold=frame,
        transcription_backend=cast(TranscriptionBackend, backend),
    )

    section(console, "Transcribing stems to MIDI", "🎹")
    info(console, f"Stems directory: {stems_dir}")
    info(console, f"Output directory: {output_dir}")
    info(console, f"Backend: {backend}")

    midi_paths = transcribe_stems(stems, config, overwrite=overwrite, backend=backend)
    _print_midis(midi_paths)
    ok(console, f"Transcribed {len(midi_paths)} stems")


@app.command("convert")
def convert_cmd(
    input: Annotated[
        Path,
        typer.Argument(help="Path to an audio file or a directory containing audio files."),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output file or directory. Default is beside the input(s).",
        ),
    ] = None,
    format: Annotated[
        Optional[str],
        typer.Option(
            "--format",
            "-f",
            help="Output format: m4a, aac, mp3, flac, wav, aiff, aifc. Inferred from --output if omitted; defaults to aifc.",
            callback=_validate_format,
        ),
    ] = None,
    bitrate: Annotated[
        str, typer.Option("--bitrate", "-b", help="Lossy bitrate, e.g. 192k, 256k, 320k.")
    ] = "256k",
    normalize: Annotated[
        bool,
        typer.Option(
            "--normalize",
            "-n",
            help="Apply constant-gain loudness normalization (uses --target-loudness and --true-peak defaults).",
        ),
    ] = False,
    target_loudness: Annotated[
        float,
        typer.Option(
            "--target-loudness",
            help="Target integrated loudness when --normalize is used.",
        ),
    ] = -18.0,
    true_peak: Annotated[
        Optional[float],
        typer.Option(
            "--true-peak",
            help="True-peak ceiling when --normalize is used.",
            show_default="-2 dBTP lossy / -1 dBTP lossless",
        ),
    ] = None,
    denoise: Annotated[
        str,
        typer.Option(
            "--denoise",
            "-d",
            help="Denoise preset: none, soft, medium, hard.",
            callback=_validate_denoise,
        ),
    ] = "none",
    filter: Annotated[
        Optional[str], typer.Option("--filter", help="Custom FFmpeg audio filter chain.")
    ] = None,
    sample_rate: Annotated[
        int, typer.Option("--sample-rate", "-r", help="Target sample rate in Hz.")
    ] = 44100,
    channels: Annotated[
        int, typer.Option("--channels", "-c", help="Number of audio channels.")
    ] = 2,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Overwrite existing output.")
    ] = False,
) -> None:
    """Convert audio file(s) to another format.

    Provide a single file or a directory. The output format is taken from
    ``--format``, then from the ``--output`` file extension, and finally
    defaults to ``aifc``.
    """
    input = input.expanduser().resolve()
    if not input.exists():
        _exit(f"Input not found: {input}")

    # Resolve format: explicit > output extension > aifc default.
    chosen_format: str = format or "aifc"
    if output is not None:
        output = output.expanduser().resolve()
        inferred = _format_from_path(output)
        if inferred is not None:
            chosen_format = inferred
        elif format is None and output.is_dir():
            chosen_format = "aifc"
    elif format is None:
        chosen_format = "aifc"

    section(console, "Converting audio", "🔄")
    info(console, f"Input: {input}")
    info(console, f"Format: {chosen_format}")
    if output:
        info(console, f"Output: {output}")

    try:
        results = convert_audio_files(
            input,
            output,
            output_format=cast(Any, chosen_format),
            bitrate=bitrate,
            normalize=normalize,
            target_loudness_lufs=target_loudness,
            true_peak_ceiling_dbtp=true_peak,
            denoise=cast(Any, denoise),
            custom_filter=filter or "",
            sample_rate=sample_rate,
            channels=channels,
            overwrite=overwrite,
        )
    except Exception as exc:
        _exit(str(exc))

    if len(results) == 1:
        kv_table(
            console,
            "Conversion result",
            summarize_conversion(next(iter(results.values()))),
        )
    else:
        table = Table(title="Converted files")
        table.add_column("Source", style="cyan")
        table.add_column("Output", style="magenta")
        table.add_column("Duration", style="green")
        for stem, result in results.items():
            table.add_row(stem, str(result.output_path), format_duration(result.info.duration))
        console.print(table)

    ok(console, f"Converted {len(results)} file(s)")


@app.command("scan")
def scan_cmd(
    input: Annotated[Path, typer.Argument(help="Path to the audio file to analyse.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print raw results as JSON.")] = False,
) -> None:
    """Analyse an audio file without modifying it."""
    input = input.expanduser().resolve()
    if not input.is_file():
        _exit(f"Input file not found: {input}")

    section(console, "Scanning audio file", "🔍")
    info(console, f"Input: {input}")

    try:
        result = scan_audio(input)
    except Exception as exc:
        _exit(str(exc))

    if json_output:
        console.print(json.dumps(_scan_to_dict(result), indent=2, ensure_ascii=False))
        return

    _print_scan(result)


@app.command("split")
def split_cmd(
    input: Annotated[Path, typer.Argument(help="Source .aif, .aiff, or .aifc file.")],
    tracks: Annotated[int, typer.Option("--tracks", help="Exact number of output tracks.")] = 0,
    min_track: Annotated[
        str, typer.Option("--min-track", help="Hard minimum track duration (MM:SS or seconds).")
    ] = "",
    max_track: Annotated[
        str, typer.Option("--max-track", help="Hard maximum track duration (MM:SS or seconds).")
    ] = "",
    target_track: Annotated[
        Optional[str],
        typer.Option("--target-track", help="Preferred track duration (MM:SS or seconds)."),
    ] = None,
    gap: Annotated[
        float, typer.Option("--gap", "-g", help="Minimum silence duration detected by FFmpeg.")
    ] = 0.5,
    threshold: Annotated[
        str,
        typer.Option(
            "--threshold",
            "-t",
            help="FFmpeg silence threshold, e.g. -40dB.",
            callback=_validate_threshold,
        ),
    ] = "-40dB",
    padding: Annotated[
        float,
        typer.Option("--padding", "-p", help="Audio retained on each side of a selected silence."),
    ] = 0.02,
    edge_guard: Annotated[
        float,
        typer.Option(
            "--edge-guard", "-e", help="Window used to recognise leading and trailing silence."
        ),
    ] = 10.0,
    output_dir: Annotated[
        Optional[Path], typer.Option("--output-dir", "-o", help="Output directory.")
    ] = None,
    prefix: Annotated[
        Optional[str], typer.Option("--prefix", help="Output filename prefix.")
    ] = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Replace existing output files.")
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", "-n", help="Detect and print boundaries without writing files."),
    ] = False,
) -> None:
    """Split a long recording into tracks using detected silences."""
    input = input.expanduser().resolve()
    if not input.is_file():
        _exit(f"Input file not found: {input}")
    if tracks <= 0:
        _exit("--tracks is required and must be greater than zero")
    if not min_track or not max_track:
        _exit("--min-track and --max-track are required")

    from .audio_split import _format_time as fmt_time

    try:
        from .audio_split import _parse_time

        min_seconds = _parse_time(min_track)
        max_seconds = _parse_time(max_track)
        target_seconds = _parse_time(target_track) if target_track else None
    except ValueError as exc:
        _exit(str(exc))

    section(console, "Splitting recording into tracks", "✂️")
    info(console, f"Input: {input}")
    info(console, f"Requested tracks: {tracks}")
    info(console, f"Allowed length: {fmt_time(min_seconds)}–{fmt_time(max_seconds)}")
    info(console, f"Detection: gap {gap:g}s, threshold {threshold}")

    try:
        result = split_recording(
            input,
            tracks,
            min_track=min_seconds,
            max_track=max_seconds,
            target_track=target_seconds,
            gap=gap,
            threshold_db=threshold,
            padding=padding,
            edge_guard=edge_guard,
            output_dir=output_dir,
            prefix=prefix,
            overwrite=overwrite,
            dry_run=dry_run,
        )
    except Exception as exc:
        _exit(str(exc))

    table = Table(title="Track boundaries")
    table.add_column("#", style="cyan")
    table.add_column("Start", style="magenta")
    table.add_column("End", style="magenta")
    table.add_column("Duration", style="green")
    for track in result.tracks:
        table.add_row(
            str(track.number), fmt_time(track.start), fmt_time(track.end), fmt_time(track.duration)
        )
    console.print(table)

    if dry_run:
        info(console, "Dry run complete; no files were written.")
        return

    _print_paths("Written tracks", result.output_paths)
    ok(console, f"Wrote {len(result.output_paths)} tracks")


@app.command("trim")
def trim_cmd(
    input: Annotated[Path, typer.Argument(help="Source AIFF/AIFC audio file.")],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output path. Defaults to INPUT_trimmed.aifc."),
    ] = None,
    min_silence: Annotated[
        float, typer.Option("--min-silence", "-m", help="Minimum edge silence duration in seconds.")
    ] = 1.5,
    threshold: Annotated[
        str,
        typer.Option(
            "--threshold",
            "-t",
            help="FFmpeg silencedetect threshold.",
            callback=_validate_threshold,
        ),
    ] = "-45dB",
    padding: Annotated[
        float, typer.Option("--padding", "-p", help="Padding retained at each edge.")
    ] = 0.020,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Overwrite existing output.")
    ] = False,
) -> None:
    """Trim long silence from both ends of an AIFF/AIFC file."""
    input = input.expanduser().resolve()
    if not input.is_file():
        _exit(f"Input file not found: {input}")

    section(console, "Trimming edge silence", "🎵")
    info(console, f"Input: {input}")
    info(console, f"Minimum silence: {min_silence}s")
    info(console, f"Threshold: {threshold}")

    try:
        result = trim_silence(
            input,
            min_silence=min_silence,
            threshold_db=threshold,
            padding=padding,
            output_path=output,
            overwrite=overwrite,
        )
    except Exception as exc:
        _exit(str(exc))

    info(console, f"Output: {result}")
    ok(console, "Trim complete")


@app.command()
def identify(
    audio: Annotated[
        Optional[Path], typer.Argument(help="Path to a single input audio file (aifc, m4a, etc.).")
    ] = None,
    input_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--input-dir",
            "--folder",
            "-i",
            "-d",
            help="Directory containing audio files to identify.",
        ),
    ] = None,
    pattern: Annotated[
        str,
        typer.Option(
            "--pattern",
            "-p",
            help="Pattern to match inside --input-dir. Use %% as a digit wildcard.",
        ),
    ] = "*.aifc",
    json_output: Annotated[
        bool, typer.Option("--json", help="Print the raw ShazamIO response(s) as JSON.")
    ] = False,
    rename: Annotated[
        bool,
        typer.Option("--rename", "-r", help="Rename matched files to 'artist-name_song-name.ext'."),
    ] = False,
    delay: Annotated[
        float,
        typer.Option(
            "--delay", help="Seconds to wait between batch requests to avoid rate limiting."
        ),
    ] = 3.0,
) -> None:
    """Identify the artist and song title using ShazamIO."""
    from .recognition import find_matching_files, format_track_filename, recognize_song

    if audio is not None and input_dir is not None:
        _exit("Specify either an audio file or --input-dir, not both.")

    files: list[Path] = []
    if input_dir is not None:
        input_dir = input_dir.expanduser().resolve()
        if not input_dir.is_dir():
            _exit(f"Directory not found: {input_dir}")
        files = find_matching_files(input_dir, pattern)
        if not files:
            _exit(f"No files matching '{pattern}' found in {input_dir}")
    elif audio is not None:
        audio = audio.expanduser().resolve()
        if not audio.is_file():
            _exit(f"Audio file not found: {audio}")
        files = [audio]
    else:
        _exit("Specify an audio file or --input-dir.")

    section(console, "Identifying music", "🎤")
    info(console, f"Files to identify: {len(files)}")

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
                warn(console, f"Recognition failed for {file_path.name}: {exc}")
                result = {"input_path": str(file_path), "error": str(exc)}

        if rename and "error" not in result:
            try:
                new_stem = format_track_filename(result.get("artist"), result.get("title"))
            except ValueError as exc:
                warn(console, f"Cannot rename {file_path.name}: {exc}")
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
                ok(console, f"Renamed {file_path.name} -> {target.name}")

        results.append(result)

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
    input_dir: Annotated[
        Path,
        typer.Option(
            "--input-dir", "--folder", "-i", "-d", help="Directory containing files to rename."
        ),
    ],
    remove_pattern: Annotated[
        Optional[str],
        typer.Option("--remove-pattern", "-p", help="Substring to remove from each filename."),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", "-n", help="Preview renames without applying them.")
    ] = False,
    recursive: Annotated[
        bool, typer.Option("--recursive", "-R", help="Rename files in subdirectories too.")
    ] = False,
    normalize_special_chars: Annotated[
        bool,
        typer.Option(
            "--normalize-special-chars",
            "-nsc",
            help="Remove accents and Latin special characters from filenames.",
        ),
    ] = False,
) -> None:
    """Remove a substring from and/or normalize all filenames in a folder."""
    from .renaming import rename_remove_pattern

    input_dir = input_dir.expanduser().resolve()
    section(console, "Renaming files", "🏷️")
    info(console, f"Directory: {input_dir}")

    try:
        renames = rename_remove_pattern(
            input_dir,
            remove_pattern or "",
            dry_run=dry_run,
            recursive=recursive,
            normalize=normalize_special_chars,
        )
    except Exception as exc:
        _exit(str(exc))

    if not renames:
        warn(console, "No files changed.")
        return

    action = "Would rename" if dry_run else "Renamed"
    table = Table(title=f"{action} Files")
    table.add_column("Original", style="cyan")
    table.add_column("New", style="magenta")
    for old_path, new_path in renames:
        table.add_row(old_path.name, new_path.name)
    console.print(table)
    ok(console, f"{action} {len(renames)} file(s)")


@app.command()
def install(
    update: Annotated[
        bool, typer.Option("--update", "-u", help="Upgrade/reinstall the package.")
    ] = False,
    bin_dir: Annotated[
        Path, typer.Option("--bin-dir", help="Directory for the system launcher.")
    ] = Path("/usr/local/bin"),
    editable: Annotated[
        bool, typer.Option("--editable/--no-editable", help="Install in editable mode.")
    ] = True,
) -> None:
    """Install ASP so it is available system-wide.

    Installs the package with pip and creates a launcher in ``/usr/local/bin``
    (or the directory given with ``--bin-dir``). Use ``--update`` to reinstall.
    """
    section(console, "Installing ASP", "🚀")

    project_root = Path(__file__).resolve().parents[2]
    pyproject = project_root / "pyproject.toml"
    if not pyproject.is_file():
        _exit(f"Could not find project root (expected pyproject.toml at {project_root})")

    info(console, f"Project root: {project_root}")
    info(console, f"Target bin directory: {bin_dir}")

    pip = [sys.executable, "-m", "pip"]
    cmd = pip + (["install", "--upgrade"] if update else ["install"])
    if editable:
        cmd.append("-e")
    cmd.append(str(project_root))

    info(console, f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        _exit(f"pip install failed (exit {exc.returncode})")

    # Find the installed asp console script.
    venv_bin = Path(sys.executable).parent
    script_source = venv_bin / "asp"
    if not script_source.is_file():
        _exit(f"Installed script not found at {script_source}")

    bin_dir = bin_dir.expanduser()
    bin_dir.mkdir(parents=True, exist_ok=True)
    launcher = bin_dir / "asp"

    if launcher.exists() or launcher.is_symlink():
        if update:
            info(console, f"Replacing existing launcher: {launcher}")
            launcher.unlink(missing_ok=True)
        else:
            _exit(f"Launcher already exists: {launcher}. Use --update to replace it.")

    try:
        launcher.symlink_to(script_source)
    except OSError as exc:
        _exit(f"Could not create launcher at {launcher}: {exc}. Try running with sudo.")

    ok(console, f"ASP installed at {launcher}")
    ok(console, f"Console script: {script_source}")
    info(console, "You can now run 'asp' from anywhere.")


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


def _print_paths(title: str, paths: list[Path]) -> None:
    table = Table(title=title)
    table.add_column("Path", style="magenta")
    for path in paths:
        table.add_row(str(path))
    console.print(table)


def _scan_to_dict(result: ScanResult) -> dict[str, Any]:
    return {
        "path": str(result.path),
        "duration_seconds": result.duration_seconds,
        "file_size_bytes": result.file_size_bytes,
        "format_name": result.format_name,
        "codec_name": result.codec_name,
        "sample_rate": result.sample_rate,
        "channels": result.channels,
        "channel_layout": result.channel_layout,
        "sample_format": result.sample_format,
        "declared_bits": result.declared_bits,
        "measured_bits": result.measured_bits,
        "samples": result.samples,
        "sample_peak_db": result.sample_peak_db,
        "sample_headroom_db": result.sample_headroom_db,
        "true_peak_db": result.true_peak_db,
        "true_peak_headroom_db": result.true_peak_headroom_db,
        "rms_level_db": result.rms_level_db,
        "rms_peak_db": result.rms_peak_db,
        "rms_trough_db": result.rms_trough_db,
        "integrated_loudness_lufs": result.integrated_loudness_lufs,
        "loudness_range_lu": result.loudness_range_lu,
        "loudness_threshold_lufs": result.loudness_threshold_lufs,
        "dc_offset": result.dc_offset,
        "crest_factor_db": result.crest_factor_db,
        "channel_stats": [
            {
                "channel": c.channel,
                "peak_db": c.peak_db,
                "rms_db": c.rms_db,
                "dc_offset": c.dc_offset,
            }
            for c in result.channel_stats
        ],
        "clipping_assessment": result.clipping_assessment,
    }


def _print_scan(result: ScanResult) -> None:
    kv_table(
        console,
        "File",
        [
            ("Path", result.path),
            ("Container", result.format_name or "unknown"),
            ("File size", human_size(result.file_size_bytes)),
            ("Duration", format_duration(result.duration_seconds)),
            ("Overall bitrate", _format_bitrate(result.info.bit_rate)),
        ],
    )
    kv_table(
        console,
        "Audio stream",
        [
            ("Codec", result.codec_name or "unknown"),
            ("Sample rate", f"{result.sample_rate} Hz" if result.sample_rate else "unknown"),
            ("Channels", result.channels or "unknown"),
            ("Channel layout", result.channel_layout or "unknown"),
            ("Sample format", result.sample_format or "unknown"),
            (
                "Declared bit depth",
                f"{result.declared_bits} bit" if result.declared_bits else "unknown",
            ),
            ("Measured bit depth", result.measured_bits or "unknown"),
            ("Analysed samples", result.samples or "unknown"),
        ],
    )
    kv_table(
        console,
        "Levels",
        [
            (
                "Sample peak",
                f"{result.sample_peak_db} dBFS" if result.sample_peak_db else "unknown",
            ),
            (
                "Sample headroom",
                f"{result.sample_headroom_db} dB" if result.sample_headroom_db else "unknown",
            ),
            ("True peak", f"{result.true_peak_db} dBTP" if result.true_peak_db else "unknown"),
            (
                "True-peak headroom",
                f"{result.true_peak_headroom_db} dB" if result.true_peak_headroom_db else "unknown",
            ),
            ("RMS level", f"{result.rms_level_db} dBFS" if result.rms_level_db else "unknown"),
            (
                "Maximum windowed RMS",
                f"{result.rms_peak_db} dBFS" if result.rms_peak_db else "unknown",
            ),
            (
                "Minimum windowed RMS",
                f"{result.rms_trough_db} dBFS" if result.rms_trough_db else "unknown",
            ),
        ],
    )
    kv_table(
        console,
        "Loudness",
        [
            (
                "Integrated loudness",
                f"{result.integrated_loudness_lufs} LUFS"
                if result.integrated_loudness_lufs
                else "unknown",
            ),
            (
                "Loudness range",
                f"{result.loudness_range_lu} LU" if result.loudness_range_lu else "unknown",
            ),
            (
                "Loudness threshold",
                f"{result.loudness_threshold_lufs} LUFS"
                if result.loudness_threshold_lufs
                else "unknown",
            ),
        ],
    )
    kv_table(
        console,
        "Signal statistics",
        [
            ("DC offset", result.dc_offset or "unknown"),
            (
                "Crest factor",
                f"{result.crest_factor_db} dB" if result.crest_factor_db else "unknown",
            ),
        ],
    )

    if result.channel_stats:
        table = Table(title="Per-channel measurements")
        table.add_column("Channel", style="cyan")
        table.add_column("Peak dBFS", style="magenta")
        table.add_column("RMS dBFS", style="magenta")
        table.add_column("DC offset", style="magenta")
        for ch in result.channel_stats:
            table.add_row(ch.channel, ch.peak_db, ch.rms_db, ch.dc_offset)
        console.print(table)

    info(console, f"Clipping assessment: {result.clipping_assessment}")


def _format_bitrate(bit_rate: int | None) -> str:
    if bit_rate is None or bit_rate <= 0:
        return "unknown"
    return f"{bit_rate / 1000:.0f} kb/s"


if __name__ == "__main__":
    app()
