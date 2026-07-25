"""Polyphonic transcription of separated stems using Spotify's basic-pitch."""

from __future__ import annotations

import contextlib
import io
import logging
import warnings
from pathlib import Path
from typing import Any, Callable

from .config import PipelineConfig
from .utils import ensure_dirs, safe_stem_name

logger = logging.getLogger(__name__)


def _basic_pitch_imports() -> tuple[str, Callable[..., Any]]:
    """Lazy import basic-pitch to avoid heavy dependencies/warnings on CLI startup."""
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict_and_save

    return ICASSP_2022_MODEL_PATH, predict_and_save


def transcribe_stem(
    audio_path: Path,
    output_dir: Path,
    config: PipelineConfig,
    label: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Transcribe a single stem audio file to MIDI with basic-pitch.

    Args:
        audio_path: Path to the stem audio file.
        output_dir: Directory for the generated MIDI file.
        config: Pipeline configuration.
        label: Optional label used for the output filename.
        overwrite: Re-transcribe even if MIDI already exists.

    Returns:
        Path to the generated ``.mid`` file.

    Raises:
        FileNotFoundError: If the input audio file does not exist.
    """
    audio_path = audio_path.expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Stem audio not found: {audio_path}")

    ensure_dirs(output_dir)
    name = safe_stem_name(label or audio_path.stem)
    midi_path = output_dir / f"{name}.mid"

    if midi_path.exists() and not overwrite:
        logger.info("Reusing existing MIDI: %s", midi_path)
        return midi_path

    logger.info("Transcribing %s -> %s", audio_path, midi_path)
    model_path, predict_and_save = _basic_pitch_imports()

    # basic-pitch and coremltools emit a lot of noise (debug prints about
    # tensor shapes, backend availability warnings, version warnings). Capture
    # stdout/stderr and ignore warnings during inference; only surface them if
    # transcription actually fails.
    bp_out = io.StringIO()
    bp_err = io.StringIO()
    # basic-pitch and coremltools log their optional-backend/version warnings
    # through Python logging; temporarily raise their level to ERROR.
    noisy_loggers = [
        logging.getLogger(name)
        for name in ("basic_pitch", "coremltools", "basic_pitch.inference")
    ]
    old_levels = [(lg, lg.level) for lg in noisy_loggers]
    for lg, _ in old_levels:
        lg.setLevel(logging.ERROR)

    try:
        with (
            contextlib.redirect_stdout(bp_out),
            contextlib.redirect_stderr(bp_err),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore")
            predict_and_save(
                [str(audio_path)],
                output_directory=str(output_dir),
                save_midi=True,
                sonify_midi=False,
                save_model_outputs=False,
                save_notes=False,
                model_or_model_path=model_path,
                onset_threshold=config.onset_threshold,
                frame_threshold=config.frame_threshold,
                minimum_note_length=config.min_note_length_ms,
                minimum_frequency=None,
                maximum_frequency=None,
                multiple_pitch_bends=True,
                melodia_trick=True,
            )
    except Exception as exc:
        err_text = bp_err.getvalue().strip()
        out_text = bp_out.getvalue().strip()
        if err_text:
            logger.error("basic-pitch stderr:\n%s", err_text)
        if out_text:
            logger.error("basic-pitch stdout:\n%s", out_text)
        raise RuntimeError(f"basic-pitch failed for {audio_path}") from exc
    finally:
        for lg, level in old_levels:
            lg.setLevel(level)

    # basic-pitch names the output based on the input filename and appends
    # "_basic_pitch" before the extension.
    produced = output_dir / f"{audio_path.stem}_basic_pitch.mid"
    if produced.is_file() and produced != midi_path:
        produced.rename(midi_path)
    if not midi_path.is_file():
        raise FileNotFoundError(f"basic-pitch did not produce expected MIDI: {midi_path}")
    return midi_path


def transcribe_stems(
    stems: dict[str, Path],
    config: PipelineConfig,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Transcribe each separated stem into its own MIDI file.

    Args:
        stems: Mapping from stem label to audio path.
        config: Pipeline configuration.
        overwrite: Re-transcribe existing MIDI files.

    Returns:
        Mapping from stem label to MIDI path.
    """
    midi_dir = config.midi_dir
    results: dict[str, Path] = {}
    for label, audio_path in stems.items():
        midi_path = transcribe_stem(
            audio_path=audio_path,
            output_dir=midi_dir,
            config=config,
            label=label,
            overwrite=overwrite,
        )
        results[label] = midi_path
    return results
