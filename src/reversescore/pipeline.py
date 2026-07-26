"""High-level orchestration of the audio-to-score pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from music21 import stream

from .config import PipelineConfig
from .notation import build_score, export_score
from .separation import separate_stems
from .transcription import transcribe_stems
from .utils import ensure_dirs, safe_stem_name

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Output paths produced by a successful pipeline run."""

    stems: dict[str, Path]
    midi_paths: dict[str, Path]
    musicxml_path: Path
    midi_score_path: Path
    score: stream.Score


def run_pipeline(
    audio_path: Path,
    config: PipelineConfig,
    overwrite_separation: bool = False,
    overwrite_transcription: bool = False,
    transcription_backend: str | None = None,
) -> PipelineResult:
    """Run the full audio-to-score pipeline.

    Args:
        audio_path: Path to the input audio file.
        config: Pipeline configuration.
        overwrite_separation: Re-run demucs even if stems exist.
        overwrite_transcription: Re-run transcription even if MIDI exists.
        transcription_backend: Optional backend override for transcription.

    Returns:
        ``PipelineResult`` with all generated artifact paths and the music21 Score.
    """
    audio_path = audio_path.expanduser().resolve()
    ensure_dirs(config.output_dir, config.separation_dir, config.midi_dir, config.score_dir)

    # 1. Separate
    logger.info("Step 1/3: source separation")
    stems = separate_stems(audio_path, config, overwrite=overwrite_separation)

    # 2. Transcribe
    logger.info("Step 2/3: transcription")
    midi_paths = transcribe_stems(
        stems,
        config,
        overwrite=overwrite_transcription,
        backend=transcription_backend,
    )

    # 3. Notation assembly
    logger.info("Step 3/3: notation assembly")
    title = audio_path.stem
    score = build_score(midi_paths, config, title=title)

    base_name = safe_stem_name(audio_path.stem)
    musicxml_path = config.score_dir / f"{base_name}.musicxml"
    midi_score_path = config.score_dir / f"{base_name}.mid"

    export_score(score, musicxml_path, fmt="musicxml")
    export_score(score, midi_score_path, fmt="midi")

    return PipelineResult(
        stems=stems,
        midi_paths=midi_paths,
        musicxml_path=musicxml_path,
        midi_score_path=midi_score_path,
        score=score,
    )
