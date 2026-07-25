"""Tests for end-to-end pipeline orchestration."""

from pathlib import Path
from unittest import mock

from music21 import note, stream
from reversescore.config import PipelineConfig
from reversescore.pipeline import run_pipeline


def _make_stem_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")


def test_run_pipeline(tmp_path: Path) -> None:
    audio = tmp_path / "tango.wav"
    _make_stem_wav(audio)

    config = PipelineConfig(output_dir=tmp_path / "out")

    stem_paths = {
        "bass": tmp_path / "stems" / "bass.wav",
        "vocals": tmp_path / "stems" / "vocals.wav",
    }
    for p in stem_paths.values():
        _make_stem_wav(p)

    midi_paths = {
        "bass": tmp_path / "midi" / "bass.mid",
        "vocals": tmp_path / "midi" / "vocals.mid",
    }
    for p in midi_paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
        s = stream.Score()
        part = stream.Part()
        part.append(note.Note("C4", quarterLength=4))
        s.insert(0, part)
        s.write("midi", fp=str(p))

    with (
        mock.patch("reversescore.pipeline.separate_stems", return_value=stem_paths) as mock_sep,
        mock.patch("reversescore.pipeline.transcribe_stems", return_value=midi_paths) as mock_trans,
    ):
        result = run_pipeline(audio, config)

    mock_sep.assert_called_once()
    mock_trans.assert_called_once()
    assert result.musicxml_path.exists()
    assert result.midi_score_path.exists()
    assert len(result.score.parts) == 2
