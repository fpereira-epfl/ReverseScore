"""Tests for transcription."""

from pathlib import Path
from unittest import mock

from asp.config import PipelineConfig
from asp.transcription import transcribe_stem


def test_transcribe_stem_reuses_existing(tmp_path: Path) -> None:
    audio = tmp_path / "bass.wav"
    audio.write_text("fake audio")
    midi_dir = tmp_path / "midi"
    midi_dir.mkdir()
    existing = midi_dir / "bass.mid"
    existing.touch()

    config = PipelineConfig()
    mock_predict = mock.MagicMock()
    with mock.patch(
        "asp.transcription._basic_pitch_imports",
        return_value=("model", mock_predict),
    ):
        result = transcribe_stem(audio, midi_dir, config, label="bass")
        mock_predict.assert_not_called()

    assert result == existing
