"""Tests for the YourMT3 remote transcription backend."""

import base64
from pathlib import Path
from unittest import mock

import pytest
from reversescore.config import PipelineConfig
from reversescore.transcription import transcribe_stem
from reversescore.yourmt3_remote import _extract_midi_bytes, transcribe_stem_yourmt3


def _minimal_midi_bytes() -> bytes:
    """Return a tiny but valid MIDI file (header + track with one note)."""
    # MThd header (1 track) + MTrk with Note On (C4, vel 64) + Note Off + EOT
    return bytes.fromhex(
        "4d546864"  # MThd
        "00000006"  # header length
        "0001"      # format 1
        "0001"      # 1 track
        "01e0"      # 480 ticks per quarter
        "4d54726b"  # MTrk
        "0000000b"  # track length 11
        "00"        # delta 0
        "903c40"    # Note On, C4, velocity 64
        "60"        # delta 96
        "803c40"    # Note Off, C4, velocity 64
        "00ff2f00"  # delta 0, end-of-track
    )


def test_extract_midi_bytes_decodes_data_uri() -> None:
    midi = _minimal_midi_bytes()
    b64 = base64.b64encode(midi).decode("ascii")
    html = f'<div><script src="data:audio/midi;base64,{b64}"></script></div>'
    assert _extract_midi_bytes(html) == midi


def test_extract_midi_bytes_prefers_non_empty_uri() -> None:
    empty = bytes.fromhex("4d546864000000060001000101e04d54726b0000000400ff2f00")
    real = _minimal_midi_bytes()
    empty_b64 = base64.b64encode(empty).decode("ascii")
    real_b64 = base64.b64encode(real).decode("ascii")
    html = (
        f'<a href="data:audio/midi;base64,{empty_b64}"></a>'
        f'<a href="data:audio/midi;base64,{real_b64}"></a>'
    )
    assert _extract_midi_bytes(html) == real


def test_extract_midi_bytes_missing_uri_raises() -> None:
    with pytest.raises(ValueError, match="did not contain a MIDI data URI"):
        _extract_midi_bytes("<html></html>")


def test_transcribe_stem_yourmt3_reuses_existing(tmp_path: Path) -> None:
    audio = tmp_path / "other.wav"
    audio.write_bytes(b"fake audio")
    midi_dir = tmp_path / "midi"
    midi_dir.mkdir()
    existing = midi_dir / "other.mid"
    existing.touch()

    config = PipelineConfig(transcription_backend="yourmt3")
    with mock.patch("reversescore.yourmt3_remote._load_gradio_client") as mock_load:
        result = transcribe_stem_yourmt3(audio, midi_dir, config, label="other")
        mock_load.assert_not_called()

    assert result == existing


def test_transcribe_stem_yourmt3_calls_space_and_saves(tmp_path: Path) -> None:
    audio = tmp_path / "other.wav"
    audio.write_bytes(b"fake audio")
    midi_dir = tmp_path / "midi"

    midi = _minimal_midi_bytes()
    b64 = base64.b64encode(midi).decode("ascii")
    fake_html = f'<div src="data:audio/midi;base64,{b64}"></div>'

    fake_client = mock.MagicMock()
    fake_client.predict.return_value = fake_html
    fake_handle_file = mock.MagicMock(return_value={"path": str(audio)})
    fake_Client = mock.MagicMock(return_value=fake_client)

    config = PipelineConfig(transcription_backend="yourmt3")
    with mock.patch(
        "reversescore.yourmt3_remote._load_gradio_client",
        return_value=(fake_Client, fake_handle_file),
    ):
        result = transcribe_stem_yourmt3(audio, midi_dir, config, label="other")

    assert result == midi_dir / "other.mid"
    assert result.read_bytes() == midi
    fake_Client.assert_called_once_with("mimbres/YourMT3")
    fake_client.predict.assert_called_once_with(
        audio_filepath=fake_handle_file.return_value,
        api_name="/process_audio",
    )


def test_transcribe_stem_dispatcher_routes_to_yourmt3(tmp_path: Path) -> None:
    audio = tmp_path / "piano.wav"
    audio.write_bytes(b"fake audio")
    midi_dir = tmp_path / "midi"

    midi = _minimal_midi_bytes()
    b64 = base64.b64encode(midi).decode("ascii")
    fake_html = f'<div src="data:audio/midi;base64,{b64}"></div>'

    fake_client = mock.MagicMock()
    fake_client.predict.return_value = fake_html
    fake_handle_file = mock.MagicMock()
    fake_Client = mock.MagicMock(return_value=fake_client)

    config = PipelineConfig(transcription_backend="basic-pitch")
    with mock.patch(
        "reversescore.yourmt3_remote._load_gradio_client",
        return_value=(fake_Client, fake_handle_file),
    ):
        result = transcribe_stem(
            audio, midi_dir, config, label="piano", backend="yourmt3"
        )

    assert result == midi_dir / "piano.mid"
    assert result.read_bytes() == midi


def test_transcribe_stem_dispatcher_invalid_backend_raises(tmp_path: Path) -> None:
    audio = tmp_path / "piano.wav"
    audio.write_bytes(b"fake audio")
    config = PipelineConfig()
    with pytest.raises(ValueError, match="Unsupported transcription backend"):
        transcribe_stem(audio, tmp_path / "midi", config, backend="not-real")
