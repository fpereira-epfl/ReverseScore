"""Tests for tempo adjustment."""

from pathlib import Path
from unittest import mock

from asp.audio_tempo import change_tempo
from asp.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def _fake_audio_info(duration: float = 120.0):
    from asp._ffmpeg import AudioInfo

    return AudioInfo(
        duration=duration,
        codec="aac",
        sample_rate=44100,
        channels=2,
    )


def test_change_tempo_default_output(tmp_path: Path) -> None:
    audio = tmp_path / "song.m4a"
    audio.write_text("fake audio")
    expected_output = tmp_path / "song_0.85x.m4a"

    def _fake_run_ffmpeg(args: list[str]) -> object:
        expected_output.write_bytes(b"x" * 2000)
        return mock.Mock(stdout="", stderr="", returncode=0)

    with (
        mock.patch("asp.audio_tempo.require_ffmpeg"),
        mock.patch("asp.audio_tempo.has_filter", return_value=True),
        mock.patch("asp.audio_tempo.probe_audio", return_value=_fake_audio_info()),
        mock.patch("asp.audio_tempo.has_encoder", return_value=True),
        mock.patch("asp.audio_tempo.run_ffmpeg", side_effect=_fake_run_ffmpeg) as mock_run,
    ):
        result = change_tempo(audio, 0.85)
        mock_run.assert_called_once()

    assert result == expected_output
    args = mock_run.call_args[0][0]
    assert "-filter:a" in args
    assert "rubberband=tempo=0.85" in args


def test_change_tempo_custom_output(tmp_path: Path) -> None:
    audio = tmp_path / "song.m4a"
    audio.write_text("fake audio")
    output = tmp_path / "slow.mp3"

    def _fake_run_ffmpeg(args: list[str]) -> object:
        output.write_bytes(b"x" * 2000)
        return mock.Mock(stdout="", stderr="", returncode=0)

    with (
        mock.patch("asp.audio_tempo.require_ffmpeg"),
        mock.patch("asp.audio_tempo.has_filter", return_value=True),
        mock.patch("asp.audio_tempo.probe_audio", return_value=_fake_audio_info()),
        mock.patch("asp.audio_tempo.has_encoder", return_value=True),
        mock.patch("asp.audio_tempo.run_ffmpeg", side_effect=_fake_run_ffmpeg) as mock_run,
    ):
        result = change_tempo(audio, 1.25, output_path=output)
        mock_run.assert_called_once()

    assert result == output
    args = mock_run.call_args[0][0]
    assert "rubberband=tempo=1.25" in args
    assert "libmp3lame" in args


def test_change_tempo_invalid_factor(tmp_path: Path) -> None:
    audio = tmp_path / "song.m4a"
    audio.write_text("fake audio")

    with (
        mock.patch("asp.audio_tempo.require_ffmpeg"),
        mock.patch("asp.audio_tempo.has_filter", return_value=True),
        mock.patch("asp.audio_tempo.probe_audio", return_value=_fake_audio_info()),
    ):
        try:
            change_tempo(audio, -0.5)
        except Exception as exc:
            assert "positive" in str(exc).lower()


def test_tempo_cli_command(tmp_path: Path) -> None:
    audio = tmp_path / "song.m4a"
    audio.write_text("fake audio")
    output = tmp_path / "song_slow.m4a"

    with mock.patch("asp.cli.change_tempo") as mock_change:
        mock_change.return_value = output
        result = runner.invoke(
            app,
            ["tempo", str(audio), "-o", str(output), "-f", "0.85"],
        )

    assert result.exit_code == 0, result.output
    mock_change.assert_called_once_with(audio, 0.85, output_path=output, overwrite=False)
    assert "0.85" in result.output
