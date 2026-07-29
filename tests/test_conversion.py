"""Tests for ffmpeg audio conversion utilities."""

from pathlib import Path
from unittest import mock

from asp.config import PipelineConfig
from asp.conversion import convert_directory_to_wav, convert_to_wav


def test_convert_to_wav_reuses_existing(tmp_path: Path) -> None:
    audio = tmp_path / "song.m4a"
    audio.write_text("fake audio")
    output = tmp_path / "song.wav"
    output.touch()

    config = PipelineConfig(output_dir=tmp_path, ffmpeg_path="ffmpeg")
    with mock.patch("asp.conversion.run_command") as mock_run:
        result = convert_to_wav(audio, output, config)
        mock_run.assert_not_called()
    assert result == output


def test_convert_to_wav_runs_ffmpeg(tmp_path: Path) -> None:
    audio = tmp_path / "song.m4a"
    audio.write_text("fake audio")
    output = tmp_path / "song.wav"

    config = PipelineConfig(output_dir=tmp_path, ffmpeg_path="ffmpeg")
    with (
        mock.patch("asp.conversion.run_command") as mock_run,
        mock.patch("asp.conversion.ensure_dirs"),
        mock.patch("asp.conversion.find_program", return_value=Path("/usr/bin/ffmpeg")),
    ):
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""
        # Simulate ffmpeg creating the output file.
        output.touch()
        result = convert_to_wav(audio, output, config, overwrite=True)

    assert result == output
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "/usr/bin/ffmpeg"
    assert "-ar" in cmd
    assert "44100" in cmd


def test_convert_directory_to_wav(tmp_path: Path) -> None:
    (tmp_path / "a.m4a").write_text("fake")
    (tmp_path / "b.mp3").write_text("fake")
    (tmp_path / "notes.txt").write_text("not audio")

    out_dir = tmp_path / "wav"
    config = PipelineConfig(output_dir=tmp_path, ffmpeg_path="ffmpeg")

    def _fake_convert(file: Path, output: Path, cfg: PipelineConfig, **kwargs: object) -> Path:
        output.touch()
        return output

    with mock.patch("asp.conversion.convert_to_wav", side_effect=_fake_convert):
        results = convert_directory_to_wav(tmp_path, out_dir, config)

    assert len(results) == 2
    assert "a" in results
    assert "b" in results
    assert all(p.suffix == ".wav" for p in results.values())
