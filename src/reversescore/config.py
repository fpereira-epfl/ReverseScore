"""Configuration and validated settings for the ReverseScore pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineConfig(BaseSettings):
    """Validated configuration for the audio-to-score pipeline.

    Values can be provided via constructor arguments, environment variables
    prefixed with ``REVERSESCORE_``, or a ``.env`` file in the working directory.
    """

    model_config = SettingsConfigDict(
        env_prefix="REVERSESCORE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    output_dir: Path = Field(default=Path("./out"), description="Directory for all outputs")

    # Source separation
    demucs_model: str = Field(
        default="htdemucs",
        description="demucs model variant to use",
    )
    demucs_stems: list[str] = Field(
        default_factory=lambda: ["drums", "bass", "other", "vocals"],
        description="Stems to request from demucs (varies by model)",
    )
    demucs_segment: float | None = Field(
        default=None,
        description="Segment length in seconds for demucs inference",
    )

    @field_validator("demucs_model")
    @classmethod
    def _validate_demucs_model(cls, value: str) -> str:
        allowed = {"htdemucs", "htdemucs_ft", "htdemucs_6s", "hdemucs_mmi"}
        if value not in allowed:
            raise ValueError(f"demucs_model must be one of {allowed}, got {value!r}")
        return value
    demucs_device: str | None = Field(
        default=None,
        description="Device for demucs ('cpu', 'cuda', or None for auto)",
    )

    # Transcription
    onset_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="basic-pitch onset detection threshold",
    )
    frame_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="basic-pitch frame detection threshold",
    )
    min_note_length_ms: int = Field(
        default=50,
        ge=0,
        description="Minimum note length in milliseconds",
    )
    midi_tempo: int = Field(
        default=120,
        ge=1,
        description="Tempo used for intermediate MIDI files (BPM)",
    )

    # Notation cleanup
    time_signature: str = Field(
        default="4/4",
        description="Expected time signature for the score",
    )
    quantization_grid: int = Field(
        default=4,
        ge=1,
        description="music21 quantization tuple denominator (4 = 16th notes)",
    )
    min_quarter_length: float = Field(
        default=0.0625,
        gt=0.0,
        description="Smallest note value to keep after quantization",
    )

    # Tango-specific defaults
    split_bandoneon: bool = Field(
        default=True,
        description="Split bandoneon transcription into treble/bass staves when possible",
    )
    tango_meter: Literal["2/4", "4/4"] = Field(
        default="4/4",
        description="Default tango meter",
    )

    @field_validator("output_dir", mode="before")
    @classmethod
    def _resolve_path(cls, value: str | Path) -> Path:
        return Path(value).expanduser().resolve()

    @property
    def separation_dir(self) -> Path:
        """Directory where demucs stems are written."""
        return self.output_dir / "stems"

    @property
    def midi_dir(self) -> Path:
        """Directory where per-stem MIDI files are written."""
        return self.output_dir / "midi"

    @property
    def score_dir(self) -> Path:
        """Directory where final scores are written."""
        return self.output_dir / "scores"
