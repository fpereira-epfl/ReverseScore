"""Configuration and validated settings for the ASP pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict
from pydantic_settings.sources import YamlConfigSettingsSource

# Instruments the pipeline knows how to label in the score.
KNOWN_INSTRUMENTS = frozenset({"bandoneon", "violin", "piano", "voice", "bass", "drums", "guitar"})
InstrumentName = Literal["bandoneon", "violin", "piano", "voice", "bass", "drums", "guitar"]

# Supported transcription backends.
TranscriptionBackend = Literal["basic-pitch", "yourmt3"]


class PipelineConfig(BaseSettings):
    """Validated configuration for the audio-to-score pipeline.

    Values are loaded from (highest to lowest precedence):

    1. Constructor arguments / CLI flags
    2. Environment variables prefixed with ``REVERSESCORE_``
    3. A ``config.yaml`` file in the working directory
    4. A ``.env`` file in the working directory
    5. Defaults defined in this class
    """

    model_config = SettingsConfigDict(
        env_prefix="ASP_",
        env_file=".env",
        env_file_encoding="utf-8",
        yaml_file="config.yaml",
        yaml_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    input_dir: Path = Field(
        default=Path("./data/m4a"),
        description="Directory containing input audio files",
    )
    output_dir: Path = Field(
        default=Path("./out"),
        description="Directory for all pipeline outputs",
    )
    wav_output_dir: Path = Field(
        default=Path("./data/wav"),
        description="Directory where ffmpeg-converted WAV files are written",
    )

    # External binaries
    ffmpeg_path: str = Field(
        default="ffmpeg",
        description="Path or name of the ffmpeg executable",
    )

    # Source separation
    demucs_model: str = Field(
        default="htdemucs",
        description="demucs model variant to use",
    )
    demucs_stems: list[str] = Field(
        default_factory=lambda: ["drums", "bass", "other", "vocals"],
        description="Stems to request from demucs (varies by model)",
    )

    # Instrument hints / exclusions. These do not improve source separation;
    # they are used to label staves correctly and skip excluded stems.
    instrument_hints: list[InstrumentName] = Field(
        default_factory=list,
        description="Instruments expected to be present in the audio",
    )
    instrument_exclusions: list[InstrumentName] = Field(
        default_factory=list,
        description="Instruments known to be absent from the audio",
    )

    demucs_segment: float | None = Field(
        default=None,
        description="Segment length in seconds for demucs inference",
    )
    demucs_device: str | None = Field(
        default=None,
        description="Device for demucs ('cpu', 'cuda', or None for auto)",
    )

    @field_validator("demucs_model")
    @classmethod
    def _validate_demucs_model(cls, value: str) -> str:
        allowed = {"htdemucs", "htdemucs_ft", "htdemucs_6s", "hdemucs_mmi"}
        if value not in allowed:
            raise ValueError(f"demucs_model must be one of {allowed}, got {value!r}")
        return value

    @field_validator("time_signature")
    @classmethod
    def _validate_time_signature(cls, value: str) -> str:
        from music21 import meter

        try:
            meter.TimeSignature(value)
        except Exception as exc:
            raise ValueError(f"Invalid time signature: {value!r}") from exc
        return value

    @field_validator("instrument_hints", "instrument_exclusions")
    @classmethod
    def _validate_instrument_list(cls, value: list[str]) -> list[str]:
        invalid = [v for v in value if v not in KNOWN_INSTRUMENTS]
        if invalid:
            raise ValueError(
                f"Invalid instruments: {invalid}. Must be one of {sorted(KNOWN_INSTRUMENTS)}."
            )
        return value

    @model_validator(mode="after")
    def _set_default_stems_for_model(self) -> PipelineConfig:
        """Use 6-stem defaults when the htdemucs_6s model is selected."""
        default_4 = ["drums", "bass", "other", "vocals"]
        default_6 = ["drums", "bass", "other", "vocals", "guitar", "piano"]
        if self.demucs_model == "htdemucs_6s" and self.demucs_stems == default_4:
            self.demucs_stems = default_6
        return self

    def is_instrument_excluded(self, name: InstrumentName) -> bool:
        """Return True if the instrument is explicitly excluded."""
        return name in self.instrument_exclusions

    # Transcription
    transcription_backend: Literal["basic-pitch", "yourmt3"] = Field(
        default="basic-pitch",
        description="Transcription backend to use per stem",
    )
    yourmt3_space_id: str = Field(
        default="mimbres/YourMT3",
        description="HuggingFace Space ID for the YourMT3 remote backend",
    )
    yourmt3_timeout_seconds: int = Field(
        default=600,
        ge=10,
        description="Maximum time to wait for a YourMT3 Space transcription",
    )

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

    @field_validator("output_dir", "input_dir", "wav_output_dir", mode="before")
    @classmethod
    def _resolve_path(cls, value: str | Path) -> Path:
        return Path(value).expanduser().resolve()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Load config.yaml between env vars and dotenv defaults."""
        yaml_source = YamlConfigSettingsSource(settings_cls)
        return (
            init_settings,
            env_settings,
            yaml_source,
            dotenv_settings,
        )

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

    @property
    def wav_dir(self) -> Path:
        """Directory where converted WAV files are written."""
        return self.wav_output_dir
