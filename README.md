# ReverseScore

A Python pipeline for transcribing Argentine tango audio into clean MusicXML and MIDI scores for MuseScore.

## Pipeline

```
tango audio
   │
   ▼
demucs ──> isolated stems (bass, other/vocals/drums, ...)
   │
   ▼
basic-pitch per stem ──> MIDI per stem
   │
   ▼
music21 ──> quantize, set meter, merge, cleanup ──> MusicXML + MIDI
```

## Features

- **Source separation** with Meta's `demucs`
- **Polyphonic transcription** with Spotify's `basic-pitch`
- **Notation cleanup** with `music21`: time signature, 16th-note quantization, pitch-bend stripping, instrument/clef assignment
- **Tango-aware defaults**: 4/4 or 2/4 meter, 16th-note grid, bandoneon treble/bass split, marcato accents on bass/drums
- **CLI** built with `typer` and `rich`
- **Configuration** via CLI flags, environment variables, or `.env`

## Requirements

- Python >=3.9, <3.13 (basic-pitch and torch do not yet support 3.13)
- `uv` or `pip` for package management
- A working audio file (WAV, FLAC, MP3, etc.)

## Installation

```bash
# Clone the repository
git clone https://github.com/ReverseScore/ReverseScore.git
cd ReverseScore

# Create a virtual environment with Python 3.11/3.12
python3.11 -m venv .venv
source .venv/bin/activate

# Install the package
pip install -e ".[dev]"
```

Or with `uv`:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

The `asp` CLI works even when the virtual environment is not activated (e.g. `.venv/bin/asp`), because it searches the interpreter's own `bin` directory for tools like `demucs`.

### Install system-wide

After installing in a virtual environment, you can make `asp` available from anywhere with:

```bash
asp install
```

This creates a launcher in `/usr/local/bin/asp`. Run `asp install --update` to replace an existing launcher.

## Usage

### Full pipeline

```bash
asp transcribe data/wav/angelica_bjbn5mice4k.wav --time-signature 4/4
```

By default the command organizes outputs under `./data/trans/<song>/`:

- `./data/trans/angelica_bjbn5mice4k/stems/` — isolated audio stems
- `./data/trans/angelica_bjbn5mice4k/midi/` — one MIDI file per stem
- `./data/trans/angelica_bjbn5mice4k/scores/` — final `.musicxml` and `.mid`

### CLI options

```bash
asp transcribe --help
```

Key options:

| Flag | Description | Default |
|------|-------------|---------|
| `-o, --output-dir` | Output directory | `./data/trans/<song>/` |
| `-t, --time-signature` | Expected meter | `4/4` |
| `-g, --grid` | Quantization grid (4 = 16ths) | `4` |
| `--onset` | basic-pitch onset threshold | `0.5` |
| `--frame` | basic-pitch frame threshold | `0.3` |
| `--demucs-model` | demucs model name | `htdemucs` |
| `--split-bandoneon / --no-split-bandoneon` | Split bandoneon part | `True` |
| `--overwrite` | Re-run intermediate steps | `False` |
| `--bandoneon / --no-bandoneon` | Hint bandoneon presence/absence | inferred |
| `--violin / --no-violin` | Hint violin presence/absence | inferred |
| `--piano / --no-piano` | Hint piano presence/absence | inferred |
| `--voice / --no-voice` | Hint voice presence/absence | inferred |
| `--bass / --no-bass` | Hint bass presence/absence | inferred |
| `--drums / --no-drums` | Hint drums presence/absence | inferred |

### Convert audio files

ASP can convert a single file or a whole directory. The output format is inferred from `--output` or `--format`, and defaults to M4A (AAC in an MP4 container):

```bash
# Convert a single file to M4A/AAC (default)
asp convert data/m4a/angelica_BJBn5MICe4k.m4a

# Convert to WAV
asp convert data/m4a/angelica_BJBn5MICe4k.m4a --format wav

# Convert to a specific format
asp convert data/m4a/angelica_BJBn5MICe4k.m4a --format flac

# Convert AIFF/AIFC to AAC/M4A
asp convert recording.aifc --output recording.m4a

# Batch-convert a directory to M4A/AAC
asp convert data/m4a --output ./data/m4a

# Batch-convert a directory to WAV
asp convert data/m4a --output ./data/wav --format wav

# Convert with normalization and denoising
asp convert recording.aifc --output cleaned.m4a --normalize --denoise soft
```

Then transcribe the WAV:

```bash
asp transcribe ./data/wav/angelica_bjbn5mice4k.wav --time-signature 4/4
```

With instrument hints (e.g. a typical tango orquesta with no singer or drums):

```bash
asp transcribe ./data/wav/angelica_bjbn5mice4k.wav \
    --bandoneon --violin --piano --bass --no-voice --no-drums
```

### Run individual steps

Separate only:

```bash
asp separate path/to/tango.wav -o ./out
```

Transcribe existing stems:

```bash
asp stems-to-midi ./out/stems/tango/htdemucs/ -o ./out
```

### Audio file processing

ASP also includes utilities for working with audio files directly:

```bash
# Analyse an audio file
asp scan data/wav/angelica_bjbn5mice4k.wav

# Convert between formats with optional normalization/denoise
asp convert recording.aifc --output cleaned.m4a --normalize --denoise soft

# Convert to M4A/AAC (default)
asp convert recording.wav --output recording.m4a

# Trim edge silence
asp trim recording.aifc --min-silence 1.0 --threshold -40dB

# Split a long recording into tracks
asp split recording.aifc --tracks 10 --min-track 2:00 --max-track 4:00
```

## Configuration (`config.yaml`)

Create a `config.yaml` in the working directory to set defaults:

```yaml
# Directory containing input audio files (e.g. m4a, mp3, flac).
input_dir: ./data/m4a

# Directory for all pipeline outputs (stems, MIDI, scores).
output_dir: ./out

# Directory where ffmpeg-converted WAV files are written.
wav_output_dir: ./data/wav

# Path or command name for the ffmpeg executable.
ffmpeg_path: ffmpeg

# demucs model to use for source separation.
# Use htdemucs_6s to also separate piano from the "other" stem.
demucs_model: htdemucs

# Expected time signature for the score.
time_signature: "4/4"

# Quantization grid denominator (4 = 16th notes).
quantization_grid: 4

# Instrument hints help label staves correctly and skip excluded stems.
# Valid values: bandoneon, violin, piano, voice, bass, drums.
instrument_hints: []
instrument_exclusions: []
```

Override precedence: CLI flags > environment variables (`ASP_*`) > `config.yaml` > `.env` > defaults.

## Project Structure

```
ReverseScore/
├── src/asp/
│   ├── __init__.py
│   ├── cli.py              # Typer + Rich CLI
│   ├── config.py           # Pydantic settings (loads config.yaml)
│   ├── conversion.py       # ffmpeg m4a/mp3 -> WAV
│   ├── audio_convert.py    # General audio conversion (normalize, denoise)
│   ├── audio_scan.py       # Audio file analysis
│   ├── audio_split.py      # Split recordings by silence
│   ├── audio_trim.py       # Trim edge silence
│   ├── _ffmpeg.py          # Shared FFmpeg helpers
│   ├── _console.py         # Shared Rich/emoji helpers
│   ├── separation.py       # demucs wrapper
│   ├── transcription.py    # basic-pitch wrapper
│   ├── notation.py         # music21 quantization & export
│   ├── pipeline.py         # End-to-end orchestration
│   ├── recognition.py      # ShazamIO song identification
│   ├── renaming.py         # Bulk filename renaming
│   ├── yourmt3_remote.py   # YourMT3 remote backend
│   └── utils.py            # Shared helpers
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_conversion.py
│   ├── test_notation.py
│   ├── test_separation.py
│   ├── test_transcription.py
│   ├── test_recognition.py
│   ├── test_renaming.py
│   ├── test_utils.py
│   ├── test_pipeline.py
│   └── test_yourmt3.py
├── config.yaml             # User-editable defaults
├── pyproject.toml
└── README.md
```

## Development

Run tests:

```bash
pytest
```

Lint and format with `ruff`:

```bash
ruff check src tests
ruff format src tests
```

Type-check with `mypy`:

```bash
mypy src
```

## Notes and Limitations

- Dense tango polyphony is hard to transcribe perfectly. Treat the output as a draft and expect manual cleanup in MuseScore.
- The default `htdemucs` model groups bandoneon, piano, and violin into the `other` stem. More advanced orchestrations may benefit from custom separation models or manual splitting in MuseScore.
- Quantization uses a 16th-note grid by default. Rubato passages may need additional human editing.
- Pitch bends are currently stripped. Future versions may map them to glissando or grace notes.
- On first transcription, `basic-pitch` may print warnings about optional backends (TensorFlow, ONNX, TFLite). On macOS it uses CoreML; on Linux/Windows install `onnxruntime` or a TensorFlow build of basic-pitch if needed.

## License

MIT
