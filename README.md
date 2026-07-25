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

The `reversescore` CLI works even when the virtual environment is not activated (e.g. `.venv/bin/reversescore`), because it searches the interpreter's own `bin` directory for tools like `demucs`.

## Usage

### Full pipeline

```bash
reversescore transcribe path/to/tango.wav -o ./out --time-signature 4/4
```

The command produces:

- `./out/stems/` — isolated audio stems
- `./out/midi/` — one MIDI file per stem
- `./out/scores/` — final `tango.musicxml` and `tango.mid`

### CLI options

```bash
reversescore transcribe --help
```

Key options:

| Flag | Description | Default |
|------|-------------|---------|
| `-o, --output-dir` | Output directory | `./out` |
| `-t, --time-signature` | Expected meter | `4/4` |
| `-g, --grid` | Quantization grid (4 = 16ths) | `4` |
| `--onset` | basic-pitch onset threshold | `0.5` |
| `--frame` | basic-pitch frame threshold | `0.3` |
| `--demucs-model` | demucs model name | `htdemucs` |
| `--split-bandoneon / --no-split-bandoneon` | Split bandoneon part | `True` |
| `--overwrite` | Re-run intermediate steps | `False` |

### Run individual steps

Separate only:

```bash
reversescore separate path/to/tango.wav -o ./out
```

Transcribe existing stems:

```bash
reversescore stems-to-midi ./out/stems/tango/htdemucs/ -o ./out
```

## Project Structure

```
ReverseScore/
├── src/reversescore/
│   ├── __init__.py
│   ├── cli.py              # Typer + Rich CLI
│   ├── config.py           # Pydantic settings
│   ├── separation.py       # demucs wrapper
│   ├── transcription.py    # basic-pitch wrapper
│   ├── notation.py         # music21 quantization & export
│   ├── pipeline.py         # End-to-end orchestration
│   └── utils.py            # Shared helpers
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_notation.py
│   ├── test_separation.py
│   └── test_transcription.py
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
