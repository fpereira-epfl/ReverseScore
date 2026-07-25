"""Notation cleanup, quantization, and score assembly with music21."""

from __future__ import annotations

import contextlib
import io
import logging
from pathlib import Path

import music21
from music21 import articulations, converter, instrument, meter, note, stream

from .config import PipelineConfig
from .utils import ensure_dirs

logger = logging.getLogger(__name__)


def _instrument(name: str) -> instrument.Instrument:
    """Return a music21 Instrument for a known instrument name."""
    if name == "bandoneon":
        return instrument.Instrument(
            instrumentName="Bandoneon",
            instrumentAbbreviation="Band.",
            midiProgram=21,  # Accordion is the closest General MIDI match.
        )
    mapping: dict[str, instrument.Instrument] = {
        "violin": instrument.Violin(),
        "piano": instrument.Piano(),
        "voice": instrument.Vocalist(),
        "bass": instrument.Contrabass(),
        "drums": instrument.Percussion(),
        "guitar": instrument.Guitar(),
    }
    return mapping.get(name, _instrument("bandoneon"))


def _clef_for(name: str) -> music21.clef.Clef:
    """Return a sensible clef for an instrument name."""
    name_lower = name.lower()
    if name_lower in {"bass", "bandoneon_bass"}:
        return music21.clef.BassClef()
    if name_lower == "drums":
        return music21.clef.PercussionClef()
    return music21.clef.TrebleClef()


def _resolve_stem_instrument(label: str, config: PipelineConfig) -> str:
    """Map a demucs stem label to an instrument name using user hints.

    The ``other``/``bandoneon_violin`` stem is ambiguous. If the user hints
    contain exactly one of {bandoneon, violin}, we label it that instrument.
    Otherwise we keep a combined label.
    """
    direct_map: dict[str, str] = {
        "drums": "drums",
        "bass": "bass",
        "voice": "voice",
        "piano": "piano",
        "guitar": "guitar",
    }
    if label in direct_map:
        return direct_map[label]

    # Combined melodic stem from htdemucs.
    hints = set(config.instrument_hints)
    if label in {"bandoneon_violin", "other", "bandoneon_piano_violin"}:
        candidates = hints & {"bandoneon", "violin", "piano", "guitar"}
        if len(candidates) == 1:
            return candidates.pop()
        if "bandoneon" in hints and "violin" not in hints and "piano" not in hints:
            return "bandoneon"
        if "violin" in hints and "bandoneon" not in hints and "piano" not in hints:
            return "violin"
        if "piano" in hints and "bandoneon" not in hints and "violin" not in hints:
            return "piano"
        return "bandoneon_violin"

    return label


def _is_excluded(label: str, config: PipelineConfig) -> bool:
    """Return True if the resolved instrument for a stem is excluded."""
    resolved = _resolve_stem_instrument(label, config)
    return config.is_instrument_excluded(resolved)  # type: ignore[arg-type]


def load_midi(path: Path) -> stream.Score:
    """Load a MIDI file into a music21 Score.

    Some transcription tools (e.g. basic-pitch with ``multiple_pitch_bends=True``)
    write many tiny tracks for a single instrument. If we detect an unusually
    large number of parts, merge them into one part so the final score does not
    explode into dozens of staves.
    """
    logger.debug("Loading MIDI: %s", path)
    score = converter.parse(str(path))
    if len(score.parts) <= 1:
        return score

    logger.debug("MIDI has %d parts; merging into one", len(score.parts))
    merged = stream.Part()
    for part in score.parts:
        if merged.getInstrument(returnDefault=None) is None:
            inst = part.getInstrument(returnDefault=None)
            if inst is not None:
                merged.insert(0, inst)
        if not list(merged.flatten().getElementsByClass("Clef")):
            clefs = list(part.flatten().getElementsByClass("Clef"))
            if clefs:
                merged.insert(0, clefs[0])
        for el in part.flatten().notesAndRests:
            merged.insert(el.offset, el)

    merged_score = stream.Score()
    merged_score.insert(0, merged)
    return merged_score


def set_time_signature(score: stream.Score, time_signature: str) -> stream.Score:
    """Insert the requested time signature at the start of every part."""
    ts = meter.TimeSignature(time_signature)
    for part in score.parts:
        measure_zero = part.measure(0)
        if measure_zero is None:
            part.insert(0, ts)
        else:
            existing = list(measure_zero.getElementsByClass("TimeSignature"))
            if existing:
                measure_zero.remove(existing[0])
            measure_zero.insert(0, ts)
    return score


def quantize_score(
    score: stream.Score,
    quantization_grid: int = 4,
    min_quarter_length: float = 0.0625,
    recurse: bool = True,
) -> stream.Score:
    """Quantize note/rest offsets and durations to a fixed grid.

    Args:
        score: music21 Score to quantize.
        quantization_grid: music21 tuple denominator; 4 = 16th notes.
        min_quarter_length: Smallest note value to retain (quarter-note units).
        recurse: Apply recursively to all contained streams.

    Returns:
        Quantized score.
    """
    logger.info(
        "Quantizing to 1/%d quarter-note grid (min length %.4f)",
        quantization_grid,
        min_quarter_length,
    )
    score.quantize(
        (quantization_grid,),
        processOffsets=True,
        processDurations=True,
        recurse=recurse,
        inPlace=True,
    )

    # Remove notes that are too short after quantization.
    for el in score.recurse().notesAndRests:
        if el.duration.quarterLength < min_quarter_length:
            el.duration.quarterLength = min_quarter_length
    return score


def strip_pitch_bends(score: stream.Score) -> stream.Score:
    """Remove pitch-bend events; basic-pitch writes them as microtonal accidentals.

    Tango transcriptions read more cleanly in MuseScore without microtones.
    Future work can convert pitch bends to grace-note glissandi.
    """
    for part in score.parts:
        for measure in part.getElementsByClass("Measure"):
            for pb in list(measure.getElementsByClass("PitchBend")):
                measure.remove(pb)
    return score


def _simplify_durations(score: stream.Score) -> stream.Score:
    """Replace notes/rests with complex durations with expressible simple values.

    MusicXML cannot represent arbitrary fractional durations (e.g. 2.5 beats)
    as a single note type. music21 marks those as ``type='complex'`` and will
    raise ``MusicXMLExportException``. We split them into standard durations
    (tied for pitched notes) before export.
    """
    for el in list(score.recurse().notesAndRests):
        if el.duration.type != "complex":
            continue
        split = el.splitAtDurations()
        if not split:
            continue
        parent = el.activeSite
        if parent is None:
            continue
        offset = el.offset
        parent.remove(el)
        running_offset = offset
        if el.isNote:
            for i, piece in enumerate(split):
                if i == 0:
                    piece.tie = music21.tie.Tie("start")
                elif i == len(split) - 1:
                    piece.tie = music21.tie.Tie("stop")
                else:
                    piece.tie = music21.tie.Tie("continue")
                piece.articulations = list(el.articulations)
                piece.expressions = list(el.expressions)
        for piece in split:
            parent.insert(running_offset, piece)
            running_offset += piece.duration.quarterLength
    return score


def _rebuild_parts(score: stream.Score, time_signature: str) -> stream.Score:
    """Flatten every part and rebuild measures cleanly.

    MIDI import can leave overlapping measures, implicit voices, or gaps that
    trigger ``StreamException`` or ``MusicXMLExportException`` during export.
    Rebuilding from a flat note/rest stream with an explicit meter avoids those
    pathologies.
    """
    ts = meter.TimeSignature(time_signature)
    new_score = stream.Score()

    for md in score.getElementsByClass("Metadata"):
        new_score.insert(0, md)

    for part in score.parts:
        new_part = stream.Part()
        new_part.partName = part.partName
        new_part.partAbbreviation = part.partAbbreviation

        # Preserve the original instrument and clef when possible.
        inst = part.getInstrument(returnDefault=None)
        if inst is not None:
            new_part.insert(0, inst)
        clefs = list(part.flatten().getElementsByClass("Clef"))
        if clefs:
            new_part.insert(0, clefs[0])
        else:
            new_part.insert(0, music21.clef.TrebleClef())
        new_part.insert(0, ts)

        # Copy only notes and rests, preserving offset, duration, and articulations.
        for el in part.flatten().notesAndRests:
            if el.isNote:
                new_el = note.Note(el.pitch, quarterLength=el.duration.quarterLength)
                new_el.articulations = list(el.articulations)
                new_el.expressions = list(el.expressions)
            else:
                new_el = note.Rest(quarterLength=el.duration.quarterLength)
            new_part.insert(el.offset, new_el)

        new_part.makeMeasures(inPlace=True)
        new_part.makeTies(inPlace=True)
        new_score.insert(0, new_part)

    return new_score


def assign_instrument(part: stream.Part, label: str, config: PipelineConfig) -> stream.Part:
    """Set a sensible instrument/clef for a part based on its stem label and hints."""
    instrument_name = _resolve_stem_instrument(label, config)
    inst = _instrument(instrument_name)
    part.insert(0, inst)
    part.insert(0, _clef_for(instrument_name))
    return part


def add_marcato_accents(part: stream.Part) -> stream.Part:
    """Accent the first attack of each measure for bass and drum parts.

    This captures the tango marcato feel without over-marking every note.
    """
    for measure in part.getElementsByClass("Measure"):
        notes = list(measure.notes)
        if notes:
            first = notes[0]
            if not any(isinstance(a, articulations.Accent) for a in first.articulations):
                first.articulations.append(articulations.Accent())
    return part


def merge_stem_midis(
    midi_paths: dict[str, Path],
    config: PipelineConfig,
    title: str | None = None,
) -> stream.Score:
    """Merge per-stem MIDI files into a single quantized score.

    Args:
        midi_paths: Mapping from stem label to MIDI path.
        config: Pipeline configuration.
        title: Optional score title.

    Returns:
        A music21 Score with one Part per stem.
    """
    score = stream.Score()
    if title:
        score.insert(0, music21.metadata.Metadata(title=title))

    for label, midi_path in midi_paths.items():
        if _is_excluded(label, config):
            logger.info("Skipping excluded stem: %s", label)
            continue
        stem_score = load_midi(midi_path)
        for part in stem_score.parts:
            assign_instrument(part, label, config)
            # Tag the part with the resolved instrument for easier cleanup.
            resolved = _resolve_stem_instrument(label, config)
            part.partAbbreviation = resolved[:8]
            if resolved in ("bass", "drums"):
                add_marcato_accents(part)
            score.insert(0, part)

    score = set_time_signature(score, config.time_signature)
    score = quantize_score(
        score,
        quantization_grid=config.quantization_grid,
        min_quarter_length=config.min_quarter_length,
    )
    score = strip_pitch_bends(score)
    return score


def split_bandoneon_part(score: stream.Score) -> stream.Score:
    """Split a combined bandoneon/piano/violin part into treble and bass staves.

    This is a heuristic split based on pitch range. Notes below B3 go to the
    bass staff, others to the treble staff. In MuseScore the user can refine
    this further.

    Args:
        score: A music21 Score.

    Returns:
        Score with split staves.
    """
    B3 = music21.pitch.Pitch("B3")
    new_score = stream.Score()

    combined_labels = {"bandone", "bandoneon", "bandoneon_violin", "bandoneon_piano_violin"}
    for part in score.parts:
        if part.partAbbreviation not in combined_labels:
            new_score.insert(0, part)
            continue

        treble_part = stream.Part()
        treble_part.id = f"{part.id}_treble"
        treble_part.partAbbreviation = "band_tre"
        treble_part.insert(0, _instrument("bandoneon"))
        treble_part.insert(0, music21.clef.TrebleClef())

        bass_part = stream.Part()
        bass_part.id = f"{part.id}_bass"
        bass_part.partAbbreviation = "band_bas"
        bass_part.insert(0, _instrument("bandoneon"))
        bass_part.insert(0, music21.clef.BassClef())

        for measure in part.getElementsByClass("Measure"):
            treble_measure = stream.Measure(number=measure.number)
            bass_measure = stream.Measure(number=measure.number)
            for el in measure:
                if isinstance(el, note.Note):
                    target = bass_measure if el.pitch <= B3 else treble_measure
                    target.insert(el.offset, el)
                elif isinstance(el, note.Rest):
                    # Place rests in both staves to preserve measure alignment.
                    treble_measure.insert(el.offset, note.Rest(quarterLength=el.duration.quarterLength))
                    bass_measure.insert(el.offset, note.Rest(quarterLength=el.duration.quarterLength))
                elif isinstance(el, meter.TimeSignature):
                    treble_measure.insert(0, el)
                    bass_measure.insert(0, el)

            if list(treble_measure.notes):
                treble_part.append(treble_measure)
            if list(bass_measure.notes):
                bass_part.append(bass_measure)

        if list(treble_part.flatten().notes):
            new_score.insert(0, treble_part)
        if list(bass_part.flatten().notes):
            new_score.insert(0, bass_part)

    return new_score


def export_score(
    score: stream.Score,
    output_path: Path,
    fmt: str = "musicxml",
) -> Path:
    """Export a music21 Score to MusicXML or MIDI.

    Args:
        score: music21 Score.
        output_path: Destination path (extension controls format if ``fmt`` is not given).
        fmt: music21 format string (``musicxml`` or ``midi``).

    Returns:
        Path to the exported file.

    Raises:
        Exception: Re-raised if both standard and fallback export attempts fail.
    """
    ensure_dirs(output_path.parent)
    logger.info("Exporting %s score to %s", fmt, output_path)

    # With many parts (6 stems + bandoneon split) music21 writes a flood of
    # "we are out of midi channels! help!" messages directly to stderr via
    # environLocal.warn(). They are harmless for notation export; capture and
    # discard them, surfacing the text only if export actually fails.
    write_err = io.StringIO()
    try:
        with contextlib.redirect_stderr(write_err):
            try:
                written = score.write(fmt=fmt, fp=str(output_path))
            except Exception:
                if fmt != "musicxml":
                    raise
                logger.warning(
                    "Standard MusicXML export failed (often a meter/voice mismatch). "
                    "Retrying with makeNotation=False. The file may need manual cleanup."
                )
                written = score.write(fmt=fmt, fp=str(output_path), makeNotation=False)
    except Exception:
        err_text = write_err.getvalue().strip()
        if err_text:
            logger.error("MusicXML/MIDI export stderr:\n%s", err_text)
        raise
    return Path(written)


def build_score(
    midi_paths: dict[str, Path],
    config: PipelineConfig,
    title: str | None = None,
    split_bandoneon: bool | None = None,
) -> stream.Score:
    """High-level helper: merge, quantize, optionally split, and return a score."""
    score = merge_stem_midis(midi_paths, config, title=title)
    if split_bandoneon is None:
        split_bandoneon = config.split_bandoneon
    if split_bandoneon:
        score = split_bandoneon_part(score)

    # Final cleanup so MusicXML export cannot choke on complex durations or
    # messy measure structures imported from MIDI.
    score = _simplify_durations(score)
    score = _rebuild_parts(score, config.time_signature)
    return score
