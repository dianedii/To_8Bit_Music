import tempfile
from pathlib import Path

import numpy as np

from src.melody_extractor import extract_melody
from src.note_simplifier import simplify_notes
from src.synthesizer import synthesize
from src.utils import export_audio, get_output_path


def test_end_to_end_synthetic():
    # 构造一个简单音符序列，模拟已转录的结果
    notes = [
        (60, 0.0, 0.4, 100),   # C4
        (60, 0.42, 0.8, 100),  # 同音，应合并
        (64, 0.85, 1.0, 100),  # E4
        (67, 1.05, 1.4, 100),  # G4
    ]
    melody = extract_melody(notes)
    simplified = simplify_notes(melody, strength=50)
    audio = synthesize(simplified, duration=2.0, sample_rate=44100, purity=30, volume=80)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "out.wav"
        export_audio(audio, str(out_path), sample_rate=44100)
        assert out_path.exists()
        assert out_path.stat().st_size > 0


def test_get_output_path():
    p = get_output_path("C:/Music/song.mp3", "wav")
    assert p.suffix == ".wav"
    p = get_output_path("C:/Music/song.mp3", "mp3")
    assert p.suffix == ".mp3"
