import numpy as np
import tempfile
from pathlib import Path

from src.melody_extractor import extract_melody
from src.note_simplifier import simplify_notes
from src.pop_synthesizer import synthesize_pop_chip
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


def test_pop_chip_end_to_end_synthetic():
    sr = 44100
    duration = 1.0
    t = np.arange(int(duration * sr)) / sr
    audio = np.zeros_like(t)
    for start, freq in [(0.0, 440.0), (0.3, 523.25), (0.6, 659.25)]:
        mask = (t >= start) & (t < start + 0.25)
        env = np.exp(-(t[mask] - start) / 0.05)
        audio[mask] += np.sin(2 * np.pi * freq * t[mask]) * env

    out = synthesize_pop_chip(audio, sample_rate=sr, volume=80)
    assert out.shape[0] == 2
    assert out.dtype == np.float32
    assert np.max(np.abs(out)) > 0

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "out.wav"
        export_audio(out, str(out_path), sample_rate=sr)
        assert out_path.exists()
        assert out_path.stat().st_size > 0
