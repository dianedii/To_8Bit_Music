import numpy as np
import tempfile
from pathlib import Path

from src.melody_extractor import extract_melody
from src.note_simplifier import simplify_notes
from src.pop_synthesizer import synthesize_pop_chip
from src.synthesizer import synthesize
from src.utils import export_audio, get_output_path


from src.pop_synthesizer import _synthesize_from_notes


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


def test_pop_chip_with_melody_extraction():
    """从转录结果中提取主旋律后，高音旋律应获得 lead 处理。"""
    sr = 44100
    duration = 1.2
    mono = np.zeros(int(duration * sr), dtype=np.float64)

    # 模拟钢琴转录：低音和弦较短较弱，高音主旋律长且响
    notes = [
        (48, 0.0, 0.3, 60),    # C3 低音（短、弱）
        (55, 0.0, 0.3, 60),    # G3 低音（短、弱）
        (64, 0.0, 0.3, 70),    # E4 中音和弦
        (67, 0.0, 0.3, 70),    # G4 中音和弦
        (72, 0.0, 0.8, 100),   # C5 旋律（长、响）
        (74, 0.4, 0.8, 100),   # D5 旋律
        (76, 0.8, 1.2, 100),   # E5 旋律（高音区）
        (79, 1.2, 1.6, 100),   # G5 旋律（高音区）
    ]
    melody = extract_melody(notes)
    melody_pitches = [n[0] for n in melody]
    # 主旋律应包含高音线
    assert any(p >= 72 for p in melody_pitches), f"Expected high melody notes, got {melody_pitches}"

    out = _synthesize_from_notes(notes, mono, sr, melody_notes=melody)
    assert out.shape == (2, len(mono))
    assert out.dtype == np.float32
    assert np.max(np.abs(out)) > 0

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "out.wav"
        export_audio(out, str(out_path), sample_rate=sr)
        assert out_path.exists()
        assert out_path.stat().st_size > 0
