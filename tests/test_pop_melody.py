import numpy as np
import pytest

from src.pop_melody import _pyin_to_notes


def generate_pure_tone(freq, duration, sr):
    t = np.arange(int(duration * sr)) / sr
    return np.sin(2 * np.pi * freq * t) * 0.5


def test_pyin_to_notes_detects_single_tone():
    sr = 44100
    audio = generate_pure_tone(440.0, 0.6, sr)
    notes = _pyin_to_notes(audio, sr, hop_length=512, min_note_duration=0.05)
    assert len(notes) >= 1
    midi, onset, offset, velocity = notes[0]
    assert abs(midi - 69) <= 1  # A4
    assert onset < 0.1
    assert offset > 0.5
    assert 1 <= velocity <= 127


def test_pyin_to_notes_empty_audio():
    notes = _pyin_to_notes(np.array([], dtype=np.float64), 44100)
    assert notes == []


def test_pyin_to_notes_silence():
    sr = 44100
    audio = np.zeros(int(sr * 0.5), dtype=np.float64)
    notes = _pyin_to_notes(audio, sr)
    assert notes == []
