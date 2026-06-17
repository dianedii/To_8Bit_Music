import numpy as np
import pytest

from src.synthesizer import synthesize


def test_synthesize_shape_and_range():
    notes = [
        (60, 0.0, 0.1, 80),  # C4, 100ms
    ]
    audio = synthesize(notes, duration=0.2, sample_rate=44100, purity=0, volume=80)
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert len(audio) == int(0.2 * 44100)
    assert np.max(np.abs(audio)) <= 1.0


def test_synthesize_silence_after_last_note():
    notes = [(60, 0.0, 0.05, 80)]
    audio = synthesize(notes, duration=0.2, sample_rate=44100, purity=0, volume=80)
    # 音符结束后应有静音
    tail = audio[int(0.1 * 44100):]
    assert np.max(np.abs(tail)) < 0.1


def test_synthesize_empty_notes():
    audio = synthesize([], duration=0.2, sample_rate=44100)
    assert len(audio) == int(0.2 * 44100)
    assert np.max(np.abs(audio)) == 0.0


def test_synthesize_note_beyond_duration():
    notes = [(60, 0.0, 1.0, 80)]
    audio = synthesize(notes, duration=0.2, sample_rate=44100)
    assert len(audio) == int(0.2 * 44100)
    assert np.max(np.abs(audio)) <= 1.0


def test_synthesize_zero_volume():
    notes = [(60, 0.0, 0.1, 80)]
    audio = synthesize(notes, duration=0.2, sample_rate=44100, volume=0)
    assert np.max(np.abs(audio)) == 0.0


def test_synthesize_purity_applies_vibrato():
    notes = [(60, 0.0, 0.5, 80)]
    audio_pure = synthesize(notes, duration=0.5, sample_rate=44100, purity=0)
    audio_impure = synthesize(notes, duration=0.5, sample_rate=44100, purity=100)
    assert not np.allclose(audio_pure, audio_impure)


def test_invalid_duration_raises():
    with pytest.raises(ValueError):
        synthesize([], duration=0, sample_rate=44100)


def test_invalid_sample_rate_raises():
    with pytest.raises(ValueError):
        synthesize([], duration=1.0, sample_rate=0)


def test_volume_clamping():
    notes = [(60, 0.0, 0.1, 80)]
    audio_clamped = synthesize(notes, duration=0.2, sample_rate=44100, volume=150)
    audio_max = synthesize(notes, duration=0.2, sample_rate=44100, volume=100)
    assert np.allclose(audio_clamped, audio_max)


def test_overlapping_notes_no_clip():
    notes = [
        (60, 0.0, 0.2, 127),
        (60, 0.05, 0.2, 127),
    ]
    audio = synthesize(notes, duration=0.2, sample_rate=44100)
    assert np.max(np.abs(audio)) <= 1.0
