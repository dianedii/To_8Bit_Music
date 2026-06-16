import numpy as np

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
