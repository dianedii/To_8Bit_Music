import numpy as np
import pytest

from src.pop_synthesizer import _detect_onsets, _segment_audio


def generate_click_tone(freq, duration, sr, onset_times):
    """生成在指定时刻出现纯音的测试音频。"""
    t = np.arange(int(duration * sr)) / sr
    audio = np.zeros_like(t)
    env_decay = 0.1
    for ot in onset_times:
        mask = (t >= ot) & (t < ot + env_decay)
        env = np.exp(-(t[mask] - ot) / 0.03)
        audio[mask] += np.sin(2 * np.pi * freq * t[mask]) * env
    return audio


def test_detect_onsets_finds_note_starts():
    sr = 44100
    onset_times = [0.2, 0.6, 1.0]
    audio = generate_click_tone(440.0, 1.5, sr, onset_times)
    onsets = _detect_onsets(audio, sr)
    assert len(onsets) >= 3
    for expected in onset_times:
        assert any(np.abs(onsets - expected) < 0.03)


def test_detect_onsets_silent_audio():
    sr = 44100
    silent_audio = np.zeros(sr * 2)
    onsets = _detect_onsets(silent_audio, sr)
    assert len(onsets) == 0


def test_segment_audio_empty_onsets():
    sr = 44100
    audio = generate_click_tone(440.0, 1.0, sr, [0.0, 0.3, 0.7])
    segments = _segment_audio(audio, sr, np.array([]), min_note_duration=0.05)
    assert segments == [(0, len(audio))]


def test_segment_audio_overlapping_onsets_deduplicated():
    sr = 44100
    audio = generate_click_tone(440.0, 1.0, sr, [0.0, 0.3, 0.7])
    # Provide onsets closer than min_note_duration
    dense_onsets = np.array([0.0, 0.01, 0.02, 0.3, 0.31, 0.7])
    segments = _segment_audio(audio, sr, dense_onsets, min_note_duration=0.05)
    # Should deduplicate and still produce valid segments
    assert len(segments) >= 2
    for start, end in segments:
        assert 0 <= start < end <= len(audio)
        assert (end - start) / sr >= 0.05


def test_segment_audio_short_audio():
    sr = 44100
    # Audio shorter than min_note_duration
    short_audio = np.zeros(int(sr * 0.01))
    segments = _segment_audio(short_audio, sr, np.array([0.0]), min_note_duration=0.05)
    assert segments == []


def test_segment_audio_splits_by_onsets():
    sr = 44100
    onset_times = [0.0, 0.3, 0.7]
    audio = generate_click_tone(440.0, 1.0, sr, onset_times)
    segments = _segment_audio(audio, sr, onset_times, min_note_duration=0.05)
    assert len(segments) >= 2
    for start, end in segments:
        assert 0 <= start < end <= len(audio)
        assert (end - start) / sr >= 0.05
