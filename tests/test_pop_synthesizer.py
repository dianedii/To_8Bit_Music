import numpy as np
import pytest

from src.pop_synthesizer import _detect_onsets, _segment_audio, _extract_stable_pitches


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


def test_extract_stable_pitches_single_tone():
    sr = 44100
    duration = 0.3
    freq = 440.0
    t = np.arange(int(duration * sr)) / sr
    segment = np.sin(2 * np.pi * freq * t) * 0.5
    pitches = _extract_stable_pitches(segment, sr, n_voices=4, hop_length=512)
    assert len(pitches) >= 1
    midi, velocity = pitches[0]
    assert abs(midi - 69) <= 1
    assert 0 < velocity <= 127


def test_extract_stable_pitches_two_tones():
    sr = 44100
    duration = 0.3
    t = np.arange(int(duration * sr)) / sr
    segment = (
        np.sin(2 * np.pi * 440.0 * t) * 0.4
        + np.sin(2 * np.pi * 659.25 * t) * 0.4
    )
    pitches = _extract_stable_pitches(segment, sr, n_voices=4, hop_length=512)
    assert len(pitches) >= 2
    midis = [p[0] for p in pitches[:2]]
    assert any(abs(m - 69) <= 1 for m in midis)
    assert any(abs(m - 76) <= 1 for m in midis)


from src.pop_synthesizer import _merge_consecutive_notes, _synthesize_events


def test_merge_consecutive_notes_joins_same_pitch():
    notes = [
        (69, 0.0, 0.3, 100),
        (69, 0.32, 0.6, 100),
        (72, 0.6, 0.9, 100),
    ]
    merged = _merge_consecutive_notes(notes, gap_threshold=0.05)
    assert len(merged) == 2
    assert merged[0] == (69, 0.0, 0.6, 100)
    assert merged[1] == (72, 0.6, 0.9, 100)


def test_merge_consecutive_notes_keeps_different_pitches():
    notes = [
        (69, 0.0, 0.3, 100),
        (72, 0.32, 0.6, 100),
    ]
    merged = _merge_consecutive_notes(notes, gap_threshold=0.05)
    assert len(merged) == 2


def test_synthesize_events_shape_and_tail_silence():
    sr = 44100
    duration = 0.5
    notes = [
        (69, 0.0, 0.2, 100),
        (72, 0.25, 0.45, 100),
    ]
    audio = _synthesize_events(notes, duration, sr, waveform='square')
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float64
    assert len(audio) == int(duration * sr)
    assert np.max(np.abs(audio)) <= 1.0
    tail = audio[int(0.46 * sr):]
    assert np.max(np.abs(tail)) < 0.1


def test_synthesize_events_pitch_stability():
    """验证同一音符片段内音高稳定（无颤音/抖动）。"""
    sr = 44100
    duration = 0.3
    notes = [(69, 0.0, 0.25, 100)]
    audio = _synthesize_events(notes, duration, sr, waveform='square')
    autocorr = np.correlate(audio[: int(0.2 * sr)], audio[: int(0.2 * sr)], mode='full')
    autocorr = autocorr[len(autocorr) // 2:]
    peak = np.argmax(autocorr[100:]) + 100
    estimated_period = peak / sr
    if estimated_period > 0:
        estimated_freq = 1.0 / estimated_period
        assert 420 <= estimated_freq <= 460
