import numpy as np
import pytest

from src.pop_synthesizer import (
    _synthesize_events,
    synthesize_pop_chip,
    _synthesize_from_notes,
    _apply_legato,
    _apply_lowpass,
    _bandlimited_waveform,
)


def test_synthesize_pop_chip_main_melody_only():
    sr = 22050
    duration = 1.0
    t = np.arange(int(sr * duration)) / sr
    audio = np.zeros_like(t)
    for start, freq in [(0.0, 261.63), (0.35, 329.63), (0.7, 392.0)]:
        mask = (t >= start) & (t < start + 0.25)
        audio[mask] += 0.3 * np.sin(2 * np.pi * freq * t[mask])

    out = synthesize_pop_chip(audio, sample_rate=sr, volume=80)
    assert out.shape[0] == 2
    assert out.shape[1] > 0
    assert out.dtype == np.float32
    assert np.max(np.abs(out)) > 0.1


def test_synthesize_pop_chip_outputs_stable_chip():
    sr = 44100
    duration = 1.0
    t = np.arange(int(duration * sr)) / sr
    audio = np.zeros_like(t)
    for start, freq in [(0.0, 440.0), (0.3, 523.25), (0.6, 659.25)]:
        mask = (t >= start) & (t < start + 0.25)
        env = np.exp(-(t[mask] - start) / 0.05)
        audio[mask] += np.sin(2 * np.pi * freq * t[mask]) * env

    out = synthesize_pop_chip(
        audio,
        sample_rate=sr,
        volume=90,
    )
    assert out.ndim == 2
    assert out.shape[0] == 2
    assert out.shape[1] == len(audio)
    assert out.dtype == np.float32
    assert np.max(np.abs(out)) <= 1.0


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
    min_lag = int(0.5 * sr / 440.0)  # about half period
    peak = np.argmax(autocorr[min_lag:]) + min_lag
    estimated_period = peak / sr
    if estimated_period > 0:
        estimated_freq = 1.0 / estimated_period
        assert 420 <= estimated_freq <= 460


def test_apply_legato_merges_close_notes():
    notes = [
        (60.0, 0.0, 0.45, 100),
        (62.0, 0.48, 0.9, 100),
    ]
    merged = _apply_legato(notes, threshold=0.05)
    assert len(merged) == 2
    assert merged[0][2] == 0.48  # first note extended to second onset


def test_apply_legato_keeps_separated_notes():
    notes = [
        (60.0, 0.0, 0.3, 100),
        (62.0, 0.5, 0.8, 100),
    ]
    merged = _apply_legato(notes, threshold=0.05)
    assert merged[0][2] == 0.3


def test_apply_legato_skips_overlapping_notes():
    notes = [
        (60.0, 0.0, 0.5, 100),
        (62.0, 0.45, 0.9, 100),
    ]
    merged = _apply_legato(notes, threshold=0.05)
    assert len(merged) == 2
    assert merged[0][2] == 0.5  # first note not extended


def test_apply_legato_keeps_max_velocity():
    notes = [
        (60.0, 0.0, 0.45, 80),
        (62.0, 0.48, 0.9, 100),
    ]
    merged = _apply_legato(notes, threshold=0.05)
    assert merged[0][3] == 100


def test_apply_legato_chains_multiple_notes():
    notes = [
        (60.0, 0.0, 0.25, 100),
        (62.0, 0.28, 0.55, 100),
        (64.0, 0.58, 0.85, 100),
    ]
    merged = _apply_legato(notes, threshold=0.05)
    assert len(merged) == 3
    assert merged[0][2] == 0.28
    assert merged[1][2] == 0.58


def test_apply_lowpass_reduces_high_freq():
    sr = 44100
    t = np.arange(sr) / sr
    audio = np.sin(2 * np.pi * 12000 * t)
    filtered = _apply_lowpass(audio, sr, cutoff=8000)
    assert np.max(np.abs(filtered)) <= np.max(np.abs(audio))
    fft_in = np.abs(np.fft.rfft(audio))
    fft_out = np.abs(np.fft.rfft(filtered))
    # Compare energy in the stopband (above cutoff) relative to passband
    cutoff_bin = int(8000 / (sr / len(fft_in)))
    assert fft_out[cutoff_bin:].sum() < fft_in[cutoff_bin:].sum()


def test_bandlimited_waveform_sine():
    sr = 44100
    freq = 440.0
    t = np.arange(int(sr * 0.1)) / sr
    phase = 2 * np.pi * freq * t
    wave = _bandlimited_waveform(phase, freq, sr, 'sine')
    np.testing.assert_allclose(wave, np.sin(phase), atol=1e-6)


def test_synthesize_events_triangle_has_fewer_harmonics():
    sr = 44100
    notes = [(69, 0.0, 0.2, 100)]  # A4
    square = _synthesize_events(notes, 0.2, sr, waveform='square')
    triangle = _synthesize_events(notes, 0.2, sr, waveform='triangle')
    fft_sq = np.abs(np.fft.rfft(square))
    fft_tri = np.abs(np.fft.rfft(triangle))
    assert fft_tri[2000:].sum() < fft_sq[2000:].sum()


def test_synthesize_from_notes_backward_compatible_without_melody():
    """未提供 melody_notes 时保持旧的按音高分层行为。"""
    sr = 44100
    duration = 0.5
    notes = [
        (60, 0.0, 0.2, 100),
        (72, 0.25, 0.45, 100),
    ]
    mono = np.zeros(int(duration * sr), dtype=np.float64)

    out = _synthesize_from_notes(notes, mono, sr, volume=80)
    assert out.shape == (2, int(duration * sr))
    assert out.dtype == np.float32
    assert np.max(np.abs(out)) > 0


def test_synthesize_from_notes_melody_high_notes_get_lead_treatment():
    """高音主旋律（MIDI 76+）在提供 melody_notes 时应获得 richer 的 lead 处理。"""
    sr = 44100
    duration = 1.0
    notes = [
        (48, 0.0, 0.5, 80),    # C3 伴奏
        (52, 0.0, 0.5, 75),    # E3 伴奏
        (55, 0.0, 0.5, 70),    # G3 伴奏
        (64, 0.0, 0.3, 90),    # E4 伴奏
        (76, 0.0, 0.4, 100),   # E5 旋律
        (79, 0.5, 0.9, 100),   # G5 旋律
        (84, 1.0, 1.4, 100),   # C6 旋律
    ]
    melody = [
        (76, 0.0, 0.4, 100),
        (79, 0.5, 0.9, 100),
        (84, 1.0, 1.4, 100),
    ]
    mono = np.zeros(int(duration * sr), dtype=np.float64)

    out_with = _synthesize_from_notes(notes, mono, sr, volume=80, melody_notes=melody)
    out_without = _synthesize_from_notes(notes, mono, sr, volume=80, melody_notes=None)

    assert out_with.shape == (2, len(mono))
    assert out_without.shape == (2, len(mono))

    rms_with = np.sqrt(np.mean(out_with ** 2))
    rms_without = np.sqrt(np.mean(out_without ** 2))
    # 旋律感知版本因为有额外的八度层，能量应明显更高
    assert rms_with > rms_without * 1.1


def test_synthesize_from_notes_melody_note_deduplication():
    """旋律音不应在伴奏层中被重复计算。"""
    sr = 44100
    duration = 0.5
    notes = [
        (60, 0.0, 0.3, 100),
        (72, 0.2, 0.5, 100),
    ]
    melody = [(60, 0.0, 0.3, 100)]
    mono = np.zeros(int(duration * sr), dtype=np.float64)

    out = _synthesize_from_notes(notes, mono, sr, melody_notes=melody)
    assert out.shape == (2, len(mono))
    assert np.max(np.abs(out)) > 0


def test_synthesize_events_release_is_longer():
    sr = 44100
    notes = [(69, 0.0, 0.1, 100)]
    audio = _synthesize_events(notes, 0.1, sr)
    # With a longer release, amplitude should still be decaying near the end
    assert abs(audio[-1]) < abs(audio[-int(0.020 * sr)])
