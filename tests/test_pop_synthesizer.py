import numpy as np
import pytest

from src.pop_synthesizer import (
    _synthesize_events,
    synthesize_pop_chip,
    _apply_legato,
    _apply_lowpass,
    _bandlimited_waveform,
)


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


def test_synthesize_pop_chip_main_melody_only():
    sr = 22050
    duration = 1.0
    t = np.arange(int(sr * duration)) / sr
    audio = np.zeros_like(t)
    for start, freq in [(0.0, 261.63), (0.35, 329.63), (0.7, 392.0)]:
        mask = (t >= start) & (t < start + 0.25)
        audio[mask] += 0.3 * np.sin(2 * np.pi * freq * t[mask])

    out = synthesize_pop_chip(audio, sample_rate=sr, chip_mix=1.0, waveform='triangle')
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
        waveform='square',
        chip_mix=1.0,
        n_voices=4,
        hop_length=512,
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


def test_synthesize_events_release_is_longer():
    sr = 44100
    notes = [(69, 0.0, 0.1, 100)]
    audio = _synthesize_events(notes, 0.1, sr)
    # With a longer release, amplitude should still be decaying near the end
    assert abs(audio[-1]) < abs(audio[-int(0.020 * sr)])
