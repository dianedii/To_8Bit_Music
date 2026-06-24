"""
流行 8-bit / 复音芯片合成器

参考 8-bit 音乐生成器的 clean 模式思想：
- 基于原音频频谱峰值跟踪多个声部
- 用带限方波/三角波合成每个声部
- 保留原曲的和声/伴奏结构，兼顾流行乐美感
- 输出为立体声（左右相同），比纯正 FC 单声部更丰满
"""

import numpy as np
from scipy import signal
from typing import Literal

import librosa


def _audio_to_mono_float(audio: np.ndarray) -> np.ndarray:
    """将音频转为单声道 float64 [-1, 1]。"""
    if audio.ndim == 2:
        audio = np.mean(audio, axis=0)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float64) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float64) / 2147483648.0
    elif audio.dtype == np.float32:
        audio = audio.astype(np.float64)
    return audio


def _rms_envelope(audio: np.ndarray, window: int = 2048, floor: float = 0.25) -> np.ndarray:
    """用滑动窗口计算 RMS 包络，避免芯片音色在安静段落完全消失。"""
    squared = audio ** 2
    rms = np.sqrt(np.convolve(squared, np.ones(window) / window, mode='same'))
    rms = np.clip(rms, 1e-6, 1.0)
    envelope = rms / np.max(rms)
    envelope = np.clip(envelope, floor, 1.0)
    return envelope


def _gentle_compress(audio: np.ndarray, threshold: float = 0.6, ratio: float = 2.0) -> np.ndarray:
    """柔和动态压缩，保留音乐性的同时控制峰值。"""
    abs_audio = np.abs(audio)
    gain = np.ones_like(audio)
    over = abs_audio > threshold
    if np.any(over):
        attenuation = threshold + (abs_audio[over] - threshold) / ratio
        gain[over] = attenuation / abs_audio[over]
    return audio * gain


def _midi_to_freq(midi_note: float) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def _freq_to_midi(freq: float) -> float:
    if freq <= 0:
        return 0.0
    return 69.0 + 12.0 * np.log2(freq / 440.0)


def _apply_lowpass(
    audio: np.ndarray,
    sample_rate: int,
    cutoff: float = 8000.0,
) -> np.ndarray:
    """对音频做 6dB/oct 低通滤波，软化高频。"""
    if cutoff <= 0 or cutoff >= sample_rate / 2:
        return audio
    b, a = signal.butter(1, cutoff / (sample_rate / 2), btype='low')
    return signal.filtfilt(b, a, audio)


def _bandlimited_waveform(phase: np.ndarray, freq: float, sample_rate: int, waveform: str) -> np.ndarray:
    """根据瞬时基频限制谐波数，生成带限方波/三角波/锯齿波/正弦波。"""
    if waveform == 'sine':
        return np.sin(phase)

    max_harm = int(np.floor((sample_rate / 2.5) / max(freq, 20.0)))
    max_harm = max(1, min(max_harm, 40))

    wave = np.zeros_like(phase)
    if waveform == 'square':
        for n in range(1, max_harm + 1, 2):
            wave += (4.0 / (np.pi * n)) * np.sin(n * phase)
    elif waveform == 'triangle':
        for n in range(1, max_harm + 1, 2):
            sign = (-1) ** ((n - 1) // 2)
            wave += (8.0 / (np.pi ** 2 * n ** 2)) * sign * np.sin(n * phase)
    elif waveform == 'sawtooth':
        for n in range(1, max_harm + 1):
            wave += (2.0 / (np.pi * n)) * ((-1) ** (n + 1)) * np.sin(n * phase)
    else:
        wave = np.sign(np.sin(phase))

    return np.clip(wave, -1.0, 1.0)


def _apply_legato(
    notes: list[tuple[float, float, float, int]],
    threshold: float = 0.05,
) -> list[tuple[float, float, float, int]]:
    """合并间隔 ≤ threshold 的相邻音符，前一个音符延长到后一个 onset。"""
    if not notes:
        return []
    notes = sorted(notes, key=lambda n: n[1])
    merged = [list(notes[0])]
    for pitch, onset, offset, velocity in notes[1:]:
        prev = merged[-1]
        gap = onset - prev[2]
        if 0 <= gap <= threshold:
            prev[2] = onset
            prev[3] = max(prev[3], velocity)
        merged.append([pitch, onset, offset, velocity])
    return [tuple(n) for n in merged]


def _synthesize_events(
    notes: list[tuple[float, float, float, int]],
    duration: float,
    sample_rate: int,
    waveform: str = 'triangle',
) -> np.ndarray:
    """将复音音符事件合成为稳定音高的芯片音频。"""
    num_samples = int(duration * sample_rate)
    audio = np.zeros(num_samples, dtype=np.float64)

    for pitch, onset, offset, velocity in notes:
        freq = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
        start_idx = max(0, int(onset * sample_rate))
        end_idx = min(num_samples, int(offset * sample_rate))
        if start_idx >= end_idx:
            continue

        length = end_idx - start_idx
        t = np.arange(length) / sample_rate
        phase = 2.0 * np.pi * freq * t
        wave = _bandlimited_waveform(phase, freq, sample_rate, waveform)

        env = np.ones(length, dtype=np.float64)
        attack_samples = min(int(0.010 * sample_rate), length)
        release_samples = min(int(0.040 * sample_rate), length)

        env[:attack_samples] = np.linspace(0.0, 1.0, attack_samples)
        release_start_idx = max(0, length - release_samples)
        start_amp = env[release_start_idx]
        env[-release_samples:] = np.linspace(start_amp, 0.0, release_samples)

        amp = (velocity / 127.0) * 0.25
        audio[start_idx:end_idx] += wave * env * amp

    audio = np.clip(audio, -1.0, 1.0)
    return audio


def synthesize_pop_chip(
    audio: np.ndarray,
    sample_rate: int = 44100,
    waveform: Literal['square', 'triangle', 'sawtooth', 'sine'] = 'triangle',
    chip_mix: float = 0.6,
    n_voices: int = 6,
    hop_length: int = 512,
    min_note_duration: float = 0.05,
    pitch_quantize_strength: float = 1.0,
    f0_median_size: int = 3,
    legato_threshold: float = 0.05,
    lowpass_cutoff: float = 8000.0,
) -> np.ndarray:
    """
    基于主旋律提取的流行 8-bit 合成。

    Args:
        audio: 输入音频，numpy 数组（单声道/立体声，任意 dtype）
        sample_rate: 采样率
        waveform: 合成器波形
        chip_mix: 合成器层混合比例（0~1）
        n_voices: 和声点缀最大声部数
        hop_length: STFT 帧移
        min_note_duration: 最短音符时长（秒）
        pitch_quantize_strength: 音高量化强度
        f0_median_size: f0 中值滤波窗口
        legato_threshold: 连音间隔阈值（秒）
        lowpass_cutoff: 低通截止频率（Hz）

    Returns:
        合成后的立体声 float32 音频，shape (2, N)
    """
    from src.pop_melody import (
        _pyin_to_notes,
        _split_candidate_lines,
        _apply_hard_filters,
        _extract_main_melody,
        _extract_harmony_voice,
    )

    mono = _audio_to_mono_float(audio)
    if len(mono) == 0:
        return np.zeros((2, 0), dtype=np.float32)

    notes = _pyin_to_notes(
        mono,
        sample_rate,
        hop_length=hop_length,
        min_note_duration=min_note_duration,
        pitch_quantize_strength=pitch_quantize_strength,
        f0_median_size=f0_median_size,
    )

    lines = _split_candidate_lines(notes)
    lines = _apply_hard_filters(lines)
    main_line = _extract_main_melody(lines)

    other_lines = [line for line in lines if line is not main_line]
    harmony = _extract_harmony_voice(main_line, other_lines, max_voices=n_voices, volume_ratio=0.6)

    main_legato = _apply_legato(main_line, threshold=legato_threshold)
    events = sorted(main_legato + harmony, key=lambda n: n[1])

    duration = len(mono) / sample_rate
    synth = _synthesize_events(events, duration, sample_rate, waveform=waveform)

    if lowpass_cutoff > 0:
        synth = _apply_lowpass(synth, sample_rate, cutoff=lowpass_cutoff)

    peak = np.max(np.abs(synth))
    if peak > 1e-9:
        synth = synth / peak * 0.98

    envelope = _rms_envelope(mono, floor=0.50)
    envelope = np.cbrt(envelope)
    if len(envelope) == len(synth):
        synth = synth * envelope

    out = (1.0 - chip_mix) * mono + chip_mix * synth
    out = out.astype(np.float64)

    out = _gentle_compress(out, threshold=0.55, ratio=2.5)
    peak = np.max(np.abs(out))
    if peak > 1e-9:
        out = out / peak * 0.98
    out = np.clip(out, -1.0, 1.0).astype(np.float32)
    return np.stack([out, out], axis=0)
