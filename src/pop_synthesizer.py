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


def _detect_onsets(audio: np.ndarray, sample_rate: int, hop_length: int = 512, wait: int = 3) -> np.ndarray:
    """基于频谱通量检测音符起始点，返回秒级时间数组。"""
    onset_frames = librosa.onset.onset_detect(
        y=audio,
        sr=sample_rate,
        hop_length=hop_length,
        wait=wait,
        units='frames',
    )
    return librosa.frames_to_time(onset_frames, sr=sample_rate, hop_length=hop_length)


def _segment_audio(
    audio: np.ndarray,
    sample_rate: int,
    onsets: np.ndarray,
    min_note_duration: float = 0.05,
) -> list[tuple[int, int]]:
    """按 onset 切分音频为样本索引片段，过滤过短片段。"""
    if len(onsets) == 0:
        return [(0, len(audio))]

    sorted_onsets = np.sort(onsets)
    merged = [sorted_onsets[0]]
    for o in sorted_onsets[1:]:
        if o - merged[-1] >= min_note_duration:
            merged.append(o)
        else:
            pass

    boundaries = [0] + list(merged) + [len(audio) / sample_rate]
    segments = []
    for i in range(len(boundaries) - 1):
        start = int(boundaries[i] * sample_rate)
        end = int(boundaries[i + 1] * sample_rate)
        start = max(0, min(start, len(audio)))
        end = max(0, min(end, len(audio)))
        if end - start >= int(min_note_duration * sample_rate):
            segments.append((start, end))
    return segments


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


def _midi_to_freq(midi_note: int) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def _freq_to_midi(freq: float) -> float:
    if freq <= 0:
        return 0.0
    return 69.0 + 12.0 * np.log2(freq / 440.0)


def _bandlimited_waveform(phase: np.ndarray, freq: float, sample_rate: int, waveform: str) -> np.ndarray:
    """根据瞬时基频限制谐波数，生成带限方波/三角波/锯齿波。"""
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


def _extract_stable_pitches(
    segment: np.ndarray,
    sample_rate: int,
    n_voices: int = 4,
    hop_length: int = 512,
) -> list[tuple[int, int]]:
    """从单一片段内提取稳定的 1~n_voices 个音高。"""
    if len(segment) < hop_length:
        return []

    n_fft = min(2048, max(512, len(segment)))
    stft = np.abs(librosa.stft(segment, n_fft=n_fft, hop_length=hop_length))
    avg_spectrum = np.mean(stft, axis=1)
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)

    peaks, _ = signal.find_peaks(
        avg_spectrum,
        height=np.max(avg_spectrum) * 0.05,
        distance=max(1, int(50 / (sample_rate / n_fft))),
    )
    if len(peaks) == 0:
        return []

    peak_infos = [(int(p), float(avg_spectrum[p])) for p in peaks if freqs[p] > 40.0]
    peak_infos.sort(key=lambda x: x[1], reverse=True)
    peak_infos = peak_infos[:n_voices]

    rms = np.sqrt(np.mean(segment ** 2))
    velocity = int(np.clip(rms * 127 * 4, 1, 127))

    pitches = []
    for peak_idx, amp in peak_infos:
        freq = freqs[peak_idx]
        midi_float = 69.0 + 12.0 * np.log2(freq / 440.0)
        midi = int(np.round(np.clip(midi_float, 0, 127)))
        pitches.append((midi, velocity))

    seen = set()
    unique = []
    for midi, vel in pitches:
        if midi not in seen:
            seen.add(midi)
            unique.append((midi, vel))
    return unique


def _merge_consecutive_notes(
    notes: list[tuple[int, float, float, int]],
    gap_threshold: float = 0.05,
) -> list[tuple[int, float, float, int]]:
    """合并相邻片段中相同音高的音符，避免重复 attack。"""
    if not notes:
        return []

    notes = sorted(notes, key=lambda n: n[1])
    merged = [list(notes[0])]
    for pitch, onset, offset, velocity in notes[1:]:
        prev = merged[-1]
        if pitch == prev[0] and (onset - prev[2]) <= gap_threshold:
            prev[2] = max(prev[2], offset)
            prev[3] = max(prev[3], velocity)
        else:
            merged.append([pitch, onset, offset, velocity])
    return [tuple(n) for n in merged]


def _synthesize_events(
    notes: list[tuple[int, float, float, int]],
    duration: float,
    sample_rate: int,
    waveform: str = 'square',
) -> np.ndarray:
    """将复音音符事件合成为稳定音高的芯片音频。"""
    num_samples = int(duration * sample_rate)
    audio = np.zeros(num_samples, dtype=np.float64)

    for pitch, onset, offset, velocity in notes:
        freq = _midi_to_freq(pitch)
        start_idx = max(0, int(onset * sample_rate))
        end_idx = min(num_samples, int(offset * sample_rate))
        if start_idx >= end_idx:
            continue

        length = end_idx - start_idx
        t = np.arange(length) / sample_rate
        phase = 2.0 * np.pi * freq * t
        wave = _bandlimited_waveform(phase, freq, sample_rate, waveform)

        env = np.ones(length, dtype=np.float64)
        attack_samples = min(int(0.005 * sample_rate), length)
        decay_samples = min(int(0.015 * sample_rate), max(0, length - attack_samples))
        release_samples = min(int(0.02 * sample_rate), length)

        env[:attack_samples] = np.linspace(0.0, 1.0, attack_samples)
        if decay_samples > 0:
            env[attack_samples:attack_samples + decay_samples] = np.linspace(
                1.0, 0.85, decay_samples
            )
        if release_samples > 0:
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
    waveform: Literal['square', 'triangle', 'sawtooth'] = 'square',
    chip_mix: float = 0.75,
    n_voices: int = 4,
    hop_length: int = 512,
    min_note_duration: float = 0.05,
    pitch_stabilize: float = 1.0,
) -> np.ndarray:
    """
    基于原音频的流行 8-bit 合成。

    Args:
        audio: 输入音频，numpy 数组（单声道/立体声，任意 dtype）
        sample_rate: 采样率
        waveform: 合成器波形
        chip_mix: 合成器层混合比例（0~1）
        n_voices: 同时跟踪的声部数
        hop_length: STFT 帧移
        min_note_duration: 最小音符时长（秒）
        pitch_stabilize: 音高稳定因子（预留，暂未使用）

    Returns:
        合成后的立体声 float32 音频，shape (2, N)
    """
    mono = _audio_to_mono_float(audio)
    if len(mono) == 0:
        return np.zeros((2, 0), dtype=np.float32)

    onsets = _detect_onsets(mono, sample_rate, hop_length=hop_length)
    segments = _segment_audio(mono, sample_rate, onsets, min_note_duration=min_note_duration)

    raw_notes = []
    for start, end in segments:
        segment = mono[start:end]
        pitches = _extract_stable_pitches(segment, sample_rate, n_voices=n_voices, hop_length=hop_length)
        onset_time = start / sample_rate
        offset_time = end / sample_rate
        for midi, velocity in pitches:
            raw_notes.append((midi, onset_time, offset_time, velocity))

    notes = _merge_consecutive_notes(raw_notes, gap_threshold=min_note_duration)

    duration = len(mono) / sample_rate
    synth = _synthesize_events(notes, duration, sample_rate, waveform=waveform)

    # Peak normalize synth layer
    peak = np.max(np.abs(synth))
    if peak > 1e-9:
        synth = synth / peak * 0.98

    # RMS envelope modulation
    envelope = _rms_envelope(mono, floor=0.50)
    envelope = np.cbrt(envelope)
    if len(envelope) == len(synth):
        synth = synth * envelope

    # Mix original audio with chip synth layer
    out = (1.0 - chip_mix) * mono + chip_mix * synth
    out = out.astype(np.float64)

    # Gentle compression + peak normalization
    out = _gentle_compress(out, threshold=0.55, ratio=2.5)
    peak = np.max(np.abs(out))
    if peak > 1e-9:
        out = out / peak * 0.98
    out = np.clip(out, -1.0, 1.0).astype(np.float32)
    return np.stack([out, out], axis=0)
