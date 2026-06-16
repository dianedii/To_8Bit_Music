import numpy as np
from typing import List, Tuple


def midi_to_freq(midi_note: int) -> float:
    """MIDI 音符转频率。"""
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def synthesize(
    notes: List[Tuple[int, float, float, int]],
    duration: float,
    sample_rate: int = 44100,
    purity: int = 0,
    volume: int = 80,
) -> np.ndarray:
    """
    将音符序列合成为 FC 方波音频。
    purity: 0 ~ 100，0=纯方波，100=轻微 vibrato + release
    volume: 0 ~ 100
    """
    num_samples = int(duration * sample_rate)
    audio = np.zeros(num_samples, dtype=np.float64)

    max_amp = (volume / 100.0) * 0.45  # 留 headroom 防止破音
    vibrato_depth = (purity / 100.0) * 0.5  # 半音范围内的轻微颤音
    release_time = (purity / 100.0) * 0.03  # 30ms 以内的 release

    for pitch, onset, offset, velocity in notes:
        freq = midi_to_freq(pitch)
        start_idx = max(0, int(onset * sample_rate))
        end_idx = min(num_samples, int(offset * sample_rate))
        if start_idx >= end_idx:
            continue

        t = np.arange(end_idx - start_idx) / sample_rate
        # 轻微 vibrato：频率微调
        if vibrato_depth > 0:
            vibrato = 1.0 + vibrato_depth * 0.01 * np.sin(2 * np.pi * 6.0 * t)
            phase = np.cumsum(2 * np.pi * freq * vibrato / sample_rate)
        else:
            phase = 2 * np.pi * freq * t

        wave = np.sign(np.sin(phase)).astype(np.float64)

        # 应用包络：快速 attack + 可选 release
        env = np.ones_like(t)
        attack_samples = min(10, len(t))
        env[:attack_samples] = np.linspace(0.0, 1.0, attack_samples)
        if release_time > 0:
            release_samples = min(int(release_time * sample_rate), len(t))
            env[-release_samples:] = np.linspace(1.0, 0.0, release_samples)

        note_amp = max_amp * (velocity / 127.0)
        audio[start_idx:end_idx] += wave * env * note_amp

    # 软限幅防止破音
    audio = np.tanh(audio)
    return audio.astype(np.float32)
