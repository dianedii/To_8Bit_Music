import numpy as np
from typing import List, Tuple


def midi_to_freq(midi_note: int) -> float:
    """MIDI 音符转频率。"""
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def _bandlimited_square(phase: np.ndarray, freq: float, sample_rate: int) -> np.ndarray:
    """
    带限方波：根据基频限制谐波数量，既保留方波电子感，
    又避免高频谐波过多导致刺耳/过亮。
    """
    max_harm = int(np.floor((sample_rate / 2.5) / freq))
    max_harm = max(1, min(max_harm, 40))
    wave = np.zeros_like(phase)
    for n in range(1, max_harm + 1, 2):
        wave += (4.0 / (np.pi * n)) * np.sin(n * phase)
    return np.clip(wave, -1.0, 1.0)


def synthesize(
    notes: List[Tuple[int, float, float, int]],
    duration: float,
    sample_rate: int = 44100,
    purity: int = 0,
    volume: int = 80,
) -> np.ndarray:
    """
    将音符序列合成为 8-bit 风格音频。
    purity: 0 ~ 100，0=纯方波，100=轻微 vibrato + release
    volume: 0 ~ 100
    """
    if duration <= 0:
        raise ValueError("duration must be > 0")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    volume = max(0, min(100, volume))
    purity = max(0, min(100, purity))

    num_samples = int(duration * sample_rate)
    audio = np.zeros(num_samples, dtype=np.float64)

    base_amp = (volume / 100.0) * 0.5  # 单音符基础振幅
    vibrato_depth = (purity / 100.0) * 0.5  # 半音范围内的轻微颤音
    release_time = (purity / 100.0) * 0.03  # 30ms 以内的 release

    for pitch, onset, offset, velocity in notes:
        velocity = max(0, min(127, velocity))
        if offset <= onset:
            continue
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

        # 使用带限方波，降低刺耳高频
        wave = _bandlimited_square(phase, freq, sample_rate)

        # 应用包络：快速 attack + 可选 release
        env = np.ones_like(t)
        attack_samples = min(10, len(t))
        env[:attack_samples] = np.linspace(0.0, 1.0, attack_samples)
        if release_time > 0:
            release_samples = min(int(release_time * sample_rate), len(t))
            env[-release_samples:] = np.linspace(1.0, 0.0, release_samples)

        note_amp = base_amp * (velocity / 127.0)
        audio[start_idx:end_idx] += wave * env * note_amp

    # 峰值归一化到 0.95，保证响度与商业音频接近
    peak = np.max(np.abs(audio))
    if peak > 1e-9:
        audio = audio / peak * 0.95
    audio = np.clip(audio, -1.0, 1.0)
    return audio.astype(np.float32)
