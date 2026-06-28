"""
流行 8-bit / 芯片合成器（目标样例风格）

不再暴露“选波”类参数，内部固定一套面向 goals/ 示例听感的多层合成方案：
- 方波主旋 + 高八度亮度层 + 高两个八度点缀
- 三角波低音贝斯
- 可选和声层
- 基于 onset 的 8-bit 噪声鼓点
- 总线 EQ、压缩/削波、响度归一化
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


def _midi_to_freq(midi_note: float) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def _freq_to_midi(freq: float) -> float:
    if freq <= 0:
        return 0.0
    return 69.0 + 12.0 * np.log2(freq / 440.0)


def _bandlimited_waveform(phase: np.ndarray, freq: float, sample_rate: int, waveform: str) -> np.ndarray:
    """根据瞬时基频限制谐波数，生成带限方波/三角波。"""
    if waveform == 'sine':
        return np.sin(phase)

    nyq = sample_rate / 2.2
    max_harm = int(np.floor(nyq / max(freq, 20.0)))
    max_harm = max(1, min(max_harm, 80))

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


def _synthesize_layer(
    notes: list[tuple[float, float, float, int]],
    duration: float,
    sample_rate: int,
    waveform: str = 'square',
    amp_scale: float = 1.0,
    attack_s: float = 0.002,
    release_s: float = 0.04,
    decay_s: float = 0.25,
) -> np.ndarray:
    """将复音音符事件合成为稳定音高的芯片音频层（带指数衰减，避免长音拖尾嗡嗡）。"""
    num_samples = int(duration * sample_rate)
    audio = np.zeros(num_samples, dtype=np.float64)

    for pitch, onset, offset, velocity in notes:
        if offset <= onset:
            continue
        freq = _midi_to_freq(pitch)
        start_idx = max(0, int(onset * sample_rate))
        end_idx = min(num_samples, int(offset * sample_rate))
        if start_idx >= end_idx:
            continue

        length = end_idx - start_idx
        t = np.arange(length) / sample_rate
        phase = 2.0 * np.pi * freq * t
        wave = _bandlimited_waveform(phase, freq, sample_rate, waveform)

        # attack + 指数衰减 + 结尾 release
        env = np.exp(-t / max(decay_s, 0.001))
        attack_samples = min(int(attack_s * sample_rate), length)
        if attack_samples > 0:
            env[:attack_samples] = np.linspace(0.0, 1.0, attack_samples)

        release_samples = min(int(release_s * sample_rate), length)
        if release_samples > 0:
            release_start_idx = max(0, length - release_samples)
            env[release_start_idx:] *= np.linspace(1.0, 0.0, length - release_start_idx)

        amp = (velocity / 127.0) * amp_scale
        audio[start_idx:end_idx] += wave * env * amp

    return audio


def _add_8bit_drums(
    audio: np.ndarray,
    y: np.ndarray,
    sample_rate: int,
    strength: float = 0.25,
) -> np.ndarray:
    """根据原曲 onset 触发短促带通噪声，模拟 8-bit 鼓点/镲片。"""
    if strength <= 0:
        return audio

    onset_env = librosa.onset.onset_strength(y=y, sr=sample_rate, hop_length=512)
    hop = 512
    times = np.arange(len(onset_env)) * hop
    interp_onset = np.interp(np.arange(len(audio)), times, onset_env)
    interp_onset = interp_onset[:len(audio)]
    max_onset = np.max(interp_onset)
    if max_onset <= 0:
        return audio
    interp_onset = interp_onset / max_onset

    peaks, _ = signal.find_peaks(
        interp_onset,
        distance=int(sample_rate * 0.15),
        prominence=0.15,
    )

    out = audio.copy()
    for p in peaks:
        dur = int(sample_rate * 0.025)
        if p + dur > len(out):
            continue
        burst = np.random.randn(dur)
        # 快速衰减包络
        env = np.exp(-np.arange(dur) / (sample_rate * 0.008))
        burst = burst * env
        # 带通让噪声更像 8-bit hihat/snare
        sos = signal.butter(4, [2500 / (sample_rate / 2), 9000 / (sample_rate / 2)], btype='band', output='sos')
        burst = signal.sosfilt(sos, burst)
        out[p:p + dur] += burst * strength

    return out


def _shape_bus(
    audio: np.ndarray,
    sample_rate: int,
    highpass_hz: float = 60.0,
    high_shelf_band: tuple[float, float] = (3000.0, 8000.0),
    high_shelf_gain: float = 0.65,
) -> np.ndarray:
    """总线塑形：去除超低频、提亮高频、轻度压缩控制 crest。"""
    # 高通，去除浑浊
    if highpass_hz > 0:
        sos_hp = signal.butter(2, highpass_hz / (sample_rate / 2), btype='high', output='sos')
        audio = signal.sosfilt(sos_hp, audio)

    # 高频带提升
    lo, hi = high_shelf_band
    sos_band = signal.butter(2, [lo / (sample_rate / 2), hi / (sample_rate / 2)], btype='band', output='sos')
    bright = signal.sosfilt(sos_band, audio)
    audio = audio + bright * high_shelf_gain

    # 适度压缩，把 crest 从自然 5~7 压到接近商业 8-bit 的 4 左右
    abs_audio = np.abs(audio)
    threshold = 0.45
    ratio = 4.0
    gain = np.ones_like(audio)
    over = abs_audio > threshold
    if np.any(over):
        target = threshold + (abs_audio[over] - threshold) / ratio
        gain[over] = target / abs_audio[over]
    audio = audio * gain

    return audio


def _peak_normalize_and_limit(
    audio: np.ndarray,
    max_peak: float = 0.98,
    soft_clip: float = 0.75,
) -> np.ndarray:
    """峰值归一化+软限幅，保留自然动态的同时获得商业响度。"""
    if soft_clip > 0:
        # 软削波：先把超过 soft_clip 的部分 tanh 压缩，而不是硬切
        abs_audio = np.abs(audio)
        over = abs_audio > soft_clip
        if np.any(over):
            audio = audio.copy()
            scaled = (abs_audio[over] - soft_clip) / soft_clip
            audio[over] = np.sign(audio[over]) * soft_clip * (1.0 + np.tanh(scaled))

    peak = np.max(np.abs(audio))
    if peak > 1e-9:
        audio = audio / peak * max_peak
    return np.clip(audio, -max_peak, max_peak)


def _synthesize_from_notes(
    notes: list[tuple[int, float, float, int]],
    mono: np.ndarray,
    sample_rate: int,
    volume: int = 80,
    melody_notes: list[tuple[int, float, float, int]] | None = None,
) -> np.ndarray:
    """从已提取的音符事件列表出发，按音高分层合成目标样例风格 8-bit。

    Args:
        notes: 所有转录出的音符事件。
        mono: 原曲单声道音频，用于鼓点提取。
        sample_rate: 采样率。
        volume: 输出音量 0~100。
        melody_notes: 可选的主旋律线。提供时，旋律音无论音高都会获得带亮度层的
            主旋处理；其余音符作为背景伴奏。为 None 时保持旧的按音高分层行为。
    """
    duration = len(mono) / sample_rate

    if not notes:
        # 没有音符时仍保留鼓点和原曲节奏感
        synth = np.zeros(len(mono), dtype=np.float64)
        synth = _add_8bit_drums(synth, mono, sample_rate, strength=0.25)
        synth = _shape_bus(synth, sample_rate)
        synth = _peak_normalize_and_limit(synth, max_peak=0.98, soft_clip=0.75)
        volume = max(0, min(100, volume))
        synth = synth * (volume / 100.0)
        synth = np.clip(synth, -1.0, 1.0)
        return np.stack([synth, synth], axis=0).astype(np.float32)

    # 清理：过滤弱音 + 限制最大音符长度，避免踏板/延音造成持续嗡嗡
    max_note_duration = 0.8
    min_velocity = 35

    def _clean(note_list: list[tuple[int, float, float, int]]) -> list[tuple[int, float, float, int]]:
        return [
            (p, t0, min(t1, t0 + max_note_duration), max(v, min_velocity))
            for p, t0, t1, v in note_list
            if v >= min_velocity
        ]

    cleaned_notes = _clean(notes)
    cleaned_melody = _clean(melody_notes) if melody_notes else []

    # 区分旋律与伴奏：用 (pitch, onset) 匹配，offset 可能在清理时被截断
    melody_set = {(p, t0) for p, t0, _, _ in cleaned_melody}
    melody = [n for n in cleaned_notes if (n[0], n[1]) in melody_set]
    accompaniment = [n for n in cleaned_notes if (n[0], n[1]) not in melody_set]

    if melody:
        # ---- 旋律优先：主旋律无论音高都获得带八度层的 lead 处理 ----
        # 伴奏层保持与旧版接近的音量，但被排除在外的旋律音不再重复出现
        accomp_bass = [(p, t0, t1, v) for p, t0, t1, v in accompaniment if p < 50]
        accomp_lead = [(p, t0, t1, v) for p, t0, t1, v in accompaniment if 50 <= p < 75]
        accomp_high = [(p, t0, t1, min(127, int(v * 0.55))) for p, t0, t1, v in accompaniment if p >= 75]

        accomp_lead_high = [(min(127, p + 12), t0, t1, min(127, int(v * 0.45))) for p, t0, t1, v in accomp_lead]
        accomp_lead_higher = [(min(127, p + 24), t0, t1, min(127, int(v * 0.20))) for p, t0, t1, v in accomp_lead]

        melody_low = [(p, t0, t1, v) for p, t0, t1, v in melody if p < 50]
        melody_mid = [(p, t0, t1, v) for p, t0, t1, v in melody if 50 <= p < 75]
        melody_high = [(p, t0, t1, v) for p, t0, t1, v in melody if p >= 75]

        def _octave_layers(src: list[tuple[int, float, float, int]], v_high: float, v_higher: float):
            high = [(min(127, p + 12), t0, t1, min(127, int(v * v_high))) for p, t0, t1, v in src]
            higher = [(min(127, p + 24), t0, t1, min(127, int(v * v_higher))) for p, t0, t1, v in src]
            return high, higher

        melody_low_high, melody_low_higher = _octave_layers(melody_low, 0.45, 0.20)
        melody_mid_high, melody_mid_higher = _octave_layers(melody_mid, 0.45, 0.20)
        melody_high_high, melody_high_higher = _octave_layers(melody_high, 0.45, 0.20)

        # 伴奏层：保持原配比
        synth_bass = _synthesize_layer(accomp_bass, duration, sample_rate, waveform='triangle', amp_scale=0.9, decay_s=0.50)
        synth_accomp_lead = _synthesize_layer(accomp_lead, duration, sample_rate, waveform='square', amp_scale=0.9, decay_s=0.25)
        synth_accomp_lead_high = _synthesize_layer(accomp_lead_high, duration, sample_rate, waveform='square', amp_scale=0.5, decay_s=0.20)
        synth_accomp_lead_higher = _synthesize_layer(accomp_lead_higher, duration, sample_rate, waveform='square', amp_scale=0.22, decay_s=0.15)
        synth_accomp_high = _synthesize_layer(accomp_high, duration, sample_rate, waveform='square', amp_scale=0.45, decay_s=0.15)

        # 旋律层：统一按 lead 处理，高音也能获得八度亮度层
        synth_melody_low = _synthesize_layer(melody_low, duration, sample_rate, waveform='square', amp_scale=0.9, decay_s=0.25)
        synth_melody_low_high = _synthesize_layer(melody_low_high, duration, sample_rate, waveform='square', amp_scale=0.5, decay_s=0.20)
        synth_melody_low_higher = _synthesize_layer(melody_low_higher, duration, sample_rate, waveform='square', amp_scale=0.22, decay_s=0.15)

        synth_melody_mid = _synthesize_layer(melody_mid, duration, sample_rate, waveform='square', amp_scale=0.9, decay_s=0.25)
        synth_melody_mid_high = _synthesize_layer(melody_mid_high, duration, sample_rate, waveform='square', amp_scale=0.5, decay_s=0.20)
        synth_melody_mid_higher = _synthesize_layer(melody_mid_higher, duration, sample_rate, waveform='square', amp_scale=0.22, decay_s=0.15)

        synth_melody_high = _synthesize_layer(melody_high, duration, sample_rate, waveform='square', amp_scale=0.9, decay_s=0.25)
        synth_melody_high_high = _synthesize_layer(melody_high_high, duration, sample_rate, waveform='square', amp_scale=0.5, decay_s=0.20)
        synth_melody_high_higher = _synthesize_layer(melody_high_higher, duration, sample_rate, waveform='square', amp_scale=0.22, decay_s=0.15)

        synth = (
            synth_bass * 0.75
            + synth_accomp_lead
            + synth_accomp_lead_high * 0.8
            + synth_accomp_lead_higher * 0.5
            + synth_accomp_high * 0.6
            + synth_melody_low
            + synth_melody_low_high * 0.8
            + synth_melody_low_higher * 0.5
            + synth_melody_mid
            + synth_melody_mid_high * 0.8
            + synth_melody_mid_higher * 0.5
            + synth_melody_high
            + synth_melody_high_high * 0.8
            + synth_melody_high_higher * 0.5
        )
    else:
        # 无主旋信息时退回到旧行为，保持兼容
        bass_notes = [(p, t0, t1, v) for p, t0, t1, v in cleaned_notes if p < 50]
        lead_notes = [(p, t0, t1, v) for p, t0, t1, v in cleaned_notes if 50 <= p < 75]
        high_notes = [(p, t0, t1, min(127, int(v * 0.55))) for p, t0, t1, v in cleaned_notes if p >= 75]

        lead_high = [(min(127, p + 12), t0, t1, min(127, int(v * 0.45))) for p, t0, t1, v in lead_notes]
        lead_higher = [(min(127, p + 24), t0, t1, min(127, int(v * 0.20))) for p, t0, t1, v in lead_notes]

        synth_bass = _synthesize_layer(bass_notes, duration, sample_rate, waveform='triangle', amp_scale=0.9, decay_s=0.50)
        synth_lead = _synthesize_layer(lead_notes, duration, sample_rate, waveform='square', amp_scale=0.9, decay_s=0.25)
        synth_lead_high = _synthesize_layer(lead_high, duration, sample_rate, waveform='square', amp_scale=0.5, decay_s=0.20)
        synth_lead_higher = _synthesize_layer(lead_higher, duration, sample_rate, waveform='square', amp_scale=0.22, decay_s=0.15)
        synth_high = _synthesize_layer(high_notes, duration, sample_rate, waveform='square', amp_scale=0.45, decay_s=0.15)

        synth = synth_bass * 0.75 + synth_lead + synth_lead_high * 0.8 + synth_lead_higher * 0.5 + synth_high * 0.6

    synth = _add_8bit_drums(synth, mono, sample_rate, strength=0.20)
    synth = _shape_bus(synth, sample_rate, highpass_hz=120.0)
    synth = _peak_normalize_and_limit(synth, max_peak=0.98, soft_clip=0.70)

    # 额外输出增益，让默认音量下的 RMS 落在目标样例范围
    output_gain = 1.0
    volume = max(0, min(100, volume))
    synth = synth * output_gain * (volume / 100.0)
    synth = np.clip(synth, -1.0, 1.0)

    return np.stack([synth, synth], axis=0).astype(np.float32)


def synthesize_pop_chip(
    audio: np.ndarray,
    sample_rate: int = 44100,
    volume: int = 80,
) -> np.ndarray:
    """
    基于主旋律提取的流行 8-bit 合成（目标样例风格，快速 pyin 路径）。

    Args:
        audio: 输入音频，numpy 数组（单声道/立体声，任意 dtype）
        sample_rate: 采样率
        volume: 整体音量 0~100

    Returns:
        合成后的立体声 float32 音频，shape (2, N)
    """
    from src.pop_melody import _pyin_to_notes

    mono = _audio_to_mono_float(audio)
    if len(mono) == 0:
        return np.zeros((2, 0), dtype=np.float32)

    notes = _pyin_to_notes(
        mono,
        sample_rate,
        hop_length=512,
        min_note_duration=0.03,
        pitch_quantize_strength=1.0,
        f0_median_size=3,
    )

    return _synthesize_from_notes(notes, mono, sample_rate, volume=volume)


# ---- 以下兼容旧测试/旧调用，保持向后兼容 ----

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


def _synthesize_events(
    notes: list[tuple[float, float, float, int]],
    duration: float,
    sample_rate: int,
    waveform: str = 'square',
) -> np.ndarray:
    """旧版复音合成入口，现委托给 _synthesize_layer 并保留长尾 release。"""
    return _synthesize_layer(
        notes,
        duration,
        sample_rate,
        waveform=waveform,
        amp_scale=0.25,
        attack_s=0.010,
        release_s=0.040,
    )
