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


def _match_peaks(prev_peaks, candidates, max_dist: float = 0.35):
    """将当前帧候选峰值与上一帧按频率最近邻匹配。"""
    matched = [None] * len(prev_peaks)
    used = set()
    for i, (pf, pa) in enumerate(prev_peaks):
        best_j = None
        best_dist = float('inf')
        for j, (cf, ca) in enumerate(candidates):
            if j in used or cf <= 0:
                continue
            dist = abs(np.log(cf / pf)) if pf > 0 else abs(cf - pf)
            if dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j is not None and best_dist < max_dist:
            matched[i] = candidates[best_j]
            used.add(best_j)
    return matched, used


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


def synthesize_pop_chip(
    audio: np.ndarray,
    sample_rate: int = 44100,
    waveform: Literal['square', 'triangle', 'sawtooth'] = 'square',
    chip_mix: float = 0.75,
    n_voices: int = 4,
    hop_length: int = 512,
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

    Returns:
        合成后的单声道 float32 音频
    """
    mono = _audio_to_mono_float(audio)
    if len(mono) == 0:
        return np.zeros(0, dtype=np.float32)

    stft = librosa.stft(mono, hop_length=hop_length)
    mag = np.abs(stft)
    freqs = librosa.fft_frequencies(sr=sample_rate)
    n_frames = mag.shape[1]

    # 每帧提取候选峰值并跟踪声部
    tracked = []
    for t in range(n_frames):
        frame = mag[:, t]
        peaks, _ = signal.find_peaks(frame, height=np.max(frame) * 0.015, distance=2)
        candidates = []
        if len(peaks) > 0:
            for p in peaks[np.argsort(frame[peaks])[::-1]]:
                if freqs[p] > 40.0:
                    candidates.append((float(freqs[p]), float(frame[p])))
        candidates = candidates[:n_voices * 3]

        if t == 0:
            init = candidates[:n_voices]
            while len(init) < n_voices:
                init.append((0.0, 0.0))
            tracked.append(init)
        else:
            prev = tracked[t - 1]
            matched, used = _match_peaks(prev, candidates, max_dist=0.35)
            filled = []
            for i in range(n_voices):
                if matched[i] is not None:
                    filled.append(matched[i])
                else:
                    appended = False
                    for j, c in enumerate(candidates):
                        if j not in used:
                            filled.append(c)
                            used.add(j)
                            appended = True
                            break
                    if not appended:
                        pf, pa = prev[i]
                        filled.append((pf, pa * 0.7))
            tracked.append(filled)

    # 准备每个声部的频率/振幅序列
    voice_data = []
    for i in range(n_voices):
        vf = np.array([tracked[t][i][0] for t in range(n_frames)])
        va = np.array([tracked[t][i][1] for t in range(n_frames)])
        # 填补零频率
        for t in range(1, n_frames):
            if vf[t] <= 0 and vf[t - 1] > 0:
                vf[t] = vf[t - 1]
                va[t] = va[t] * 0.5
        # 中值滤波平滑
        if n_frames >= 5:
            vf = signal.medfilt(vf, kernel_size=5)
            va = signal.medfilt(va, kernel_size=5)
        voice_data.append((vf, va))

    times = librosa.times_like(mag, sr=sample_rate, hop_length=hop_length)
    t_full = np.arange(len(mono)) / sample_rate

    synth = np.zeros(len(mono), dtype=np.float64)
    global_max_amp = max(1e-9, max(np.max(va) for _, va in voice_data))

    for i in range(n_voices):
        vf, va = voice_data[i]
        f_interp = np.interp(t_full, times, vf)
        a_interp = np.interp(t_full, times, va)
        f_interp = np.clip(f_interp, 20.0, 8000.0)
        a_interp = a_interp / global_max_amp
        fixed_phase_offset = (i * 2.0944) % (2 * np.pi)
        phase = 2.0 * np.pi * np.cumsum(f_interp) / sample_rate + fixed_phase_offset
        wave = _bandlimited_waveform(phase, np.median(f_interp), sample_rate, waveform)
        synth += wave * a_interp

    peak = np.max(np.abs(synth))
    if peak > 1e-9:
        synth = synth / peak * 0.98

    # 用原音频包络调制，保留动态；对包络做开立方，既提升安静段落响度又保留层次感
    envelope = _rms_envelope(mono, floor=0.50)
    envelope = np.cbrt(envelope)
    synth = synth * envelope

    # 混合原音频与芯片合成层
    out = (1.0 - chip_mix) * mono + chip_mix * synth
    out = out.astype(np.float64)

    # 柔和压缩 + 峰值归一化，让响度接近商业参考
    out = _gentle_compress(out, threshold=0.55, ratio=2.5)
    peak = np.max(np.abs(out))
    if peak > 1e-9:
        out = out / peak * 0.98
    out = np.clip(out, -1.0, 1.0).astype(np.float32)
    # 输出立体声（左右相同），与参考 8.mp3 一致
    return np.stack([out, out], axis=0)
