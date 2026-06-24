import numpy as np
import librosa
from scipy import ndimage


def _smooth_f0(
    f0: np.ndarray,
    voiced_flag: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """对 voiced 帧的 f0 做中值滤波，去除单帧毛刺，unvoiced 帧保持原值。"""
    if len(f0) == 0:
        return f0
    kernel_size = max(3, kernel_size | 1)
    smoothed = ndimage.median_filter(f0, size=kernel_size, mode='nearest')
    result = f0.copy()
    result[voiced_flag] = smoothed[voiced_flag]
    return result


def _pyin_to_notes(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int = 512,
    min_note_duration: float = 0.05,
    pitch_quantize_strength: float = 1.0,
    f0_median_size: int = 3,
) -> list[tuple[float, float, float, int]]:
    """用 pYIN 将音频转为音符事件列表。

    Args:
        audio: 单声道音频数组
        sample_rate: 采样率
        hop_length: STFT 帧移
        min_note_duration: 最短音符时长（秒）
        pitch_quantize_strength: 音高量化强度，1.0=完全量化到半音，0.0=保留原始 float MIDI
        f0_median_size: f0 中值滤波窗口大小

    返回 [(midi_pitch, onset_time, offset_time, velocity), ...]
    其中 midi_pitch 在 strength < 1.0 时可能为 float
    """
    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sample_rate,
        hop_length=hop_length,
    )
    f0 = _smooth_f0(f0, voiced_flag, kernel_size=f0_median_size)
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sample_rate, hop_length=hop_length)

    notes = []
    if len(f0) == 0:
        return notes

    current_pitch = None
    start_time = None
    velocities = []

    def _finalize_note(end_time: float):
        nonlocal current_pitch, start_time, velocities
        if current_pitch is None:
            return
        duration = end_time - start_time
        if duration >= min_note_duration:
            notes.append((current_pitch, start_time, end_time, int(np.mean(velocities))))
        current_pitch = None
        velocities = []

    for i in range(len(f0)):
        if not voiced_flag[i] or np.isnan(f0[i]):
            _finalize_note(times[i])
            continue

        midi_float = librosa.hz_to_midi(f0[i])
        if pitch_quantize_strength >= 1.0:
            midi = round(midi_float)
        else:
            quantized = round(midi_float)
            midi = quantized + (1.0 - pitch_quantize_strength) * (midi_float - quantized)

        vel = int(np.clip(voiced_prob[i] * 127, 1, 127)) if not np.isnan(voiced_prob[i]) else 64

        if current_pitch is None:
            current_pitch = midi
            start_time = times[i]
            velocities = [vel]
        elif abs(midi - current_pitch) <= 0.5 * pitch_quantize_strength + 0.05:
            velocities.append(vel)
        else:
            _finalize_note(times[i])
            current_pitch = midi
            start_time = times[i]
            velocities = [vel]

    if current_pitch is not None:
        _finalize_note(times[-1])

    return notes


def _split_candidate_lines(
    notes: list[tuple[float, float, float, int]],
) -> list[list[tuple[float, float, float, int]]]:
    """将音符事件拆分为若干条不重叠的连续单音候选线。

    每条线内的音符按时间顺序排列，且任意相邻音符不重叠。
    """
    if not notes:
        return []

    notes = sorted(notes, key=lambda n: n[1])
    lines: list[list[tuple[float, float, float, int]]] = [[notes[0]]]

    for note in notes[1:]:
        onset = note[1]
        best_line = None
        best_gap = float('inf')
        for line in lines:
            last = line[-1]
            if last[2] <= onset:
                gap = onset - last[2]
                if gap < best_gap:
                    best_gap = gap
                    best_line = line
        if best_line is not None:
            best_line.append(note)
        else:
            lines.append([note])

    return lines


def _score_melody_line(
    line: list[tuple[float, float, float, int]],
    bar_duration: float = 2.0,
) -> float:
    """按歌唱平滑度、乐句重复度、音域居中、力度稳定性打分。"""
    if len(line) < 2:
        return 0.0

    pitches = [int(round(n[0])) for n in line]
    onsets = [n[1] for n in line]
    offsets = [n[2] for n in line]
    velocities = [n[3] for n in line]
    durations = [offsets[i] - onsets[i] for i in range(len(line))]

    intervals = [abs(pitches[i] - pitches[i - 1]) for i in range(1, len(pitches))]
    small_interval_ratio = sum(1 for iv in intervals if 0 <= iv <= 5) / len(intervals) if intervals else 0.0
    large_jump_ratio = sum(1 for iv in intervals if iv > 12) / len(intervals) if intervals else 0.0

    gaps = [onsets[i] - offsets[i - 1] for i in range(1, len(line))]
    rest_ratio = sum(1 for g in gaps if g > 0.05) / len(gaps) if gaps else 0.0
    no_rest_penalty = 0.2 if rest_ratio == 0 else 0.0

    long_note_ratio = sum(1 for d in durations if d >= 0.3) / len(durations) if durations else 0.0
    sixteenth_note_ratio = sum(1 for d in durations if d < 0.15) / len(durations) if durations else 0.0

    smoothness = (
        0.4 * small_interval_ratio
        + 0.3 * rest_ratio
        + 0.3 * long_note_ratio
        - 0.4 * sixteenth_note_ratio
        - 0.4 * large_jump_ratio
        - no_rest_penalty
    )
    smoothness = float(np.clip(smoothness, 0.0, 1.0))

    avg_duration = np.mean(durations) if durations else 0.25
    notes_per_bar = max(2, int(bar_duration / max(avg_duration, 0.01)))
    # 用音程序列方向（上升/下降/持平）比较重复度；注意重复音标记为 0
    contour = [np.sign(pitches[i] - pitches[i - 1]) for i in range(1, len(pitches))]

    # 降低 notes_per_bar 要求，使常见旋律都能进入重复度检测
    notes_per_bar = min(notes_per_bar, max(2, len(contour) // 2))

    repetition_score = 0.0
    if len(contour) >= notes_per_bar * 2:
        matches = 0
        segments = 0
        for start in range(0, len(contour) - notes_per_bar * 2 + 1, notes_per_bar):
            seg1 = contour[start:start + notes_per_bar]
            seg2 = contour[start + notes_per_bar:start + notes_per_bar * 2]
            if len(seg1) == len(seg2) and len(seg1) > 0:
                similarity = sum(1 for a, b in zip(seg1, seg2) if a == b) / len(seg1)
                matches += similarity
                segments += 1
        repetition_score = matches / segments if segments > 0 else 0.0
    repetition_score = float(np.clip(repetition_score, 0.0, 1.0))

    target_low = librosa.note_to_midi('C3')
    target_high = librosa.note_to_midi('F5')
    in_range = sum(1 for p in pitches if target_low <= p <= target_high) / len(pitches)
    pitch_range_score = float(np.clip(in_range, 0.0, 1.0))

    vel_diffs = [abs(velocities[i] - velocities[i - 1]) for i in range(1, len(velocities))]
    vel_variance = np.mean(vel_diffs) / 127.0 if vel_diffs else 0.0
    velocity_score = float(np.clip(1.0 - vel_variance, 0.0, 1.0))

    return (
        0.45 * smoothness
        + 0.30 * repetition_score
        + 0.15 * pitch_range_score
        + 0.10 * velocity_score
    )


def _apply_hard_filters(
    lines: list[list[tuple[float, float, float, int]]],
    bar_duration: float = 2.0,
) -> list[list[tuple[float, float, float, int]]]:
    """硬性过滤：碎音装饰线与重复伴奏线淘汰。"""
    filtered = []
    for line in lines:
        if len(line) == 0:
            continue

        durations = [n[2] - n[1] for n in line]
        short_ratio = sum(1 for d in durations if d < 0.15) / len(durations)
        if short_ratio > 0.8:
            continue

        pitches = [int(round(n[0])) for n in line]
        avg_duration = np.mean(durations) if durations else 0.25
        notes_per_bar = max(2, int(bar_duration / max(avg_duration, 0.01)))

        if len(pitches) >= notes_per_bar * 8:
            pattern = pitches[:notes_per_bar]
            repeats = 0
            for i in range(notes_per_bar, len(pitches), notes_per_bar):
                segment = pitches[i:i + notes_per_bar]
                if len(segment) == len(pattern) and all(a == b for a, b in zip(pattern, segment)):
                    repeats += 1
                else:
                    break
            if repeats >= 7:
                continue

        filtered.append(line)
    return filtered


_CONSONANT_INTERVALS = {3, 4, 7, 8, 12}


def _extract_main_melody(
    lines: list[list[tuple[float, float, float, int]]],
) -> list[tuple[float, float, float, int]]:
    """从候选线中选出主旋律线。"""
    if not lines:
        return []
    scored = [(_score_melody_line(line), line) for line in lines]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _extract_harmony_voice(
    main_line: list[tuple[float, float, float, int]],
    other_lines: list[list[tuple[float, float, float, int]]],
    max_voices: int = 1,
    volume_ratio: float = 0.6,
) -> list[tuple[float, float, float, int]]:
    """从剩余候选线中提取与主旋律形成和谐音程的点缀音符。"""
    if max_voices <= 0 or not other_lines:
        return []

    scored = [(_score_melody_line(line), line) for line in other_lines]
    scored.sort(key=lambda x: x[0], reverse=True)

    harmony_notes = []
    voices_added = 0
    for _, line in scored:
        if voices_added >= max_voices:
            break
        for note in line:
            pitch, onset, offset, _ = note
            for main_note in main_line:
                m_pitch, m_onset, m_offset, _ = main_note
                if onset < m_offset and offset > m_onset:
                    interval = abs(int(round(pitch)) - int(round(m_pitch))) % 12
                    if interval in _CONSONANT_INTERVALS:
                        harmony_notes.append((pitch, onset, offset, int(127 * volume_ratio)))
                        break
        voices_added += 1

    harmony_notes.sort(key=lambda n: n[1])
    return harmony_notes
