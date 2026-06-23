import numpy as np
import librosa


def _pyin_to_notes(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int = 512,
    min_note_duration: float = 0.05,
) -> list[tuple[int, float, float, int]]:
    """用 pYIN 将音频转为音符事件列表。

    返回 [(midi_pitch, onset_time, offset_time, velocity), ...]
    """
    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sample_rate,
        hop_length=hop_length,
    )
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sample_rate, hop_length=hop_length)

    notes = []
    if len(f0) == 0:
        return notes

    current_pitch = None
    start_time = None
    velocities = []

    def _finalize_note():
        nonlocal current_pitch, start_time, velocities
        if current_pitch is None:
            return
        duration = times[i] - start_time
        if duration >= min_note_duration:
            notes.append((current_pitch, start_time, times[i], int(np.mean(velocities))))
        current_pitch = None
        velocities = []

    for i in range(len(f0)):
        if not voiced_flag[i] or np.isnan(f0[i]):
            _finalize_note()
            continue

        midi = int(np.round(librosa.hz_to_midi(f0[i])))
        # 用 voiced_prob 作为置信度/力度代理；后续可结合 RMS 能量改进
        vel = int(np.clip(voiced_prob[i] * 127, 1, 127))

        if current_pitch is None:
            current_pitch = midi
            start_time = times[i]
            velocities = [vel]
        elif abs(midi - current_pitch) <= 1:
            velocities.append(vel)
        else:
            _finalize_note()
            current_pitch = midi
            start_time = times[i]
            velocities = [vel]

    if current_pitch is not None:
        duration = times[-1] - start_time
        if duration >= min_note_duration:
            notes.append((current_pitch, start_time, times[-1], int(np.mean(velocities))))

    return notes


def _split_candidate_lines(
    notes: list[tuple[int, float, float, int]],
) -> list[list[tuple[int, float, float, int]]]:
    """将音符事件拆分为若干条不重叠的连续单音候选线。

    每条线内的音符按时间顺序排列，且任意相邻音符不重叠。
    """
    if not notes:
        return []

    notes = sorted(notes, key=lambda n: n[1])
    lines: list[list[tuple[int, float, float, int]]] = [[notes[0]]]

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
