import numpy as np
import librosa


def _pyin_to_notes(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int = 512,
    min_note_duration: float = 0.05,
) -> list[tuple[int, float, float, int]]:
    """用 pYIN 将音频转为音符事件列表。"""
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

    for i in range(len(f0)):
        if not voiced_flag[i] or np.isnan(f0[i]):
            if current_pitch is not None:
                duration = times[i] - start_time
                if duration >= min_note_duration:
                    notes.append((current_pitch, start_time, times[i], int(np.mean(velocities))))
                current_pitch = None
                velocities = []
            continue

        midi = int(np.round(librosa.hz_to_midi(f0[i])))
        vel = int(np.clip(voiced_prob[i] * 127, 1, 127))

        if current_pitch is None:
            current_pitch = midi
            start_time = times[i]
            velocities = [vel]
        elif abs(midi - current_pitch) <= 1:
            velocities.append(vel)
        else:
            duration = times[i] - start_time
            if duration >= min_note_duration:
                notes.append((current_pitch, start_time, times[i], int(np.mean(velocities))))
            current_pitch = midi
            start_time = times[i]
            velocities = [vel]

    if current_pitch is not None:
        duration = times[-1] - start_time
        if duration >= min_note_duration:
            notes.append((current_pitch, start_time, times[-1], int(np.mean(velocities))))

    return notes
