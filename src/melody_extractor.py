from typing import List, Tuple


def _score_note(pitch: int, duration: float, velocity: int) -> float:
    """综合音符时长、力度和音高打分。"""
    return duration * (velocity / 127.0) + (pitch / 127.0) * 1.5


def extract_melody(
    notes: List[Tuple[int, float, float, int]],
    window_ms: int = 50,
) -> List[Tuple[int, float, float, int]]:
    """
    从多声部钢琴音符中提取单声部主旋律。
    notes: [(pitch, onset, offset, velocity), ...]
    返回: [(pitch, onset, offset, velocity), ...]
    """
    if not notes:
        return []

    notes = sorted(notes, key=lambda n: n[1])
    window_s = window_ms / 1000.0
    melody = []
    current_group = [notes[0]]
    current_end = notes[0][2]

    def _pick_best(group):
        best = max(group, key=lambda n: _score_note(n[0], n[2] - n[1], n[3]))
        return best

    for note in notes[1:]:
        # 如果当前音符与当前组的时间窗重叠或紧邻，则加入同一组
        if note[1] <= current_end + window_s:
            current_group.append(note)
            current_end = max(current_end, note[2])
        else:
            melody.append(_pick_best(current_group))
            current_group = [note]
            current_end = note[2]

    melody.append(_pick_best(current_group))
    return melody
