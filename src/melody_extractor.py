from typing import List, Tuple


def _score_note(pitch: int, duration: float, velocity: int) -> float:
    """综合音符时长、力度和音高打分。"""
    # 主旋律通常位于高音区，因此给音高一个适度的固定奖励，使其在时长和力度相近时优先被选中。
    # Formula: duration * (velocity / 127.0) + (pitch / 127.0) * 1.5
    # Rationale: duration is the primary factor; normalized velocity scales duration;
    # pitch gets a fixed 1.5x bonus so higher notes win when duration/velocity are
    # similar (since melodies tend to be in higher registers).
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

    notes = [
        n for n in notes
        if n[2] > n[1] and n[3] > 0
    ]

    if not notes:
        return []

    notes = sorted(notes, key=lambda n: n[1])
    window_s = window_ms / 1000.0
    melody = []
    current_group = [notes[0]]
    current_end = notes[0][2]

    for note in notes[1:]:
        # 如果当前音符与当前组的时间窗重叠或紧邻，则加入同一组
        if note[1] <= current_end + window_s:
            current_group.append(note)
            current_end = max(current_end, note[2])
        else:
            melody.append(max(current_group, key=lambda n: _score_note(n[0], n[2] - n[1], n[3])))
            current_group = [note]
            current_end = note[2]

    melody.append(max(current_group, key=lambda n: _score_note(n[0], n[2] - n[1], n[3])))
    return melody
