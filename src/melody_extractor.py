from typing import List, Tuple


def _score_note(pitch: int, duration: float, velocity: int) -> float:
    """综合音符时长、力度和音高打分。

    主旋律通常音高偏高、时值适中、力度较强。这里让时长和力度占主导，
    音高作为适度奖励；同时惩罚过长的音符（多为和弦/低音延音），避免
    它们长期霸占主旋律位置。
    """
    score = duration * (velocity / 127.0) + (pitch / 127.0) * 1.0
    if duration > 2.0:
        score *= 0.3
    elif duration > 1.0:
        score *= 0.7
    return score


def extract_melody(
    notes: List[Tuple[int, float, float, int]],
    window_ms: int = 50,
) -> List[Tuple[int, float, float, int]]:
    """
    从多声部钢琴音符中提取单声部主旋律。

    采用事件驱动方式维护当前活跃音符集合，在每个音符开始/结束事件点
    重新评估主旋律。通过"切换阈值"避免频繁跳变，通过"最大旋律音符时长"
    防止被某个长伴奏音（如延音低音/和弦）长期霸占主旋律位置，从而得到
    更连续、可辨的主旋律线。

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

    # 构建开始/结束事件，同一时间先处理结束再处理开始
    events = []
    for n in notes:
        events.append((n[1], 1, n))   # on
        events.append((n[2], 0, n))   # off
    events.sort(key=lambda e: (e[0], e[1]))

    active: List[Tuple[int, float, float, int]] = []
    current_note: Tuple[int, float, float, int] | None = None
    current_start: float | None = None
    melody: List[Tuple[int, float, float, int]] = []

    switch_threshold = 0.02
    # 单个旋律音最长持续 0.5 秒，超过强制重新评估，避免旋律音拖太长
    max_melody_note_duration = 0.5

    def _pick_best(candidates: List[Tuple[int, float, float, int]]) -> Tuple[int, float, float, int]:
        return max(
            candidates,
            key=lambda n: _score_note(n[0], n[2] - n[1], n[3]),
        )

    idx = 0
    while idx < len(events):
        time = events[idx][0]
        # 批量处理同一时刻的所有事件，避免同时开始的音符产生零时长片段
        while idx < len(events) and events[idx][0] == time:
            _, event_type, note = events[idx]
            if event_type == 1:
                active.append(note)
            else:
                try:
                    active.remove(note)
                except ValueError:
                    pass
            idx += 1

        if active:
            best = _pick_best(active)
            if current_note is None:
                current_note = best
                current_start = time
            else:
                current_score = _score_note(
                    current_note[0], current_note[2] - current_note[1], current_note[3]
                )
                best_score = _score_note(best[0], best[2] - best[1], best[3])
                duration_now = time - current_start
                # 当前旋律音持续过长，强制结束并重新选择，避免长音拖尾
                force_switch = duration_now > max_melody_note_duration
                score_better = (
                    best_score > current_score * (1 + switch_threshold)
                    and best[0] != current_note[0]
                )
                if force_switch or score_better:
                    end_time = min(time, current_start + max_melody_note_duration)
                    melody.append((current_note[0], current_start, end_time, current_note[3]))
                    current_note = best
                    current_start = end_time if force_switch else time
        else:
            if current_note is not None:
                end_time = min(time, current_start + max_melody_note_duration)
                melody.append((current_note[0], current_start, end_time, current_note[3]))
                current_note = None
                current_start = None

    return melody
