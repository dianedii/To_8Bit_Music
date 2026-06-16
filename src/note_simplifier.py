from typing import List, Tuple


def simplify_notes(
    notes: List[Tuple[int, float, float, int]],
    strength: int = 50,
) -> List[Tuple[int, float, float, int]]:
    """
    简化音符序列：过滤碎音、合并同音、移除颤音/装饰音、限制密度。
    strength: 0 ~ 100
    """
    if not notes:
        return []

    # 根据强度计算阈值
    min_duration = 0.02 + (1 - strength / 100.0) * 0.08  # 50 -> 0.06s
    merge_gap = 0.01 + (1 - strength / 100.0) * 0.04     # 50 -> 0.03s
    ornament_window = 0.05 + (1 - strength / 100.0) * 0.1  # 50 -> 0.10s
    max_notes_per_beat = max(2, int(8 - (strength / 100.0) * 6))  # 50 -> 5

    notes = sorted(notes, key=lambda n: n[1])

    # 1. 过滤碎音
    filtered = [n for n in notes if (n[2] - n[1]) >= min_duration]

    # 2. 合并相邻同音
    merged = []
    for note in filtered:
        if merged and note[0] == merged[-1][0] and (note[1] - merged[-1][2]) <= merge_gap:
            prev = merged[-1]
            merged[-1] = (prev[0], prev[1], max(prev[2], note[2]), max(prev[3], note[3]))
        else:
            merged.append(note)

    # 3. 移除颤音/装饰音：极短时间内多次跳音，只保留最长的一个
    cleaned = []
    i = 0
    while i < len(merged):
        window_notes = [merged[i]]
        j = i + 1
        while j < len(merged) and (merged[j][1] - window_notes[-1][1]) <= ornament_window:
            window_notes.append(merged[j])
            j += 1
        if len(window_notes) > 1:
            # 保留时长最长的音符
            best = max(window_notes, key=lambda n: n[2] - n[1])
            cleaned.append(best)
            i = j
        else:
            cleaned.append(merged[i])
            i += 1

    # 4. 密度限制：每拍（0.5s）最多保留 N 个音符
    final = []
    beat_start = cleaned[0][1] if cleaned else 0
    beat_notes = []
    for note in cleaned:
        if note[1] >= beat_start + 0.5:
            beat_notes.sort(key=lambda n: (n[2] - n[1], n[3]), reverse=True)
            final.extend(beat_notes[:max_notes_per_beat])
            beat_start = note[1]
            beat_notes = [note]
        else:
            beat_notes.append(note)
    if beat_notes:
        beat_notes.sort(key=lambda n: (n[2] - n[1], n[3]), reverse=True)
        final.extend(beat_notes[:max_notes_per_beat])

    return sorted(final, key=lambda n: n[1])
