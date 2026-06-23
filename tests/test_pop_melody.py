import numpy as np
import pytest

from src.pop_melody import _pyin_to_notes, _split_candidate_lines, _score_melody_line, _apply_hard_filters, _extract_main_melody, _extract_harmony_voice


def generate_pure_tone(freq, duration, sr):
    t = np.arange(int(duration * sr)) / sr
    return np.sin(2 * np.pi * freq * t) * 0.5


def test_pyin_to_notes_detects_single_tone():
    sr = 44100
    audio = generate_pure_tone(440.0, 0.6, sr)
    notes = _pyin_to_notes(audio, sr, hop_length=512, min_note_duration=0.05)
    assert len(notes) >= 1
    midi, onset, offset, velocity = notes[0]
    assert abs(midi - 69) <= 1  # A4
    assert onset < 0.1
    assert offset > 0.5
    assert 1 <= velocity <= 127


def test_pyin_to_notes_empty_audio():
    notes = _pyin_to_notes(np.array([], dtype=np.float64), 44100)
    assert notes == []


def test_pyin_to_notes_silence():
    sr = 44100
    audio = np.zeros(int(sr * 0.5), dtype=np.float64)
    notes = _pyin_to_notes(audio, sr)
    assert notes == []


def test_split_candidate_lines_separates_overlapping():
    notes = [
        (60, 0.0, 0.5, 100),  # 低音
        (72, 0.0, 0.5, 100),  # 高音，与低音重叠
        (62, 0.6, 1.0, 100),  # 低音后续
        (74, 0.6, 1.0, 100),  # 高音后续
    ]
    lines = _split_candidate_lines(notes)
    assert len(lines) == 2
    pitches_a = [n[0] for n in lines[0]]
    pitches_b = [n[0] for n in lines[1]]
    assert set(pitches_a) == {60, 62}
    assert set(pitches_b) == {72, 74}


def test_split_candidate_lines_empty():
    assert _split_candidate_lines([]) == []


def test_split_candidate_lines_real_overlap():
    # 真正的重叠：低音持续到 0.7，高音从 0.5 开始
    notes = [
        (60, 0.0, 0.7, 100),
        (72, 0.5, 1.0, 100),
        (62, 1.1, 1.5, 100),  # 低音后续，不重叠
    ]
    lines = _split_candidate_lines(notes)
    assert len(lines) == 2
    # 62 应该跟 72（gap 更小），而不是跟 60（虽然 60 也是低音但 gap 更大）
    # 这是贪心策略的预期行为：优先最小 gap
    line_with_60 = [line for line in lines if any(n[0] == 60 for n in line)][0]
    line_with_72 = [line for line in lines if any(n[0] == 72 for n in line)][0]
    # 60 单独一条线（因为 62 被 72 抢走了）
    assert len(line_with_60) == 1
    # 72 和 62 在同一条线（gap 更小）
    assert any(n[0] == 62 for n in line_with_72)
    assert len(line_with_72) == 2


def test_score_melody_line_prefers_smooth():
    smooth_line = [
        (60, 0.0, 0.4, 100),
        (62, 0.5, 0.9, 100),
        (64, 1.0, 1.4, 100),
    ]
    ornament_line = [
        (72, 0.0, 0.1, 100),
        (74, 0.1, 0.2, 100),
        (76, 0.2, 0.3, 100),
        (77, 0.3, 0.4, 100),
    ]
    smooth_score = _score_melody_line(smooth_line)
    ornament_score = _score_melody_line(ornament_line)
    assert smooth_score > ornament_score


def test_score_melody_line_repetition():
    # 有重复的 4 音符模式
    repeated_line = [
        (60, 0.0, 0.2, 100),
        (62, 0.2, 0.4, 100),
        (64, 0.4, 0.6, 100),
        (65, 0.6, 0.8, 100),
        (60, 0.8, 1.0, 100),
        (62, 1.0, 1.2, 100),
        (64, 1.2, 1.4, 100),
        (65, 1.4, 1.6, 100),
    ]
    non_repeated_line = [
        (60, 0.0, 0.2, 100),
        (65, 0.2, 0.4, 100),
        (70, 0.4, 0.6, 100),
        (75, 0.6, 0.8, 100),
    ]
    assert _score_melody_line(repeated_line) > _score_melody_line(non_repeated_line)


def test_score_melody_line_pitch_range():
    in_range_line = [(60, i * 0.3, i * 0.3 + 0.25, 100) for i in range(6)]  # C4 附近
    out_of_range_line = [(40, i * 0.3, i * 0.3 + 0.25, 100) for i in range(6)]  # 很低
    assert _score_melody_line(in_range_line) > _score_melody_line(out_of_range_line)


def test_score_melody_line_velocity_stability():
    stable_line = [(60, i * 0.3, i * 0.3 + 0.25, 100) for i in range(6)]
    unstable_line = [(60, i * 0.3, i * 0.3 + 0.25, 80 if i % 2 == 0 else 120) for i in range(6)]
    assert _score_melody_line(stable_line) > _score_melody_line(unstable_line)


def test_score_melody_line_short_line_returns_zero():
    assert _score_melody_line([(60, 0.0, 0.3, 100)]) == 0.0


def test_score_melody_line_large_jumps_penalized():
    smooth_line = [(60, 0.0, 0.3, 100), (62, 0.3, 0.6, 100), (64, 0.6, 0.9, 100)]
    jumpy_line = [(60, 0.0, 0.3, 100), (72, 0.3, 0.6, 100), (60, 0.6, 0.9, 100)]  # 八度跳
    assert _score_melody_line(smooth_line) > _score_melody_line(jumpy_line)


def test_apply_hard_filters_removes_ornament_line():
    lines = [
        [(72, 0.0, 0.05, 100), (74, 0.05, 0.10, 100), (76, 0.10, 0.15, 100)],
        [(60, 0.0, 0.4, 100), (62, 0.5, 0.9, 100)],
    ]
    filtered = _apply_hard_filters(lines)
    assert len(filtered) == 1
    assert filtered[0][0][0] == 60


def test_apply_hard_filters_removes_repeating_accompaniment():
    # 8 小节完全重复的分解和弦，每小节 4 个音符
    repeating = [(60 + i % 4, i * 0.125, i * 0.125 + 0.1, 100) for i in range(32)]
    melody = [(72, 0.0, 0.5, 100), (74, 0.6, 1.1, 100)]
    filtered = _apply_hard_filters([repeating, melody])
    assert len(filtered) == 1
    assert filtered[0][0][0] == 72


def test_extract_main_melody_returns_highest_scored():
    lines = [
        [(72, 0.0, 0.05, 100), (74, 0.05, 0.10, 100)],  # 碎音装饰
        [(60, 0.0, 0.4, 100), (62, 0.5, 0.9, 100)],      # 主旋律
    ]
    filtered = _apply_hard_filters(lines)
    main = _extract_main_melody(filtered)
    assert len(main) == 2
    assert main[0][0] == 60


def test_extract_harmony_voice_returns_consonant():
    main_line = [(60, 0.0, 0.5, 100)]  # C4
    other_line = [(64, 0.0, 0.5, 100)]  # E4，大三度
    harmony = _extract_harmony_voice(main_line, [other_line], max_voices=1, volume_ratio=0.6)
    assert len(harmony) == 1
    assert harmony[0][0] == 64
    assert harmony[0][3] == int(127 * 0.6)

