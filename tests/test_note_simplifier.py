from src.note_simplifier import simplify_notes


def test_filter_short_notes():
    notes = [
        (60, 0.0, 0.05, 80),   # 碎音，应被过滤
        (60, 0.1, 0.6, 80),    # 正常音符
    ]
    simplified = simplify_notes(notes, strength=50)
    assert len(simplified) == 1
    assert simplified[0] == (60, 0.1, 0.6, 80)


def test_merge_same_pitch():
    notes = [
        (60, 0.0, 0.4, 80),
        (60, 0.42, 0.8, 80),  # 间隔很小，应合并
    ]
    simplified = simplify_notes(notes, strength=50)
    assert len(simplified) == 1
    assert simplified[0][0] == 60
    assert abs(simplified[0][2] - 0.8) < 0.01
