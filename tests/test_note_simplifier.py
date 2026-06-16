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
    simplified[0][0] == 60
    assert abs(simplified[0][2] - 0.8) < 0.01


def test_remove_ornament():
    notes = [
        (60, 0.0, 0.10, 80),   # C
        (62, 0.02, 0.12, 80),  # D
        (60, 0.04, 0.14, 80),  # C
        (62, 0.06, 0.16, 80),  # D
    ]
    simplified = simplify_notes(notes, strength=50)
    assert len(simplified) == 1
    assert simplified[0][0] == 60


def test_does_not_destroy_chord():
    notes = [
        (60, 0.0, 0.5, 80),   # C
        (64, 0.0, 0.5, 80),   # E (same start time, different pitch)
    ]
    simplified = simplify_notes(notes, strength=50)
    assert len(simplified) == 2
    pitches = {n[0] for n in simplified}
    assert pitches == {60, 64}


def test_density_limit():
    notes = []
    for i in range(20):
        notes.append((60 + i % 2, i * 0.02, i * 0.02 + 0.1, 80))
    simplified = simplify_notes(notes, strength=50)
    # 20 notes within 0.5s, max_notes_per_beat at strength=50 is 5
    assert len(simplified) <= 5


def test_empty_input():
    assert simplify_notes([]) == []


def test_unsorted_input():
    notes = [
        (60, 0.2, 0.5, 80),
        (62, 0.0, 0.3, 80),
    ]
    simplified = simplify_notes(notes, strength=50)
    assert len(simplified) == 2
    assert simplified[0][1] <= simplified[1][1]
