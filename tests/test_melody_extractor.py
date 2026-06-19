from src.melody_extractor import extract_melody


def test_extract_melody_keeps_highest_longest():
    # 同一时间窗内有两个音符，应保留音高更高的主旋律音
    notes = [
        (48, 0.0, 0.6, 80),  # 低音
        (72, 0.0, 0.6, 90),  # 高音，应被选中
    ]
    melody = extract_melody(notes, window_ms=50)
    assert len(melody) == 1
    assert melody[0][0] == 72


def test_extract_melody_removes_overlapping():
    # 两个时间窗不重叠，应保留两个音符
    notes = [
        (60, 0.0, 0.3, 80),
        (64, 0.5, 0.8, 80),
    ]
    melody = extract_melody(notes, window_ms=50)
    assert len(melody) == 2


def test_extract_melody_empty_returns_empty():
    assert extract_melody([]) == []


def test_extract_melody_single_note():
    notes = [(60, 0.0, 0.5, 80)]
    melody = extract_melody(notes)
    assert melody == [(60, 0.0, 0.5, 80)]


def test_extract_melody_filters_invalid_duration():
    notes = [
        (60, 0.0, 0.5, 80),
        (64, 0.5, 0.5, 80),   # duration == 0 (offset == onset)
        (67, 0.5, 0.4, 80),   # duration < 0 (offset < onset)
        (72, 1.0, 1.5, 80),   # valid note in second window
        (74, 1.5, 2.0, -10),  # velocity < 0
    ]
    melody = extract_melody(notes, window_ms=50)
    assert len(melody) == 2
    assert melody[0] == (60, 0.0, 0.5, 80)
    assert melody[1] == (72, 1.0, 1.5, 80)  # only valid note in second window


def test_extract_melody_preserves_full_tuple():
    notes = [
        (60, 0.0, 0.4, 80),
        (72, 0.0, 0.4, 90),  # 同时结束的高音，应被完整保留
    ]
    melody = extract_melody(notes, window_ms=50)
    assert len(melody) == 1
    assert melody[0] == (72, 0.0, 0.4, 90)
