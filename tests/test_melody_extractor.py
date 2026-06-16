from src.melody_extractor import extract_melody


def test_extract_melody_keeps_highest_longest():
    # 同一时间窗内有两个音符，应保留更长、更高的主旋律音
    notes = [
        (48, 0.0, 1.0, 80),  # 低音，较长
        (72, 0.0, 0.6, 90),  # 高音，较短但音高优势大
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
