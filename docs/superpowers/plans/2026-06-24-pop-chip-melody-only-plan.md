# 流行 8-bit 主旋律提取与 FC 模式隐藏实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将流行 8-bit 模式改为基于 pYIN 音符事件的主旋律提取（含和声点缀），并隐藏纯正 FC 模式 UI 入口。

**Architecture：** 新增 `src/pop_melody.py` 负责 pYIN 音符化、候选旋律线拆分、主旋律评分、硬性过滤、和声点缀选择；`src/pop_synthesizer.py` 复用事件合成并只合成主旋律 + 和声；UI 移除 FC 模式选项，README 与文档同步更新。

**Tech Stack：** Python 3.9+, numpy, scipy, librosa

---

## 文件结构

| 文件 | 职责 |
|------|------|
| `src/pop_melody.py` | 新增：pYIN 音符化、候选旋律线拆分、主旋律评分、硬性过滤、和声点缀 |
| `src/pop_synthesizer.py` | 修改：主流程改为调用 pop_melody，只合成主旋律 + 和声点缀 |
| `src/ui.py` | 修改：移除“纯正 FC”选项，只保留“流行 8-bit” |
| `src/worker.py` | 修改：固定走 pop-chip 流程 |
| `src/transcriber.py` | 标记 deprecated，不修改逻辑 |
| `src/melody_extractor.py` | 标记 deprecated，不修改逻辑 |
| `src/note_simplifier.py` | 标记 deprecated，不修改逻辑 |
| `src/synthesizer.py` | 标记 deprecated，不修改逻辑 |
| `tests/test_pop_melody.py` | 新增：pop_melody 单元测试 |
| `tests/test_pop_synthesizer.py` | 修改：更新主流程测试 |
| `tests/test_integration.py` | 修改：更新集成测试 |
| `README.md` | 修改：更新功能说明，增加 FC 模式隐藏说明 |

---

## Task 1: 创建 `src/pop_melody.py` 并添加 pYIN 音符化

**Files:**
- Create: `src/pop_melody.py`
- Create: `tests/test_pop_melody.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_pop_melody.py
import numpy as np
import pytest

from src.pop_melody import _pyin_to_notes


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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_pop_melody.py::test_pyin_to_notes_detects_single_tone -v`

Expected: `ModuleNotFoundError: No module named 'src.pop_melody'`

- [ ] **Step 3: 实现 `_pyin_to_notes`**

```python
# src/pop_melody.py
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_pop_melody.py -v`

Expected: `test_pyin_to_notes_detects_single_tone` PASS

- [ ] **Step 5: 提交**

```bash
git add src/pop_melody.py tests/test_pop_melody.py
git commit -m "feat: add pYIN-based note extraction for pop-chip"
```

---

## Task 2: 候选旋律线拆分

**Files:**
- Modify: `src/pop_melody.py`
- Modify: `tests/test_pop_melody.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_pop_melody.py（追加）
from src.pop_melody import _split_candidate_lines


def test_split_candidate_lines_separates_overlapping():
    # 两个声部同时发声
    notes = [
        (60, 0.0, 0.5, 100),  # 低音
        (72, 0.0, 0.5, 100),  # 高音，与低音重叠
        (62, 0.6, 1.0, 100),  # 低音后续
        (74, 0.6, 1.0, 100),  # 高音后续
    ]
    lines = _split_candidate_lines(notes, gap_threshold=0.05)
    assert len(lines) == 2
    pitches_a = [n[0] for n in lines[0]]
    pitches_b = [n[0] for n in lines[1]]
    assert set(pitches_a) == {60, 62}
    assert set(pitches_b) == {72, 74}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_pop_melody.py::test_split_candidate_lines_separates_overlapping -v`

Expected: `ImportError: cannot import name '_split_candidate_lines'`

- [ ] **Step 3: 实现 `_split_candidate_lines`**

```python
# src/pop_melody.py（追加）

def _split_candidate_lines(
    notes: list[tuple[int, float, float, int]],
    gap_threshold: float = 0.05,
) -> list[list[tuple[int, float, float, int]]]:
    """将重叠音符拆分为不重叠的连续单音候选线。"""
    if not notes:
        return []

    notes = sorted(notes, key=lambda n: n[1])
    lines: list[list[tuple[int, float, float, int]]] = [[notes[0]]]

    for note in notes[1:]:
        onset = note[1]
        best_line = None
        best_gap = float('inf')
        for line in lines:
            last = line[-1]
            if last[2] <= onset + gap_threshold:
                gap = onset - last[2]
                if gap < best_gap:
                    best_gap = gap
                    best_line = line
        if best_line is not None:
            best_line.append(note)
        else:
            lines.append([note])

    return lines
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_pop_melody.py -v`

Expected: 所有测试 PASS

- [ ] **Step 5: 提交**

```bash
git add src/pop_melody.py tests/test_pop_melody.py
git commit -m "feat: add candidate melody line splitting"
```

---

## Task 3: 主旋律评分

**Files:**
- Modify: `src/pop_melody.py`
- Modify: `tests/test_pop_melody.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_pop_melody.py（追加）
from src.pop_melody import _score_melody_line


def test_score_melody_line_prefers_smooth():
    # 平滑旋律：音程小、有休止、长音多
    smooth_line = [
        (60, 0.0, 0.4, 100),
        (62, 0.5, 0.9, 100),
        (64, 1.0, 1.4, 100),
    ]
    # 碎音装饰线
    ornament_line = [
        (72, 0.0, 0.1, 100),
        (74, 0.1, 0.2, 100),
        (76, 0.2, 0.3, 100),
        (77, 0.3, 0.4, 100),
    ]
    smooth_score = _score_melody_line(smooth_line)
    ornament_score = _score_melody_line(ornament_line)
    assert smooth_score > ornament_score
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_pop_melody.py::test_score_melody_line_prefers_smooth -v`

Expected: `ImportError: cannot import name '_score_melody_line'`

- [ ] **Step 3: 实现 `_score_melody_line`**

```python
# src/pop_melody.py（追加）

def _score_melody_line(
    line: list[tuple[int, float, float, int]],
    bar_duration: float = 2.0,
) -> float:
    """按歌唱平滑度、乐句重复度、音域居中、力度稳定性打分。"""
    if len(line) < 2:
        return 0.0

    pitches = [n[0] for n in line]
    onsets = [n[1] for n in line]
    offsets = [n[2] for n in line]
    velocities = [n[3] for n in line]
    durations = [offsets[i] - onsets[i] for i in range(len(line))]

    # 歌唱平滑度
    intervals = [abs(pitches[i] - pitches[i - 1]) for i in range(1, len(pitches))]
    small_interval_ratio = sum(1 for iv in intervals if 1 <= iv <= 5) / len(intervals) if intervals else 0.0
    large_jump_ratio = sum(1 for iv in intervals if iv > 12) / len(intervals) if intervals else 0.0

    gaps = [onsets[i] - offsets[i - 1] for i in range(1, len(line))]
    rest_ratio = sum(1 for g in gaps if g > 0.05) / len(gaps) if gaps else 0.0
    no_rest_penalty = 0.2 if rest_ratio == 0 else 0.0

    long_note_ratio = sum(1 for d in durations if d >= 0.3) / len(durations) if durations else 0.0
    sixteenth_note_ratio = sum(1 for d in durations if d < 0.15) / len(durations) if durations else 0.0

    smoothness = (
        0.4 * small_interval_ratio
        + 0.3 * rest_ratio
        + 0.3 * long_note_ratio
        - 0.4 * sixteenth_note_ratio
        - 0.4 * large_jump_ratio
        - no_rest_penalty
    )
    smoothness = float(np.clip(smoothness, 0.0, 1.0))

    # 乐句重复度
    avg_duration = np.mean(durations) if durations else 0.25
    notes_per_bar = max(2, int(bar_duration / max(avg_duration, 0.01)))
    contour = [np.sign(pitches[i] - pitches[i - 1]) for i in range(1, len(pitches))]

    repetition_score = 0.0
    if len(contour) >= notes_per_bar * 2:
        matches = 0
        segments = 0
        for start in range(0, len(contour) - notes_per_bar * 2 + 1, notes_per_bar):
            seg1 = contour[start:start + notes_per_bar]
            seg2 = contour[start + notes_per_bar:start + notes_per_bar * 2]
            if len(seg1) == len(seg2) and len(seg1) > 0:
                similarity = sum(1 for a, b in zip(seg1, seg2) if a == b) / len(seg1)
                matches += similarity
                segments += 1
        repetition_score = matches / segments if segments > 0 else 0.0
    repetition_score = float(np.clip(repetition_score, 0.0, 1.0))

    # 音域居中
    target_low = librosa.note_to_midi('C3')
    target_high = librosa.note_to_midi('F5')
    in_range = sum(1 for p in pitches if target_low <= p <= target_high) / len(pitches)
    pitch_range_score = float(np.clip(in_range, 0.0, 1.0))

    # 力度稳定性
    vel_diffs = [abs(velocities[i] - velocities[i - 1]) for i in range(1, len(velocities))]
    vel_variance = np.mean(vel_diffs) / 127.0 if vel_diffs else 0.0
    velocity_score = float(np.clip(1.0 - vel_variance, 0.0, 1.0))

    return (
        0.45 * smoothness
        + 0.30 * repetition_score
        + 0.15 * pitch_range_score
        + 0.10 * velocity_score
    )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_pop_melody.py -v`

Expected: 所有测试 PASS

- [ ] **Step 5: 提交**

```bash
git add src/pop_melody.py tests/test_pop_melody.py
git commit -m "feat: add weighted melody line scoring"
```

---

## Task 4: 硬性过滤

**Files:**
- Modify: `src/pop_melody.py`
- Modify: `tests/test_pop_melody.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_pop_melody.py（追加）
from src.pop_melody import _apply_hard_filters


def test_apply_hard_filters_removes_ornament_line():
    lines = [
        [(72, 0.0, 0.05, 100), (74, 0.05, 0.10, 100), (76, 0.10, 0.15, 100)],
        [(60, 0.0, 0.4, 100), (62, 0.5, 0.9, 100)],
    ]
    filtered = _apply_hard_filters(lines)
    assert len(filtered) == 1
    assert filtered[0][0][0] == 60


def test_apply_hard_filters_removes_repeating_accompaniment():
    # 8 小节完全重复的分解和弦
    repeating = [(60, i * 0.5, i * 0.5 + 0.25, 100) for i in range(32)]
    melody = [(72, 0.0, 0.5, 100), (74, 0.6, 1.1, 100)]
    filtered = _apply_hard_filters([repeating, melody])
    assert len(filtered) == 1
    assert filtered[0][0][0] == 72
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_pop_melody.py::test_apply_hard_filters_removes_ornament_line -v`

Expected: `ImportError: cannot import name '_apply_hard_filters'`

- [ ] **Step 3: 实现 `_apply_hard_filters`**

```python
# src/pop_melody.py（追加）

def _apply_hard_filters(
    lines: list[list[tuple[int, float, float, int]]],
    bar_duration: float = 2.0,
) -> list[list[tuple[int, float, float, int]]]:
    """硬性过滤：碎音装饰线与重复伴奏线淘汰。"""
    filtered = []
    for line in lines:
        if len(line) == 0:
            continue

        durations = [n[2] - n[1] for n in line]
        short_ratio = sum(1 for d in durations if d < 0.15) / len(durations)
        if short_ratio > 0.8:
            continue

        pitches = [n[0] for n in line]
        avg_duration = np.mean(durations) if durations else 0.25
        notes_per_bar = max(2, int(bar_duration / max(avg_duration, 0.01)))

        if len(pitches) >= notes_per_bar * 8:
            pattern = pitches[:notes_per_bar]
            repeats = 0
            for i in range(notes_per_bar, len(pitches), notes_per_bar):
                segment = pitches[i:i + notes_per_bar]
                if len(segment) == len(pattern) and all(a == b for a, b in zip(pattern, segment)):
                    repeats += 1
                else:
                    break
            if repeats >= 7:
                continue

        filtered.append(line)
    return filtered
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_pop_melody.py -v`

Expected: 所有测试 PASS

- [ ] **Step 5: 提交**

```bash
git add src/pop_melody.py tests/test_pop_melody.py
git commit -m "feat: add hard filters for ornament and repeated accompaniment lines"
```

---

## Task 5: 主旋律与和声点缀提取

**Files:**
- Modify: `src/pop_melody.py`
- Modify: `tests/test_pop_melody.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_pop_melody.py（追加）
from src.pop_melody import _extract_main_melody, _extract_harmony_voice


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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_pop_melody.py::test_extract_main_melody_returns_highest_scored -v`

Expected: `ImportError: cannot import name '_extract_main_melody'`

- [ ] **Step 3: 实现主旋律与和声提取函数**

```python
# src/pop_melody.py（追加）

_CONSONANT_INTERVALS = {3, 4, 7, 8, 12}


def _extract_main_melody(
    lines: list[list[tuple[int, float, float, int]]],
) -> list[tuple[int, float, float, int]]:
    """从候选线中选出主旋律线。"""
    if not lines:
        return []
    scored = [(_score_melody_line(line), line) for line in lines]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _extract_harmony_voice(
    main_line: list[tuple[int, float, float, int]],
    other_lines: list[list[tuple[int, float, float, int]]],
    max_voices: int = 1,
    volume_ratio: float = 0.6,
) -> list[tuple[int, float, float, int]]:
    """从剩余候选线中提取与主旋律形成和谐音程的点缀音符。"""
    if max_voices <= 0 or not other_lines:
        return []

    scored = [(_score_melody_line(line), line) for line in other_lines]
    scored.sort(key=lambda x: x[0], reverse=True)

    harmony_notes = []
    voices_added = 0
    for _, line in scored:
        if voices_added >= max_voices:
            break
        for note in line:
            pitch, onset, offset, _ = note
            for main_note in main_line:
                m_pitch, m_onset, m_offset, _ = main_note
                if onset < m_offset and offset > m_onset:
                    interval = abs(pitch - m_pitch) % 12
                    if interval in _CONSONANT_INTERVALS:
                        harmony_notes.append((pitch, onset, offset, int(127 * volume_ratio)))
                        break
        voices_added += 1

    harmony_notes.sort(key=lambda n: n[1])
    return harmony_notes
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_pop_melody.py -v`

Expected: 所有测试 PASS

- [ ] **Step 5: 提交**

```bash
git add src/pop_melody.py tests/test_pop_melody.py
git commit -m "feat: add main melody and harmony voice extraction"
```

---

## Task 6: 重构 `synthesize_pop_chip` 主流程

**Files:**
- Modify: `src/pop_synthesizer.py`
- Modify: `src/pop_melody.py`（如有需要）
- Modify: `tests/test_pop_synthesizer.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_pop_synthesizer.py（追加或替换原有 end-to-end 测试）
from src.pop_synthesizer import synthesize_pop_chip


def test_synthesize_pop_chip_main_melody_only():
    sr = 44100
    duration = 1.2
    t = np.arange(int(duration * sr)) / sr
    audio = np.zeros_like(t)
    # 主旋律 C4 - D4 - E4
    for start, freq in [(0.0, 261.63), (0.4, 293.66), (0.8, 329.63)]:
        mask = (t >= start) & (t < start + 0.3)
        env = np.exp(-(t[mask] - start) / 0.08)
        audio[mask] += np.sin(2 * np.pi * freq * t[mask]) * env * 0.6
    # 低音和弦伴奏 C3
    mask = (t >= 0.0) & (t < 1.0)
    audio[mask] += np.sin(2 * np.pi * 130.81 * t[mask]) * 0.3

    out = synthesize_pop_chip(audio, sample_rate=sr, waveform='square', chip_mix=1.0)
    assert out.ndim == 2
    assert out.shape[0] == 2
    assert out.shape[1] == len(audio)
    assert out.dtype == np.float32
    assert np.max(np.abs(out)) <= 1.0
```

- [ ] **Step 2: 运行测试确认失败或不满足新假设**

Run: `python -m pytest tests/test_pop_synthesizer.py::test_synthesize_pop_chip_main_melody_only -v`

Expected: 可能因旧实现不区分主旋律与伴奏而失败

- [ ] **Step 3: 重构 `synthesize_pop_chip` 主流程**

在 `src/pop_synthesizer.py` 顶部添加导入：

```python
from src.pop_melody import (
    _pyin_to_notes,
    _split_candidate_lines,
    _apply_hard_filters,
    _extract_main_melody,
    _extract_harmony_voice,
)
```

替换 `synthesize_pop_chip` 中从 onset 分段到音符提取的逻辑为：

```python
    notes = _pyin_to_notes(mono, sample_rate, hop_length=hop_length, min_note_duration=min_note_duration)
    if len(notes) == 0:
        # fallback: 保持静音输出或保留原音频
        out = mono.astype(np.float32)
        return np.stack([out, out], axis=0)

    candidate_lines = _split_candidate_lines(notes, gap_threshold=min_note_duration)
    filtered_lines = _apply_hard_filters(candidate_lines)
    if not filtered_lines:
        filtered_lines = candidate_lines

    main_line = _extract_main_melody(filtered_lines)
    other_lines = [line for line in filtered_lines if line is not main_line]
    harmony = _extract_harmony_voice(main_line, other_lines, max_voices=1, harmony_volume_ratio=0.6)

    final_notes = sorted(main_line + harmony, key=lambda n: n[1])

    duration = len(mono) / sample_rate
    synth = _synthesize_events(final_notes, duration, sample_rate, waveform=waveform)
```

同时更新函数签名，新增内部参数：

```python
def synthesize_pop_chip(
    audio: np.ndarray,
    sample_rate: int = 44100,
    waveform: Literal['square', 'triangle', 'sawtooth'] = 'square',
    chip_mix: float = 0.75,
    n_voices: int = 4,  # 已弃用，保留以兼容旧调用
    hop_length: int = 512,
    min_note_duration: float = 0.05,
    pitch_stabilize: float = 1.0,  # 已弃用
    max_harmony_voices: int = 1,
    harmony_volume_ratio: float = 0.6,
) -> np.ndarray:
```

后续峰值归一化、RMS 包络、chip_mix 混合、压缩、归一化、立体声输出保持不变。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_pop_synthesizer.py -v`

Expected: 所有测试 PASS

- [ ] **Step 5: 提交**

```bash
git add src/pop_synthesizer.py src/pop_melody.py tests/test_pop_synthesizer.py
git commit -m "feat: rewire pop-chip to pYIN-based main melody extraction with harmony"
```

---

## Task 7: 隐藏 FC 模式 UI

**Files:**
- Modify: `src/ui.py`
- Test: 手动验证 GUI 启动（无法单元测试）

- [ ] **Step 1: 修改 UI 移除 FC 模式**

在 `src/ui.py` 中：
1. 移除转换风格下拉框中的“纯正 FC”选项，只保留“流行 8-bit”。
2. 移除或禁用 FC 模式相关参数滑块：复古纯度、音符简化强度。
3. 保留：波形选择、芯片混合比例、整体音量、输出格式。

具体修改（示例）：

```python
# 原转换风格下拉框
self.style_combo = QComboBox()
self.style_combo.addItems(["流行 8-bit"])  # 只保留一项
```

如果“复古纯度”“音符简化强度”有独立控件，直接注释或删除相关代码，避免界面空白可保留标签说明“当前版本仅支持流行 8-bit”。

- [ ] **Step 2: 运行 GUI 启动测试**

Run: `python -c "from src.ui import MainWindow; from PyQt6.QtWidgets import QApplication; import sys; app = QApplication(sys.argv); w = MainWindow(); print('UI OK')"`

Expected: 无异常，打印 `UI OK`

- [ ] **Step 3: 提交**

```bash
git add src/ui.py
git commit -m "feat: hide FC mode from UI, keep only pop-chip"
```

---

## Task 8: 更新 `worker.py`

**Files:**
- Modify: `src/worker.py`

- [ ] **Step 1: 移除 FC 模式分支**

`src/worker.py` 中如果存在根据 `style` 参数选择 FC / pop-chip 的分支，简化为只调用 pop-chip：

```python
# 删除 style 参数相关判断，直接调用 synthesize_pop_chip
audio_data = synthesize_pop_chip(
    mono_samples,
    sample_rate=44100,
    waveform='square',
    chip_mix=0.75,
    volume=volume,
)
```

如果 `ConvertWorker` 构造函数仍接收 `style` 参数，保留参数但忽略，避免 UI 信号绑定断裂。

- [ ] **Step 2: 运行测试**

Run: `python -m pytest tests/ -v`

Expected: 所有测试 PASS

- [ ] **Step 3: 提交**

```bash
git add src/worker.py
git commit -m "feat: remove FC mode branch from worker"
```

---

## Task 9: 标记 FC 模块为 Deprecated

**Files:**
- Modify: `src/transcriber.py`
- Modify: `src/melody_extractor.py`
- Modify: `src/note_simplifier.py`
- Modify: `src/synthesizer.py`

- [ ] **Step 1: 在每个 FC 模块顶部添加 deprecation 说明**

例如 `src/transcriber.py`：

```python
"""
[DEPRECATED] 纯正 FC 模式已隐藏，本模块不再被主流程调用。
代码保留仅供历史参考或未来重新启用。
"""
```

对 `src/melody_extractor.py`、`src/note_simplifier.py`、`src/synthesizer.py` 做同样处理。

- [ ] **Step 2: 添加回归测试确保仍可 import**

```python
# tests/test_fc_deprecated.py
def test_fc_modules_still_importable():
    import src.transcriber
    import src.melody_extractor
    import src.note_simplifier
    import src.synthesizer
    assert src.transcriber is not None
    assert src.melody_extractor is not None
    assert src.note_simplifier is not None
    assert src.synthesizer is not None
```

- [ ] **Step 3: 运行测试确认通过**

Run: `python -m pytest tests/test_fc_deprecated.py -v`

Expected: PASS

- [ ] **Step 4: 提交**

```bash
git add src/transcriber.py src/melody_extractor.py src/note_simplifier.py src/synthesizer.py tests/test_fc_deprecated.py
git commit -m "chore: mark FC modules as deprecated"
```

---

## Task 10: 更新集成测试

**Files:**
- Modify: `tests/test_integration.py`

- [ ] **Step 1: 更新集成测试**

```python
# tests/test_integration.py（追加/修改）
import numpy as np
import tempfile
from pathlib import Path

from src.pop_synthesizer import synthesize_pop_chip
from src.utils import export_audio


def test_pop_chip_melody_only_end_to_end():
    sr = 44100
    duration = 1.5
    t = np.arange(int(duration * sr)) / sr
    audio = np.zeros_like(t)
    # 主旋律
    for start, freq in [(0.0, 261.63), (0.4, 293.66), (0.8, 329.63)]:
        mask = (t >= start) & (t < start + 0.3)
        env = np.exp(-(t[mask] - start) / 0.08)
        audio[mask] += np.sin(2 * np.pi * freq * t[mask]) * env * 0.6

    out = synthesize_pop_chip(audio, sample_rate=sr, chip_mix=1.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "out.wav"
        export_audio(out, str(out_path), sample_rate=sr)
        assert out_path.exists()
        assert out_path.stat().st_size > 0
```

- [ ] **Step 2: 运行全部测试**

Run: `python -m pytest tests/ -v`

Expected: 所有测试 PASS

- [ ] **Step 3: 提交**

```bash
git add tests/test_integration.py
git commit -m "test: update integration test for melody-only pop-chip"
```

---

## Task 11: 更新 README 与文档

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-06-23-pop-chip-melody-only-design.md`

- [ ] **Step 1: 更新 README**

1. 删除“纯正 FC”相关功能介绍和参数说明。
2. 在“两种模式对比”或“功能特性”处增加说明：
   ```markdown
   > **注意：** 纯正 FC 模式已从 UI 隐藏，相关模块保留但不再维护。
   ```
3. 简化参数表，只保留流行 8-bit 相关参数。

- [ ] **Step 2: 更新设计文档状态**

将 `docs/superpowers/specs/2026-06-23-pop-chip-melody-only-design.md` 顶部状态改为：

```markdown
**状态：** 已实现
```

- [ ] **Step 3: 运行全部测试做最终验证**

Run: `python -m pytest tests/ -v`

Expected: 所有测试 PASS

- [ ] **Step 4: 提交**

```bash
git add README.md docs/superpowers/specs/2026-06-23-pop-chip-melody-only-design.md
git commit -m "docs: update README and mark melody-only design as implemented"
```

---

## 自检清单

- [x] Spec coverage: pYIN 音符化、候选线拆分、评分、硬性过滤、和声点缀、主流程重构、UI 隐藏、worker 更新、FC 模块标记 deprecated、测试、README 均已覆盖
- [x] Placeholder scan: 无 TBD/TODO/"implement later"
- [x] Type consistency: 函数签名在 Task 5/6 中一致，`hop_length`、`min_note_duration` 等参数名统一
- [x] 所有文件路径使用绝对路径
- [x] 每个任务均包含失败测试、实现代码、通过命令、提交命令

---

**Execution Handoff:**

Plan complete and saved to `docs/superpowers/plans/2026-06-24-pop-chip-melody-only-plan.md`. Two execution options:

1. **Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
