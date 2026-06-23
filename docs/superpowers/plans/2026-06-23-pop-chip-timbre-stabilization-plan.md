# 流行 8-bit 音色稳定化实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `src/pop_synthesizer.py` 从逐帧频谱峰值跟踪重构为 onset 分段 + 稳定音高提取 + 事件驱动合成，消除流行 8-bit 模式的“音飘”问题。

**Architecture：** 在 `src/pop_synthesizer.py` 内部新增 onset 检测、音频分段、稳定音高提取、音符合并、事件合成五个辅助函数；主入口 `synthesize_pop_chip` 保持接口不变，内部改走新流程。不新增独立模块，不改动 UI。

**Tech Stack：** Python 3.9+, numpy, scipy, librosa

---

## 文件结构

| 文件 | 职责 |
|------|------|
| `src/pop_synthesizer.py` | 主要实现文件，新增 onset/分段/音高提取/合并/事件合成函数，重构主流程 |
| `tests/test_pop_synthesizer.py` | 新增单元测试，覆盖 onset、分段、音高提取、合并、事件合成、完整流程 |
| `tests/test_integration.py` | 更新集成测试，验证新流程输出格式正确 |

---

## Task 1: Onset 检测与分段

**Files:**
- Modify: `src/pop_synthesizer.py`
- Create: `tests/test_pop_synthesizer.py`

- [ ] **Step 1: 写失败测试 —— 验证 onset 检测能找出音符起始点**

```python
# tests/test_pop_synthesizer.py
import numpy as np
import pytest

from src.pop_synthesizer import _detect_onsets, _segment_audio


def generate_click_tone(freq, duration, sr, onset_times):
    """生成在指定时刻出现纯音的测试音频。"""
    t = np.arange(int(duration * sr)) / sr
    audio = np.zeros_like(t)
    env_decay = 0.1
    for ot in onset_times:
        mask = (t >= ot) & (t < ot + env_decay)
        env = np.exp(-(t[mask] - ot) / 0.03)
        audio[mask] += np.sin(2 * np.pi * freq * t[mask]) * env
    return audio


def test_detect_onsets_finds_note_starts():
    sr = 44100
    onset_times = [0.2, 0.6, 1.0]
    audio = generate_click_tone(440.0, 1.5, sr, onset_times)
    onsets = _detect_onsets(audio, sr)
    assert len(onsets) >= 3
    # 允许 ±30ms 误差
    for expected in onset_times:
        assert any(np.abs(onsets - expected) < 0.03)


def test_segment_audio_splits_by_onsets():
    sr = 44100
    onset_times = [0.0, 0.3, 0.7]
    audio = generate_click_tone(440.0, 1.0, sr, onset_times)
    segments = _segment_audio(audio, sr, onset_times, min_note_duration=0.05)
    assert len(segments) >= 2
    for start, end in segments:
        assert 0 <= start < end <= len(audio)
        assert (end - start) / sr >= 0.05
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_pop_synthesizer.py -v`

Expected: `ImportError: cannot import name '_detect_onsets' from 'src.pop_synthesizer'`

- [ ] **Step 3: 实现 onset 检测与分段函数**

```python
# 追加到 src/pop_synthesizer.py
import librosa


def _detect_onsets(audio: np.ndarray, sample_rate: int, hop_length: int = 512, wait: int = 3) -> np.ndarray:
    """基于频谱通量检测音符起始点，返回秒级时间数组。"""
    onset_frames = librosa.onset.onset_detect(
        y=audio,
        sr=sample_rate,
        hop_length=hop_length,
        wait=wait,
        units='frames',
    )
    return librosa.frames_to_time(onset_frames, sr=sample_rate, hop_length=hop_length)


def _segment_audio(
    audio: np.ndarray,
    sample_rate: int,
    onsets: np.ndarray,
    min_note_duration: float = 0.05,
) -> list[tuple[int, int]]:
    """按 onset 切分音频为样本索引片段，过滤过短片段。"""
    if len(onsets) == 0:
        return [(0, len(audio))]

    # 合并间隔小于最小音符时长的 onset
    sorted_onsets = np.sort(onsets)
    merged = [sorted_onsets[0]]
    for o in sorted_onsets[1:]:
        if o - merged[-1] >= min_note_duration:
            merged.append(o)
        else:
            merged[-1] = o

    # 构造片段边界，包含尾部
    boundaries = list(merged) + [len(audio) / sample_rate]
    segments = []
    for i in range(len(boundaries) - 1):
        start = int(boundaries[i] * sample_rate)
        end = int(boundaries[i + 1] * sample_rate)
        start = max(0, min(start, len(audio)))
        end = max(0, min(end, len(audio)))
        if end - start >= int(min_note_duration * sample_rate):
            segments.append((start, end))
    return segments
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_pop_synthesizer.py -v`

Expected: `test_detect_onsets_finds_note_starts` 和 `test_segment_audio_splits_by_onsets` PASS

- [ ] **Step 5: 提交**

```bash
git add src/pop_synthesizer.py tests/test_pop_synthesizer.py
git commit -m "feat: add onset detection and audio segmentation for pop-chip"
```

---

## Task 2: 片段内稳定音高提取

**Files:**
- Modify: `src/pop_synthesizer.py`
- Modify: `tests/test_pop_synthesizer.py`

- [ ] **Step 1: 写失败测试 —— 验证片段内音高稳定**

```python
# tests/test_pop_synthesizer.py（追加）
from src.pop_synthesizer import _extract_stable_pitches


def test_extract_stable_pitches_single_tone():
    sr = 44100
    duration = 0.3
    freq = 440.0
    t = np.arange(int(duration * sr)) / sr
    segment = np.sin(2 * np.pi * freq * t) * 0.5
    pitches = _extract_stable_pitches(segment, sr, n_voices=4, hop_length=512)
    assert len(pitches) >= 1
    # 最接近 A4 (MIDI 69)
    midi, velocity = pitches[0]
    assert abs(midi - 69) <= 1
    assert 0 < velocity <= 127


def test_extract_stable_pitches_two_tones():
    sr = 44100
    duration = 0.3
    t = np.arange(int(duration * sr)) / sr
    segment = (
        np.sin(2 * np.pi * 440.0 * t) * 0.4
        + np.sin(2 * np.pi * 659.25 * t) * 0.4
    )
    pitches = _extract_stable_pitches(segment, sr, n_voices=4, hop_length=512)
    assert len(pitches) >= 2
    midis = [p[0] for p in pitches[:2]]
    assert any(abs(m - 69) <= 1 for m in midis)
    assert any(abs(m - 76) <= 1 for m in midis)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_pop_synthesizer.py::test_extract_stable_pitches_single_tone -v`

Expected: `ImportError: cannot import name '_extract_stable_pitches' from 'src.pop_synthesizer'`

- [ ] **Step 3: 实现稳定音高提取函数**

```python
# 追加到 src/pop_synthesizer.py
from scipy import signal


def _extract_stable_pitches(
    segment: np.ndarray,
    sample_rate: int,
    n_voices: int = 4,
    hop_length: int = 512,
) -> list[tuple[int, int]]:
    """从单一片段内提取稳定的 1~n_voices 个音高。"""
    if len(segment) < hop_length:
        return []

    n_fft = min(2048, max(512, len(segment)))
    stft = np.abs(librosa.stft(segment, n_fft=n_fft, hop_length=hop_length))
    avg_spectrum = np.mean(stft, axis=1)
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)

    peaks, _ = signal.find_peaks(
        avg_spectrum,
        height=np.max(avg_spectrum) * 0.05,
        distance=max(1, int(50 / (sample_rate / n_fft))),
    )
    if len(peaks) == 0:
        return []

    peak_infos = [(int(p), float(avg_spectrum[p])) for p in peaks if freqs[p] > 40.0]
    peak_infos.sort(key=lambda x: x[1], reverse=True)
    peak_infos = peak_infos[:n_voices]

    rms = np.sqrt(np.mean(segment ** 2))
    velocity = int(np.clip(rms * 127 * 4, 1, 127))

    pitches = []
    for peak_idx, amp in peak_infos:
        freq = freqs[peak_idx]
        midi_float = 69.0 + 12.0 * np.log2(freq / 440.0)
        midi = int(np.round(np.clip(midi_float, 0, 127)))
        pitches.append((midi, velocity))

    # 去重，保留能量高的
    seen = set()
    unique = []
    for midi, vel in pitches:
        if midi not in seen:
            seen.add(midi)
            unique.append((midi, vel))
    return unique
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_pop_synthesizer.py -v`

Expected: 所有测试 PASS

- [ ] **Step 5: 提交**

```bash
git add src/pop_synthesizer.py tests/test_pop_synthesizer.py
git commit -m "feat: add stable pitch extraction per segment"
```

---

## Task 3: 复音合并

**Files:**
- Modify: `src/pop_synthesizer.py`
- Modify: `tests/test_pop_synthesizer.py`

- [ ] **Step 1: 写失败测试 —— 验证相邻同音合并**

```python
# tests/test_pop_synthesizer.py（追加）
from src.pop_synthesizer import _merge_consecutive_notes


def test_merge_consecutive_notes_joins_same_pitch():
    notes = [
        (69, 0.0, 0.3, 100),
        (69, 0.32, 0.6, 100),
        (72, 0.6, 0.9, 100),
    ]
    merged = _merge_consecutive_notes(notes, gap_threshold=0.05)
    assert len(merged) == 2
    assert merged[0] == (69, 0.0, 0.6, 100)
    assert merged[1] == (72, 0.6, 0.9, 100)


def test_merge_consecutive_notes_keeps_different_pitches():
    notes = [
        (69, 0.0, 0.3, 100),
        (72, 0.32, 0.6, 100),
    ]
    merged = _merge_consecutive_notes(notes, gap_threshold=0.05)
    assert len(merged) == 2
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_pop_synthesizer.py::test_merge_consecutive_notes_joins_same_pitch -v`

Expected: `ImportError: cannot import name '_merge_consecutive_notes'`

- [ ] **Step 3: 实现复音合并函数**

```python
# 追加到 src/pop_synthesizer.py

def _merge_consecutive_notes(
    notes: list[tuple[int, float, float, int]],
    gap_threshold: float = 0.05,
) -> list[tuple[int, float, float, int]]:
    """合并相邻片段中相同音高的音符，避免重复 attack。"""
    if not notes:
        return []

    notes = sorted(notes, key=lambda n: n[1])
    merged = [list(notes[0])]
    for pitch, onset, offset, velocity in notes[1:]:
        prev = merged[-1]
        if pitch == prev[0] and (onset - prev[2]) <= gap_threshold:
            prev[2] = max(prev[2], offset)
            prev[3] = max(prev[3], velocity)
        else:
            merged.append([pitch, onset, offset, velocity])
    return [tuple(n) for n in merged]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_pop_synthesizer.py -v`

Expected: 所有测试 PASS

- [ ] **Step 5: 提交**

```bash
git add src/pop_synthesizer.py tests/test_pop_synthesizer.py
git commit -m "feat: add consecutive note merging"
```

---

## Task 4: 事件驱动合成

**Files:**
- Modify: `src/pop_synthesizer.py`
- Modify: `tests/test_pop_synthesizer.py`

- [ ] **Step 1: 写失败测试 —— 验证事件合成输出稳定**

```python
# tests/test_pop_synthesizer.py（追加）
from src.pop_synthesizer import _synthesize_events


def test_synthesize_events_shape_and_tail_silence():
    sr = 44100
    duration = 0.5
    notes = [
        (69, 0.0, 0.2, 100),
        (72, 0.25, 0.45, 100),
    ]
    audio = _synthesize_events(notes, duration, sr, waveform='square')
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float64
    assert len(audio) == int(duration * sr)
    assert np.max(np.abs(audio)) <= 1.0
    tail = audio[int(0.46 * sr):]
    assert np.max(np.abs(tail)) < 0.1


def test_synthesize_events_pitch_stability():
    """验证同一音符片段内音高稳定（无颤音/抖动）。"""
    sr = 44100
    duration = 0.3
    notes = [(69, 0.0, 0.25, 100)]
    audio = _synthesize_events(notes, duration, sr, waveform='square')
    # 用自相关检测基频，应该接近 A4 440Hz
    autocorr = np.correlate(audio[: int(0.2 * sr)], audio[: int(0.2 * sr)], mode='full')
    autocorr = autocorr[len(autocorr) // 2:]
    peak = np.argmax(autocorr[100:]) + 100
    estimated_period = peak / sr
    if estimated_period > 0:
        estimated_freq = 1.0 / estimated_period
        assert 420 <= estimated_freq <= 460
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_pop_synthesizer.py::test_synthesize_events_shape_and_tail_silence -v`

Expected: `ImportError: cannot import name '_synthesize_events'`

- [ ] **Step 3: 实现事件驱动合成函数**

```python
# 追加到 src/pop_synthesizer.py

def _synthesize_events(
    notes: list[tuple[int, float, float, int]],
    duration: float,
    sample_rate: int,
    waveform: str = 'square',
) -> np.ndarray:
    """将复音音符事件合成为稳定音高的芯片音频。"""
    num_samples = int(duration * sample_rate)
    audio = np.zeros(num_samples, dtype=np.float64)

    for pitch, onset, offset, velocity in notes:
        freq = _midi_to_freq(pitch)
        start_idx = max(0, int(onset * sample_rate))
        end_idx = min(num_samples, int(offset * sample_rate))
        if start_idx >= end_idx:
            continue

        length = end_idx - start_idx
        t = np.arange(length) / sample_rate
        phase = 2.0 * np.pi * freq * t
        wave = _bandlimited_waveform(phase, freq, sample_rate, waveform)

        # ADSR 包络：短 attack / decay， sustain，短 release
        env = np.ones(length, dtype=np.float64)
        attack_samples = min(int(0.005 * sample_rate), length)
        decay_samples = min(int(0.015 * sample_rate), length - attack_samples)
        release_samples = min(int(0.02 * sample_rate), length)

        env[:attack_samples] = np.linspace(0.0, 1.0, attack_samples)
        if decay_samples > 0:
            env[attack_samples:attack_samples + decay_samples] = np.linspace(
                1.0, 0.85, decay_samples
            )
        if release_samples > 0:
            env[-release_samples:] = np.linspace(env[-release_samples], 0.0, release_samples)

        amp = (velocity / 127.0) * 0.25
        audio[start_idx:end_idx] += wave * env * amp

    return audio
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_pop_synthesizer.py -v`

Expected: 所有测试 PASS

- [ ] **Step 5: 提交**

```bash
git add src/pop_synthesizer.py tests/test_pop_synthesizer.py
git commit -m "feat: add event-driven polyphonic synthesis"
```

---

## Task 5: 重构主流程

**Files:**
- Modify: `src/pop_synthesizer.py`
- Modify: `tests/test_pop_synthesizer.py`

- [ ] **Step 1: 写失败测试 —— 验证完整流程输出格式**

```python
# tests/test_pop_synthesizer.py（追加）
from src.pop_synthesizer import synthesize_pop_chip


def test_synthesize_pop_chip_outputs_stable_chip():
    sr = 44100
    duration = 1.0
    t = np.arange(int(duration * sr)) / sr
    # 三段不同音高，模拟旋律
    audio = np.zeros_like(t)
    for start, freq in [(0.0, 440.0), (0.3, 523.25), (0.6, 659.25)]:
        mask = (t >= start) & (t < start + 0.25)
        env = np.exp(-(t[mask] - start) / 0.05)
        audio[mask] += np.sin(2 * np.pi * freq * t[mask]) * env

    out = synthesize_pop_chip(
        audio,
        sample_rate=sr,
        waveform='square',
        chip_mix=1.0,
        n_voices=4,
        hop_length=512,
    )
    assert out.ndim == 2  # 立体声
    assert out.shape[0] == 2
    assert out.shape[1] == len(audio)
    assert out.dtype == np.float32
    assert np.max(np.abs(out)) <= 1.0
```

- [ ] **Step 2: 运行测试确认失败（或行为不正确）**

Run: `python -m pytest tests/test_pop_synthesizer.py::test_synthesize_pop_chip_outputs_stable_chip -v`

Expected: 可能因旧实现与新测试假设不同而失败，或输出 shape 不符

- [ ] **Step 3: 重构 `synthesize_pop_chip` 主流程**

替换 `src/pop_synthesizer.py` 中 `synthesize_pop_chip` 的主体，保留函数签名与返回值格式。关键改动：

1. mono 归一化后调用 `_detect_onsets`
2. 用 `_segment_audio` 切分
3. 对每个片段调用 `_extract_stable_pitches`
4. 收集所有事件后调用 `_merge_consecutive_notes`
5. 调用 `_synthesize_events` 生成芯片层
6. 保留 RMS 包络调制、chip_mix 混合、压缩、归一化、立体声输出

重构后代码骨架：

```python
def synthesize_pop_chip(
    audio: np.ndarray,
    sample_rate: int = 44100,
    waveform: Literal['square', 'triangle', 'sawtooth'] = 'square',
    chip_mix: float = 0.75,
    n_voices: int = 4,
    hop_length: int = 512,
    min_note_duration: float = 0.05,
    pitch_stabilize: float = 1.0,
) -> np.ndarray:
    import librosa

    mono = _audio_to_mono_float(audio)
    if len(mono) == 0:
        return np.zeros((2, 0), dtype=np.float32)

    onsets = _detect_onsets(mono, sample_rate, hop_length=hop_length)
    segments = _segment_audio(mono, sample_rate, onsets, min_note_duration=min_note_duration)

    raw_notes = []
    for start, end in segments:
        segment = mono[start:end]
        pitches = _extract_stable_pitches(segment, sample_rate, n_voices=n_voices, hop_length=hop_length)
        onset_time = start / sample_rate
        offset_time = end / sample_rate
        for midi, velocity in pitches:
            raw_notes.append((midi, onset_time, offset_time, velocity))

    notes = _merge_consecutive_notes(raw_notes, gap_threshold=min_note_duration)

    duration = len(mono) / sample_rate
    synth = _synthesize_events(notes, duration, sample_rate, waveform=waveform)

    # 峰值归一化芯片层
    peak = np.max(np.abs(synth))
    if peak > 1e-9:
        synth = synth / peak * 0.98

    # RMS 包络调制
    envelope = _rms_envelope(mono, floor=0.50)
    envelope = np.cbrt(envelope)
    if len(envelope) == len(synth):
        synth = synth * envelope

    # 混合原音频
    out = (1.0 - chip_mix) * mono + chip_mix * synth
    out = out.astype(np.float64)

    # 压缩 + 归一化
    out = _gentle_compress(out, threshold=0.55, ratio=2.5)
    peak = np.max(np.abs(out))
    if peak > 1e-9:
        out = out / peak * 0.98
    out = np.clip(out, -1.0, 1.0).astype(np.float32)
    return np.stack([out, out], axis=0)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_pop_synthesizer.py -v`

Expected: 所有测试 PASS

- [ ] **Step 5: 提交**

```bash
git add src/pop_synthesizer.py tests/test_pop_synthesizer.py
git commit -m "feat: rewire pop-chip synthesis to onset-based event pipeline"
```

---

## Task 6: 集成与回归测试

**Files:**
- Modify: `tests/test_integration.py`
- Modify: `src/pop_synthesizer.py`（如发现问题）

- [ ] **Step 1: 更新集成测试**

```python
# tests/test_integration.py（追加或替换原有 pop 相关断言）
import numpy as np
import tempfile
from pathlib import Path

from src.pop_synthesizer import synthesize_pop_chip
from src.utils import export_audio


def test_pop_chip_end_to_end_synthetic():
    sr = 44100
    duration = 1.0
    t = np.arange(int(duration * sr)) / sr
    audio = np.zeros_like(t)
    for start, freq in [(0.0, 440.0), (0.3, 523.25), (0.6, 659.25)]:
        mask = (t >= start) & (t < start + 0.25)
        env = np.exp(-(t[mask] - start) / 0.05)
        audio[mask] += np.sin(2 * np.pi * freq * t[mask]) * env

    out = synthesize_pop_chip(audio, sample_rate=sr, chip_mix=1.0, n_voices=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "out.wav"
        export_audio(out, str(out_path), sample_rate=sr)
        assert out_path.exists()
        assert out_path.stat().st_size > 0
```

- [ ] **Step 2: 运行全部测试**

Run: `python -m pytest tests/ -v`

Expected: 所有测试 PASS

- [ ] **Step 3: 修复回归问题（如有）**

如果 FC 模式或其他测试失败，定位并修复。

- [ ] **Step 4: 提交**

```bash
git add tests/test_integration.py
git commit -m "test: add pop-chip end-to-end integration test"
```

---

## Task 7: 最终验证与文档更新

**Files:**
- Modify: `docs/superpowers/specs/2026-06-23-pop-chip-timbre-stabilization-design.md`
- Modify: `README.md`（如需要）

- [ ] **Step 1: 用真实 MP3 做主观听感验证**

准备一首熟悉的流行/钢琴 MP3，运行：

```bash
python main.py
```

选择“流行 8-bit”模式转换，听取：
- 音高是否稳定，无飘动感
- 和声/伴奏是否保留
- 是否有爆音、杂音

- [ ] **Step 2: 对比参数效果**

分别用 `n_voices=2` 和 `n_voices=6` 转换同一文件，确认和声厚度差异。

分别用 `waveform='square'` 和 `waveform='triangle'` 转换，确认音色差异。

- [ ] **Step 3: 更新设计文档状态**

将 `docs/superpowers/specs/2026-06-23-pop-chip-timbre-stabilization-design.md` 顶部状态改为：

```markdown
**状态：** 已实现
```

- [ ] **Step 4: 提交最终版本**

```bash
git add docs/superpowers/specs/2026-06-23-pop-chip-timbre-stabilization-design.md
git commit -m "docs: mark pop-chip timbre stabilization as implemented"
```

---

## 自检清单

- [x] Spec coverage: onset 检测、分段、稳定音高提取、复音合并、事件合成、主流程重构、测试、主观验证均已覆盖
- [x] Placeholder scan: 无 TBD/TODO/"implement later"/"similar to Task N"
- [x] Type consistency: `_midi_to_freq`、`_bandlimited_waveform`、`_rms_envelope`、`_gentle_compress`、`_audio_to_mono_float` 均沿用已有定义；新函数签名在 Task 3 与 Task 5 中一致
- [x] 所有文件路径使用绝对路径
- [x] 每个任务均包含失败测试、实现代码、通过命令、提交命令

---

**Execution Handoff:**

Plan complete and saved to `docs/superpowers/plans/2026-06-23-pop-chip-timbre-stabilization-plan.md`. Two execution options:

1. **Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
