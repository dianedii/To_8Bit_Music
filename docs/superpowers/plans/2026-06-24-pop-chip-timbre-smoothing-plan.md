# 流行 8-bit 音色柔化与音高定位优化实现计划

> **状态：** 本文档为历史计划。实际实现已根据用户参考样例重新设计，移除了波形/chip_mix 控件，改用 `piano_transcription_inference` + 多层方波/三角波合成，并新增视频输入支持。详见 `2026-06-24-pop-chip-timbre-smoothing-design.md` 第 7 节与更新后的 `README.md`。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让流行 8-bit 输出更丝滑、主旋律音高定位更准确，同时从 UI 隐藏 FC 模式。

**Architecture:** 重写 `src/pop_synthesizer.py`，接入 `src/pop_melody.py` 的主旋律提取流程；通过默认三角波、legato、柔和 ADSR、低通滤波降低粗硬感；通过 f0 平滑和 pYIN 音符化提升定位精度。

**Tech Stack:** Python, numpy, scipy, librosa, PyQt6, pytest

---

## File Structure

| 文件 | 职责 |
|------|------|
| `src/pop_melody.py` | 扩展 pYIN 音符化，支持 f0 平滑与量化强度 |
| `src/pop_synthesizer.py` | 接入主旋律提取，新增 legato/ADSR/低通，移除旧 onset/segment 代码 |
| `src/ui.py` | 隐藏 FC 模式，暴露波形选择与 chip_mix 滑块 |
| `src/worker.py` | 把 waveform、chip_mix 透传给合成器 |
| `tests/test_pop_melody.py` | 新增 f0 平滑与量化强度测试 |
| `tests/test_pop_synthesizer.py` | 更新/新增 legato/低通/波形/集成测试 |
| `tests/test_integration.py` | 更新 pop-chip 端到端测试 |
| `README.md` | 更新功能说明 |

---

### Task 1: f0 中值平滑

**Files:**
- Modify: `src/pop_melody.py`
- Test: `tests/test_pop_melody.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from src.pop_melody import _smooth_f0


def test_smooth_f0_removes_single_frame_spike():
    f0 = np.array([440.0, 441.0, 880.0, 439.0, 440.0])
    voiced = np.array([True, True, True, True, True])
    smoothed = _smooth_f0(f0, voiced, kernel_size=3)
    assert smoothed[2] == 440.0
    np.testing.assert_array_equal(smoothed[voiced], np.array([440.0, 441.0, 440.0, 439.0, 440.0]))


def test_smooth_f0_keeps_unvoiced_nan():
    f0 = np.array([440.0, np.nan, 880.0, np.nan, 440.0])
    voiced = np.array([True, False, True, False, True])
    smoothed = _smooth_f0(f0, voiced, kernel_size=3)
    assert np.isnan(smoothed[1])
    assert np.isnan(smoothed[3])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pop_melody.py::test_smooth_f0_removes_single_frame_spike -v`
Expected: FAIL with "name '_smooth_f0' is not defined"

- [ ] **Step 3: Write minimal implementation**

在 `src/pop_melody.py` 顶部新增：

```python
from scipy import ndimage
```

在 `src/pop_melody.py` 中新增函数：

```python
def _smooth_f0(
    f0: np.ndarray,
    voiced_flag: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """对 voiced 帧的 f0 做中值滤波，去除单帧毛刺，unvoiced 帧保持原值。"""
    if len(f0) == 0:
        return f0
    kernel_size = max(3, kernel_size | 1)
    smoothed = ndimage.median_filter(f0, size=kernel_size, mode='nearest')
    result = f0.copy()
    result[voiced_flag] = smoothed[voiced_flag]
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pop_melody.py::test_smooth_f0_removes_single_frame_spike tests/test_pop_melody.py::test_smooth_f0_keeps_unvoiced_nan -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_pop_melody.py src/pop_melody.py
git commit -m "feat: add f0 median smoothing for pitch stability"
```

---

### Task 2: 扩展 pYIN 音符化支持量化强度

**Files:**
- Modify: `src/pop_melody.py`
- Test: `tests/test_pop_melody.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import librosa
from src.pop_melody import _pyin_to_notes


def test_pyin_to_notes_quantizes_to_semitones():
    sr = 22050
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.3 * np.sin(2 * np.pi * 445.0 * t)
    notes = _pyin_to_notes(audio, sr, hop_length=512, pitch_quantize_strength=1.0)
    assert len(notes) >= 1
    assert all(n[0] == 69 for n in notes)  # A4 = MIDI 69
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pop_melody.py::test_pyin_to_notes_quantizes_to_semitones -v`
Expected: FAIL because `_pyin_to_notes` does not accept `pitch_quantize_strength`

- [ ] **Step 3: Write minimal implementation**

修改 `src/pop_melody.py` 中 `_pyin_to_notes` 的签名：

```python
def _pyin_to_notes(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int = 512,
    min_note_duration: float = 0.05,
    pitch_quantize_strength: float = 1.0,
    f0_median_size: int = 3,
) -> list[tuple[float, float, float, int]]:
```

在函数体中，pYIN 调用之后加入平滑：

```python
    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sample_rate,
        hop_length=hop_length,
    )
    f0 = _smooth_f0(f0, voiced_flag, kernel_size=f0_median_size)
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sample_rate, hop_length=hop_length)
```

修改循环内的音高计算：

```python
        midi_float = librosa.hz_to_midi(f0[i])
        if pitch_quantize_strength >= 1.0:
            midi = round(midi_float)
        else:
            quantized = round(midi_float)
            midi = quantized + (1.0 - pitch_quantize_strength) * (midi_float - quantized)

        vel = int(np.clip(voiced_prob[i] * 127, 1, 127))
```

并将 `current_pitch` 的比较改为处理 float：

```python
        if current_pitch is None:
            current_pitch = midi
            start_time = times[i]
            velocities = [vel]
        elif abs(midi - current_pitch) <= 0.5 * pitch_quantize_strength + 0.05:
            velocities.append(vel)
        else:
            _finalize_note()
            current_pitch = midi
            start_time = times[i]
            velocities = [vel]
```

音符返回类型改为 `list[tuple[float, float, float, int]]`，但 `notes.append` 处保持：

```python
            notes.append((current_pitch, start_time, times[i], int(np.mean(velocities))))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pop_melody.py::test_pyin_to_notes_quantizes_to_semitones -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_pop_melody.py src/pop_melody.py
git commit -m "feat: extend pyin note extraction with quantization strength"
```

---

### Task 3: 候选线评分与过滤兼容 float pitch

**Files:**
- Modify: `src/pop_melody.py`
- Test: `tests/test_pop_melody.py`

- [ ] **Step 1: Write the failing test**

```python
from src.pop_melody import _score_melody_line, _apply_hard_filters


def test_score_melody_line_accepts_float_pitch():
    line = [
        (60.2, 0.0, 0.3, 100),
        (62.1, 0.35, 0.6, 100),
        (64.0, 0.65, 0.9, 100),
    ]
    score = _score_melody_line(line)
    assert 0.0 <= score <= 1.0


def test_hard_filters_accepts_float_pitch():
    lines = [
        [(60.0, 0.0, 0.05, 100), (62.0, 0.06, 0.11, 100)],
        [(60.2, 0.0, 0.3, 100), (64.1, 0.35, 0.65, 100)],
    ]
    filtered = _apply_hard_filters(lines)
    assert len(filtered) == 1
    assert abs(filtered[0][0][0] - 60.2) < 0.01
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pop_melody.py::test_score_melody_line_accepts_float_pitch -v`
Expected: FAIL or type error if current code assumes int

- [ ] **Step 3: Write minimal implementation**

修改 `src/pop_melody.py`：

在 `_score_melody_line` 中：

```python
    pitches = [int(round(n[0])) for n in line]
```

在 `_apply_hard_filters` 中：

```python
        pitches = [int(round(n[0])) for n in line]
```

在 `_extract_harmony_voice` 中：

```python
                    interval = abs(int(round(pitch)) - int(round(m_pitch))) % 12
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pop_melody.py::test_score_melody_line_accepts_float_pitch tests/test_pop_melody.py::test_hard_filters_accepts_float_pitch -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_pop_melody.py src/pop_melody.py
git commit -m "fix: make melody scoring and filtering compatible with float pitches"
```

---

### Task 4: Legato 处理

**Files:**
- Modify: `src/pop_synthesizer.py`
- Test: `tests/test_pop_synthesizer.py`

- [ ] **Step 1: Write the failing test**

```python
from src.pop_synthesizer import _apply_legato


def test_apply_legato_merges_close_notes():
    notes = [
        (60.0, 0.0, 0.45, 100),
        (62.0, 0.48, 0.9, 100),
    ]
    merged = _apply_legato(notes, threshold=0.05)
    assert len(merged) == 2
    assert merged[0][2] == 0.48  # first note extended to second onset


def test_apply_legato_keeps_separated_notes():
    notes = [
        (60.0, 0.0, 0.3, 100),
        (62.0, 0.5, 0.8, 100),
    ]
    merged = _apply_legato(notes, threshold=0.05)
    assert merged[0][2] == 0.3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pop_synthesizer.py::test_apply_legato_merges_close_notes -v`
Expected: FAIL with "name '_apply_legato' is not defined"

- [ ] **Step 3: Write minimal implementation**

在 `src/pop_synthesizer.py` 中新增：

```python
def _apply_legato(
    notes: list[tuple[float, float, float, int]],
    threshold: float = 0.05,
) -> list[tuple[float, float, float, int]]:
    """合并间隔 ≤ threshold 的相邻音符，前一个音符延长到后一个 onset，两者都保留。"""
    if not notes:
        return []
    notes = sorted(notes, key=lambda n: n[1])
    merged = [list(notes[0])]
    for pitch, onset, offset, velocity in notes[1:]:
        prev = merged[-1]
        gap = onset - prev[2]
        if 0 <= gap <= threshold:
            prev[2] = onset
            prev[3] = max(prev[3], velocity)
        merged.append([pitch, onset, offset, velocity])
    return [tuple(n) for n in merged]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pop_synthesizer.py::test_apply_legato_merges_close_notes tests/test_pop_synthesizer.py::test_apply_legato_keeps_separated_notes -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_pop_synthesizer.py src/pop_synthesizer.py
git commit -m "feat: add legato merging for smoother melody phrasing"
```

---

### Task 5: 低通滤波与正弦波支持

**Files:**
- Modify: `src/pop_synthesizer.py`
- Test: `tests/test_pop_synthesizer.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from src.pop_synthesizer import _apply_lowpass, _bandlimited_waveform


def test_apply_lowpass_reduces_high_freq():
    sr = 44100
    t = np.arange(sr) / sr
    audio = np.sin(2 * np.pi * 12000 * t)
    filtered = _apply_lowpass(audio, sr, cutoff=8000)
    assert np.max(np.abs(filtered)) <= np.max(np.abs(audio))
    # Energy above cutoff should drop
    fft_in = np.abs(np.fft.rfft(audio))
    fft_out = np.abs(np.fft.rfft(filtered))
    assert fft_out[-100:].sum() < fft_in[-100:].sum()


def test_bandlimited_waveform_sine():
    sr = 44100
    freq = 440.0
    t = np.arange(int(sr * 0.1)) / sr
    phase = 2 * np.pi * freq * t
    wave = _bandlimited_waveform(phase, freq, sr, 'sine')
    np.testing.assert_allclose(wave, np.sin(phase), atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pop_synthesizer.py::test_apply_lowpass_reduces_high_freq -v`
Expected: FAIL with "name '_apply_lowpass' is not defined"

- [ ] **Step 3: Write minimal implementation**

在 `src/pop_synthesizer.py` 中修改 `_bandlimited_waveform`：

```python
def _bandlimited_waveform(phase: np.ndarray, freq: float, sample_rate: int, waveform: str) -> np.ndarray:
    """根据瞬时基频限制谐波数，生成带限方波/三角波/锯齿波/正弦波。"""
    if waveform == 'sine':
        return np.sin(phase)

    max_harm = int(np.floor((sample_rate / 2.5) / max(freq, 20.0)))
    max_harm = max(1, min(max_harm, 40))

    wave = np.zeros_like(phase)
    if waveform == 'square':
        for n in range(1, max_harm + 1, 2):
            wave += (4.0 / (np.pi * n)) * np.sin(n * phase)
    elif waveform == 'triangle':
        for n in range(1, max_harm + 1, 2):
            sign = (-1) ** ((n - 1) // 2)
            wave += (8.0 / (np.pi ** 2 * n ** 2)) * sign * np.sin(n * phase)
    elif waveform == 'sawtooth':
        for n in range(1, max_harm + 1):
            wave += (2.0 / (np.pi * n)) * ((-1) ** (n + 1)) * np.sin(n * phase)
    else:
        wave = np.sign(np.sin(phase))

    return np.clip(wave, -1.0, 1.0)
```

在 `src/pop_synthesizer.py` 中新增：

```python
def _apply_lowpass(
    audio: np.ndarray,
    sample_rate: int,
    cutoff: float = 8000.0,
) -> np.ndarray:
    """对音频做 6dB/oct 低通滤波，软化高频。"""
    if cutoff <= 0 or cutoff >= sample_rate / 2:
        return audio
    b, a = signal.butter(1, cutoff / (sample_rate / 2), btype='low')
    return signal.filtfilt(b, a, audio)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pop_synthesizer.py::test_apply_lowpass_reduces_high_freq tests/test_pop_synthesizer.py::test_bandlimited_waveform_sine -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_pop_synthesizer.py src/pop_synthesizer.py
git commit -m "feat: add lowpass filter and sine waveform option"
```

---

### Task 6: 改进事件合成 ADSR

**Files:**
- Modify: `src/pop_synthesizer.py`
- Test: `tests/test_pop_synthesizer.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from src.pop_synthesizer import _synthesize_events


def test_synthesize_events_triangle_has_fewer_harmonics():
    sr = 44100
    notes = [(69, 0.0, 0.2, 100)]  # A4
    square = _synthesize_events(notes, 0.2, sr, waveform='square')
    triangle = _synthesize_events(notes, 0.2, sr, waveform='triangle')
    fft_sq = np.abs(np.fft.rfft(square))
    fft_tri = np.abs(np.fft.rfft(triangle))
    # Triangle should have less energy in high partials
    assert fft_tri[2000:].sum() < fft_sq[2000:].sum()


def test_synthesize_events_release_is_longer():
    sr = 44100
    notes = [(69, 0.0, 0.1, 100)]
    audio = _synthesize_events(notes, 0.1, sr)
    # Last samples should ramp down (not abruptly cut)
    assert audio[-1] < audio[-int(0.005 * sr)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pop_synthesizer.py::test_synthesize_events_triangle_has_fewer_harmonics -v`
Expected: FAIL or assertion error depending on current defaults

- [ ] **Step 3: Write minimal implementation**

修改 `src/pop_synthesizer.py` 中 `_synthesize_events`：

```python
def _synthesize_events(
    notes: list[tuple[float, float, float, int]],
    duration: float,
    sample_rate: int,
    waveform: str = 'triangle',
) -> np.ndarray:
    """将复音音符事件合成为稳定音高的芯片音频。"""
    num_samples = int(duration * sample_rate)
    audio = np.zeros(num_samples, dtype=np.float64)

    for pitch, onset, offset, velocity in notes:
        freq = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
        start_idx = max(0, int(onset * sample_rate))
        end_idx = min(num_samples, int(offset * sample_rate))
        if start_idx >= end_idx:
            continue

        length = end_idx - start_idx
        t = np.arange(length) / sample_rate
        phase = 2.0 * np.pi * freq * t
        wave = _bandlimited_waveform(phase, freq, sample_rate, waveform)

        env = np.ones(length, dtype=np.float64)
        attack_samples = min(int(0.010 * sample_rate), length)
        release_samples = min(int(0.040 * sample_rate), length)

        env[:attack_samples] = np.linspace(0.0, 1.0, attack_samples)
        release_start_idx = max(0, length - release_samples)
        start_amp = env[release_start_idx]
        env[-release_samples:] = np.linspace(start_amp, 0.0, release_samples)

        amp = (velocity / 127.0) * 0.25
        audio[start_idx:end_idx] += wave * env * amp

    audio = np.clip(audio, -1.0, 1.0)
    return audio
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pop_synthesizer.py::test_synthesize_events_triangle_has_fewer_harmonics tests/test_pop_synthesizer.py::test_synthesize_events_release_is_longer -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_pop_synthesizer.py src/pop_synthesizer.py
git commit -m "feat: soften ADSR and default waveform to triangle"
```

---

### Task 7: 重构 synthesize_pop_chip 主流程

**Files:**
- Modify: `src/pop_synthesizer.py`
- Test: `tests/test_pop_synthesizer.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from src.pop_synthesizer import synthesize_pop_chip


def test_synthesize_pop_chip_main_melody_only():
    sr = 22050
    duration = 1.0
    t = np.arange(int(sr * duration)) / sr
    # Simple melody: C4 - E4 - G4, clean sine-like input
    audio = np.zeros_like(t)
    for start, freq in [(0.0, 261.63), (0.35, 329.63), (0.7, 392.0)]:
        mask = (t >= start) & (t < start + 0.25)
        audio[mask] += 0.3 * np.sin(2 * np.pi * freq * t[mask])

    out = synthesize_pop_chip(audio, sample_rate=sr, chip_mix=1.0, waveform='triangle')
    assert out.shape[0] == 2
    assert out.shape[1] > 0
    assert out.dtype == np.float32
    assert np.max(np.abs(out)) > 0.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pop_synthesizer.py::test_synthesize_pop_chip_main_melody_only -v`
Expected: FAIL because main flow still uses old onset/segment path

- [ ] **Step 3: Write minimal implementation**

重写 `src/pop_synthesizer.py` 中 `synthesize_pop_chip`：

```python
def synthesize_pop_chip(
    audio: np.ndarray,
    sample_rate: int = 44100,
    waveform: Literal['square', 'triangle', 'sawtooth', 'sine'] = 'triangle',
    chip_mix: float = 0.6,
    n_voices: int = 6,
    hop_length: int = 512,
    min_note_duration: float = 0.05,
    pitch_quantize_strength: float = 1.0,
    f0_median_size: int = 3,
    legato_threshold: float = 0.05,
    lowpass_cutoff: float = 8000.0,
) -> np.ndarray:
    """
    基于主旋律提取的流行 8-bit 合成。

    Args:
        audio: 输入音频，numpy 数组（单声道/立体声，任意 dtype）
        sample_rate: 采样率
        waveform: 合成器波形
        chip_mix: 合成器层混合比例（0~1）
        n_voices: pYIN/旋律提取内部参数（兼容性保留）
        hop_length: STFT 帧移
        min_note_duration: 最短音符时长（秒）
        pitch_quantize_strength: 音高量化强度
        f0_median_size: f0 中值滤波窗口
        legato_threshold: 连音间隔阈值（秒）
        lowpass_cutoff: 低通截止频率（Hz）

    Returns:
        合成后的立体声 float32 音频，shape (2, N)
    """
    from src.pop_melody import (
        _pyin_to_notes,
        _split_candidate_lines,
        _apply_hard_filters,
        _extract_main_melody,
        _extract_harmony_voice,
    )

    mono = _audio_to_mono_float(audio)
    if len(mono) == 0:
        return np.zeros((2, 0), dtype=np.float32)

    notes = _pyin_to_notes(
        mono,
        sample_rate,
        hop_length=hop_length,
        min_note_duration=min_note_duration,
        pitch_quantize_strength=pitch_quantize_strength,
        f0_median_size=f0_median_size,
    )

    lines = _split_candidate_lines(notes)
    lines = _apply_hard_filters(lines)
    main_line = _extract_main_melody(lines)

    other_lines = [line for line in lines if line is not main_line]
    harmony = _extract_harmony_voice(main_line, other_lines, max_voices=1, volume_ratio=0.6)

    main_legato = _apply_legato(main_line, threshold=legato_threshold)
    events = sorted(main_legato + harmony, key=lambda n: n[1])

    duration = len(mono) / sample_rate
    synth = _synthesize_events(events, duration, sample_rate, waveform=waveform)

    if lowpass_cutoff > 0:
        synth = _apply_lowpass(synth, sample_rate, cutoff=lowpass_cutoff)

    peak = np.max(np.abs(synth))
    if peak > 1e-9:
        synth = synth / peak * 0.98

    envelope = _rms_envelope(mono, floor=0.50)
    envelope = np.cbrt(envelope)
    if len(envelope) == len(synth):
        synth = synth * envelope

    out = (1.0 - chip_mix) * mono + chip_mix * synth
    out = out.astype(np.float64)

    out = _gentle_compress(out, threshold=0.55, ratio=2.5)
    peak = np.max(np.abs(out))
    if peak > 1e-9:
        out = out / peak * 0.98
    out = np.clip(out, -1.0, 1.0).astype(np.float32)
    return np.stack([out, out], axis=0)
```

并移除不再使用的旧函数：`_detect_onsets`、`_segment_audio`、`_extract_stable_pitches`、`_merge_consecutive_notes`。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pop_synthesizer.py::test_synthesize_pop_chip_main_melody_only -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_pop_synthesizer.py src/pop_synthesizer.py
git commit -m "feat: rewrite pop-chip synthesis around main melody extraction"
```

---

### Task 8: UI 隐藏 FC 并暴露新参数

**Files:**
- Modify: `src/ui.py`
- Test: 手动验证（暂无 UI 单元测试，通过集成测试覆盖）

- [ ] **Step 1: 修改 UI**

在 `src/ui.py` 中：

1. 移除“复古纯度”和“音符简化强度”相关代码。
2. 把“转换风格”下拉框改为只保留“流行 8-bit”或直接移除。
3. 增加波形选择下拉框：`["三角波", "方波", "锯齿波", "正弦波"]`，映射到 `triangle/square/sawtooth/sine`。
4. 增加 `chip_mix` 滑块，范围 0~100，默认值 60。
5. `_start_convert` 中把 `waveform` 和 `chip_mix` 传给 `ConvertWorker`。

关键改动示意（完整代码需替换整个文件）：

```python
        # 音色波形
        waveform_layout = QHBoxLayout()
        waveform_layout.addWidget(QLabel("芯片波形"))
        self.waveform_combo = QComboBox()
        self.waveform_combo.addItems(["三角波", "方波", "锯齿波", "正弦波"])
        self.waveform_combo.setCurrentText("三角波")
        waveform_layout.addWidget(self.waveform_combo)
        waveform_layout.addStretch()
        layout.addLayout(waveform_layout)

        # 芯片混合比例
        layout.addWidget(QLabel("芯片音色占比"))
        self.chip_mix_slider = QSlider(Qt.Orientation.Horizontal)
        self.chip_mix_slider.setRange(0, 100)
        self.chip_mix_slider.setValue(60)
        layout.addWidget(self.chip_mix_slider)
        chip_mix_layout = QHBoxLayout()
        chip_mix_layout.addWidget(QLabel("原声"))
        chip_mix_layout.addStretch()
        chip_mix_layout.addWidget(QLabel("芯片"))
        layout.addLayout(chip_mix_layout)
```

`_start_convert` 中：

```python
        waveform_map = {
            "三角波": "triangle",
            "方波": "square",
            "锯齿波": "sawtooth",
            "正弦波": "sine",
        }
        waveform = waveform_map[self.waveform_combo.currentText()]
        chip_mix = self.chip_mix_slider.value() / 100.0

        self.worker = ConvertWorker(
            input_path=self.input_path,
            purity=0,
            simplification=0,
            volume=self.volume_slider.value(),
            output_format=self.format_combo.currentText(),
            mode="pop",
            waveform=waveform,
            chip_mix=chip_mix,
        )
```

- [ ] **Step 2: 运行集成测试**

Run: `pytest tests/test_integration.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add src/ui.py
git commit -m "feat: hide FC mode and expose waveform and chip_mix controls"
```

---

### Task 9: Worker 透传 waveform 与 chip_mix

**Files:**
- Modify: `src/worker.py`
- Test: `tests/test_integration.py`

- [ ] **Step 1: Write the failing test**

更新 `tests/test_integration.py` 中 `test_pop_chip_end_to_end_synthetic`：

```python
def test_pop_chip_end_to_end_synthetic():
    sr = 44100
    duration = 1.0
    t = np.arange(int(duration * sr)) / sr
    audio = np.zeros_like(t)
    for start, freq in [(0.0, 440.0), (0.3, 523.25), (0.6, 659.25)]:
        mask = (t >= start) & (t < start + 0.25)
        env = np.exp(-(t[mask] - start) / 0.05)
        audio[mask] += np.sin(2 * np.pi * freq * t[mask]) * env

    out = synthesize_pop_chip(audio, sample_rate=sr, chip_mix=1.0, waveform='triangle')

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "out.wav"
        export_audio(out, str(out_path), sample_rate=sr)
        assert out_path.exists()
        assert out_path.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_integration.py::test_pop_chip_end_to_end_synthetic -v`
Expected: FAIL if worker signature mismatch

- [ ] **Step 3: Write minimal implementation**

修改 `src/worker.py`：

```python
    def __init__(
        self,
        input_path: str,
        purity: int,
        simplification: int,
        volume: int,
        output_format: str,
        mode: str = "pop",
        waveform: str = "triangle",
        chip_mix: float = 0.6,
    ):
        super().__init__()
        self.input_path = input_path
        self.purity = purity
        self.simplification = simplification
        self.volume = volume
        self.output_format = output_format
        self.mode = mode
        self.waveform = waveform
        self.chip_mix = chip_mix
```

在 `run()` 的 pop 分支中：

```python
                audio_data = synthesize_pop_chip(
                    audio_np,
                    sample_rate=audio.frame_rate,
                    waveform=self.waveform,
                    chip_mix=self.chip_mix,
                )
```

移除 `n_voices=6` 硬编码。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_integration.py::test_pop_chip_end_to_end_synthetic -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_integration.py src/worker.py
git commit -m "feat: pass waveform and chip_mix from worker to synthesizer"
```

---

### Task 10: 更新 README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 更新功能说明**

在 `README.md` 中：

1. 移除 FC 模式介绍，改为说明 FC 已隐藏。
2. 说明流行 8-bit 模式现在基于主旋律提取。
3. 列出 UI 控件：波形选择、芯片音色占比、音量、输出格式。

示例改动：

```markdown
## 功能

- **流行 8-bit**：自动提取主旋律 + 少量和声，用三角波/方波/锯齿波/正弦波合成，保留流行音乐美感。
- 支持 MP3 / WAV 输出。
- 可调整芯片音色占比与波形。

> **注意**：纯正 FC 模式已从 UI 隐藏，相关代码保留但不再被主流程调用。
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README for pop-chip timbre smoothing and hidden FC mode"
```

---

### Task 11: 全量回归测试

**Files:**
- All test files

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -q`
Expected: all tests pass

- [ ] **Step 2: Fix any failures**

If failures exist, diagnose and fix. Common issues:
- `tests/test_pop_synthesizer.py` old tests expecting old API: update or remove.
- Import errors from removed functions: update tests to use new API.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "test: update regression tests for new pop-chip flow"
```

---

## Self-Review Checklist

1. **Spec coverage:**
   - f0 平滑 → Task 1
   - pYIN 音符化/量化强度 → Task 2
   - float pitch 兼容 → Task 3
   - legato → Task 4
   - 低通/正弦波 → Task 5
   - 柔和 ADSR/默认三角波 → Task 6
   - 主旋律驱动流程 → Task 7
   - UI 调整 → Task 8
   - Worker 参数透传 → Task 9
   - README → Task 10
   - 回归测试 → Task 11

2. **Placeholder scan:** 无 TBD/TODO/"implement later"。

3. **Type consistency:** `_pyin_to_notes` 返回 `list[tuple[float, float, float, int]]`；`synthesize_pop_chip` 内部音符事件均为 float pitch；评分/过滤/和声函数已统一使用 `int(round(pitch))`。

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-24-pop-chip-timbre-smoothing-plan.md`.

Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
