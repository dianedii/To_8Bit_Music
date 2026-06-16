# FC/NES 8-bit 芯片音乐转换器实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现一款 PyQt6 桌面 GUI 工具，将纯钢琴 MP3 通过「分离-识别-重合成」全链路转换为纯正 FC/NES 8-bit 芯片音乐。

**Architecture：** 采用模块化流水线：转录模块将 MP3 转为音符事件 → 主旋律提取模块聚类选主导声部 → 音符简化模块清理碎音与装饰音 → 合成模块从零生成 FC 方波 → GUI 与后台线程协调进度与输出。所有模块通过纯 Python 数据对象交互，不保留原始音频采样。

**Tech Stack：** Python 3.9+, PyQt6, piano_transcription_inference, numpy, scipy, pydub/lameenc

---

## 目录结构

```
8Bit-New/
├── main.py                      # 程序入口
├── requirements.txt             # 依赖清单
├── setup_env.py                 # 依赖检测与国内镜像安装脚本
├── src/
│   ├── __init__.py
│   ├── ui.py                    # PyQt6 主窗口
│   ├── worker.py                # QThread 后台转换
│   ├── transcriber.py           # 钢琴音频转录
│   ├── melody_extractor.py      # 主旋律提取
│   ├── note_simplifier.py       # 音符简化
│   ├── synthesizer.py           # FC 方波合成
│   └── utils.py                 # 依赖检测、音频导出、通用工具
├── tests/
│   ├── __init__.py
│   ├── test_melody_extractor.py
│   ├── test_note_simplifier.py
│   ├── test_synthesizer.py
│   └── test_utils.py
└── docs/superpowers/specs/2026-06-16-fc-8bit-converter-design.md
```

---

## Task 1: 项目初始化与 requirements.txt

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Modify: `.gitignore`

- [ ] **Step 1: 创建 requirements.txt**

```text
PyQt6>=6.4.0
piano_transcription_inference>=0.2
numpy>=1.23.0
scipy>=1.9.0
pydub>=0.25.1
```

- [ ] **Step 2: 创建 src/__init__.py 与 tests/__init__.py**

两个文件均为空文件，仅用于包识别。

```bash
# 创建空文件（使用 Write 工具写入空内容或空字符串）
```

- [ ] **Step 3: 更新 .gitignore**

```gitignore
__pycache__/
*.pyc
*.pyo
*.egg-info/
.env/
.venv/
*.wav
*.mp3
!tests/fixtures/*.mp3
!tests/fixtures/*.wav
.superpowers/brainstorm/
```

- [ ] **Step 4: 提交**

```bash
git add requirements.txt src/__init__.py tests/__init__.py .gitignore
git commit -m "chore: initialize project structure and requirements"
```

---

## Task 2: 依赖检测与国内镜像安装脚本

**Files:**
- Create: `setup_env.py`
- Test: `tests/test_utils.py`（此任务只测试依赖检测函数）

- [ ] **Step 1: 写测试 —— 检测已安装与未安装包**

```python
# tests/test_utils.py
import sys
import importlib
from unittest.mock import patch

from src.utils import is_package_installed


def test_is_package_installed_true():
    # numpy 在 requirements 中，通常测试环境未安装，这里用一个肯定存在的内置包测试
    assert is_package_installed('sys') is True


def test_is_package_installed_false():
    assert is_package_installed('nonexistent_package_abc123') is False
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_utils.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.utils'`

- [ ] **Step 3: 实现 utils.py 中的依赖检测与安装函数**

```python
# src/utils.py
import importlib
import subprocess
import sys
from pathlib import Path


MIRROR_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"


def is_package_installed(package_name: str) -> bool:
    """检查 Python 包是否已安装。"""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


def install_packages(package_names: list[str], use_mirror: bool = True) -> None:
    """使用 pip 安装指定包，可选国内镜像。"""
    cmd = [sys.executable, "-m", "pip", "install"]
    if use_mirror:
        cmd.extend(["--index-url", MIRROR_INDEX_URL])
    cmd.extend(package_names)
    subprocess.check_call(cmd)


def get_output_path(input_path: str, output_format: str) -> Path:
    """根据输入文件和输出格式生成输出路径。"""
    p = Path(input_path)
    suffix = ".wav" if output_format.lower() == "wav" else ".mp3"
    return p.with_suffix(suffix)


def open_folder(path: Path) -> None:
    """打开文件所在文件夹。"""
    import platform
    folder = path.parent
    system = platform.system()
    if system == "Windows":
        subprocess.run(["explorer", str(folder)], check=False)
    elif system == "Darwin":
        subprocess.run(["open", str(folder)], check=False)
    else:
        subprocess.run(["xdg-open", str(folder)], check=False)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest tests/test_utils.py -v
```

Expected: 两个测试均 PASS

- [ ] **Step 5: 实现 setup_env.py 一键安装入口**

```python
# setup_env.py
from src.utils import is_package_installed, install_packages


REQUIRED_PACKAGES = [
    "PyQt6",
    "piano_transcription_inference",
    "numpy",
    "scipy",
    "pydub",
]


def main():
    missing = [pkg for pkg in REQUIRED_PACKAGES if not is_package_installed(pkg)]
    if not missing:
        print("所有依赖已安装。")
        return
    print(f"检测到缺失依赖: {', '.join(missing)}")
    print("正在通过国内镜像安装...")
    install_packages(missing, use_mirror=True)
    print("安装完成。")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 提交**

```bash
git add src/utils.py setup_env.py tests/test_utils.py .gitignore
git commit -m "feat: add dependency check and mirror-based installer"
```

---

## Task 3: 钢琴音频转录模块

**Files:**
- Create: `src/transcriber.py`
- Test: `tests/test_transcriber.py`

- [ ] **Step 1: 写测试 —— 验证转录结果格式**

```python
# tests/test_transcriber.py
from unittest.mock import patch, MagicMock
import numpy as np

from src.transcriber import transcribe_to_notes


def test_transcribe_to_notes_format():
    # Mock piano_transcription_inference 的输出
    mock_notes = [
        {"midi_note": 60, "onset_time": 0.0, "offset_time": 0.5, "velocity": 80},
        {"midi_note": 64, "onset_time": 0.5, "offset_time": 1.0, "velocity": 75},
    ]
    with patch('src.transcriber.transcribe') as mock_transcribe:
        mock_transcribe.return_value = {"est_note_events": mock_notes}
        notes = transcribe_to_notes("dummy.mp3")
    assert len(notes) == 2
    assert notes[0] == (60, 0.0, 0.5, 80)
    assert notes[1] == (64, 0.5, 1.0, 75)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_transcriber.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.transcriber'`

- [ ] **Step 3: 实现 transcriber.py**

```python
# src/transcriber.py
from pathlib import Path
from typing import List, Tuple

try:
    from piano_transcription_inference import PianoTranscription, sample_rate, window_size
    PTI_AVAILABLE = True
except ImportError:
    PianoTranscription = None
    sample_rate = 16000
    window_size = 2048
    PTI_AVAILABLE = False


def transcribe_to_notes(audio_path: str) -> List[Tuple[int, float, float, int]]:
    """
    将音频文件转录为音符事件列表。
    返回: [(midi_pitch, onset_time, offset_time, velocity), ...]
    """
    if not PTI_AVAILABLE:
        raise RuntimeError("piano_transcription_inference 未安装，请先运行 setup_env.py")

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    transcriptor = PianoTranscription(
        instrument="piano",
        device="cpu",
    )
    # piano_transcription_inference 的 transcribe 方法返回字典
    result = transcriptor.transcribe(str(audio_path))
    note_events = result.get("est_note_events", [])

    notes = []
    for event in note_events:
        pitch = int(event["midi_note"])
        onset = float(event["onset_time"])
        offset = float(event["offset_time"])
        velocity = int(event.get("velocity", 80))
        notes.append((pitch, onset, offset, velocity))

    return notes
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest tests/test_transcriber.py -v
```

Expected: `test_transcribe_to_notes_format` PASS

- [ ] **Step 5: 提交**

```bash
git add src/transcriber.py tests/test_transcriber.py
git commit -m "feat: add piano transcription module"
```

---

## Task 4: 主旋律提取模块

**Files:**
- Create: `src/melody_extractor.py`
- Test: `tests/test_melody_extractor.py`

- [ ] **Step 1: 写测试 —— 简单主旋律提取**

```python
# tests/test_melody_extractor.py
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
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_melody_extractor.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.melody_extractor'`

- [ ] **Step 3: 实现 melody_extractor.py**

```python
# src/melody_extractor.py
from typing import List, Tuple


def _score_note(pitch: int, duration: float, velocity: int) -> float:
    """综合音符时长、力度和音高打分。"""
    return duration * (velocity / 127.0) + (pitch / 127.0) * 0.3


def extract_melody(
    notes: List[Tuple[int, float, float, int]],
    window_ms: int = 50,
) -> List[Tuple[int, float, float, int]]:
    """
    从多声部钢琴音符中提取单声部主旋律。
    notes: [(pitch, onset, offset, velocity), ...]
    返回: [(pitch, onset, offset, velocity), ...]
    """
    if not notes:
        return []

    notes = sorted(notes, key=lambda n: n[1])
    window_s = window_ms / 1000.0
    melody = []
    current_group = [notes[0]]
    current_end = notes[0][2]

    def _pick_best(group):
        best = max(group, key=lambda n: _score_note(n[0], n[2] - n[1], n[3]))
        return best

    for note in notes[1:]:
        # 如果当前音符与当前组的时间窗重叠或紧邻，则加入同一组
        if note[1] <= current_end + window_s:
            current_group.append(note)
            current_end = max(current_end, note[2])
        else:
            melody.append(_pick_best(current_group))
            current_group = [note]
            current_end = note[2]

    melody.append(_pick_best(current_group))
    return melody
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest tests/test_melody_extractor.py -v
```

Expected: 两个测试均 PASS

- [ ] **Step 5: 提交**

```bash
git add src/melody_extractor.py tests/test_melody_extractor.py
git commit -m "feat: add melody extraction module"
```

---

## Task 5: 音符简化模块

**Files:**
- Create: `src/note_simplifier.py`
- Test: `tests/test_note_simplifier.py`

- [ ] **Step 1: 写测试 —— 过滤碎音与合并同音**

```python
# tests/test_note_simplifier.py
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
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_note_simplifier.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.note_simplifier'`

- [ ] **Step 3: 实现 note_simplifier.py**

```python
# src/note_simplifier.py
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
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest tests/test_note_simplifier.py -v
```

Expected: 两个测试均 PASS

- [ ] **Step 5: 提交**

```bash
git add src/note_simplifier.py tests/test_note_simplifier.py
git commit -m "feat: add note simplification module"
```

---

## Task 6: FC 方波合成模块

**Files:**
- Create: `src/synthesizer.py`
- Test: `tests/test_synthesizer.py`

- [ ] **Step 1: 写测试 —— 验证波形输出形状与格式**

```python
# tests/test_synthesizer.py
import numpy as np

from src.synthesizer import synthesize


def test_synthesize_shape_and_range():
    notes = [
        (60, 0.0, 0.1, 80),  # C4, 100ms
    ]
    audio = synthesize(notes, duration=0.2, sample_rate=44100, purity=0, volume=80)
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert len(audio) == int(0.2 * 44100)
    assert np.max(np.abs(audio)) <= 1.0


def test_synthesize_silence_after_last_note():
    notes = [(60, 0.0, 0.05, 80)]
    audio = synthesize(notes, duration=0.2, sample_rate=44100, purity=0, volume=80)
    # 音符结束后应有静音
    tail = audio[int(0.1 * 44100):]
    assert np.max(np.abs(tail)) < 0.1
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_synthesizer.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.synthesizer'`

- [ ] **Step 3: 实现 synthesizer.py**

```python
# src/synthesizer.py
import numpy as np
from typing import List, Tuple


def midi_to_freq(midi_note: int) -> float:
    """MIDI 音符转频率。"""
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def synthesize(
    notes: List[Tuple[int, float, float, int]],
    duration: float,
    sample_rate: int = 44100,
    purity: int = 0,
    volume: int = 80,
) -> np.ndarray:
    """
    将音符序列合成为 FC 方波音频。
    purity: 0 ~ 100，0=纯方波，100=轻微 vibrato + release
    volume: 0 ~ 100
    """
    num_samples = int(duration * sample_rate)
    audio = np.zeros(num_samples, dtype=np.float64)

    max_amp = (volume / 100.0) * 0.45  # 留 headroom 防止破音
    vibrato_depth = (purity / 100.0) * 0.5  # 半音范围内的轻微颤音
    release_time = (purity / 100.0) * 0.03  # 30ms 以内的 release

    for pitch, onset, offset, velocity in notes:
        freq = midi_to_freq(pitch)
        start_idx = max(0, int(onset * sample_rate))
        end_idx = min(num_samples, int(offset * sample_rate))
        if start_idx >= end_idx:
            continue

        t = np.arange(end_idx - start_idx) / sample_rate
        # 轻微 vibrato：频率微调
        if vibrato_depth > 0:
            vibrato = 1.0 + vibrato_depth * 0.01 * np.sin(2 * np.pi * 6.0 * t)
            phase = np.cumsum(2 * np.pi * freq * vibrato / sample_rate)
        else:
            phase = 2 * np.pi * freq * t

        wave = np.sign(np.sin(phase)).astype(np.float64)

        # 应用包络：快速 attack + 可选 release
        env = np.ones_like(t)
        attack_samples = min(10, len(t))
        env[:attack_samples] = np.linspace(0.0, 1.0, attack_samples)
        if release_time > 0:
            release_samples = min(int(release_time * sample_rate), len(t))
            env[-release_samples:] = np.linspace(1.0, 0.0, release_samples)

        note_amp = max_amp * (velocity / 127.0)
        audio[start_idx:end_idx] += wave * env * note_amp

    # 软限幅防止破音
    audio = np.tanh(audio)
    return audio.astype(np.float32)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest tests/test_synthesizer.py -v
```

Expected: 两个测试均 PASS

- [ ] **Step 5: 提交**

```bash
git add src/synthesizer.py tests/test_synthesizer.py
git commit -m "feat: add FC square wave synthesizer"
```

---

## Task 7: 音频导出与 MP3/WAV 支持

**Files:**
- Modify: `src/utils.py`
- Test: `tests/test_utils.py`

- [ ] **Step 1: 写测试 —— 验证 WAV 导出**

```python
# tests/test_utils.py（追加）
import numpy as np
import tempfile
from pathlib import Path

from src.utils import export_audio


def test_export_wav():
    samples = np.sin(2 * np.pi * 440 * np.arange(4410) / 44100).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test.wav"
        export_audio(samples, str(out_path), sample_rate=44100)
        assert out_path.exists()
        assert out_path.stat().st_size > 0
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_utils.py::test_export_wav -v
```

Expected: `AttributeError: module 'src.utils' has no attribute 'export_audio'`

- [ ] **Step 3: 在 utils.py 中追加 export_audio 函数**

```python
# src/utils.py（追加到文件末尾）
import wave
import struct

import numpy as np


def export_audio(audio: np.ndarray, output_path: str, sample_rate: int = 44100) -> Path:
    """导出音频为 WAV 或 MP3（依赖 pydub）。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 先写出临时 WAV
    wav_path = output_path.with_suffix(".wav")
    # 确保音频在 [-1, 1] 范围内并转为 16bit
    clipped = np.clip(audio, -1.0, 1.0)
    int16_audio = (clipped * 32767).astype(np.int16)

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_audio.tobytes())

    if output_path.suffix.lower() == ".mp3":
        try:
            from pydub import AudioSegment
            segment = AudioSegment.from_wav(str(wav_path))
            segment.export(str(output_path), format="mp3", bitrate="192k")
            wav_path.unlink(missing_ok=True)
        except Exception as e:
            # MP3 导出失败时保留 WAV 并抛出提示
            raise RuntimeError(f"MP3 导出失败，已保留 WAV: {wav_path}. 错误: {e}")

    return output_path
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest tests/test_utils.py -v
```

Expected: 所有测试 PASS

- [ ] **Step 5: 提交**

```bash
git add src/utils.py tests/test_utils.py
git commit -m "feat: add WAV/MP3 audio export"
```

---

## Task 8: 转换工作流与 GUI 集成

**Files:**
- Create: `src/worker.py`
- Create: `src/ui.py`
- Create: `main.py`
- Test: `tests/test_worker.py`（可选，主要做集成测试）

- [ ] **Step 1: 实现 worker.py 后台转换线程**

```python
# src/worker.py
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal

from src.transcriber import transcribe_to_notes
from src.melody_extractor import extract_melody
from src.note_simplifier import simplify_notes
from src.synthesizer import synthesize
from src.utils import export_audio, get_output_path


class ConvertWorker(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished_success = pyqtSignal(str)
    finished_error = pyqtSignal(str)

    def __init__(
        self,
        input_path: str,
        purity: int,
        simplification: int,
        volume: int,
        output_format: str,
    ):
        super().__init__()
        self.input_path = input_path
        self.purity = purity
        self.simplification = simplification
        self.volume = volume
        self.output_format = output_format

    def run(self):
        try:
            self.status.emit("正在读取音频并检查时长...")
            self.progress.emit(5)

            from pydub import AudioSegment
            audio = AudioSegment.from_file(self.input_path)
            duration_sec = len(audio) / 1000.0
            if duration_sec > 600:
                self.finished_error.emit("曲目时长超过 10 分钟限制。")
                return

            self.status.emit("正在转录音符...")
            self.progress.emit(20)
            notes = transcribe_to_notes(self.input_path)

            self.status.emit("正在提取主旋律...")
            self.progress.emit(50)
            melody = extract_melody(notes)

            self.status.emit("正在简化音符...")
            self.progress.emit(70)
            simplified = simplify_notes(melody, strength=self.simplification)

            self.status.emit("正在合成 8-bit 音频...")
            self.progress.emit(85)
            output_path = get_output_path(self.input_path, self.output_format)
            audio_data = synthesize(
                simplified,
                duration=duration_sec + 0.5,
                sample_rate=44100,
                purity=self.purity,
                volume=self.volume,
            )

            self.status.emit("正在导出文件...")
            self.progress.emit(95)
            export_audio(audio_data, str(output_path), sample_rate=44100)

            self.progress.emit(100)
            self.finished_success.emit(str(output_path))
        except Exception as e:
            self.finished_error.emit(f"转换失败: {e}")
```

- [ ] **Step 2: 实现 ui.py 主窗口**

```python
# src/ui.py
import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QSlider, QComboBox, QFileDialog,
    QMessageBox,
)
from PyQt6.QtCore import Qt

from src.worker import ConvertWorker
from src.utils import open_folder


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FC 8-bit 芯片音乐转换器")
        self.setMinimumWidth(480)

        self.input_path = ""
        self.worker: Optional[ConvertWorker] = None

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

        # 文件选择
        file_layout = QHBoxLayout()
        self.path_label = QLabel("未选择文件")
        self.path_label.setWordWrap(True)
        file_layout.addWidget(self.path_label, stretch=1)
        self.select_btn = QPushButton("选择文件")
        self.select_btn.clicked.connect(self._select_file)
        file_layout.addWidget(self.select_btn)
        layout.addLayout(file_layout)

        # 复古纯度
        layout.addWidget(QLabel("复古纯度"))
        self.purity_slider = QSlider(Qt.Orientation.Horizontal)
        self.purity_slider.setRange(0, 100)
        self.purity_slider.setValue(0)
        layout.addWidget(self.purity_slider)
        purity_layout = QHBoxLayout()
        purity_layout.addWidget(QLabel("纯 FC 方波"))
        purity_layout.addStretch()
        purity_layout.addWidget(QLabel("轻微润色"))
        layout.addLayout(purity_layout)

        # 音符简化强度
        layout.addWidget(QLabel("音符简化强度"))
        self.simplify_slider = QSlider(Qt.Orientation.Horizontal)
        self.simplify_slider.setRange(0, 100)
        self.simplify_slider.setValue(50)
        layout.addWidget(self.simplify_slider)
        simplify_layout = QHBoxLayout()
        simplify_layout.addWidget(QLabel("保留原样"))
        simplify_layout.addStretch()
        simplify_layout.addWidget(QLabel("极致简化"))
        layout.addLayout(simplify_layout)

        # 整体音量
        layout.addWidget(QLabel("整体音量"))
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)
        layout.addWidget(self.volume_slider)
        volume_layout = QHBoxLayout()
        volume_layout.addWidget(QLabel("轻"))
        volume_layout.addStretch()
        volume_layout.addWidget(QLabel("响"))
        layout.addLayout(volume_layout)

        # 输出格式
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("输出格式"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["MP3", "WAV"])
        format_layout.addWidget(self.format_combo)
        format_layout.addStretch()
        layout.addLayout(format_layout)

        # 转换按钮
        self.convert_btn = QPushButton("开始转换")
        self.convert_btn.clicked.connect(self._start_convert)
        layout.addWidget(self.convert_btn)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # 状态提示
        self.status_label = QLabel("状态：等待选择文件...")
        layout.addWidget(self.status_label)

        # 打开文件夹按钮
        self.open_folder_btn = QPushButton("打开文件所在文件夹")
        self.open_folder_btn.setEnabled(False)
        self.open_folder_btn.clicked.connect(self._open_result_folder)
        layout.addWidget(self.open_folder_btn)

        self.output_path = ""

    def _select_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 MP3 文件", "", "MP3 文件 (*.mp3)"
        )
        if path:
            self.input_path = path
            self.path_label.setText(path)
            self.status_label.setText("状态：准备就绪")

    def _start_convert(self):
        if not self.input_path:
            QMessageBox.warning(self, "提示", "请先选择 MP3 文件")
            return

        self.convert_btn.setEnabled(False)
        self.open_folder_btn.setEnabled(False)
        self.progress_bar.setValue(0)

        self.worker = ConvertWorker(
            input_path=self.input_path,
            purity=self.purity_slider.value(),
            simplification=self.simplify_slider.value(),
            volume=self.volume_slider.value(),
            output_format=self.format_combo.currentText(),
        )
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self.status_label.setText)
        self.worker.finished_success.connect(self._on_success)
        self.worker.finished_error.connect(self._on_error)
        self.worker.start()

    def _on_success(self, output_path: str):
        self.output_path = output_path
        self.status_label.setText(f"转换完成: {output_path}")
        self.convert_btn.setEnabled(True)
        self.open_folder_btn.setEnabled(True)
        QMessageBox.information(self, "完成", f"已保存到:\n{output_path}")

    def _on_error(self, message: str):
        self.status_label.setText(message)
        self.convert_btn.setEnabled(True)
        self.open_folder_btn.setEnabled(False)
        QMessageBox.critical(self, "错误", message)

    def _open_result_folder(self):
        if self.output_path:
            open_folder(Path(self.output_path))


def run_app():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
```

- [ ] **Step 3: 实现 main.py 入口**

```python
# main.py
from src.ui import run_app

if __name__ == "__main__":
    run_app()
```

- [ ] **Step 4: 提交**

```bash
git add src/worker.py src/ui.py main.py
git commit -m "feat: add PyQt6 GUI and conversion worker"
```

---

## Task 9: 依赖启动检测与入口优化

**Files:**
- Modify: `main.py`
- Modify: `src/utils.py`

- [ ] **Step 1: 在 utils.py 中添加批量依赖检测函数**

```python
# src/utils.py（追加）
REQUIRED_PACKAGES = [
    ("PyQt6", "PyQt6"),
    ("piano_transcription_inference", "piano_transcription_inference"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pydub", "pydub"),
]


def check_all_dependencies() -> list[str]:
    """返回缺失的依赖包名列表。"""
    missing = []
    for import_name, install_name in REQUIRED_PACKAGES:
        if not is_package_installed(import_name):
            missing.append(install_name)
    return missing
```

- [ ] **Step 2: 更新 main.py，启动时检测依赖**

```python
# main.py
import sys

from src.utils import check_all_dependencies, install_packages


def main():
    missing = check_all_dependencies()
    if missing:
        print(f"检测到缺失依赖: {', '.join(missing)}")
        print("正在通过国内镜像自动安装...")
        try:
            install_packages(missing, use_mirror=True)
        except Exception as e:
            print(f"自动安装失败: {e}")
            print("请手动运行: python setup_env.py")
            sys.exit(1)

    from src.ui import run_app
    run_app()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 提交**

```bash
git add main.py src/utils.py
git commit -m "feat: auto-install missing dependencies on startup"
```

---

## Task 10: 集成测试与验收

**Files:**
- Create: `tests/test_integration.py`
- Create: `tests/fixtures/`（放置一个短钢琴 MP3 样本）

- [ ] **Step 1: 准备测试音频**

由于 `piano_transcription_inference` 需要真实钢琴音频做完整集成测试，可准备一个 3–5 秒的短钢琴 MP3 放到 `tests/fixtures/piano_short.mp3`。

如果没有真实样本，可先用 synthesizer 生成一个已知音符的 WAV，假装是输入（但无法测试转录模块）。

- [ ] **Step 2: 写集成测试**

```python
# tests/test_integration.py
import tempfile
from pathlib import Path

import numpy as np

from src.melody_extractor import extract_melody
from src.note_simplifier import simplify_notes
from src.synthesizer import synthesize
from src.utils import export_audio, get_output_path


def test_end_to_end_synthetic():
    # 构造一个简单音符序列，模拟已转录的结果
    notes = [
        (60, 0.0, 0.4, 100),   # C4
        (60, 0.42, 0.8, 100),  # 同音，应合并
        (64, 0.85, 1.0, 100),  # E4
        (67, 1.05, 1.4, 100),  # G4
    ]
    melody = extract_melody(notes)
    simplified = simplify_notes(melody, strength=50)
    audio = synthesize(simplified, duration=2.0, sample_rate=44100, purity=30, volume=80)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "out.wav"
        export_audio(audio, str(out_path), sample_rate=44100)
        assert out_path.exists()
        assert out_path.stat().st_size > 0


def test_get_output_path():
    p = get_output_path("C:/Music/song.mp3", "wav")
    assert p.suffix == ".wav"
    p = get_output_path("C:/Music/song.mp3", "mp3")
    assert p.suffix == ".mp3"
```

- [ ] **Step 3: 运行全部测试**

```bash
python -m pytest tests/ -v
```

Expected: 所有单元测试 PASS，集成测试 PASS（如果没有真实 MP3，可跳过需要转录的测试）。

- [ ] **Step 4: 提交**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests"
```

---

## Task 11: 最终联调与性能验收

- [ ] **Step 1: 运行 GUI**

```bash
python main.py
```

Expected: GUI 正常启动，无崩溃。

- [ ] **Step 2: 选择一首 ≤3 分钟的钢琴 MP3 进行转换**

观察：
- 进度条从 0 到 100 更新
- 状态提示正常切换
- 转换完成弹出保存路径
- 输出文件可播放，听感为纯 8-bit 方波，无钢琴音色残留

- [ ] **Step 3: 测试时长限制**

准备或选择一个超过 10 分钟的 MP3，点击转换，应弹出友好提示并终止。

- [ ] **Step 4: 参数差异验证**

分别用「复古纯度=0」和「复古纯度=100」转换同一文件，确认听感有差异（后者有轻微颤音/润色）。

分别用「音符简化强度=0」和「音符简化强度=100」转换同一文件，确认后者更干净、碎音更少。

- [ ] **Step 5: 提交最终版本**

```bash
git add -A
git commit -m "feat: complete FC 8-bit converter v1"
```

---

## 风险缓解与回退方案

| 风险 | 缓解措施 |
|------|----------|
| `piano_transcription_inference` 安装失败或模型下载失败 | `setup_env.py` 提供明确错误信息；main.py 启动时自动安装；首次下载后模型缓存复用 |
| CPU 推理超过 180 秒 | 后台线程 + 进度条避免界面卡死；转换耗时在日志中记录；接近上限时提示用户 |
| 主旋律提取效果不佳 | GUI 提供「音符简化强度」实时调节；算法参数预留，便于迭代优化 |
| MP3 导出失败 | fallback 到 WAV 并提示用户；WAV 为纯 Python 标准库实现，不依赖外部编码器 |
| 破音/杂音 | 合成阶段做 attack/release ramp + 软限幅；音量默认 80% 留 headroom |

---

## 建议提交节点

1. `chore: initialize project structure and requirements`
2. `feat: add dependency check and mirror-based installer`
3. `feat: add piano transcription module`
4. `feat: add melody extraction module`
5. `feat: add note simplification module`
6. `feat: add FC square wave synthesizer`
7. `feat: add WAV/MP3 audio export`
8. `feat: add PyQt6 GUI and conversion worker`
9. `feat: auto-install missing dependencies on startup`
10. `test: add integration tests`
11. `feat: complete FC 8-bit converter v1`
