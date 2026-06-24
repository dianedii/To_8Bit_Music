# 8-bit 芯片音乐转换器

将现代音乐（尤其是钢琴独奏/流行乐）一键转换为 8-bit / Chiptune 风格的桌面工具。

- **流行 8-bit**：自动提取主旋律 + 少量和声，用三角波/方波/锯齿波/正弦波合成，保留流行音乐美感。
- 支持 MP3 / WAV 输出。
- 可调整芯片音色占比与波形。

> 本项目采用 Python + PyQt6 构建，所有处理均在本地完成，无需联网。

> **注意**：纯正 FC 模式已从 UI 隐藏，相关代码保留但不再被主流程调用。

---

## 功能特性

- 🎹 **流行 8-bit**：自动提取主旋律 + 少量和声，用三角波/方波/锯齿波/正弦波合成
- 🖥️ **图形界面**：基于 PyQt6，无需命令行
- 🎛️ **可调参数**：波形选择、芯片音色占比、整体音量、输出格式
- 🎵 **输入支持**：MP3（时长 ≤ 10 分钟）
- 💾 **输出支持**：MP3（192kbps） / WAV（16bit / 44.1kHz）
- ⚡ **自动依赖安装**：启动时检测并安装缺失依赖
- ✅ **单元测试覆盖**：核心模块均有测试

---

## 技术路线

### 流行 8-bit 模式

```
MP3 输入
    ↓
STFT 频谱分析
    ↓
多声部频谱峰值跟踪
    ↓
旋律提取 + 和声保留
    ↓
波形合成（三角波 / 方波 / 锯齿波 / 正弦波）
    ↓
芯片音色占比混合
    ↓
RMS 包络调制 + 柔和压缩
    ↓
MP3 / WAV 输出
```

---

## 安装

### 环境要求

- Python 3.9+
- Windows / macOS / Linux

### 安装依赖

```bash
# 方式一：运行一键安装脚本
python setup_env.py

# 方式二：手动安装
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

`requirements.txt` 包含：

```text
PyQt6>=6.4.0
piano_transcription_inference>=0.2
numpy>=1.23.0
scipy>=1.9.0
pydub>=0.25.1
```

---

## 使用方法

### 启动 GUI

```bash
python main.py
```

界面操作：

1. 点击「选择文件」选择 MP3
2. 调节参数：
   - **波形选择**：三角波 / 方波 / 锯齿波 / 正弦波
   - **芯片音色占比**：0 = 原声，100 = 纯芯片音色
   - **整体音量**：0 ~ 100
3. 选择输出格式：MP3 / WAV
4. 点击「开始转换」
5. 完成后可点击「打开文件所在文件夹」

### 命令行

目前主要通过 GUI 使用。如需命令行调用，可参考 `src/worker.py` 自行封装。

---

## 项目结构

```
8Bit-New/
├── main.py                      # 程序入口
├── setup_env.py                 # 依赖检测与一键安装
├── requirements.txt             # 依赖清单
├── src/
│   ├── __init__.py
│   ├── ui.py                    # PyQt6 主窗口
│   ├── worker.py                # QThread 后台转换线程
│   ├── transcriber.py           # 钢琴音频转录（FC 模式，已隐藏）
│   ├── melody_extractor.py      # 主旋律提取
│   ├── note_simplifier.py       # 音符简化（FC 模式，已隐藏）
│   ├── synthesizer.py           # FC 方波合成（已隐藏）
│   ├── pop_synthesizer.py       # 流行 8-bit 多声部合成
│   └── utils.py                 # 依赖检测、音频导出、通用工具
├── tests/                       # 单元测试
├── docs/
│   └── superpowers/
│       ├── specs/2026-06-16-fc-8bit-converter-design.md
│       └── plans/2026-06-16-fc-8bit-converter-plan.md
└── README.md                    # 本文档
```

---

## 运行测试

```bash
python -m pytest tests/ -v
```

---

## 参数说明

| 参数 | 范围 | 说明 |
|------|------|------|
| 波形选择 | 三角波 / 方波 / 锯齿波 / 正弦波 | 选择合成器基础波形 |
| 芯片音色占比 | 0 ~ 100 | 0 = 原声，100 = 纯芯片音色 |
| 整体音量 | 0 ~ 100 | 最终输出 gain |
| 输出格式 | MP3 / WAV | MP3 使用 192kbps CBR |

---

## 注意事项

1. **时长限制**：单首曲目 ≤ 10 分钟，超出会提示并终止。
2. **输出覆盖保护**：当输出格式与输入格式同为 MP3 时，输出文件名会自动添加 `_8bit` 后缀，避免覆盖原文件。
3. **依赖安装**：启动时会自动检测缺失依赖并通过国内镜像安装。

---

## 致谢

- `piano_transcription_inference`：钢琴音频转录模型
- `PyQt6`：图形界面框架
- `numpy` / `scipy` / `pydub` / `librosa`：数值计算与音频处理

---

## License

MIT License
