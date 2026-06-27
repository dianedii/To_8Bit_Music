# 8-bit 芯片音乐转换器

将现代音乐（尤其是钢琴独奏 / 流行乐）一键转换为 8-bit / Chiptune 风格的桌面工具。

- **流行 8-bit**：自动提取主旋律与和声骨架，用多层三角波/方波合成 + 8-bit 噪声鼓点，目标听感贴近参考样例。
- **视频友好**：支持直接传入 MP4 / MKV / MOV 等视频文件，会自动提取音频为 MP3 后再转换。
- **零波形选择**：不再让用户挑选“方波/三角波/锯齿波”，合成器内部根据参考样例风格自动配器。
- 支持 **MP3 / WAV** 输出。

> 本项目采用 Python + PyQt6 构建，所有处理均在本地完成，无需联网（首次运行需下载钢琴转录模型）。

> **注意**：纯正 FC 模式已从 UI 隐藏，相关代码保留但不再被主流程调用。

---

## 功能特性

- 🎹 **流行 8-bit**：基于 `piano_transcription_inference` 钢琴转录，自动分层合成主旋律、低音、亮度层
- 🥁 **自动鼓点**：根据原曲 onset 触发短促带通噪声，模拟 8-bit 鼓/镲
- 🖥️ **图形界面**：基于 PyQt6，无需命令行
- 🎛️ **可调参数**：整体音量、输出格式（所有音色参数内部自动优化）
- 🎵 **输入支持**：MP3 / MP4 / M4V / MOV / MKV / AVI / FLV / WebM（时长 ≤ 10 分钟）
- 💾 **输出支持**：MP3（192kbps） / WAV（16bit / 44.1kHz）
- ⚡ **自动依赖安装**：启动时检测并安装缺失依赖
- ✅ **单元测试覆盖**：核心模块均有测试

---

## 安装

### 环境要求

- Python 3.9+
- Windows / macOS / Linux
- ffmpeg（视频转 MP3 及 MP3 导出依赖，需加入系统 PATH）

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
librosa>=0.10.0
pydub>=0.25.1
```

> 首次进行钢琴转录时，会自动从 Zenodo 下载模型文件（约 170 MB）到 `~/piano_transcription_inference_data/`。

---

## 使用方法

### 启动 GUI

```bash
python main.py
```

界面操作：

1. 点击「选择文件」选择 **MP3 或视频文件**（MP4 / MOV / MKV 等）
2. 调节 **整体音量**（0 ~ 100）
3. 选择 **输出格式**：MP3 / WAV
4. 点击「开始转换」
5. 完成后可点击「打开文件所在文件夹」

> 若输入为视频，程序会先在同目录生成 `{原文件名}.mp3`，再基于该 MP3 进行 8-bit 转换。

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
│   ├── transcriber.py           # 钢琴音频转录（piano_transcription_inference）
│   ├── melody_extractor.py      # 主旋律提取（FC 模式，已隐藏）
│   ├── note_simplifier.py       # 音符简化（FC 模式，已隐藏）
│   ├── synthesizer.py           # FC 方波合成（已隐藏）
│   ├── pop_synthesizer.py       # 流行 8-bit 多层合成引擎
│   ├── pop_melody.py            # 基于 pYIN 的主旋律提取（备用入口）
│   └── utils.py                 # 依赖检测、音频导出、通用工具
├── tests/                       # 单元测试
├── docs/
│   └── superpowers/
│       ├── specs/               # 设计文档
│       └── plans/               # 实现计划
└── README.md                    # 本文档
```

---

## 流行 8-bit 技术路线

```
MP3 / MP4 输入
    ↓
视频文件自动提取为 MP3（pydub + ffmpeg）
    ↓
piano_transcription_inference 转录为音符事件
    ↓
清理与分层：过滤弱音、限制音符长度、按音高分为低音/中音/高音
    ↓
多层合成
  - 三角波低音贝斯
  - 方波主旋律
  - 方波 +12 / +24 八度亮度层
  - 基于 onset 的 8-bit 噪声鼓点
    ↓
总线塑形：高通、高频提升、压缩、峰值归一化、软限幅
    ↓
MP3 / WAV 输出
```

### 音色设计（内部固定，无需用户选择）

| 层级 | 波形 | 作用 | 衰减 |
|------|------|------|------|
| 低音 | 三角波 | 贝斯 / 低频节奏 | 0.5s |
| 主旋律 | 方波 | 中音主旋 | 0.25s |
| 亮度 +12 | 方波 | 增加明亮度 | 0.20s |
| 点缀 +24 | 方波 | 高频空气感 | 0.15s |
| 高音区 | 方波 | 75 键以上音符 | 0.15s |

所有音符均带短 attack 与指数衰减，避免长音拖尾产生的“嗡嗡”低频。

---

## 运行测试

```bash
python -m pytest tests/ -v
```

---

## 参数说明

| 参数 | 范围 | 说明 |
|------|------|------|
| 整体音量 | 0 ~ 100 | 最终输出 gain |
| 输出格式 | MP3 / WAV | MP3 使用 192kbps CBR |

音色、波形、层配比等已由内部引擎根据 `goals/` 参考样例风格自动优化，UI 不再暴露。

---

## 注意事项

1. **时长限制**：单首曲目 ≤ 10 分钟，超出会提示并终止。
2. **输出覆盖保护**：当输出格式与输入格式同为 MP3 时，输出文件名会自动添加 `_8bit` 后缀，避免覆盖原文件。
3. **依赖安装**：启动时会自动检测缺失依赖并通过国内镜像安装。
4. **模型下载**：首次转录需下载钢琴转录模型，请保持网络畅通。

---

## 致谢

- `piano_transcription_inference`：钢琴音频转录模型
- `PyQt6`：图形界面框架
- `numpy` / `scipy` / `pydub` / `librosa`：数值计算与音频处理

---

## License

MIT License
