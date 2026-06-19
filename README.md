# 8-bit 芯片音乐转换器

将现代音乐（尤其是钢琴独奏/流行乐）一键转换为 8-bit / Chiptune 风格的桌面工具。支持两种风格：

- **流行 8-bit**：兼顾 8-bit 电子味与流行乐的和声美感，适合钢琴曲、流行歌曲。
- **纯正 FC**：严格遵循 FC/NES 芯片音乐规则，仅保留单声部主旋律，输出标准方波。

> 本项目采用 Python + PyQt6 构建，所有处理均在本地完成，无需联网。

---

## 功能特性

- 🎹 **两种转换风格**：流行 8-bit（复音合成） / 纯正 FC（单声部主旋律）
- 🖥️ **图形界面**：基于 PyQt6，无需命令行
- 🎛️ **可调参数**：复古纯度、音符简化强度、整体音量、输出格式
- 🎵 **输入支持**：MP3（时长 ≤ 10 分钟）
- 💾 **输出支持**：MP3（192kbps） / WAV（16bit / 44.1kHz）
- ⚡ **自动依赖安装**：启动时检测并安装缺失依赖
- ✅ **单元测试覆盖**：核心模块均有测试

---

## 两种模式对比

| 特性 | 流行 8-bit | 纯正 FC |
|------|-----------|---------|
| 处理思路 | 频谱峰值跟踪 + 多声部方波合成 | 钢琴转录 → 主旋律提取 → 单声部方波 |
| 和声/伴奏 | 保留 | 剔除 |
| 音色 | 带限方波，柔和电子感 | 标准 50% 占空比方波 |
| 适合音乐 | 钢琴独奏、流行歌曲、复音音乐 | 追求纯正红白机芯片质感 |
| 处理速度 | 快（无需 AI 推理） | 较慢（依赖钢琴转录模型） |

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
带限方波合成（6 声部）
    ↓
RMS 包络调制 + 柔和压缩
    ↓
MP3 / WAV 输出
```

### 纯正 FC 模式

```
MP3 钢琴音频
    ↓
piano_transcription_inference 钢琴转录模型
    ↓
MIDI 音符事件
    ↓
主旋律提取（时序聚类 + 打分）
    ↓
音符简化（碎音过滤 / 同音合并 / 装饰音移除）
    ↓
FC 方波合成
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

> 首次使用**纯正 FC 模式**时，需要下载约 165MB 的钢琴转录模型。程序会自动通过 `urllib` 下载（Windows 无需 wget）。

---

## 使用方法

### 启动 GUI

```bash
python main.py
```

界面操作：

1. 点击「选择文件」选择 MP3
2. 选择「转换风格」：
   - **流行 8-bit**：默认推荐，兼顾美感与电子味
   - **纯正 FC**：纯正红白机风格
3. 调节参数（纯正 FC 模式下生效）：
   - 复古纯度：0 = 纯方波，100 = 轻微 vibrato + release
   - 音符简化强度：0 = 保留原样，100 = 极致简化
   - 整体音量：0 ~ 100
4. 选择输出格式：MP3 / WAV
5. 点击「开始转换」
6. 完成后可点击「打开文件所在文件夹」

### 命令行（仅流行 8-bit 示例）

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
│   ├── transcriber.py           # 钢琴音频转录（FC 模式）
│   ├── melody_extractor.py      # 主旋律提取（FC 模式）
│   ├── note_simplifier.py       # 音符简化（FC 模式）
│   ├── synthesizer.py           # FC 方波合成
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

### 流行 8-bit 模式

| 参数 | 说明 |
|------|------|
| 转换风格 | 选择「流行 8-bit」 |
| 输出格式 | MP3 / WAV |

### 纯正 FC 模式

| 参数 | 范围 | 说明 |
|------|------|------|
| 复古纯度 | 0 ~ 100 | 0=纯方波，100=轻微 vibrato + 柔和 release |
| 音符简化强度 | 0 ~ 100 | 控制碎音过滤、同音合并、装饰音移除强度 |
| 整体音量 | 0 ~ 100 | 最终输出 gain |
| 输出格式 | MP3 / WAV | MP3 使用 192kbps CBR |

---

## 注意事项

1. **纯正 FC 模式首次使用**：需要下载约 165MB 转录模型，请保持网络畅通。
2. **时长限制**：单首曲目 ≤ 10 分钟，超出会提示并终止。
3. **输出覆盖保护**：当输出格式与输入格式同为 MP3 时，输出文件名会自动添加 `_8bit` 后缀，避免覆盖原文件。
4. **依赖安装**：启动时会自动检测缺失依赖并通过国内镜像安装。

---

## 致谢

- `piano_transcription_inference`：钢琴音频转录模型
- `PyQt6`：图形界面框架
- `numpy` / `scipy` / `pydub` / `librosa`：数值计算与音频处理

---

## License

MIT License
