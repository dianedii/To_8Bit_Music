# 流行 8-bit 音色柔化与音高定位精度优化设计文档

**日期：** 2026-06-24  
**主题：** 让流行 8-bit 输出更丝滑、主旋律音高定位更准确  
**状态：** 已确认，待实现

---

## 1. 背景与问题

当前 `src/pop_synthesizer.py` 仍采用 onset 分段 + 频谱峰值跟踪的方式提取音高，并同时合成多个声部。这带来两个突出问题：

1. **电子声偏粗**：多声部叠加 + 方波谐波丰富，导致输出听感硬、杂。
2. **主旋律音定位不准**：片段内平均频谱峰值后再量化到 MIDI，容易偏离真实音符中心；多个声部同时转换也让主旋律被淹没。

用户希望音色往“稍微丝滑”方向优化，同时提升主旋律的音高定位准确度。

---

## 2. 目标

- 流行 8-bit 模式只输出稳定的主旋律 + 最多一条和声点缀。
- 默认音色更柔和、顺滑，减少“粗硬”感。
- 主旋律音符边界和音高更接近真实演奏。
- 保留 8-bit 核心味道，UI 仍允许用户切换更硬的波形。
- 保持输出格式不变：立体声 float32，shape `(2, N)`。

---

## 3. 技术路线

```
MP3 输入
  ↓
AudioSegment → mono float64
  ↓
pYIN 基频跟踪 → 连续音符事件 (pitch, onset, offset, velocity)
  ↓
对 f0 做轻量中值滤波，去除毛刺
  ↓
拆分为候选旋律线（不重叠连续单音链条）
  ↓
按加权规则打分 + 硬性过滤
  ↓
选出 1 条主旋律 + 最多 1 条和声点缀
  ↓
事件驱动合成
  - 默认三角波
  - legato：短间隔音符直接连音，不重新触发 attack
  - 柔和 ADSR：attack 10ms，release 40ms
  - 轻量低通滤波：cutoff ~8kHz
  ↓
chip_mix 混合原音频（默认 0.6）
  ↓
压缩 + 峰值归一化
  ↓
输出 WAV/MP3
```

---

## 4. 架构与模块

### 4.1 主要改动

重写 `src/pop_synthesizer.py`：

| 函数 | 职责 |
|------|------|
| `_smooth_f0` | 对 pYIN 输出的 f0 做中值滤波，去除单帧毛刺 |
| `_pyin_to_notes` | 用 librosa.pyin 将音频转为音符事件（复用/扩展 `src/pop_melody.py`） |
| `_split_candidate_lines` | 将重叠音符拆成不重叠的连续单音候选线 |
| `_score_melody_line` | 按四维加权规则给候选线打分 |
| `_apply_hard_filters` | 硬性过滤：碎音线、重复伴奏线淘汰 |
| `_extract_main_melody` | 返回最高分主旋律线 |
| `_extract_harmony_voice` | 返回一条和声点缀线 |
| `_apply_legato` | 合并间隔 ≤ 阈值的相邻同/近音，减少断点 |
| `_synthesize_events` | 事件驱动合成，支持 legato/ADSR/低通 |
| `_apply_lowpass` | 轻量低通滤波，软化高频 |
| `synthesize_pop_chip` | 主流程，串联上述步骤 |

### 4.2 UI 调整

`src/ui.py`：

- 隐藏“纯正 FC”选项（保留代码）。
- 移除“复古纯度”“音符简化强度”滑块。
- 保留并暴露：
  - 波形选择：方波 / 三角波 / 锯齿波 / 正弦波
  - `chip_mix` 滑块（0~100，默认 60）
  - 整体音量、输出格式

### 4.3 Worker 调整

`src/worker.py`：

- 将 UI 上的 `waveform`、`chip_mix` 透传给 `synthesize_pop_chip`。
- 流行模式下固定走 pop-chip 流程。

---

## 5. 关键算法

### 5.1 f0 平滑

pYIN 输出的 f0 序列偶尔会有单帧跳变。在进入音符化之前，先对 voiced 帧做中值滤波：

```python
f0_smooth = scipy.signal.medfilt(f0, kernel_size=3)
```

只替换 voiced 帧，unvoiced 帧保持 NaN。这样不会引入延迟，但能去掉大部分毛刺。

### 5.2 音符化改进

在现有 `_pyin_to_notes` 基础上：

- 使用平滑后的 f0。
- `pitch_quantize_strength=1.0`：将每帧 f0 完全量化到最近的 MIDI 半音。
- 同一 MIDI 音高连续出现且时长 ≥ `min_note_duration` 才形成一个音符。
- 音高变化或进入 unvoiced 时切分音符。

### 5.3 主旋律提取

复用 `src/pop_melody.py` 的候选线拆分、评分、过滤、主旋律提取、和声提取逻辑。

### 5.4 Legato（连音）

合成前对主旋律线做一次连音处理：

```python
def _apply_legato(notes, threshold=0.05):
    # 相邻音符间隔 ≤ threshold 秒时，直接延长前一个音符到下一个音符的 offset
    # 不重新触发 attack
```

只对主旋律应用 legato，和声点缀保持独立触发，避免和声被主旋律“吞掉”。

### 5.5 合成柔化

#### 5.5.1 默认三角波

三角波的谐波按 `1/n²` 衰减，比方波柔和。UI 仍允许用户切回方波获得更硬的 8-bit 味道。

#### 5.5.2 新增正弦波选项

正弦波无谐波，最平滑，可作为“极致丝滑”选项。

#### 5.5.3 ADSR 调整

| 阶段 | 旧值 | 新值 |
|------|------|------|
| attack | 5ms | 10ms |
| decay | 15ms | 15ms |
| sustain | 0.85 | 0.85 |
| release | 20ms | 40ms |

#### 5.5.4 低通滤波

合成完成后对芯片层做 6dB/oct 低通：

```python
b, a = scipy.signal.butter(1, cutoff / (sr / 2), btype='low')
synth = scipy.signal.filtfilt(b, a, synth)
```

cutoff 默认 8000 Hz，保留明亮感但削掉刺耳高频。

### 5.6 混合与动态

- `chip_mix` 默认从 0.75 降到 0.6，让原音频占比更高，芯片层作为“染色层”。
- RMS 包络调制、柔和压缩、峰值归一化流程保留。

---

## 6. 参数映射

### 6.1 新增/调整 UI 参数

| UI 控件 | 内部参数 | 默认值 | 说明 |
|---------|----------|--------|------|
| 波形选择 | `waveform` | `'triangle'` | 方波/三角波/锯齿波/正弦波 |
| 芯片混合 | `chip_mix` | 0.6 | 0~1，芯片层占比 |
| 整体音量 | `volume` | 0.8 | 最终响度 |

### 6.2 内部参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hop_length` | 512 | pYIN 帧移 |
| `min_note_duration` | 0.05s | 最短音符时长 |
| `pitch_quantize_strength` | 1.0 | 1.0=完全量化到半音 |
| `f0_median_size` | 3 | f0 中值滤波窗口 |
| `legato_threshold` | 0.05s | 连音间隔阈值 |
| `lowpass_cutoff` | 8000Hz | 低通截止频率 |
| `max_harmony_voices` | 1 | 和声点缀最大声部数 |
| `harmony_volume_ratio` | 0.6 | 和声相对主旋律音量 |

---

## 7. 数据流示例

输入：一首 30 秒钢琴独奏 MP3

```
pYIN 输出：约 80 ~ 150 个音符事件
f0 平滑后毛刺减少，音符边界更清晰
候选线拆分：约 5 ~ 12 条候选线
硬性过滤：淘汰 2 ~ 4 条碎音/伴奏线
主旋律评分：选出 1 条最高分线
legato 合并：主旋律中相邻短间隔音符连接
和声点缀：选出 1 条线（如存在）
最终合成事件：约 60 ~ 100 个音符
三角波 + 柔和 ADSR + 低通滤波 → 更丝滑输出
```

---

## 8. 测试策略

### 8.1 单元测试

- `_smooth_f0`：验证单帧毛刺被平滑，unvoiced 帧保持 NaN。
- `_pyin_to_notes`：用 440Hz 合成音频验证输出为 A4（MIDI 69）。
- `_apply_legato`：验证间隔 ≤ 阈值的相邻音符被合并。
- `_apply_lowpass`：验证输出高频能量低于输入。
- `_synthesize_events`：三角波输出比方波高频谐波少。

### 8.2 集成测试

- 用合成钢琴音频跑完整流程，输出 WAV 文件。
- 验证输出为单声部为主 + 少量和声。
- 验证输出 shape 为 `(2, N)`，dtype float32。

### 8.3 回归测试

- FC 相关模块仍可正常 import。
- 原有 utils 测试不受影响。
- `tests/test_pop_melody.py` 全部通过。

---

## 9. 风险与回退

| 风险 | 应对 |
|------|------|
| pYIN 速度慢于 STFT 峰值 | 已在可接受范围；如后续成为瓶颈，可对长音频分段处理 |
| 三角波/正弦波让 8-bit 味道太淡 | UI 保留方波选项，用户可自行切换 |
| legato 过度合并导致音符糊掉 | 阈值默认 50ms，仅合并非常短的间隔；可后续暴露参数 |
| 复杂伴奏下主旋律提取不稳 | 主攻钢琴独奏/纯音乐；后续可考虑加前奏/间奏检测 |
| 低通滤波让声音发闷 | cutoff 默认 8kHz 较高，保留明亮感；用户可通过波形选择补偿 |

---

## 10. 验收标准

- [ ] 流行 8-bit 模式默认音色比方波旧版明显柔和。
- [ ] 主旋律音高定位更准确，无飘忽感。
- [ ] 只输出主旋律 + 最多一条和声，杂乱感降低。
- [ ] UI 隐藏 FC 模式，暴露波形选择和 chip_mix。
- [ ] 全部测试通过（目标 ≥ 63 个）。
- [ ] README 与文档更新完成。

---

## 11. 相关文件

- `src/pop_synthesizer.py`：主要实现文件，大改。
- `src/pop_melody.py`：复用主旋律提取逻辑。
- `src/ui.py`：调整 UI 控件。
- `src/worker.py`：透传新参数。
- `tests/test_pop_synthesizer.py`：更新/新增测试。
- `tests/test_integration.py`：更新集成测试。
- `README.md`：更新功能说明。
- `docs/superpowers/specs/2026-06-24-pop-chip-timbre-smoothing-design.md`：本文档。

---

**License:** MIT
