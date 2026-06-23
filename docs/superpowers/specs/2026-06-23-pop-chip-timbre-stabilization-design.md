# 流行 8-bit 音色稳定化设计文档

**日期：** 2026-06-23  
**主题：** 解决流行 8-bit 模式下“音飘”问题  
**状态：** 已确认，待实现

---

## 1. 背景与问题

当前 `src/pop_synthesizer.py` 采用 STFT 逐帧频谱峰值跟踪，多声部独立插值合成。由于每帧峰值频率独立计算，相邻帧之间频率会抖动、声部 ID 会交换，导致合成结果出现“音不稳、音飘”的听感。

本设计将流行 8-bit 模式从“逐帧跟踪”改为 **onset 分段 + 稳定音高提取 + 事件驱动合成**，在保留和声/伴奏的前提下，让音符音高稳定、音符感更强。

---

## 2. 目标

- 消除流行 8-bit 模式下的音高漂移/抖动
- 保留多声部和声结构
- 保持对外接口不变，现有 UI 无需改动
- 新增参数先内部默认，后续根据听感决定是否暴露到 UI

---

## 3. 技术路线

```
MP3 输入
  ↓
AudioSegment → mono float64
  ↓
onset 检测（librosa.onset_detect）
  ↓
按 onset 切分成音符片段
  ↓
每个片段内提取稳定的主导音高（1~n_voices 个）
  ↓
生成复音音符事件: [(pitch, onset, offset, velocity), ...]
  ↓
事件驱动合成（带 ADSR 包络）
  ↓
RMS 包络调制 + 柔和压缩 + 峰值归一化
  ↓
输出 WAV/MP3
```

---

## 4. 架构与模块

主要改动集中在 `src/pop_synthesizer.py`，不新增独立模块，保持对外接口不变。

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
    ...
```

新增内部辅助函数：

| 函数 | 职责 |
|------|------|
| `_audio_to_mono_float` | 音频归一化（保留） |
| `_detect_onsets` | 基于频谱通量检测音符起始点 |
| `_segment_audio` | 按 onset 切分音频片段 |
| `_extract_stable_pitches` | 单一片段内提取稳定主导音高 |
| `_merge_consecutive_notes` | 合并相邻片段的同名持续音 |
| `_synthesize_events` | 复音事件驱动合成 |
| `_rms_envelope` / `_gentle_compress` | 响度调制与动态控制（保留） |

---

## 5. 关键算法

### 5.1 Onset 检测

使用 `librosa.onset_detect` 基于频谱通量检测音符起始点。

参数：
- `hop_length=512`
- `wait=3`：相邻 onset 至少间隔 3 帧，避免过密
- 合并间隔小于 `min_note_duration` 的相邻 onset

### 5.2 片段内稳定音高提取

对每个片段 `[onset_i, onset_{i+1}]`：

1. 截取片段音频并做 STFT
2. 沿时间轴取平均频谱，减少单帧抖动
3. 用 `scipy.signal.find_peaks` 找前 N 个能量峰值
4. 将频率量化到最近的 MIDI 半音
5. 用片段 RMS 能量映射 velocity

每个片段输出最多 `n_voices` 个音符事件。

### 5.3 复音合并

相邻片段若出现相同音高（量化后相同 MIDI），且间隔小于阈值，合并为持续音符，避免每个片段都重新触发 attack。

### 5.4 事件驱动合成

复用现有 `_bandlimited_waveform` 生成基础波形，但改为按事件触发：

- 每个音符事件独立生成波形
- 应用 ADSR 包络：短 attack、轻微 decay、sustain、自然 release
- 多音符线性叠加
- 叠加后用原音频 RMS 包络调制响度，保留原曲动态
- 混合原音频与合成层（`chip_mix`）
- 柔和压缩 + 峰值归一化

---

## 6. 参数映射

当前 UI 参数直接映射：

| UI 参数 | 内部参数 | 说明 |
|---------|---------|------|
| 转换风格 | - | 固定为流行 8-bit |
| 输出格式 | - | MP3 / WAV |

`pop_synthesizer.py` 内部新增参数（先不暴露到 UI）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_voices` | 4 | 同时保留的音高数 |
| `waveform` | `'square'` | 合成波形 |
| `chip_mix` | 0.75 | 芯片层混合比例 |
| `hop_length` | 512 | STFT 帧移 |
| `min_note_duration` | 0.05s | 最短音符时长，过滤过密 onset |
| `pitch_stabilize` | 1.0 | 音高量化强度（1.0=完全量化到半音） |

---

## 7. 数据流示例

输入：一段钢琴/流行 MP3

```
原音频时长: 10s
onset 数量: ~40 个（取决于音乐密度）
片段数: ~40
每片段音高: 最多 4 个
最终事件数: ~100 个
```

合成时每个事件有固定音高，不再逐帧插值，音高稳定性显著提升。

---

## 8. 测试策略

### 8.1 单元测试

- 用已知频率的合成测试音频（如 440Hz + 659Hz 双音）验证输出片段音高稳定
- 验证 onset 检测不会漏掉明显音符起始
- 验证相邻同音合并逻辑

### 8.2 回归测试

- 纯正 FC 模式不受影响
- `synthesize_pop_chip` 返回仍为 `np.ndarray`，shape 与原先一致（立体声）

### 8.3 集成测试

- 用真实 MP3 跑完整流程，输出文件可正常播放
- 主观听感：音高稳定，无飘动感

---

## 9. 风险与回退

| 风险 | 应对 |
|------|------|
| Onset 漏检快速连音 | 调低 `wait`、缩短 `hop_length` |
| 和弦过厚导致浑浊 | 降低 `n_voices` 或做力度平衡 |
| 效果不达预期 | 保留原逐帧峰值代码路径作为参数化 fallback |
| 计算量增加 | onset 检测与片段 STFT 均为轻量操作，整体仍快于钢琴转录 |

---

## 10. 验收标准

- [ ] 流行 8-bit 模式输出音高稳定，无漂移/抖动
- [ ] 保留多声部和声结构
- [ ] 输出仍为可播放 MP3/WAV，响度与商业音频接近
- [ ] 纯正 FC 模式行为不变
- [ ] 现有单元测试与集成测试全部通过

---

## 11. 后续可选扩展

- 将 `waveform`、`n_voices`、`min_note_duration` 等参数暴露到 UI
- 增加噪声通道（noise channel）模拟 NES 鼓点
- 增加琶音器（arpeggio）效果
- 增加滑音（portamento）参数化控制

---

## 12. 相关文件

- `src/pop_synthesizer.py`：主要实现文件
- `src/worker.py`：调用入口，无需改动
- `tests/test_pop_synthesizer.py`：新增/更新测试
- `tests/test_integration.py`：集成验证

---

**License:** MIT
