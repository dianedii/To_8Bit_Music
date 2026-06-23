# 流行 8-bit 主旋律提取与 FC 模式隐藏设计文档

**日期：** 2026-06-23  
**主题：** 流行 8-bit 模式改为音符事件驱动的主旋律提取，并隐藏 FC 模式  
**状态：** 已确认，待实现

---

## 1. 背景与问题

当前实现已将流行 8-bit 模式从逐帧频谱峰值跟踪改为 onset 分段 + 稳定音高提取 + 事件驱动合成。但测试反馈显示：当输入为钢琴独奏或纯音乐时，多个声部（主旋律 + 副旋律/伴奏）同时被转换，导致输出听感杂乱。

本设计将流行 8-bit 模式进一步改为**以主旋律为核心、最多保留一条和声点缀**的单声部为主输出，并**隐藏纯正 FC 模式**（代码保留但不再暴露）。

---

## 2. 目标

- 流行 8-bit 模式只输出稳定的主旋律，显著降低杂乱感
- 保留少量和声点缀，避免输出过于空洞
- 针对钢琴独奏 / 纯音乐优化
- 隐藏 FC 模式 UI 入口，相关模块标记为 deprecated
- 保持输出格式不变：立体声 float32，shape `(2, N)`

---

## 3. 技术路线

```
MP3 输入
  ↓
AudioSegment → mono float64
  ↓
pYIN 基频跟踪 → 连续音符事件 (pitch, onset, offset, velocity)
  ↓
拆分为候选旋律线（不重叠连续单音链条）
  ↓
按加权规则打分 + 硬性过滤
  ↓
选出主旋律线 + 选一条和声点缀线
  ↓
事件驱动合成（主旋律 + 和声点缀）
  ↓
RMS 包络调制 + chip_mix 混合 + 压缩 + 归一化
  ↓
输出 WAV/MP3
```

---

## 4. 架构与模块

### 4.1 主要改动

修改 `src/pop_synthesizer.py`：

| 函数 | 职责 |
|------|------|
| `_pyin_to_notes` | 用 librosa.pyin 将音频转为音符事件 |
| `_split_candidate_lines` | 将重叠音符拆成不重叠的连续单音候选线 |
| `_score_melody_line` | 按四维加权规则给候选线打分 |
| `_apply_hard_filters` | 硬性过滤：碎音线、重复伴奏线淘汰 |
| `_extract_main_melody` | 返回最高分主旋律线 |
| `_extract_harmony_voice` | 返回一条和声点缀线 |
| `_synthesize_events` | 事件驱动合成（已存在，复用） |
| `synthesize_pop_chip` | 主流程，改为只合成主旋律 + 和声点缀 |

### 4.2 FC 模式隐藏

- `src/ui.py`：移除“纯正 FC”下拉选项，仅保留“流行 8-bit”
- `src/worker.py`：固定调用 pop-chip 流程
- `src/transcriber.py`、`melody_extractor.py`、`note_simplifier.py`、`synthesizer.py`：**保留文件，标记 deprecated**
- `README.md` 与 `docs/`：增加 FC 模式隐藏说明

---

## 5. 关键算法

### 5.1 pYIN 音符化

使用 `librosa.pyin` 获取基频序列：

```python
f0, voiced_flag, voiced_prob = librosa.pyin(
    audio,
    fmin=librosa.note_to_hz('C2'),
    fmax=librosa.note_to_hz('C7'),
    sr=sample_rate,
    hop_length=hop_length,
)
```

将连续的 voiced 段落按音高稳定性切分为音符事件：
- 同一音高（量化到半音）持续超过阈值 → 一个音符
- 音高变化 → 新音符
- 静音 / unvoiced → 音符边界

输出：`[(pitch, onset, offset, velocity), ...]`

### 5.2 候选旋律线拆分

将音符事件按时间顺序排列，构建不重叠的连续单音链条：

1. 按 onset 排序
2. 当前音符与下一条候选线中最后一个音符的 offset 重叠或间隔小于阈值时，不能加入同一条线
3. 每个音符只能选择加入一条线，优先加入最近结束的那条线
4. 最终得到若干条 candidate line

### 5.3 主旋律评分

对每条 candidate line 按四维打分，总分最高者为主旋律。

#### 5.3.1 歌唱平滑度（45%）

```
score_smooth = w1 * small_interval_ratio
             + w2 * rest_ratio
             + w3 * long_note_ratio
             - w4 * sixteenth_note_ratio
             - w5 * large_jump_ratio
             - w6 * no_rest_penalty
```

- `small_interval_ratio`：相邻音程 ≤ 5 半音的占比
- `rest_ratio`：有休止符分句的占比
- `long_note_ratio`：四分音符及以上长度占比
- `sixteenth_note_ratio`：十六分音符占比
- `large_jump_ratio`：相邻音程 > 8 度（如 12 半音以上）的占比
- `no_rest_penalty`：完全没有休止时的固定惩罚

#### 5.3.2 乐句重复度（30%）

- 将候选线按小节长度切片（默认每小节 2 秒，即约 120 BPM）
- 用轮廓相似度（音程走向）比较各片段
- 完整 4 小节片段重复次数越多，分数越高
- 无限循环的完全重复音型会在硬性过滤中淘汰

#### 5.3.3 音域居中（15%）

- 目标音域：C3 ~ F5（MIDI 48 ~ 77）
- 计算音符落在该区间内的比例
- 比例越高，分数越高

#### 5.3.4 力度稳定性（10%）

- 计算相邻音符 velocity 差值的方差
- 方差越小，分数越高
- 乐句开头稍重、结尾稍轻的模式额外加分

### 5.4 硬性过滤

候选线在评分前先经过硬性过滤，被淘汰者不参与排名：

1. **碎音线过滤**：如果一条线中 80% 以上音符时长短于 0.1s（约十六分音符 @120BPM），判定为装饰音/副旋律，淘汰。
2. **重复伴奏过滤**：如果一条线连续 8 小节以上音符形状完全一致（音程序列相同），判定为分解和弦伴奏，淘汰。

### 5.5 和声点缀线选择

选出主旋律后，从剩余候选线中选一条和声点缀线：

1. 排除已被硬性过滤的线
2. 优先选择与主旋律同时发声、音程关系以 3 度/5 度/8 度为主的线
3. 限制和声音数量：每个时刻最多 1 个和声音
4. 和声音量为主旋律的 50% ~ 70%，避免喧宾夺主

### 5.6 合成

将主旋律线与和声点缀线合并为最终音符事件列表，调用 `_synthesize_events` 合成，后续 RMS 包络调制、chip_mix、压缩、归一化流程保持不变。

---

## 6. 参数映射

### 6.1 内部参数（暂不暴露到 UI）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `fmin` | C2 (~65Hz) | pYIN 最低频率 |
| `fmax` | C7 (~2093Hz) | pYIN 最高频率 |
| `hop_length` | 512 | STFT 帧移 |
| `min_note_duration` | 0.05s | 最短音符时长 |
| `max_harmony_voices` | 1 | 和声点缀最大声部数 |
| `harmony_volume_ratio` | 0.6 | 和声相对主旋律音量 |

### 6.2 UI 参数调整

移除以下参数：
- 转换风格下拉框（只剩“流行 8-bit”）
- 复古纯度（FC 参数）
- 音符简化强度（FC 参数）

保留：
- 波形选择（方波/三角波/锯齿波）
- 芯片混合比例 `chip_mix`
- 整体音量
- 输出格式

---

## 7. 数据流示例

输入：一首 30 秒钢琴独奏 MP3

```
pYIN 输出：约 80 ~ 150 个音符事件
候选线拆分：约 5 ~ 12 条候选线
硬性过滤：淘汰 2 ~ 4 条碎音/伴奏线
主旋律评分：选出 1 条最高分线
和声点缀：选出 1 条线（如存在）
最终合成事件：约 60 ~ 100 个音符
```

---

## 8. 测试策略

### 8.1 单元测试

- `_pyin_to_notes`：用已知频率的合成音频验证输出音符音高正确
- `_split_candidate_lines`：验证重叠音符正确拆分为不重叠线
- `_score_melody_line`：用构造的候选线验证主旋律得分最高
- `_apply_hard_filters`：验证碎音线和重复线被淘汰
- `_extract_harmony_voice`：验证和声线与主旋律音程合理

### 8.2 集成测试

- 用合成钢琴音频跑完整流程，输出 WAV 文件
- 验证输出为单声部为主 + 少量和声
- 验证输出 shape 为 `(2, N)`，dtype float32

### 8.3 回归测试

- FC 相关模块（transcriber, melody_extractor, note_simplifier, synthesizer）仍可正常 import
- 原有 utils 测试不受影响

---

## 9. 风险与回退

| 风险 | 应对 |
|------|------|
| pYIN 在复杂伴奏上效果差 | 本设计主攻钢琴独奏/纯音乐；复杂伴奏可后续加人声/鼓分离预处理 |
| 乐句重复度计算对小节长度敏感 | 默认 120 BPM，可后续根据 onset 密度自适应估算 BPM |
| 和声点缀导致仍显杂乱 | 提供参数 `max_harmony_voices` 默认 1，可设为 0 实现纯单声部 |
| 用户怀念 FC 模式 | 代码保留，可通过命令行参数或配置文件重新启用 |

---

## 10. FC 模式隐藏说明

### 10.1 UI 层面

- 主界面“转换风格”下拉框只保留“流行 8-bit”
- 移除 FC 模式相关参数滑块（复古纯度、音符简化强度）

### 10.2 代码层面

- `src/transcriber.py`、`src/melody_extractor.py`、`src/note_simplifier.py`、`src/synthesizer.py` 保留原文件
- 在文件顶部或 README 中标注：
  ```
  # 已弃用（Deprecated）：纯正 FC 模式已隐藏，本模块不再被主流程调用。
  ```

### 10.3 文档层面

- `README.md` 更新：移除 FC 模式介绍，增加隐藏说明
- 本设计文档记录隐藏原因与回退方式

---

## 11. 验收标准

- [ ] 流行 8-bit 模式输出以主旋律为主，杂乱感显著降低
- [ ] 保留适量和声点缀，输出不过于空洞
- [ ] 钢琴独奏输入效果良好
- [ ] FC 模式从 UI 移除
- [ ] FC 相关模块代码保留并可正常 import
- [ ] 全部测试通过
- [ ] README 与文档更新完成

---

## 12. 相关文件

- `src/pop_synthesizer.py`：主要实现文件
- `src/ui.py`：移除 FC 模式 UI 选项
- `src/worker.py`：固定调用 pop-chip 流程
- `src/transcriber.py`、`src/melody_extractor.py`、`src/note_simplifier.py`、`src/synthesizer.py`：保留并标记 deprecated
- `tests/test_pop_synthesizer.py`：新增/更新测试
- `tests/test_integration.py`：更新集成测试
- `README.md`：更新功能说明
- `docs/superpowers/specs/2026-06-23-pop-chip-melody-only-design.md`：本文档

---

**License:** MIT
