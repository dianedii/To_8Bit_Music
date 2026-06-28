from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal

from src.transcriber import transcribe_to_notes
from src.melody_extractor import extract_melody
from src.note_simplifier import simplify_notes
from src.synthesizer import synthesize
from src.pop_synthesizer import synthesize_pop_chip, _synthesize_from_notes, _audio_to_mono_float
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
        mode: str = "pop",
    ):
        super().__init__()
        self.input_path = input_path
        self.purity = purity
        self.simplification = simplification
        self.volume = volume
        self.output_format = output_format
        self.mode = mode

    def _load_audio_numpy(self, audio_segment):
        """将 pydub AudioSegment 转为 numpy 数组。"""
        import numpy as np
        samples = np.array(audio_segment.get_array_of_samples())
        if audio_segment.channels == 2:
            samples = samples.reshape(-1, 2)
            samples = np.transpose(samples)
        max_val = float(2 ** (audio_segment.sample_width * 8 - 1))
        return samples.astype(np.float64) / max_val

    def _ensure_audio_file(self) -> str:
        """若输入是视频文件，先提取音频保存为 MP3，再返回可用的音频路径。"""
        p = Path(self.input_path)
        suffix = p.suffix.lower()
        if suffix not in {".mp4", ".m4v", ".mov", ".mkv", ".avi", ".flv", ".webm"}:
            return self.input_path

        self.status.emit("检测到视频文件，正在提取音频为 MP3...")
        from pydub import AudioSegment
        audio = AudioSegment.from_file(str(p), format=suffix.lstrip("."))

        mp3_path = p.with_suffix(".mp3")
        counter = 1
        base = p.stem
        while mp3_path.exists():
            mp3_path = p.with_name(f"{base}_{counter}.mp3")
            counter += 1

        audio.export(str(mp3_path), format="mp3", bitrate="192k")
        return str(mp3_path)

    def run(self):
        try:
            self.status.emit("正在读取音频并检查时长...")
            self.progress.emit(5)

            # 视频文件先转成 MP3
            audio_path = self._ensure_audio_file()

            from pydub import AudioSegment
            audio = AudioSegment.from_file(audio_path)
            duration_sec = len(audio) / 1000.0
            if duration_sec > 600:
                self.finished_error.emit("曲目时长超过 10 分钟限制。")
                return

            output_path = get_output_path(audio_path, self.output_format)

            if self.mode == "pop":
                self.status.emit("正在转录音符...")
                self.progress.emit(30)
                notes = transcribe_to_notes(self.input_path)

                self.status.emit("正在提取主旋律...")
                self.progress.emit(45)
                melody = extract_melody(notes)
                if not melody:
                    melody = None

                self.status.emit("正在合成流行 8-bit 音频...")
                self.progress.emit(60)
                audio_np = self._load_audio_numpy(audio)
                audio_data = _synthesize_from_notes(
                    notes,
                    _audio_to_mono_float(audio_np),
                    sample_rate=audio.frame_rate,
                    volume=self.volume,
                    melody_notes=melody,
                )

                self.status.emit("正在导出文件...")
                self.progress.emit(95)
                export_audio(audio_data, str(output_path), sample_rate=audio.frame_rate)

                self.progress.emit(100)
                self.finished_success.emit(str(output_path))
                return

            # 纯正 FC 模式：分离 → 识别 → 重合成
            self.status.emit("正在转录音符...")
            self.progress.emit(20)
            notes = transcribe_to_notes(self.input_path)

            self.status.emit("正在提取主旋律...")
            self.progress.emit(50)
            melody = extract_melody(notes)

            self.status.emit("正在简化音符...")
            self.progress.emit(70)
            simplified = simplify_notes(melody, strength=self.simplification)

            self.status.emit("正在合成 FC 8-bit 音频...")
            self.progress.emit(85)
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
