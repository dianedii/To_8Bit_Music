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
