import os
import tempfile
import urllib.request
from pathlib import Path
from typing import List, Tuple

try:
    from piano_transcription_inference import PianoTranscription
    import librosa
    PTI_AVAILABLE = True
except ImportError:
    PianoTranscription = None
    librosa = None
    PTI_AVAILABLE = False


MODEL_URL = (
    "https://zenodo.org/record/4034264/files/"
    "CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
)
MODEL_FILENAME = "note_F1=0.9677_pedal_F1=0.9186.pth"
MODEL_MIN_SIZE = 1.6e8


def _get_default_checkpoint_path() -> Path:
    return Path.home() / "piano_transcription_inference_data" / MODEL_FILENAME


def _download_checkpoint(checkpoint_path: Path) -> None:
    """使用 urllib 下载模型文件（兼容无 wget 的 Windows 环境）。"""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"正在下载模型到: {checkpoint_path}")
    print(f"模型 URL: {MODEL_URL}")

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
        print(f"\r下载进度: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} / {total_size / 1024 / 1024:.1f} MB)", end="")

    try:
        urllib.request.urlretrieve(MODEL_URL, str(checkpoint_path), _progress)
        print()  # newline after progress
    except Exception as e:
        if checkpoint_path.exists():
            checkpoint_path.unlink(missing_ok=True)
        raise RuntimeError(f"模型下载失败: {e}")


def ensure_model_checkpoint(checkpoint_path: Path | None = None) -> Path:
    """确保模型文件存在，不存在则下载。"""
    path = checkpoint_path or _get_default_checkpoint_path()
    needs_download = True
    if path.exists():
        try:
            needs_download = os.path.getsize(path) < MODEL_MIN_SIZE
        except OSError:
            needs_download = True
    if needs_download:
        _download_checkpoint(path)
    return path


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

    checkpoint_path = ensure_model_checkpoint()
    transcriptor = PianoTranscription(
        device="cpu",
        checkpoint_path=str(checkpoint_path),
    )

    # piano_transcription_inference 的 transcribe 需要 numpy 音频数组和 midi 输出路径
    audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        midi_path = tmp.name

    try:
        result = transcriptor.transcribe(audio, midi_path)
        note_events = result.get("est_note_events", [])
    finally:
        Path(midi_path).unlink(missing_ok=True)

    notes = []
    for event in note_events:
        pitch = int(event["midi_note"])
        onset = float(event["onset_time"])
        offset = float(event["offset_time"])
        velocity = int(event.get("velocity", 80))
        notes.append((pitch, onset, offset, velocity))

    return notes
