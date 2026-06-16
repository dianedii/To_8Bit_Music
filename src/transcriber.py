from pathlib import Path
from typing import List, Tuple

try:
    from piano_transcription_inference import PianoTranscription
    PTI_AVAILABLE = True
except ImportError:
    PianoTranscription = None
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
