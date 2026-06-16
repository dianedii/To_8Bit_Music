from unittest.mock import patch, MagicMock
import numpy as np

from src.transcriber import transcribe_to_notes


def test_transcribe_to_notes_format():
    # Mock piano_transcription_inference 的输出
    mock_notes = [
        {"midi_note": 60, "onset_time": 0.0, "offset_time": 0.5, "velocity": 80},
        {"midi_note": 64, "onset_time": 0.5, "offset_time": 1.0, "velocity": 75},
    ]
    with patch('src.transcriber.PianoTranscription') as MockPT, \
         patch('src.transcriber.PTI_AVAILABLE', True), \
         patch('src.transcriber.Path.exists', return_value=True):
        mock_instance = MagicMock()
        mock_instance.transcribe.return_value = {"est_note_events": mock_notes}
        MockPT.return_value = mock_instance
        notes = transcribe_to_notes("dummy.mp3")
    assert len(notes) == 2
    assert notes[0] == (60, 0.0, 0.5, 80)
    assert notes[1] == (64, 0.5, 1.0, 75)
