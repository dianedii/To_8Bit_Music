from unittest.mock import patch, MagicMock
import sys

def test_transcribe_to_notes_format():
    # Mock piano_transcription_inference in sys.modules BEFORE importing src.transcriber
    mock_piano_transcription_inference = MagicMock()
    mock_PianoTranscription = MagicMock()
    mock_piano_transcription_inference.PianoTranscription = mock_PianoTranscription

    mock_notes = [
        {"midi_note": 60, "onset_time": 0.0, "offset_time": 0.5, "velocity": 80},
        {"midi_note": 64, "onset_time": 0.5, "offset_time": 1.0, "velocity": 75},
    ]

    mock_instance = MagicMock()
    mock_instance.transcribe.return_value = {"est_note_events": mock_notes}
    mock_PianoTranscription.return_value = mock_instance

    with patch.dict('sys.modules', {'piano_transcription_inference': mock_piano_transcription_inference}):
        # Import inside the mock context so the try/except import resolves to the mock
        from src.transcriber import transcribe_to_notes
        with patch('src.transcriber.Path.exists', return_value=True):
            notes = transcribe_to_notes("dummy.mp3")

    assert len(notes) == 2
    assert notes[0] == (60, 0.0, 0.5, 80)
    assert notes[1] == (64, 0.5, 1.0, 75)
