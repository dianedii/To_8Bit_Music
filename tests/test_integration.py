import numpy as np
import tempfile
from pathlib import Path

from src.pop_synthesizer import synthesize_pop_chip
from src.utils import export_audio


def test_pop_chip_end_to_end_synthetic():
    sr = 44100
    duration = 1.0
    t = np.arange(int(duration * sr)) / sr
    audio = np.zeros_like(t)
    for start, freq in [(0.0, 440.0), (0.3, 523.25), (0.6, 659.25)]:
        mask = (t >= start) & (t < start + 0.25)
        env = np.exp(-(t[mask] - start) / 0.05)
        audio[mask] += np.sin(2 * np.pi * freq * t[mask]) * env

    out = synthesize_pop_chip(audio, sample_rate=sr, chip_mix=1.0, n_voices=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "out.wav"
        export_audio(out, str(out_path), sample_rate=sr)
        assert out_path.exists()
        assert out_path.stat().st_size > 0
