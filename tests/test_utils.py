from pathlib import Path

import numpy as np
import tempfile

from src.utils import is_package_installed, get_output_path, export_audio


def test_is_package_installed_true():
    # numpy 在 requirements 中，通常测试环境未安装，这里用一个肯定存在的内置包测试
    assert is_package_installed('sys') is True


def test_is_package_installed_false():
    assert is_package_installed('nonexistent_package_abc123') is False


def test_get_output_path_mp3():
    assert get_output_path("input.wav", "mp3") == Path("goals/input.mp3")


def test_get_output_path_wav():
    assert get_output_path("input.mp3", "wav") == Path("goals/input.wav")


def test_get_output_path_avoid_overwrite():
    # 当输入文件就在 goals/ 且格式相同时，应添加 _8bit 后缀避免覆盖
    assert get_output_path("goals/input.mp3", "mp3") == Path("goals/input_8bit.mp3")
    assert get_output_path("goals/input.wav", "wav") == Path("goals/input_8bit.wav")


def test_get_output_path_invalid_format():
    try:
        get_output_path("input.wav", "ogg")
    except ValueError as e:
        assert "不支持的输出格式" in str(e)
    else:
        raise AssertionError("Expected ValueError for unsupported format")


def test_export_wav():
    samples = np.sin(2 * np.pi * 440 * np.arange(4410) / 44100).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test.wav"
        export_audio(samples, str(out_path), sample_rate=44100)
        assert out_path.exists()
        assert out_path.stat().st_size > 0
