from pathlib import Path

from src.utils import is_package_installed, get_output_path


def test_is_package_installed_true():
    # numpy 在 requirements 中，通常测试环境未安装，这里用一个肯定存在的内置包测试
    assert is_package_installed('sys') is True


def test_is_package_installed_false():
    assert is_package_installed('nonexistent_package_abc123') is False


def test_get_output_path_mp3():
    assert get_output_path("input.wav", "mp3") == Path("input.mp3")


def test_get_output_path_wav():
    assert get_output_path("input.mp3", "wav") == Path("input.wav")


def test_get_output_path_invalid_format():
    try:
        get_output_path("input.wav", "ogg")
    except ValueError as e:
        assert "不支持的输出格式" in str(e)
    else:
        raise AssertionError("Expected ValueError for unsupported format")
