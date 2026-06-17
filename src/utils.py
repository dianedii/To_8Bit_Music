import importlib
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np


MIRROR_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"


REQUIRED_PACKAGES = [
    ("PyQt6", "PyQt6"),
    ("piano_transcription_inference", "piano_transcription_inference"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pydub", "pydub"),
]


def is_package_installed(import_name: str) -> bool:
    """检查 Python 包是否已安装。"""
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False


def install_packages(package_names: list[str], use_mirror: bool = True) -> None:
    """使用 pip 安装指定包，可选国内镜像。"""
    cmd = [sys.executable, "-m", "pip", "install"]
    if use_mirror:
        cmd.extend(["--index-url", MIRROR_INDEX_URL])
    cmd.extend(package_names)
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        if not use_mirror:
            raise
        # Retry without mirror
        cmd = [sys.executable, "-m", "pip", "install"]
        cmd.extend(package_names)
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as retry_error:
            raise subprocess.CalledProcessError(
                retry_error.returncode, retry_error.cmd,
                output=retry_error.output, stderr=retry_error.stderr,
            ) from retry_error


def get_output_path(input_path: str, output_format: str) -> Path:
    """根据输入文件和输出格式生成输出路径。"""
    p = Path(input_path)
    fmt = output_format.lower()
    if fmt == "wav":
        suffix = ".wav"
    elif fmt == "mp3":
        suffix = ".mp3"
    else:
        raise ValueError(f"不支持的输出格式: {output_format}")

    # 避免输出文件与输入文件同名导致覆盖
    if p.suffix.lower() == suffix:
        return p.with_stem(f"{p.stem}_8bit").with_suffix(suffix)
    return p.with_suffix(suffix)


def open_folder(path: Path) -> None:
    """打开文件所在文件夹。"""
    import platform
    folder = path.parent
    system = platform.system()
    if system == "Windows":
        subprocess.run(["explorer", str(folder)], check=False)
    elif system == "Darwin":
        subprocess.run(["open", str(folder)], check=False)
    else:
        subprocess.run(["xdg-open", str(folder)], check=False)


def export_audio(audio: np.ndarray, output_path: str, sample_rate: int = 44100) -> Path:
    """导出音频为 WAV 或 MP3（依赖 pydub）。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 先写出临时 WAV
    wav_path = output_path.with_suffix(".wav")
    # 确保音频在 [-1, 1] 范围内并转为 16bit
    clipped = np.clip(audio, -1.0, 1.0)
    int16_audio = (clipped * 32767).astype(np.int16)

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_audio.tobytes())

    if output_path.suffix.lower() == ".mp3":
        try:
            from pydub import AudioSegment
            segment = AudioSegment.from_wav(str(wav_path))
            segment.export(str(output_path), format="mp3", bitrate="192k")
            wav_path.unlink(missing_ok=True)
        except Exception as e:
            # MP3 导出失败时保留 WAV 并抛出提示
            raise RuntimeError(f"MP3 导出失败，已保留 WAV: {wav_path}. 错误: {e}")

    return output_path


def check_all_dependencies() -> list[str]:
    """返回缺失的依赖包名列表。"""
    missing = []
    for import_name, install_name in REQUIRED_PACKAGES:
        if not is_package_installed(import_name):
            missing.append(install_name)
    return missing
