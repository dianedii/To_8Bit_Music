import importlib
import subprocess
import sys
from pathlib import Path


MIRROR_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"


def is_package_installed(package_name: str) -> bool:
    """检查 Python 包是否已安装。"""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


def install_packages(package_names: list[str], use_mirror: bool = True) -> None:
    """使用 pip 安装指定包，可选国内镜像。"""
    cmd = [sys.executable, "-m", "pip", "install"]
    if use_mirror:
        cmd.extend(["--index-url", MIRROR_INDEX_URL])
    cmd.extend(package_names)
    subprocess.check_call(cmd)


def get_output_path(input_path: str, output_format: str) -> Path:
    """根据输入文件和输出格式生成输出路径。"""
    p = Path(input_path)
    suffix = ".wav" if output_format.lower() == "wav" else ".mp3"
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
