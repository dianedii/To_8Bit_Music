from src.utils import is_package_installed, install_packages


REQUIRED_PACKAGES = [
    "PyQt6",
    "piano_transcription_inference",
    "numpy",
    "scipy",
    "pydub",
]


def main():
    missing = [pkg for pkg in REQUIRED_PACKAGES if not is_package_installed(pkg)]
    if not missing:
        print("所有依赖已安装。")
        return
    print(f"检测到缺失依赖: {', '.join(missing)}")
    print("正在通过国内镜像安装...")
    install_packages(missing, use_mirror=True)
    print("安装完成。")


if __name__ == "__main__":
    main()
