from src.utils import is_package_installed, install_packages, REQUIRED_PACKAGES


def main():
    missing = [install_name for import_name, install_name in REQUIRED_PACKAGES if not is_package_installed(import_name)]
    if not missing:
        print("所有依赖已安装。")
        return
    print(f"检测到缺失依赖: {', '.join(missing)}")
    print("正在通过国内镜像安装...")
    install_packages(missing, use_mirror=True)
    print("安装完成。")


if __name__ == "__main__":
    main()
