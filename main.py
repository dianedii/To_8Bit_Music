import sys

from src.utils import check_all_dependencies, install_packages


def main():
    missing = check_all_dependencies()
    if missing:
        print(f"检测到缺失依赖: {', '.join(missing)}")
        print("正在通过国内镜像自动安装...")
        try:
            install_packages(missing, use_mirror=True)
        except Exception as e:
            print(f"自动安装失败: {e}")
            print("请手动运行: python setup_env.py")
            sys.exit(1)

    from src.ui import run_app
    run_app()


if __name__ == "__main__":
    main()
