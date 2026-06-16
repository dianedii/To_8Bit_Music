import sys
import importlib
from unittest.mock import patch

from src.utils import is_package_installed


def test_is_package_installed_true():
    # numpy 在 requirements 中，通常测试环境未安装，这里用一个肯定存在的内置包测试
    assert is_package_installed('sys') is True


def test_is_package_installed_false():
    assert is_package_installed('nonexistent_package_abc123') is False
