import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QSlider, QComboBox, QFileDialog,
    QMessageBox,
)
from PyQt6.QtCore import Qt

from src.worker import ConvertWorker
from src.utils import open_folder


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FC 8-bit 芯片音乐转换器")
        self.setMinimumWidth(480)

        self.input_path = ""
        self.worker: Optional[ConvertWorker] = None

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

        # 文件选择
        file_layout = QHBoxLayout()
        self.path_label = QLabel("未选择文件")
        self.path_label.setWordWrap(True)
        file_layout.addWidget(self.path_label, stretch=1)
        self.select_btn = QPushButton("选择文件")
        self.select_btn.clicked.connect(self._select_file)
        file_layout.addWidget(self.select_btn)
        layout.addLayout(file_layout)

        # 复古纯度
        layout.addWidget(QLabel("复古纯度"))
        self.purity_slider = QSlider(Qt.Orientation.Horizontal)
        self.purity_slider.setRange(0, 100)
        self.purity_slider.setValue(0)
        layout.addWidget(self.purity_slider)
        purity_layout = QHBoxLayout()
        purity_layout.addWidget(QLabel("纯 FC 方波"))
        purity_layout.addStretch()
        purity_layout.addWidget(QLabel("轻微润色"))
        layout.addLayout(purity_layout)

        # 音符简化强度
        layout.addWidget(QLabel("音符简化强度"))
        self.simplify_slider = QSlider(Qt.Orientation.Horizontal)
        self.simplify_slider.setRange(0, 100)
        self.simplify_slider.setValue(50)
        layout.addWidget(self.simplify_slider)
        simplify_layout = QHBoxLayout()
        simplify_layout.addWidget(QLabel("保留原样"))
        simplify_layout.addStretch()
        simplify_layout.addWidget(QLabel("极致简化"))
        layout.addLayout(simplify_layout)

        # 整体音量
        layout.addWidget(QLabel("整体音量"))
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)
        layout.addWidget(self.volume_slider)
        volume_layout = QHBoxLayout()
        volume_layout.addWidget(QLabel("轻"))
        volume_layout.addStretch()
        volume_layout.addWidget(QLabel("响"))
        layout.addLayout(volume_layout)

        # 输出格式
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("输出格式"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["MP3", "WAV"])
        format_layout.addWidget(self.format_combo)
        format_layout.addStretch()
        layout.addLayout(format_layout)

        # 转换按钮
        self.convert_btn = QPushButton("开始转换")
        self.convert_btn.clicked.connect(self._start_convert)
        layout.addWidget(self.convert_btn)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # 状态提示
        self.status_label = QLabel("状态：等待选择文件...")
        layout.addWidget(self.status_label)

        # 打开文件夹按钮
        self.open_folder_btn = QPushButton("打开文件所在文件夹")
        self.open_folder_btn.setEnabled(False)
        self.open_folder_btn.clicked.connect(self._open_result_folder)
        layout.addWidget(self.open_folder_btn)

        self.output_path = ""

    def _select_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 MP3 文件", "", "MP3 文件 (*.mp3)"
        )
        if path:
            self.input_path = path
            self.path_label.setText(path)
            self.status_label.setText("状态：准备就绪")

    def _start_convert(self):
        if not self.input_path:
            QMessageBox.warning(self, "提示", "请先选择 MP3 文件")
            return

        self.convert_btn.setEnabled(False)
        self.open_folder_btn.setEnabled(False)
        self.progress_bar.setValue(0)

        self.worker = ConvertWorker(
            input_path=self.input_path,
            purity=self.purity_slider.value(),
            simplification=self.simplify_slider.value(),
            volume=self.volume_slider.value(),
            output_format=self.format_combo.currentText(),
        )
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self.status_label.setText)
        self.worker.finished_success.connect(self._on_success)
        self.worker.finished_error.connect(self._on_error)
        self.worker.start()

    def _on_success(self, output_path: str):
        self.output_path = output_path
        self.status_label.setText(f"转换完成: {output_path}")
        self.convert_btn.setEnabled(True)
        self.open_folder_btn.setEnabled(True)
        QMessageBox.information(self, "完成", f"已保存到:\n{output_path}")

    def _on_error(self, message: str):
        self.status_label.setText(message)
        self.convert_btn.setEnabled(True)
        self.open_folder_btn.setEnabled(False)
        QMessageBox.critical(self, "错误", message)

    def _open_result_folder(self):
        if self.output_path:
            open_folder(Path(self.output_path))


def run_app():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
