import sys

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QDialog,
                             QVBoxLayout, QLabel, QScrollArea, QTextEdit,
                             QTabWidget, QWidget, QFrame, QHBoxLayout, QTableWidget,
                             QTableWidgetItem, QHeaderView)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidget, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt

class ExperimentResultsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("实验结果")
        self.setMinimumSize(1500, 1200)

        # 设置字体
        table_title_font = QFont()
        table_title_font.setFamily("Times New Roman")  # 设置为Times New Roman
        table_title_font.setPointSize(16)
        table_title_font.setBold(True)

        header_font = QFont()
        header_font.setFamily("Times New Roman")  # 设置为Times New Roman
        header_font.setPointSize(14)
        header_font.setBold(True)

        content_font = QFont()
        content_font.setFamily("Times New Roman")  # 设置为Times New Roman
        content_font.setPointSize(12)

        # 创建主布局
        main_layout = QVBoxLayout(self)

        # 创建滚动区域以容纳多个表格
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        # 创建滚动区域的内容窗口
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)

        ##########################
        # 添加第一个表格
        # 表5.3 在多数据集上的‌mIoU指标消融实验结果
        table1_title = QLabel("表1 在多数据集上的‌mIoU指标消融实验结果")
        table1_title.setFont(table_title_font)
        table1_title.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        content_layout.addWidget(table1_title)

        table1 = QTableWidget()
        table1.setColumnCount(5)
        table1.setHorizontalHeaderLabels(["模型配置", "SA-1B", "COCO", "PASCAL VOC", "LVIS"])

        header = table1.horizontalHeader()
        header.setFont(header_font)
        header.setSectionResizeMode(QHeaderView.Stretch)

        row_data_1 = [
            ["SAM + Bounding box", "0.9095", "0.7459", "0.7866", "0.7714"],
            ["SAM + Bounding box+ PointRend", "0.9100", "0.7454", "0.7868", "0.7720"],
            ["SAM + Free-rotating box", "0.9136", "0.7638", "0.7976", "0.7827"],
            ["SAM + Free-rotating box + PointRend", "0.9141", "0.7633", "0.7978", "0.7832"]
        ]

        table1.setRowCount(len(row_data_1))
        for row_idx, row in enumerate(row_data_1):
            for col_idx, item in enumerate(row):
                table_item = QTableWidgetItem(item)
                table_item.setFont(content_font)
                table_item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                table_item.setFlags(table_item.flags() & ~Qt.ItemIsEditable)
                table1.setItem(row_idx, col_idx, table_item)

        table1.setStyleSheet("""
            QTableWidget {
                gridline-color: #d0d0d0;
                border: 1px solid #d0d0d0;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                border: 1px solid #d0d0d0;
                padding: 4px;
            }
            QTableWidget::item {
                padding: 4px;
                border: 1px solid #d0d0d0;
            }
        """)

        content_layout.addWidget(table1)

        # 添加分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        content_layout.addWidget(line)
        ##########################

        ##########################
        # 添加第一个表格
        # 表5.3 在多数据集上的‌mIoU指标消融实验结果
        table2_title = QLabel("表2 FastSAM在多个数据集上使用两种提示方式的‌mIoU指标对比实验结果")
        table2_title.setFont(table_title_font)
        table2_title.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        content_layout.addWidget(table2_title)

        table2 = QTableWidget()
        table2.setColumnCount(5)
        table2.setHorizontalHeaderLabels(["模型", "SA-1B", "COCO", "PASCAL VOC", "LVIS"])

        header = table2.horizontalHeader()
        header.setFont(header_font)
        header.setSectionResizeMode(QHeaderView.Stretch)

        row_data_2 = [
            ["FastSAM - Bounding box", "0.6497", "0.5854", "0.6950", "0.5039"],
            ["FastSAM - Free-rotating box", "0.6536", "0.5912", "0.7085", "0.5145"]
        ]

        table2.setRowCount(len(row_data_2))
        for row_idx, row in enumerate(row_data_2):
            for col_idx, item in enumerate(row):
                table_item = QTableWidgetItem(item)
                table_item.setFont(content_font)
                table_item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                table_item.setFlags(table_item.flags() & ~Qt.ItemIsEditable)
                table2.setItem(row_idx, col_idx, table_item)

        # 设置表格样式（与第一个表格相同）
        table2.setStyleSheet(table1.styleSheet())

        content_layout.addWidget(table2)

        # 添加分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        content_layout.addWidget(line)
        ##########################

        ##########################
        # 添加第一个表格
        # 表5.3 在多数据集上的‌mIoU指标消融实验结果
        table3_title = QLabel("表3 MedSAM在MICCAI FLARE 2022上使用两种提示方式的‌mIoU指标对比实验结果")
        table3_title.setFont(table_title_font)
        table3_title.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        content_layout.addWidget(table3_title)

        table3 = QTableWidget()
        table3.setColumnCount(7)
        table3.setHorizontalHeaderLabels(["模型", "1", "2", "3", "4", "5", "平均"])

        header = table3.horizontalHeader()
        header.setFont(header_font)
        header.setSectionResizeMode(QHeaderView.Stretch)

        row_data_3 = [
            ["MedSAM - Bounding box", "0.9033", "0.9063", "0.9045", "0.9041", "0.9077", "0.9052±0.0016"],
            ["MedSAM - Free-rotating box", "0.9048", "0.9085", "0.9070", "0.9024", "0.9084", "0.9062±0.0023"]
        ]

        table3.setRowCount(len(row_data_3))
        for row_idx, row in enumerate(row_data_3):
            for col_idx, item in enumerate(row):
                table_item = QTableWidgetItem(item)
                table_item.setFont(content_font)
                table_item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                table_item.setFlags(table_item.flags() & ~Qt.ItemIsEditable)
                table3.setItem(row_idx, col_idx, table_item)

        # 设置表格样式（与第一个表格相同）
        table3.setStyleSheet(table1.styleSheet())

        content_layout.addWidget(table3)

        # 添加分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        content_layout.addWidget(line)
        ##########################

        ##########################
        # 添加第一个表格
        # 表5.3 在多数据集上的‌mIoU指标消融实验结果
        table4_title = QLabel("表4 多种模型在SA-1B数据集上多种指标的对比实验结果")
        table4_title.setFont(table_title_font)
        table4_title.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        content_layout.addWidget(table4_title)

        table4 = QTableWidget()
        table4.setColumnCount(6)
        table4.setHorizontalHeaderLabels(["模型", "mIoU↑", "Parameter↓", "Time↓", "GFLOPs↓", "FPS↑"])

        header = table4.horizontalHeader()
        header.setFont(header_font)
        header.setSectionResizeMode(QHeaderView.Stretch)

        row_data_4 = [
            ["EfficientSAM", "0.8638", "10.2M", "106.2863", "208.0684", "9.4086"],
            ["FastSAM", "0.6497", "72.2M", "57.0206", "584.3392", "17.5375"],
            ["SAM", "0.9095", "93.7M", "144.3868", "976.8252", "6.9258"],
            ["Ours", "0.9141", "94.2M", "158.6840", "977.1824", "6.3018"]
        ]

        table4.setRowCount(len(row_data_4))
        for row_idx, row in enumerate(row_data_4):
            for col_idx, item in enumerate(row):
                table_item = QTableWidgetItem(item)
                table_item.setFont(content_font)
                table_item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                table_item.setFlags(table_item.flags() & ~Qt.ItemIsEditable)
                table4.setItem(row_idx, col_idx, table_item)

        # 设置表格样式（与第一个表格相同）
        table4.setStyleSheet(table1.styleSheet())

        content_layout.addWidget(table4)

        # 添加分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        content_layout.addWidget(line)
        ##########################

        ##########################
        # 添加第一个表格
        # 表5.3 在多数据集上的‌mIoU指标消融实验结果
        table5_title = QLabel("表5 在多个数据集上的‌mIoU对比实验结果")
        table5_title.setFont(table_title_font)
        table5_title.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        content_layout.addWidget(table5_title)

        table5 = QTableWidget()
        table5.setColumnCount(5)
        table5.setHorizontalHeaderLabels(["模型", "SA-1B", "COCO", "PASCAL VOC", "LVIS"])

        header = table5.horizontalHeader()
        header.setFont(header_font)
        header.setSectionResizeMode(QHeaderView.Stretch)

        row_data_5 = [
            ["EfficientSAM", "0.8638", "0.7310", "0.7725", "0.7339"],
            ["FastSAM", "0.6497", "0.5854", "0.6950", "0.5039"],
            ["SAM", "0.9095", "0.7459", "0.7866", "0.7714"],
            ["Ours", "0.9141", "0.7633", "0.7978", "0.7832"]
        ]

        table5.setRowCount(len(row_data_5))
        for row_idx, row in enumerate(row_data_5):
            for col_idx, item in enumerate(row):
                table_item = QTableWidgetItem(item)
                table_item.setFont(content_font)
                table_item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                table_item.setFlags(table_item.flags() & ~Qt.ItemIsEditable)
                table5.setItem(row_idx, col_idx, table_item)

        # 设置表格样式（与第一个表格相同）
        table5.setStyleSheet(table1.styleSheet())

        content_layout.addWidget(table5)

        # 添加分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        content_layout.addWidget(line)
        ##########################

        # 添加间距
        content_layout.addStretch(1)

        # 将内容窗口设置到滚动区域
        scroll_area.setWidget(content_widget)

        # 将滚动区域添加到主布局
        main_layout.addWidget(scroll_area)

        # 添加关闭按钮
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.close)
        main_layout.addWidget(close_button, alignment=Qt.AlignRight)
