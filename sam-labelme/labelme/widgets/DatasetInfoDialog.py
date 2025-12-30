import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QDialog,
                             QVBoxLayout, QLabel, QScrollArea, QTextEdit,
                             QTabWidget, QWidget, QFrame, QHBoxLayout)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class DatasetInfoDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("数据集信息")
        self.setMinimumSize(800, 600)

        # 设置字体
        title_font = QFont()
        title_font.setFamily("Times New Roman")  # 设置为Times New Roman
        title_font.setPointSize(16)
        title_font.setBold(True)

        desc_font = QFont()
        title_font.setFamily("Times New Roman")  # 设置为Times New Roman
        desc_font.setPointSize(14)

        # 创建标签页
        self.tabs = QTabWidget()

        # SA-1B数据集
        sa1b_tab = QWidget()
        sa1b_layout = QVBoxLayout(sa1b_tab)

        sa1b_title = QLabel("SA-1B (Segment Anything 1-Billion)")
        sa1b_title.setFont(title_font)
        sa1b_layout.addWidget(sa1b_title)

        sa1b_desc = QTextEdit()
        sa1b_desc.setReadOnly(True)
        sa1b_desc.setFont(desc_font)
        sa1b_desc.setHtml("""
        <style>
        p, ul, li { line-height: 1.5; }
        </style>
        <p>SA-1B是由Meta AI开发的大规模分割数据集，包含超过10亿个掩码，覆盖了1100万张图像。</p>
        <p>该数据集是使用Segment Anything模型自动生成的，为计算机视觉中的分割任务提供了丰富的标注资源。</p>
        <p><b>特点：</b></p>
        <ul>
            <li>1100万张多样化图像</li>
            <li>超过10亿个高质量分割掩码</li>
            <li>涵盖广泛的对象类别和场景</li>
        </ul>
        """)
        sa1b_layout.addWidget(sa1b_desc)

        sa1b_url = QLabel("<a href='https://segment-anything.com/dataset/index.html'>下载地址（包含数据集格式）</a>")
        sa1b_url.setOpenExternalLinks(True)
        sa1b_url.setAlignment(Qt.AlignRight)
        sa1b_layout.addWidget(sa1b_url)

        self.tabs.addTab(sa1b_tab, "SA-1B")

        # COCO数据集
        coco_tab = QWidget()
        coco_layout = QVBoxLayout(coco_tab)

        coco_title = QLabel("COCO (Common Objects in Context)")
        coco_title.setFont(title_font)
        coco_layout.addWidget(coco_title)

        coco_desc = QTextEdit()
        coco_desc.setReadOnly(True)
        coco_desc.setFont(desc_font)
        coco_desc.setHtml("""
        <style>
        p, ul, li { line-height: 1.5; }
        </style>
        <p>COCO是一个广泛用于目标检测、分割和描述的大规模数据集。</p>
        <p>它包含33万张图像，其中超过20万张有标注，涵盖80个对象类别，超过150万个物体实例。</p>
        <p><b>特点：</b></p>
        <ul>
            <li>上下文环境中的对象检测与分割</li>
            <li>图像描述和场景理解</li>
            <li>关键点检测和人体姿态估计</li>
        </ul>
        """)
        coco_layout.addWidget(coco_desc)

        coco_url = QLabel("<a href='https://cocodataset.org/'>下载地址（包含数据集格式）</a>")
        coco_url.setOpenExternalLinks(True)
        coco_url.setAlignment(Qt.AlignRight)
        coco_layout.addWidget(coco_url)

        self.tabs.addTab(coco_tab, "COCO")

        # VOC数据集
        voc_tab = QWidget()
        voc_layout = QVBoxLayout(voc_tab)

        voc_title = QLabel("Pascal VOC (Pattern Analysis, Statistical Modelling and Computational Learning Visual Object Classes)")
        voc_title.setFont(title_font)
        voc_layout.addWidget(voc_title)

        voc_desc = QTextEdit()
        voc_desc.setReadOnly(True)
        voc_desc.setFont(desc_font)
        voc_desc.setHtml("""
        <style>
        p, ul, li { line-height: 1.5; }
        </style>
        <p>Pascal VOC是计算机视觉领域的经典数据集，专注于目标检测和分割任务。</p>
        <p>该数据集包含20个不同的对象类别，如人、动物、交通工具等，共有约11,500张图像。</p>
        <p><b>特点：</b></p>
        <ul>
            <li>20个预定义的对象类别</li>
            <li>精确的边界框和分割掩码</li>
            <li>广泛用于算法评估和比较</li>
        </ul>
        """)
        voc_layout.addWidget(voc_desc)

        voc_url = QLabel("<a href='http://host.robots.ox.ac.uk/pascal/VOC/'>下载地址（包含数据集格式）</a>")
        voc_url.setOpenExternalLinks(True)
        voc_url.setAlignment(Qt.AlignRight)
        voc_layout.addWidget(voc_url)

        self.tabs.addTab(voc_tab, "Pascal VOC")

        # LVIS数据集
        lvis_tab = QWidget()
        lvis_layout = QVBoxLayout(lvis_tab)

        lvis_title = QLabel("LVIS (Large Vocabulary Instance Segmentation)")
        lvis_title.setFont(title_font)
        lvis_layout.addWidget(lvis_title)

        lvis_desc = QTextEdit()
        lvis_desc.setReadOnly(True)
        lvis_desc.setFont(desc_font)
        lvis_desc.setHtml("""
        <style>
        p, ul, li { line-height: 1.5; }
        </style>
        <p>LVIS是一个专注于大规模实例分割的数据集，包含超过1200个类别。</p>
        <p>该数据集强调长尾分布，包含许多罕见类别的实例，对模型的泛化能力提出了挑战。</p>
        <p><b>特点：</b></p>
        <ul>
            <li>1203个对象类别，包括许多罕见类别</li>
            <li>约200万张图像中的实例分割标注</li>
            <li>类别频率呈现长尾分布</li>
        </ul>
        """)
        lvis_layout.addWidget(lvis_desc)

        lvis_url = QLabel("<a href='https://www.lvisdataset.org/'>下载地址（包含数据集格式）</a>")
        lvis_url.setOpenExternalLinks(True)
        lvis_url.setAlignment(Qt.AlignRight)
        lvis_layout.addWidget(lvis_url)

        self.tabs.addTab(lvis_tab, "LVIS")

        # 创建主布局
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.tabs)

        # 添加关闭按钮
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.close)
        main_layout.addWidget(close_button, alignment=Qt.AlignRight)