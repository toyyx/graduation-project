import datetime
import logging
import os
import sys
import traceback
import termcolor
from PyQt5.QtWidgets import QApplication, QMessageBox
from qtpy import QtCore

if os.name == "nt":  # Windows
    import colorama

    colorama.init()

from . import __appname__

COLORS = {
    "WARNING": "yellow",
    "INFO": "white",
    "DEBUG": "blue",
    "CRITICAL": "red",
    "ERROR": "red",
}


def show_error_popup(message):
    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Critical)
    msg_box.setText("An error occurred, see log file for details")
    msg_box.setWindowTitle("Error")
    msg_box.setInformativeText(message)
    msg_box.exec_()


class ColoredFormatter(logging.Formatter):

    def __init__(self, fmt, use_color=True):
        logging.Formatter.__init__(self, fmt)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in COLORS:
            def colored(text):
                return termcolor.colored(
                    text,
                    color=COLORS[levelname],
                    attrs={"bold": True},
                )

            record.levelname2 = colored("{:<7}".format(record.levelname))
            record.message2 = colored(record.msg)

            asctime2 = datetime.datetime.fromtimestamp(record.created)
            record.asctime2 = termcolor.colored(asctime2, color="green")

            record.module2 = termcolor.colored(record.module, color="cyan")
            record.funcName2 = termcolor.colored(record.funcName, color="cyan")
            record.lineno2 = termcolor.colored(record.lineno, color="cyan")
        return logging.Formatter.format(self, record)


logger = logging.getLogger(__appname__)
logger.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler(sys.stderr)
handler_format = ColoredFormatter(
    "%(asctime)s [%(levelname2)s] %(module2)s:%(funcName2)s:%(lineno2)s"
    "- %(message2)s"
)
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)

log_file_abs_path = os.path.abspath('autoregion_labelme_error.log')
file_handler = logging.FileHandler(log_file_abs_path)
file_handler.setFormatter(handler_format)
logger.addHandler(file_handler)

def handle_exception(type, value, traceback):
    try:
        logger.exception(f"Exception：{type.__name__}", exc_info=(type, value, traceback))
        show_error_popup(f"异常类型：{type.__name__}\n日志文件位于:{log_file_abs_path}")
    except Exception as e:
        logger.exception(f"Error occurred while handling exception: {e}", exc_info=True)
        show_error_popup(f"异常类型：Error occurred while handling exception\n日志文件位于:{log_file_abs_path}")

    #正常的异常处理会终止程序
    #original_excepthook = sys.excepthook
    #original_excepthook(type, value, traceback)

sys.excepthook = handle_exception




