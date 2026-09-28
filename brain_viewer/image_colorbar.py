"""Compact legend using exactly the same lookup table as the image slices."""
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QLinearGradient, QPainter
from PySide6.QtWidgets import QWidget
from .image_colormaps import image_lut


class ImageColorBar(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(39)
        self.palette_name, self.low, self.high = 'hot', 0., 1.

    def set_range(self, palette, low, high):
        self.palette_name, self.low, self.high = palette, low, high
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        gradient = QLinearGradient(0, 0, self.width(), 0)
        for index, color in enumerate(image_lut(self.palette_name)):
            gradient.setColorAt(index/255, QColor(*map(int, color)))
        painter.fillRect(QRectF(0, 0, self.width(), 13), gradient)
        painter.setPen(QColor('#d3e2f1'))
        rect = QRectF(0, 16, self.width(), 21)
        painter.drawText(rect, Qt.AlignmentFlag.AlignLeft, f'{self.low:g}')
        painter.drawText(rect, Qt.AlignmentFlag.AlignRight, f'{self.high:g}')
