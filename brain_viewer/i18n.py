"""Small, bidirectional catalogue; language changes preserve widgets and scene state."""
import json
from pathlib import Path
import re
from string import Formatter

LANGUAGE = "ja"
CATALOG = json.loads(Path(__file__).with_name("translations.json").read_text(encoding="utf-8"))
REVERSE = {value:key for key,value in CATALOG.items()}


def set_language(language):
    global LANGUAGE
    LANGUAGE = "en" if language == "en" else "ja"


def _patterns(catalog):
    result = []
    for source, target in catalog.items():
        if "{" not in source: continue
        regex = ""
        for literal, name, _, _ in Formatter().parse(source):
            regex += re.escape(literal)
            if name is not None: regex += "(?P<" + name + ">.*?)"
        result.append((re.compile("^"+regex+"$", re.DOTALL),target))
    return result


PATTERNS = {"en":_patterns(CATALOG), "ja":_patterns(REVERSE)}


def tr(text):
    catalog = CATALOG if LANGUAGE == "en" else REVERSE
    if text in catalog: return catalog[text]
    for pattern, replacement in PATTERNS[LANGUAGE]:
        match = pattern.fullmatch(text)
        if match: return replacement.format(**match.groupdict())
    return text


def translate_widgets(root):
    from PySide6.QtWidgets import (QWidget,QLabel,QAbstractButton,QGroupBox,QComboBox,QTabBar,QStatusBar)
    from PySide6.QtGui import QAction
    for widget in [root,*root.findChildren(QWidget),*root.findChildren(QAction)]:
        if widget.property('userText'): continue
        if isinstance(widget,(QLabel,QAbstractButton,QAction)):
            widget.setText(tr(widget.text()))
        if isinstance(widget,QGroupBox): widget.setTitle(tr(widget.title()))
        if isinstance(widget,QComboBox):
            if widget.objectName() == 'sourceImageNames':
                for i in range(widget.count()):
                    if widget.itemData(i) in ('mri','ct'): widget.setItemText(i,tr(widget.itemText(i)))
                continue
            if widget.objectName() in ('patientNames','imageNames'):
                if widget.count(): widget.setItemText(0,tr(widget.itemText(0)))
                continue
            for i in range(widget.count()):
                if i and widget.objectName() == "dataNames": continue
                widget.setItemText(i,tr(widget.itemText(i)))
        if isinstance(widget,QTabBar):
            for i in range(widget.count()):
                widget.setTabText(i,tr(widget.tabText(i)))
                widget.setTabToolTip(i,tr(widget.tabToolTip(i)))
        if isinstance(widget,QStatusBar): widget.showMessage(tr(widget.currentMessage()))
        if hasattr(widget,"toolTip"): widget.setToolTip(tr(widget.toolTip()))
