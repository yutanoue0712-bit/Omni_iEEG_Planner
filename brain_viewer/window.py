"""Independent desktop workspace; all reads and exports stay on the local PC."""
from __future__ import annotations

import json
import os
from pathlib import Path
import traceback
import time
from dataclasses import replace

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QFont, QKeySequence, QPainter
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QMenu, QInputDialog, QDialog,
    QPushButton, QScrollArea, QSlider, QSplitter, QTabBar, QTabWidget, QVBoxLayout, QWidget, QSizePolicy, QStackedWidget,
)

from .imaging import InputError, Scene, discover_inputs, load_mri, load_scene, make_demo, save_scene
from .slice_view import SlicePanel
from .surface_view import SurfacePanel
from .alignment_review import AlignmentReview
from .contact_editor import ContactEditor
from .electrode_editing import create_lead, export_records
from .layer_controls import LayerList
from .workspace_ui import WorkspaceUI
from .mri_controls import MRIControls
from .segmentation_panel import SegmentationPanel
from .results_panel import ResultsPanel
from .diffusion_panel import DiffusionPanel
from .postop_panel import PostopPanel
from . import __version__
from .i18n import tr, set_language, translate_widgets


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def log_error():
    folder = PROJECT_ROOT / "private_reports"
    folder.mkdir(exist_ok=True)
    with (folder / "viewer_error.log").open("a", encoding="utf-8") as stream:
        stream.write(f'\n[{time.strftime("%Y-%m-%d %H:%M:%S")} / PID {os.getpid()}]\n')
        traceback.print_exc(file=stream)


class Job(QThread):
    progress = Signal(str)

    def __init__(self, operation, parent=None):
        super().__init__(parent)
        self.operation = operation
        self.value = None
        self.error = None

    def run(self):
        try:
            self.value = self.operation(self.progress.emit)
        except Exception as exc:
            log_error()
            self.error = (str(exc) if isinstance(exc, InputError) else
                          "処理を完了できませんでした。入力形式・保存先を確認してください。詳細はローカルのエラーログに記録しました。")


STYLE = """
QMainWindow, QWidget { background:#101a29; color:#dce5ef; font-family:'Yu Gothic UI','Meiryo','Segoe UI'; font-size:12px; }
QLabel#brand {font-size:21px; font-weight:700; color:#f2f7fb;}
QLabel#localBadge {color:#90dfbc; border:1px solid #33564e; background:#15362f; border-radius:9px; padding:3px 9px; font-size:10px; font-weight:700;}
QLabel#muted {color:#8fa2b8; font-size:11px;}
QLabel#importFeedback {color:#f0ce96;}
QLabel#coordinate {font-family:'Consolas','Yu Gothic UI'; color:#b9e6ea; font-size:13px; padding:3px 0;}
QLabel#viewHint {color:#8297ad; font-size:10px; padding:7px 12px; background:#101a29;}
QFrame#viewPanel {border:1px solid #2a3b50; border-radius:7px; background:#101a29;}
QGroupBox {border:1px solid #2a3b50; border-radius:7px; margin-top:11px; padding:12px 10px 8px; font-weight:600;}
QGroupBox::title {subcontrol-origin:margin; left:10px; padding:0 5px; color:#a4b8ce;}
QPushButton {background:#1b2c40; border:1px solid #354a62; border-radius:5px; padding:6px 12px;}
QPushButton:hover {background:#28405a; border-color:#6a9bb0;}
QPushButton:pressed {background:#133b43;}
QPushButton:disabled {color:#52667e; border-color:#23344a;}
QPushButton#primary {background:#206d72; border-color:#36898d; color:#f1ffff; font-weight:600;}
QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit {background:#0c1624; border:1px solid #33485e; border-radius:4px; padding:4px;}
/* Reserve both native Windows 11 spin buttons. Otherwise the line edit covers Up. */
QDoubleSpinBox, QSpinBox {padding-right:28px;}
QComboBox QAbstractItemView {background:#142236; selection-background-color:#285367;}
QTabBar::tab {padding:11px 14px; color:#92a7bc; border-bottom:2px solid transparent; font-size:13px;}
QTabBar::tab:selected {color:#93e1e7; border-bottom:2px solid #69c8d2;}
QTabBar::tab:disabled {color:#4c6179;}
QTabWidget#sideTabs::pane {border:none;}
QTabWidget#sideTabs QTabBar::tab {padding:8px 9px; font-size:12px;}
QSlider::groove:horizontal {height:4px; border-radius:2px; background:#2b3e54;}
QSlider::sub-page:horizontal {background:#4a9da8; border-radius:2px;}
QSlider::handle:horizontal {background:#bce7eb; border:1px solid #4a9da8; width:11px; margin:-5px 0; border-radius:6px;}
QCheckBox {spacing:7px; padding:3px 0;}
QScrollArea {border:none;}
QSplitter::handle {background:#101a29;}
QStatusBar {border-top:1px solid #25364b; color:#9bafc4;}
QToolTip {background:#203349; color:#e9f5ff; border:1px solid #5a7794; padding:5px;}
"""


class ViewerWindow(WorkspaceUI, MRIControls, QMainWindow):
    scene_ready = Signal()
    scene_failed = Signal(str)

    def __init__(self, smoke=False, demo=False):
        super().__init__()
        self.scene: Scene | None = None
        self.ijk = np.zeros(3, dtype=int)
        self.worker: Job | None = None
        self._job_timer = QTimer(self)
        self._job_timer.setInterval(1000)
        self._job_timer.timeout.connect(self._update_job_progress)
        self._job_started = 0.
        self._job_stage = ''
        self._after_job = None
        self._smoke = smoke
        self._demo = demo
        self._closing = False
        self._restoring = False
        self.pending_tip = None
        self.setWindowTitle("Omni-iEEG planner 1.0 RC20")
        self.setStyleSheet(STYLE)
        self.setMinimumSize(1000, 670)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 14, 20, 8)
        layout.setSpacing(10)
        brand_row = QHBoxLayout()
        brand = QLabel("Omni-iEEG planner")
        brand.setObjectName("brand")
        brand_row.addWidget(brand)
        brand_row.addStretch()
        self.language_tabs = QTabBar()
        self.language_tabs.addTab("日本語")
        self.language_tabs.addTab("English")
        self.language_tabs.setToolTip("表示言語 / Interface language")
        brand_row.addWidget(self.language_tabs)
        version = QLabel("PROTOTYPE  1.0 RC20")
        version.setObjectName("muted")
        brand_row.addWidget(version)
        layout.addLayout(brand_row)
        nav_row = QHBoxLayout()
        self.workflow_tabs = QTabBar()
        for index, text in enumerate((tr("画像表示"), tr("セグメンテーション"), tr("電極プラン"), tr("術後評価"), tr("解析表示"), tr("DTI・線維"))):
            self.workflow_tabs.addTab(text)
            if index == 2:
                self.workflow_tabs.setTabEnabled(index, False)
                self.workflow_tabs.setTabToolTip(index, tr("今後追加する機能です"))
        nav_row.addWidget(self.workflow_tabs)
        nav_row.addStretch()
        self.setup_patients(nav_row, PROJECT_ROOT)
        self.save_button = QPushButton(tr("保存"))
        self.save_button.setObjectName("primary")
        self.save_button.clicked.connect(self.save_work)
        self.export_button = QPushButton(tr("画像出力"))
        self.export_button.clicked.connect(self.export_image)
        for button in (self.save_button, self.export_button):
            nav_row.addWidget(button)
        layout.addLayout(nav_row)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        self.controls = self._build_controls()
        self.control_stack = QStackedWidget()
        self.control_stack.addWidget(self.controls)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.control_stack)
        scroll.setMinimumWidth(285)
        scroll.setMaximumWidth(370)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        split.addWidget(scroll)
        self.slices = {axis: SlicePanel(axis) for axis in (2, 1, 0)}
        for axis, position in ((2, (0, 0)), (1, (0, 1)), (0, (1, 0))):
            panel = self.slices[axis]
            panel.canvas.cursor_changed.connect(self.set_cursor)
            panel.canvas.slice_step.connect(lambda delta, ax=axis: self.step_slice(ax, delta))
            panel.canvas.window_delta.connect(self.drag_window)
            panel.slider.valueChanged.connect(lambda value, ax=axis: self.move_slice(ax, value))
        self.surface = SurfacePanel()
        self.surface.point_picked.connect(self.select_world)
        self.surface.widget.window_delta.connect(self.drag_window)
        self.segmentation = SegmentationPanel(self)
        self.control_stack.addWidget(self.segmentation)
        self.results = ResultsPanel(self)
        self.control_stack.addWidget(self.results)
        self.diffusion = DiffusionPanel(self)
        self.control_stack.addWidget(self.diffusion)
        self.postop = PostopPanel(self)
        self.control_stack.addWidget(self.postop)
        # Numeric ranges can produce very wide size hints. Keep the full buttons
        # inside the sidebar instead of letting a hidden page widen every page.
        from PySide6.QtWidgets import QAbstractSpinBox
        for spin in self.control_stack.findChildren(QAbstractSpinBox):
            policy = spin.sizePolicy()
            policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
            spin.setSizePolicy(policy)
        self.workflow_tabs.currentChanged.connect(self.change_workflow)
        for panel in self.slice_panels(): panel.canvas.brush_requested.connect(self.segmentation.draw)
        from .view_workspace import ViewWorkspace
        self.views = ViewWorkspace(self)
        right=QWidget(); right_layout=QVBoxLayout(right); right_layout.setContentsMargins(0,0,0,0)
        from .result_rendering import ResultLegend
        self.result_banner=ResultLegend(); self.result_banner.hide()
        right_layout.addWidget(self.result_banner)
        from .result_display_controls import ResultPlaybackBar
        self.results.transport=ResultPlaybackBar(self.results);right_layout.addWidget(self.results.transport)
        self.right_stack=QStackedWidget(); self.right_stack.addWidget(self.views); self.right_stack.addWidget(self.results.chart)
        right_layout.addWidget(self.right_stack,1); split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([340, 1030])
        layout.addWidget(split, 1)
        self.message(tr("表示を準備しています…"))
        self.space_label = QLabel(tr("座標：患者固有 RAS / mm  "))
        self.space_label.setObjectName("muted")
        self.statusBar().addPermanentWidget(self.space_label)
        for label, shortcut, callback in ((tr("保存"), QKeySequence.StandardKey.Save, self.save_work),
                                          (tr("画像出力"), "Ctrl+E", self.export_image),
                                          (tr("表示をリセット"), "Home", self.reset_views)):
            action = QAction(label, self)
            action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(callback)
            self.addAction(action)
        available = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1460, available.width() - 60), min(960, available.height() - 65))
        self._set_busy(True)
        language = "ja"
        if not smoke:
            try:
                language = json.loads((PROJECT_ROOT / "local_data" / "preferences.json").read_text())["language"]
            except (OSError, ValueError, KeyError):
                pass
        self.language_tabs.currentChanged.connect(self.change_language)
        self.language_tabs.setCurrentIndex(1 if language == "en" else 0)
        self.change_language(self.language_tabs.currentIndex())

    def slice_panels(self):
        views=getattr(self,'views',None)
        return views.slice_panels() if views is not None else list(self.slices.values())

    def change_workflow(self, index):
        if index!=3:self.postop.clear_mask_preview()
        if index!=5:
            self.diffusion.exclusion.mode.setCurrentIndex(0)
            self.diffusion.exclusion.clear_preview()
            self.diffusion.draw_mode.setCurrentIndex(0)
        self.segmentation.finish_stroke()
        self.segmentation.extraction.clear_preview()
        self.postop.draw_mode.setCurrentIndex(0)
        self.control_stack.setCurrentIndex({3:4,4:2,5:3}.get(index,index))
        for i in range(self.control_stack.count()):
            self.control_stack.widget(i).setSizePolicy(QSizePolicy.Policy.Ignored,
                QSizePolicy.Policy.Preferred if i==self.control_stack.currentIndex() else QSizePolicy.Policy.Ignored)
        self.control_stack.adjustSize()
        if index != 1: self.segmentation.mode.setCurrentIndex(0)
        self.segmentation.update_tools()
        if index != 4: self.results.stop()
        self.results.change_view()
        if index == 5: self.diffusion.refresh_regions()
        self.diffusion.update_roi_preview()
        if index == 3:
            self.postop.refresh_masks(); self.postop.update_controls(); self.postop.apply_view()
        self.update_layers()

    def message(self, text):
        self.statusBar().showMessage(tr(text))

    def change_language(self, index):
        language = "en" if index else "ja"
        set_language(language)
        translate_widgets(self)
        self.refresh_import_feedback()
        self.diffusion.select_model()
        if self.scene is not None:
            self.select_extra()
            self.review.refresh()
            self.segmentation.refresh_info()
            self.results.refresh()
            self.postop.refresh_stages(self.postop.stage.currentData()); self.postop.update_controls()
            for panel in self.slice_panels(): panel.canvas.update()
        if not self._smoke:
            folder = PROJECT_ROOT / "local_data"
            folder.mkdir(exist_ok=True)
            (folder / "preferences.json").write_text(json.dumps({"language":language}), encoding="utf-8")

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 10, 0)
        layout.setSpacing(12)
        self.image_layers = LayerList('画像レイヤー')
        self.image_empty = QLabel(tr('画像はまだありません'))
        self.image_empty.setObjectName('muted')
        self.image_layers.rows_layout.addWidget(self.image_empty)
        mri_row = self.image_layers.add_layer('mri','基準MRI',True,100)
        ct_row = self.image_layers.add_layer('ct','CT',True,65)
        ct_row.allow_removal(lambda:self.remove_image_layer('ct'))
        self.mri_visible, self.mri_opacity = mri_row.check, mri_row.slider
        self.ct_visible, self.ct_opacity, self.ct_opacity_label = ct_row.check, ct_row.slider, ct_row.percent
        self.image_layers.changed.connect(self.update_layers)
        layout.addWidget(self.image_layers)
        image_group = QGroupBox(tr('基準MRI'))
        image_layout = QVBoxLayout(image_group)
        self.image_info = QLabel(tr("3方向の断面を準備中"))
        self.image_info.setObjectName("muted")
        self.image_info.setWordWrap(True)
        image_layout.addWidget(self.image_info)
        for title, name in ((tr("表示幅（コントラスト）"), "window_spin"), (tr("中心値（明るさ）"), "level_spin")):
            image_layout.addWidget(QLabel(title))
            spin = QDoubleSpinBox()
            spin.setDecimals(1)
            spin.setRange(-1e6, 1e6)
            spin.setSingleStep(25)
            spin.valueChanged.connect(self.update_contrast)
            setattr(self, name, spin)
            image_layout.addWidget(spin)
        self.window_spin.setMinimum(.1)
        automatic = QPushButton(tr("コントラストを自動調整"))
        automatic.clicked.connect(self.auto_contrast)
        image_layout.addWidget(automatic)
        image_layout.addWidget(QLabel(tr("右ドラッグで調整する画像")))
        self.window_target = QComboBox()
        self.window_target.setObjectName('imageNames')
        self.window_target.addItem(tr('基準MRI'), "mri")
        self.window_target.addItem("CT", "ct")
        image_layout.addWidget(self.window_target)
        self.annotation_check = QCheckBox(tr("マウス位置の解剖ラベル"))
        self.annotation_check.setChecked(True)
        self.annotation_check.toggled.connect(self.update_layers)
        image_layout.addWidget(self.annotation_check)
        layout.addWidget(image_group)
        self.import_group = self.build_import_panel()
        self.build_extra_controls()
        ct_group = QGroupBox(tr("術後CT"))
        ct_layout = QVBoxLayout(ct_group)
        self.ct_mode = QComboBox()
        self.ct_mode.addItem(tr("骨・電極を重ねる"), "bone")
        self.ct_mode.addItem(tr("CT全体を重ねる"), "full")
        self.ct_mode.addItem(tr("市松模様で照合"), "checker")
        self.ct_mode.addItem(tr("CTの骨の輪郭"), "edges")
        self.ct_mode.currentIndexChanged.connect(self.update_layers)
        ct_layout.addWidget(self.ct_mode)
        for label, attribute, value in ((tr("幅"), "ct_window", 2000), (tr("中心"), "ct_level", 1000)):
            ct_numbers = QHBoxLayout()
            ct_numbers.addWidget(QLabel(label))
            spin = QDoubleSpinBox()
            spin.setDecimals(0)
            spin.setRange(.1 if attribute == "ct_window" else -5000, 20000)
            spin.setSingleStep(100)
            spin.setValue(value)
            spin.valueChanged.connect(self.update_layers)
            setattr(self, attribute, spin)
            ct_numbers.addWidget(spin, 1)
            ct_layout.addLayout(ct_numbers)
        self.ct_info = QLabel(tr("CT未読込"))
        self.ct_info.setObjectName("muted")
        self.ct_info.setWordWrap(True)
        ct_layout.addWidget(self.ct_info)
        layout.addWidget(ct_group)
        electrode_group = QGroupBox(tr("電極・コンタクト"))
        electrode_layout = QVBoxLayout(electrode_group)
        self.contacts_check = QCheckBox(tr("コンタクト中心を表示"))
        self.contacts_check.setChecked(True)
        self.contacts_check.toggled.connect(self.update_layers)
        electrode_layout.addWidget(self.contacts_check)
        self.electrode_group = QComboBox()
        self.electrode_group.setObjectName("dataNames")
        self.electrode_group.addItem(tr("すべての電極"), "")
        self.electrode_group.currentIndexChanged.connect(self.update_contact_list)
        electrode_layout.addWidget(self.electrode_group)
        self.contact_combo = QComboBox()
        self.contact_combo.setObjectName("dataNames")
        self.contact_combo.addItem(tr("コンタクトへ移動"), None)
        self.contact_combo.activated.connect(self.select_contact)
        electrode_layout.addWidget(self.contact_combo)
        self.edit_contacts_button = QPushButton(tr("コンタクトを確認・修正"))
        self.edit_contacts_button.setObjectName("primary")
        self.edit_contacts_button.clicked.connect(self.edit_contacts)
        electrode_layout.addWidget(self.edit_contacts_button)
        self.detect_contacts_button = QPushButton(tr("CTから電極候補を作成"))
        self.detect_contacts_button.clicked.connect(self.detect_contacts)
        electrode_layout.addWidget(self.detect_contacts_button)
        self.tip_button = QPushButton(tr("1. 選択位置を先端として記録"))
        self.tip_button.clicked.connect(self.capture_tip)
        electrode_layout.addWidget(self.tip_button)
        self.last_button = QPushButton(tr("2. 最後端を記録して電極を追加"))
        self.last_button.clicked.connect(self.capture_last)
        self.last_button.setEnabled(False)
        electrode_layout.addWidget(self.last_button)
        self.electrode_info = QLabel(tr("電極未読込"))
        self.electrode_info.setObjectName("muted")
        self.electrode_info.setWordWrap(True)
        electrode_layout.addWidget(self.electrode_info)
        export_contacts = QPushButton(tr("電極座標を出力"))
        export_contacts.clicked.connect(self.export_contacts)
        electrode_layout.addWidget(export_contacts)
        self.contact_labels_button=QPushButton(tr('解剖ラベル一覧・CSV出力'))
        self.contact_labels_button.clicked.connect(self.show_contact_labels)
        electrode_layout.addWidget(self.contact_labels_button)
        layout.addWidget(electrode_group)
        surface_group = self.model_layers = LayerList('3Dモデル')
        brain_row = self.model_layers.add_layer('brain','脳表',True,100)
        nuclei_row = self.model_layers.add_layer('nuclei','視床核',False,90)
        self.brain_visible = brain_row.check
        self.opacity_slider, self.opacity_label = brain_row.slider, brain_row.percent
        self.nuclei_check, self.nuclei_opacity = nuclei_row.check, nuclei_row.slider
        self.model_layers.changed.connect(self.update_surfaces)
        self.model_layers.changed.connect(self.update_layers)
        surface_layout = surface_group.rows_layout
        self.surface_mode = QComboBox()
        self.surface_mode.addItem(tr("脳表（pial）"), "pial")
        self.surface_mode.addItem(tr("白質境界（white）"), "white")
        self.surface_mode.currentIndexChanged.connect(self.update_surfaces)
        surface_layout.addWidget(self.surface_mode)
        checks = QHBoxLayout()
        self.left_check = QCheckBox(tr("左半球"))
        self.right_check = QCheckBox(tr("右半球"))
        for check in (self.left_check, self.right_check):
            check.setChecked(True)
            check.toggled.connect(self.update_surfaces)
            checks.addWidget(check)
        surface_layout.addLayout(checks)
        self.planes_check = QCheckBox(tr("MRI断面を3Dにも表示"))
        self.planes_check.toggled.connect(self.update_surfaces)
        surface_layout.addWidget(self.planes_check)
        self.mesh_export_button = QPushButton(tr("表示中の3Dモデルを出力"))
        self.mesh_export_button.clicked.connect(self.export_mesh)
        surface_layout.addWidget(self.mesh_export_button)
        self.nuclei_2d_check = QCheckBox(tr("視床核を2Dにも重ねる"))
        self.nuclei_2d_check.toggled.connect(self.update_layers)
        surface_layout.addWidget(self.nuclei_2d_check)
        self.nucleus_combo = QComboBox()
        self.nucleus_combo.setObjectName("dataNames")
        self.nucleus_combo.addItem(tr("すべての視床核"), 0)
        self.nucleus_combo.currentIndexChanged.connect(self.update_layers)
        surface_layout.addWidget(self.nucleus_combo)
        nucleus_jump = QPushButton(tr("選択した視床核へ移動"))
        nucleus_jump.clicked.connect(self.select_nucleus)
        surface_layout.addWidget(nucleus_jump)
        self.nucleus_info = QLabel(tr("視床核未読込"))
        self.nucleus_info.setWordWrap(True)
        self.nucleus_info.setObjectName("muted")
        surface_layout.addWidget(self.nucleus_info)
        layout.addWidget(surface_group)
        coordinates = QGroupBox(tr("選択位置"))
        coord_layout = QVBoxLayout(coordinates)
        self.ras_label = QLabel("R  —\nA  —\nS  —")
        self.ras_label.setObjectName("coordinate")
        coord_layout.addWidget(self.ras_label)
        self.anatomy_label = QLabel(tr("解剖ラベル未読込"))
        self.anatomy_label.setStyleSheet("color:#a5e9d6;")
        self.anatomy_label.setWordWrap(True)
        coord_layout.addWidget(self.anatomy_label)
        self.voxel_label = QLabel(tr("元MRI voxel：—"))
        self.voxel_label.setObjectName("muted")
        self.voxel_label.setWordWrap(True)
        coord_layout.addWidget(self.voxel_label)
        annotation = QLabel(tr("患者固有の物理座標です。\nMNI座標ではありません。"))
        annotation.setObjectName("muted")
        coord_layout.addWidget(annotation)
        reset = QPushButton(tr("断面・3Dを初期表示へ"))
        reset.clicked.connect(self.reset_views)
        coord_layout.addWidget(reset)
        layout.addWidget(coordinates)
        help_text = QLabel(tr("2D：マウス位置に解剖ラベル\n右ドラッグ：window / level\nホイールで断面を移動\nCtrl＋ホイールで拡大\n\n3D：左ドラッグで回転\nダブルクリックで位置を選択"))
        help_text.setObjectName("muted")
        help_text.setWordWrap(True)
        # Keep every control reachable without a horizontally clipped sidebar.
        for group in (self.image_layers, image_group, ct_group, electrode_group, surface_group, coordinates):
            layout.removeWidget(group)
        self.side_tabs = QTabWidget()
        self.side_tabs.setObjectName("sideTabs")
        for title, groups in ((tr("画像"), (self.import_group, self.image_layers, self.extra_group, image_group, ct_group)), ("3D", (surface_group,)),
                              (tr("電極"), (electrode_group,)), (tr("位置"), (coordinates,))):
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(0, 3, 0, 0)
            page_layout.setSpacing(10)
            for group in groups:
                group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
                page_layout.addWidget(group)
            page_layout.addStretch()
            self.side_tabs.addTab(page, title)
        self.review = AlignmentReview(self)
        self.side_tabs.addTab(self.review, tr("照合"))
        layout.addWidget(self.side_tabs, 1)
        layout.addWidget(help_text)
        for label in panel.findChildren(QLabel):
            label.setWordWrap(True)
        for widget in panel.findChildren(QWidget):
            if isinstance(widget, (QComboBox, QPushButton, QDoubleSpinBox, QCheckBox, QTabWidget)):
                policy = widget.sizePolicy()
                policy.setHorizontalPolicy(QSizePolicy.Policy.Minimum if widget.property('layerCheck') else QSizePolicy.Policy.Ignored)
                widget.setSizePolicy(policy)
        return panel

    def initialize(self):
        self.surface.initialize()

    def _set_busy(self, busy):
        busy = busy or self.worker is not None or self.results.exporting
        if busy:
            self.segmentation.finish_stroke()
            self.results.stop()
            for panel in self.slice_panels(): panel.canvas._end_stroke()
        self.controls.setEnabled(not busy)
        self.segmentation.setEnabled(not busy and self.scene is not None)
        self.results.setEnabled(not busy and self.scene is not None)
        self.diffusion.setEnabled(not busy and self.scene is not None)
        self.postop.setEnabled(not busy and self.scene is not None)
        if self.results.mapping_dialog: self.results.mapping_dialog.setEnabled(not busy)
        self.language_tabs.setEnabled(not busy)
        self.workflow_tabs.setEnabled(not busy)
        self.workflow_tabs.setTabEnabled(1, self.scene is not None)
        self.workflow_tabs.setTabEnabled(3, self.scene is not None)
        self.workflow_tabs.setTabEnabled(4, self.scene is not None)
        self.workflow_tabs.setTabEnabled(5, self.scene is not None)
        self.views.setEnabled(not busy and self.scene is not None)
        for index in range(1, self.side_tabs.count()): self.side_tabs.setTabEnabled(index, self.scene is not None)
        for group in (self.image_layers, self.extra_group): group.setEnabled(not busy and self.scene is not None)
        for button in (self.save_button, self.export_button):
            button.setEnabled(not busy and self.scene is not None)
        for widget in (self.patient_combo, self.new_patient_button, self.patient_menu_button): widget.setEnabled(not busy)
        self.refresh_import_choices()

    def run_job(self, operation, callback, description, on_error=None):
        if self.worker is not None:
            return
        self._set_busy(True)
        self._job_started = time.perf_counter()
        self.worker = Job(operation, self)
        self.worker.completed_callback = callback
        self.worker.error_callback = on_error or self._job_failed
        self._update_job_progress(description)
        self.worker.progress.connect(self._update_job_progress)
        # A modal callback can run a nested Qt event loop. Finish and release the
        # worker before dispatching any callbacks, so its finished signal cannot
        # consume a pending follow-up while a source picker is still open.
        self.worker.finished.connect(self._job_finished)
        self._job_timer.start()
        self.worker.start()

    def _update_job_progress(self, text=None):
        if self.worker is None: return
        if text is not None:
            self._job_stage = text
        seconds = int(time.perf_counter()-self._job_started)
        self.message(tr('{stage}（経過 {seconds} 秒）').format(stage=tr(self._job_stage), seconds=seconds))

    def _job_finished(self):
        job = self.worker
        if job is None: return
        self._job_timer.stop()
        self.worker = None
        self._set_busy(False)
        self.message(tr('処理が完了しました。'))
        try:
            if job.error is not None:
                job.error_callback(job.error)
            else:
                job.completed_callback(job.value)
        except Exception:
            log_error()
            self._job_failed(tr('画面を更新できませんでした。詳細はローカルのエラーログに記録しました。'))
        finally:
            job.deleteLater()
        # The completed callback may already have started the next worker.
        if self.worker is not None: return
        if self._closing:
            self.close()
        elif self._after_job:
            callback, self._after_job = self._after_job, None
            QTimer.singleShot(0, callback)

    def _job_failed(self, message):
        self._closing = False
        self._after_job = None
        self.refresh_patients()
        message = tr(message)
        self.message(tr('処理を完了できませんでした：{reason}').format(reason=message))
        self.scene_failed.emit(message)
        if not self._smoke:
            QMessageBox.warning(self, tr("読み込み・保存の確認"), message)

    def load_default(self, root: Path, demo=False):
        def load(progress):
            if demo:
                return make_demo()
            from .multimodal import load_workspace
            return load_workspace(root, PROJECT_ROOT / "local_data" / "registration", progress)
        self.run_job(load, self.install_scene, tr("MRIを読み込んでいます…"))

    def update_electrode_info(self):
        scene = self.scene
        self.detect_contacts_button.setText(tr('CTから再抽出' if scene.contacts else 'CTから電極候補を作成'))
        reviewed = sum(c.status == "reviewed" for c in scene.contacts)
        self.electrode_info.setText(tr("CT由来：{groups}本 / {count}個\n確認済み：{reviewed}個\n仮の電極名・先端順は確認が必要").format(
            groups=len({c.group for c in scene.contacts}), count=len(scene.contacts), reviewed=reviewed))
        native = scene.raw_ct is not None and scene.ct_to_mri is not None
        self.edit_contacts_button.setEnabled(native and bool(scene.contacts) and all(c.ct_position is not None for c in scene.contacts))
        self.contact_labels_button.setEnabled(bool(scene.contacts))
        self.detect_contacts_button.setEnabled(native)
        self.tip_button.setEnabled(native)
        self.last_button.setEnabled(native and self.pending_tip is not None)

    def replace_electrodes(self, result):
        # Geometry is shared; updating contact actors preserves all image and camera settings.
        selected = self.electrode_group.currentData()
        self.scene.contacts = result.contacts
        self.scene.electrode_quality = result.electrode_quality
        for panel in self.slice_panels(): panel.canvas.scene = self.scene
        self.surface.refresh_contacts(self.scene)
        self.electrode_group.blockSignals(True)
        self.electrode_group.clear()
        self.electrode_group.addItem(tr("すべての電極"), "")
        for group in dict.fromkeys(c.group for c in self.scene.contacts): self.electrode_group.addItem(group, group)
        self.electrode_group.setCurrentIndex(max(0,self.electrode_group.findData(selected)))
        self.electrode_group.blockSignals(False)
        self.update_contact_list()
        self.update_electrode_info()
        self.write_status("ready")

    def edit_contacts(self, checked=False, candidate=None, group=None):
        if self.scene is None or self.scene.raw_ct is None: return
        target = candidate if candidate is not None else self.scene
        if not target.contacts:
            self.message(tr('電極候補は0個でした。CTの種類・濃度や撮像範囲を確認し、必要に応じて2点から電極を追加してください。'))
            return
        editor = ContactEditor(target, self)
        if group: editor.groups.setCurrentText(group)
        if editor.exec() == QDialog.DialogCode.Accepted:
            self.replace_electrodes(editor.scene)
        editor.deleteLater()

    def detect_contacts(self):
        if self.scene is None: return
        from .multimodal import create_ct_contacts
        cache = (self.patient_store.folder(self.patient_id)/'cache'/'electrodes' if self.patient_id
                 else PROJECT_ROOT/'local_data'/'electrodes')
        self.run_job(lambda progress: create_ct_contacts(self.scene, cache, progress),
                     self._contacts_detected, tr("元CTからコンタクト候補を作成しています…"))

    def _contacts_detected(self, candidate):
        if not candidate.contacts:
            message = tr('電極候補は0個でした。CTの種類・濃度や撮像範囲を確認し、必要に応じて2点から電極を追加してください。')
            self.message(message)
            self.electrode_info.setText(message)
            self.write_status('ready', {'last_operation': 'contact_detection', 'candidate_count': 0})
            return
        self.message(tr('電極候補 {count}個を作成しました。確認画面を開きます。').format(count=len(candidate.contacts)))
        # End the background job and re-enable the viewer before opening a modal editor.
        self._after_job = lambda: self.edit_contacts(candidate=candidate)

    def capture_tip(self):
        if self.scene is None or self.scene.raw_ct is None: return
        self.pending_tip = self.scene.world(self.ijk).copy()
        self.last_button.setEnabled(True)
        self.message(tr("先端のコンタクト中心を記録しました。最後端の中心を選んで手順2へ進んでください。"))

    def capture_last(self):
        if self.scene is None or self.pending_tip is None: return
        import nibabel as nib
        inverse = np.linalg.inv(self.scene.ct_to_mri)
        tip = nib.affines.apply_affine(inverse,self.pending_tip)
        last = nib.affines.apply_affine(inverse,self.scene.world(self.ijk))
        count, ok = QInputDialog.getInt(self, tr("電極を追加"), tr("この電極のコンタクト数"), 8, 2, 64)
        if not ok: return
        number = 1
        while f"E{number:02d}" in {c.group for c in self.scene.contacts}: number += 1
        name, ok = QInputDialog.getText(self, tr("電極を追加"), tr("電極名"), text=f"E{number:02d}")
        if not ok: return
        try:
            candidate = create_lead(self.scene,tip,last,count,name.strip())
        except InputError as exc:
            QMessageBox.warning(self,tr("電極の確認"),tr(str(exc)))
            return
        self.edit_contacts(candidate=candidate,group=name.strip())
        self.pending_tip = None
        self.last_button.setEnabled(False)

    def install_scene(self, scene: Scene):
        self.message(tr("3D表示を準備しています…"))
        try:
            self.segmentation.finish_stroke()
            self.scene = scene
            self._restoring = True
            self.refresh_mri_layers()
            for panel in self.slice_panels():
                panel.set_scene(scene)
            self.surface_mode.setCurrentIndex(0)
            self.surface_mode.model().item(1).setEnabled(all(k in scene.surfaces for k in ("white_lh", "white_rh")))
            self.left_check.setChecked(True)
            self.right_check.setChecked(True)
            self.brain_visible.setChecked(True)
            self.opacity_slider.setValue(32 if scene.contacts else 100)
            self.planes_check.setChecked(False)
            self.mri_visible.setChecked(True)
            self.mri_opacity.setValue(100)
            self.ct_visible.setChecked(scene.ct is not None)
            self.image_layers.rows['ct'].setEnabled(scene.ct is not None)
            self.ct_mode.setCurrentIndex(0)
            self.ct_opacity.setValue(65)
            self.ct_window.setValue(2000)
            self.ct_level.setValue(1000)
            self.window_target.setCurrentIndex(0)
            self.window_target.model().item(1).setEnabled(scene.ct is not None)
            self.annotation_check.setChecked(True)
            self.contacts_check.setChecked(True)
            self.nuclei_check.setChecked(False)
            self.nuclei_opacity.setValue(90)
            self.model_layers.rows['nuclei'].setEnabled(scene.nuclei is not None)
            self.nuclei_2d_check.setChecked(False)
            self.nuclei_2d_check.setEnabled(scene.nuclei is not None)
            self.nucleus_combo.clear()
            self.nucleus_combo.addItem(tr("すべての視床核"), 0)
            for number, name in sorted(scene.nuclei_names.items()):
                self.nucleus_combo.addItem(name, number)
            self.electrode_group.blockSignals(True)
            self.electrode_group.clear()
            self.electrode_group.addItem(tr("すべての電極"), "")
            for group in dict.fromkeys(c.group for c in scene.contacts):
                self.electrode_group.addItem(group, group)
            self.electrode_group.blockSignals(False)
            self.surface.set_scene(scene)
            self.segmentation.bind_scene()
            self.results.bind_scene()
            self.diffusion.bind_scene()
            self.postop.bind_scene()
            self.views.restore({})
            self.surface.presets.setCurrentIndex(0)
            self._restoring = False
            self.set_cursor(scene.initial_index())
            self.auto_contrast()
            self.update_surfaces()
            self.update_contact_list()
            self.update_layers()
            if scene.view_state:
                self.restore_state(scene.view_state)
            spacing = " × ".join(f"{v:.1f}" for v in scene.spacing)
            self.image_info.setText(tr(f"MRI  /  RAS断面\n{spacing} mm\n元のMRI：") + " × ".join(str(v) for v in scene.source_shape))
            self.ct_info.setText(tr("自動位置合わせ済み・目視確認前\nCTは金色で表示") if scene.ct is not None else tr("CT未読込"))
            if scene.quality.get('ct_input_selection_required') and scene.ct is None:
                self.ct_info.setText(tr("CTが複数あります。CTを開く操作で撮像を選択してください。"))
            if scene.ct_quality.get("review_status") == "manually_adjusted_requires_review":
                self.ct_info.setText(tr("手動補正あり・目視確認前"))
            if scene.ct_quality.get("version") == "rigid-mi-v1":
                self.ct_visible.setChecked(False)
                self.ct_info.setText(tr("旧方式のCT位置合わせです。作業フォルダから読み直して再登録してください。"))
            if scene.nuclei is not None:
                spacing = np.linalg.norm(scene.nuclei_affine[:3,:3],axis=0)
                self.nucleus_info.setText(tr("視床核：{count}ラベル\n元格子：{spacing} mm\nFreeSurferの推定領域").format(count=len(scene.nuclei_names),spacing=" × ".join(f"{s:.1f}" for s in spacing)))
            else:
                self.nucleus_info.setText(tr("視床核未読込"))
            self.review.pending_mri = None
            self.review.refresh()
            self.pending_tip = None
            self.last_button.setEnabled(False)
            self.update_electrode_info()
            if scene.contacts and not scene.view_state:
                self.side_tabs.setCurrentIndex(2)
            self.message(tr("右ドラッグでwindow調整。CTの重なりを切り替えて位置合わせを確認できます。"))
            self._set_busy(False)
            self.refresh_import_choices()
            QTimer.singleShot(150, self._scene_ready)
        except Exception:
            self._restoring = False
            log_error()
            self.clear_scene()
            self._job_failed(tr("画面を準備できませんでした。詳細はローカルのエラーログに記録しました。"))

    def _scene_ready(self):
        self.surface.render()
        self.write_status("ready")
        self.scene_ready.emit()

    def clear_scene(self):
        self.clear_import_feedback()
        self.segmentation.finish_stroke()
        self.scene = None
        for panel in self.slice_panels(): panel.clear_scene()
        self.surface.clear_scene()
        for panel in self.views.trajectories: panel.refresh()
        self.segmentation.bind_scene()
        self.results.bind_scene()
        self.diffusion.bind_scene()
        self.postop.bind_scene()
        self.workflow_tabs.setCurrentIndex(0)
        self.pending_tip = None
        self.refresh_mri_layers()
        self.refresh_import_choices()
        self.image_info.setText(tr('画像はまだありません'))
        self.ct_info.setText(tr('CT未読込'))
        self._set_busy(False)
        self.message(tr('左側から基準MRIを追加してください'))
        self.write_status('empty_patient')

    def write_status(self, phase: str, extra=None):
        folder = PROJECT_ROOT / "private_reports"
        folder.mkdir(exist_ok=True)
        report = {"pid": os.getpid(), "state": phase, "kind": self.scene.kind if self.scene else None,
                  "display_shape": list(self.scene.data.shape) if self.scene else None,
                  "surface_count": len(self.scene.surfaces) if self.scene else 0,
                  "renders": self.surface.render_count,
                  "version": __version__, "visible": self.isVisible(),
                  "nuclei": len(self.scene.nuclei_names) if self.scene else 0,
                  "ct_loaded": self.scene is not None and self.scene.ct is not None,
                  "contacts": len(self.scene.contacts) if self.scene else 0,
                  "electrode_source": self.scene.electrode_quality.get("source") if self.scene else None,
                  "reviewed_contacts": sum(c.status == "reviewed" for c in self.scene.contacts) if self.scene else 0,
                  "patient_selected": self.patient_id is not None,
                  "additional_mri_count": len(self.scene.extra_mris) if self.scene else 0,
                  "segmentation_count": len(self.scene.segmentations) if self.scene else 0,
                  "analysis_result_count": len(self.scene.results) if self.scene else 0,
                  "diffusion_count": len(self.scene.diffusions) if self.scene else 0,
                  "tract_bundle_count": len(self.scene.tracts) if self.scene else 0,
                  "annotation_loaded": self.scene is not None and self.scene.label_volume is not None}
        if extra:
            report.update(extra)
        content = json.dumps(report, indent=2)
        (folder / "viewer_session.json").write_text(content, encoding="utf-8")
        if phase == "smoke_passed":
            suffix = "synthetic" if self._demo or (self.scene and self.scene.kind == "synthetic") else "local_mri"
            (folder / f"validation_{suffix}.json").write_text(content, encoding="utf-8")

    def set_cursor(self, ijk):
        if self.scene is None:
            return
        self.ijk = np.clip(np.rint(ijk).astype(int), 0, np.array(self.scene.data.shape) - 1)
        for panel in self.slice_panels():
            panel.set_cursor(self.ijk)
        self.surface.set_cursor(self.ijk)
        world = self.scene.world(self.ijk)
        self.ras_label.setText("\n".join(f"{axis}  {value:+8.1f} mm" for axis, value in zip("RAS", world)))
        native = self.scene.native_index(world)
        outside = bool(np.any(native < -.5) or np.any(native > np.array(self.scene.source_shape) - .5))
        self.voxel_label.setText(tr("元MRI voxel（0始まり）\n") + ", ".join(f"{v:.1f}" for v in native) + (tr("\n撮像範囲外") if outside else ""))
        label, name = self.scene.annotation(self.ijk)
        self.anatomy_label.setText(name + (f"  [ {label} ]" if label else ""))

    def select_world(self, world):
        if self.scene is not None:
            if self.scene.contacts:
                positions = self.scene.contact_positions()
                distances = np.linalg.norm(positions-np.asarray(world), axis=1)
                nearest = int(np.argmin(distances))
                if distances[nearest] < 2:
                    combo_index = self.contact_combo.findData(nearest)
                    if combo_index >= 0:
                        self.contact_combo.setCurrentIndex(combo_index)
                        self.update_layers()
            self.set_cursor(self.scene.index(world))

    def drag_window(self, dx, dy):
        if self.scene is None:
            return
        extra = self.extra_layer(self.window_target.currentData()) if self.window_target.currentData() not in ('mri','ct') else None
        if extra is not None:
            extra.window = float(np.clip(extra.window*np.exp(np.clip(dx/160,-2,2)), .001, 1e9))
            extra.level += dy*extra.window/180
            self.extra_combo.setCurrentIndex(self.extra_combo.findData(extra.uid))
            self.select_extra(); self.update_layers()
            if self.workflow_tabs.currentIndex()==3: self.postop.sync_contrast()
            return
        is_ct = self.window_target.currentData() == "ct" and self.scene.ct is not None
        width_spin, center_spin = (self.ct_window, self.ct_level) if is_ct else (self.window_spin, self.level_spin)
        width = width_spin.value()
        width_spin.blockSignals(True)
        center_spin.blockSignals(True)
        width_spin.setValue(float(np.clip(width*np.exp(np.clip(dx/160, -2, 2)), 1, 20000)))
        center_spin.setValue(center_spin.value() + dy * width / 180)
        width_spin.blockSignals(False)
        center_spin.blockSignals(False)
        self.update_layers() if is_ct else self.update_contrast()

    def ct_options(self):
        return {"visible": self.ct_visible.isChecked(), "opacity": self.ct_opacity.value()/100,
                "mri_visible": self.mri_visible.isChecked(), "mri_opacity":self.mri_opacity.value()/100,
                "window": self.ct_window.value(), "level": self.ct_level.value(),
                "nuclei_visible": self.nuclei_2d_check.isChecked(), "nucleus": self.nucleus_combo.currentData() or 0,
                "mode": self.ct_mode.currentData(), "color": "amber", "extra_mris": self.extra_options(),
                "segmentation_context": {
                    "workflow": {1:'segmentation',3:'postop'}.get(self.workflow_tabs.currentIndex(),'images'),
                    "exclusion": self.postop.manual_uid}}

    def contact_options(self):
        selected = self.contact_combo.currentData()
        name = self.scene.contacts[selected].name if self.scene and isinstance(selected, int) and selected < len(self.scene.contacts) else ""
        return {"visible": self.contacts_check.isChecked(), "group": self.electrode_group.currentData() or "",
                "source": False, "tolerance": 2., "selected": name}

    def update_layers(self, *_):
        if self.scene is None or self._restoring:
            return
        ct, contacts = self.ct_options(), self.contact_options()
        has_mri = (ct['mri_visible'] and ct['mri_opacity']>0) or any(s['visible'] and s['opacity']>0 for s in ct['extra_mris'].values())
        self.ct_mode.setEnabled(self.scene.ct is not None and ct['visible'] and has_mri)
        self.ct_opacity_label.setText(f"{self.ct_opacity.value()}%")
        for panel in self.slice_panels():
            panel.canvas.set_layers(ct, contacts, self.annotation_check.isChecked())
        self.surface.set_ct_options(ct)
        self.surface.set_nuclei_options(self.nuclei_check.isChecked(),ct["nucleus"],self.nuclei_opacity.value()/100)
        self.surface.set_contact_options(contacts)
        self.surface.refresh_segmentations(self.scene)
        self.views.refresh_trajectories()
        self.results.refresh()
        if self.scene.ct is not None:
            self.message(tr("CT由来のコンタクトを表示。電極タブから元CT上で確認・修正できます。") if self.scene.contacts else
                         tr('CTを表示しています。電極候補は電極タブから作成できます。'))

    def update_contact_list(self, *_):
        if self.scene is None or self._restoring:
            return
        group = self.electrode_group.currentData()
        previous = self.contact_combo.currentData()
        self.contact_combo.blockSignals(True)
        self.contact_combo.clear()
        self.contact_combo.addItem(tr("コンタクトへ移動"), None)
        for index, contact in enumerate(self.scene.contacts):
            if not group or contact.group == group:
                self.contact_combo.addItem(contact.name, index)
        found = self.contact_combo.findData(previous)
        self.contact_combo.setCurrentIndex(max(found, 0))
        self.contact_combo.blockSignals(False)
        self.update_layers()

    def select_contact(self, index):
        if self.scene is None:
            return
        contact = self.contact_combo.itemData(index)
        if not isinstance(contact, int):
            return
        point = self.scene.contact_positions()[contact]
        self.set_cursor(self.scene.index(point))
        self.slices[2].canvas.show_annotation = True
        self.update_layers()

    def select_nucleus(self):
        if self.scene is None or self.scene.nuclei is None:
            return
        number = self.nucleus_combo.currentData()
        mask = self.scene.nuclei == number if number else self.scene.nuclei > 0
        indices = np.argwhere(mask)
        if not len(indices): return
        center = np.mean(indices, axis=0)
        # Select a real voxel of the nucleus, not a centre outside a concave region.
        nearest = indices[np.argmin(np.sum((indices-center)**2,axis=1))]
        world = self.scene.nuclei_affine[:3,:3] @ nearest + self.scene.nuclei_affine[:3,3]
        self.nuclei_check.setChecked(True)
        self.nuclei_2d_check.setChecked(True)
        self.opacity_slider.setValue(min(self.opacity_slider.value(),25))
        self.set_cursor(self.scene.index(world))
        self.slices[2].canvas.show_annotation = True
        self.update_layers()

    def move_slice(self, axis, value):
        ijk = self.ijk.copy()
        ijk[axis] = value
        self.set_cursor(ijk)

    def step_slice(self, axis, delta):
        self.move_slice(axis, int(self.ijk[axis]) + delta)

    def update_contrast(self, *_):
        if self.scene is None or self._restoring:
            return
        width, center = self.window_spin.value(), self.level_spin.value()
        low, high = center - width/2, center + width/2
        for panel in self.slice_panels():
            panel.canvas.set_contrast(low, high)
        self.surface.set_contrast(low, high)
        self.views.refresh_trajectories()

    def auto_contrast(self):
        if self.scene is None:
            return
        low, high = self.scene.contrast_limits()
        self.window_spin.blockSignals(True)
        self.level_spin.blockSignals(True)
        self.window_spin.setValue(high - low)
        self.level_spin.setValue((high + low) / 2)
        self.window_spin.blockSignals(False)
        self.level_spin.blockSignals(False)
        self.update_contrast()

    def update_surfaces(self, *_):
        if self.scene is None or self._restoring:
            return
        self.opacity_label.setText(f"{self.opacity_slider.value()}%")
        self.surface.set_surface_options(mode=self.surface_mode.currentData(), opacity=self.opacity_slider.value()/100,
                                         hemispheres={"lh": self.left_check.isChecked(), "rh": self.right_check.isChecked()},
                                         show_planes=self.planes_check.isChecked(),visible=self.brain_visible.isChecked())

    def reset_views(self):
        if self.scene is None:
            return
        self.set_cursor(self.scene.initial_index())
        for panel in self.slice_panels():
            panel.canvas.reset_view()
        self.surface.presets.setCurrentIndex(0)
        self.surface.set_camera_preset(0)

    def view_state(self) -> dict:
        return {"ijk": self.ijk.tolist(), "window": self.window_spin.value(), "level": self.level_spin.value(),
                "surface": self.surface_mode.currentData(), "opacity": self.opacity_slider.value(),
                "left": self.left_check.isChecked(), "right": self.right_check.isChecked(),
                "planes": self.planes_check.isChecked(), "camera": self.surface.camera_state(),
                "ct": self.ct_options(), "contacts": self.contact_options(),
                "image_layers":self.image_layers.state(),"model_layers":self.model_layers.state(),
                "extra_styles": dict(self.extra_styles), "extra_palettes": dict(self.extra_palettes),
                "extra_surfaces": dict(self.extra_surfaces),
                "extra_selected": self.extra_combo.currentData(),
                "diffusion": self.diffusion.state(),
                "postop": self.postop.state(),
                "panels": self.views.state(),
                "annotation": self.annotation_check.isChecked(), "window_target": self.window_target.currentData(),
                "slices": {str(axis): {"zoom": panel.canvas.zoom, "pan": [panel.canvas.pan.x(), panel.canvas.pan.y()]}
                           for axis, panel in self.slices.items()}}

    def restore_state(self, state: dict):
        try:
            self._restoring = True
            index = np.asarray(state.get("ijk", self.scene.initial_index()), dtype=float)
            if index.shape == (3,) and np.isfinite(index).all():
                self.set_cursor(index)
            self.window_spin.setValue(float(state.get("window", self.window_spin.value())))
            self.level_spin.setValue(float(state.get("level", self.level_spin.value())))
            mode = state.get("surface", "pial")
            if mode == "white" and all(k in self.scene.surfaces for k in ("white_lh", "white_rh")):
                self.surface_mode.setCurrentIndex(1)
            self.opacity_slider.setValue(int(state.get("opacity", 100)))
            self.brain_visible.setChecked(True)
            self.left_check.setChecked(bool(state.get("left", True)))
            self.right_check.setChecked(bool(state.get("right", True)))
            self.planes_check.setChecked(bool(state.get("planes", False)))
            ct = state.get("ct", {})
            self.mri_visible.setChecked(bool(ct.get('mri_visible',ct.get('mode')!='ct')))
            self.mri_opacity.setValue(round(float(ct.get('mri_opacity',1.))*100))
            self.ct_visible.setChecked(bool(ct.get("visible", self.scene.ct is not None)))
            self.ct_opacity.setValue(round(float(ct.get("opacity", .65))*100))
            self.ct_window.setValue(float(ct.get("window", 2000)))
            self.ct_level.setValue(float(ct.get("level", 1000)))
            mode = ct.get('mode','bone')
            self.ct_mode.setCurrentIndex(max(0, self.ct_mode.findData('full' if mode=='ct' else mode)))
            self.nuclei_check.setChecked(bool(ct.get("nuclei_visible",False)))
            self.nuclei_2d_check.setChecked(bool(ct.get("nuclei_visible",False)))
            self.nuclei_opacity.setValue(90)
            self.image_layers.restore(state.get('image_layers',{}))
            self.extra_styles = dict(state.get('extra_styles', {}))
            self.extra_palettes = dict(state.get('extra_palettes', {}))
            self.extra_surfaces = dict(state.get('extra_surfaces', {}))
            self.extra_combo.setCurrentIndex(max(0, self.extra_combo.findData(state.get('extra_selected'))))
            self.select_extra()
            self.diffusion.restore(state.get('diffusion',{}))
            self.postop.restore(state.get('postop',{}))
            self.model_layers.restore(state.get('model_layers',{}))
            self.nucleus_combo.setCurrentIndex(max(0,self.nucleus_combo.findData(ct.get("nucleus",0))))
            self.annotation_check.setChecked(bool(state.get("annotation", True)))
            self.window_target.setCurrentIndex(max(0, self.window_target.findData(state.get('window_target', 'mri'))))
            contacts = state.get("contacts", {})
            self.contacts_check.setChecked(bool(contacts.get("visible", True)))
            self.electrode_group.setCurrentIndex(max(0, self.electrode_group.findData(contacts.get("group", ""))))
            for axis, panel in self.slices.items():
                item = state.get("slices", {}).get(str(axis), {})
                zoom = float(item.get("zoom", 1))
                pan = np.asarray(item.get("pan", [0, 0]), dtype=float)
                if np.isfinite(zoom) and pan.shape == (2,) and np.isfinite(pan).all():
                    panel.canvas.zoom = float(np.clip(zoom, .3, 8))
                    panel.canvas.pan = QPointF(*np.clip(pan, -10000, 10000))
                    panel.canvas.update()
            self.surface.restore_camera(state.get("camera", {}))
            self.views.restore(state.get('panels',{}))
        except (ValueError, TypeError, AttributeError):
            self.message(tr("一部の表示設定を初期値に戻しました。"))
        finally:
            self._restoring = False
        self.update_contrast()
        self.update_surfaces()
        self.update_contact_list()
        selected = state.get("contacts", {}).get("selected", "")
        index = self.contact_combo.findText(selected)
        if index >= 0:
            self.contact_combo.setCurrentIndex(index)
        self.update_layers()

    def export_image(self):
        if self.scene is None:
            return
        if self.workflow_tabs.currentIndex()==4 and self.results.selected():
            self.results.export_png(); return
        default = self.export_directory()
        default.mkdir(exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(self, tr("現在の画面配置をPNGに保存"), str(default / "mri_views.png"), tr("PNG画像 (*.png)"))
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        self.surface.render()
        if self.capture_widget(self.views).save(path, "PNG"):
            self.message(tr("現在の画面配置をPNGに保存しました。"))
        else:
            QMessageBox.warning(self, tr("画像出力"), tr("保存できませんでした。保存先を確認してください。"))

    def capture_widget(self, widget):
        pixmap = widget.grab()
        painter = QPainter(pixmap)
        for surface in self.views.surface_panels():
            if surface.isVisible() and (widget is surface or widget.isAncestorOf(surface)):
                origin = surface.widget.mapTo(widget, QPoint(0, 0))
                painter.drawImage(QRectF(origin.x(), origin.y(), surface.widget.width(), surface.widget.height()),
                                  surface.capture_image())
        painter.end()
        return pixmap

    def export_mesh(self):
        if self.scene is None:
            return
        if not self.brain_visible.isChecked() or self.opacity_slider.value()==0 or (not self.left_check.isChecked() and not self.right_check.isChecked()):
            self.message(tr("出力する半球を選択してください。"))
            return
        default = self.export_directory()
        default.mkdir(exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(self, tr("表示中の脳表を出力（RAS / mm）"), str(default / "cortex.vtp"), tr("VTK脳表モデル (*.vtp)"))
        if path:
            if not path.lower().endswith(".vtp"):
                path += ".vtp"
            try:
                self.surface.export_surface(path)
                self.message(tr("3Dモデルを患者固有のRAS座標で保存しました。"))
            except Exception:
                log_error()
                QMessageBox.warning(self, tr("3D出力"), tr("保存できませんでした。保存先を確認してください。"))

    def export_contacts(self):
        if self.scene is None or not self.scene.contacts:
            return
        default = self.export_directory()
        default.mkdir(exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(self, tr("電極座標を保存（CT・MRI RAS / mm）"), str(default / "contacts.json"), tr("電極座標 (*.json)"))
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            Path(path).write_text(json.dumps(export_records(self.scene), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            self.message(tr("電極座標と解剖ラベルを保存しました。"))
        except Exception:
            log_error()
            QMessageBox.warning(self, tr("電極出力"), tr("保存できませんでした。保存先を確認してください。"))

    def show_contact_labels(self):
        if self.scene is None or not self.scene.contacts: return
        from .contact_labels_dialog import ContactLabelsDialog
        directory=self.export_directory(); directory.mkdir(exist_ok=True)
        dialog=ContactLabelsDialog(self.scene,directory,self)
        dialog.exec()
        dialog.deleteLater()

    def closeEvent(self, event):
        self.results.stop()
        if self.results.exporting:
            self.results._close_after_export=True
            self.results.cancel_export(); event.ignore(); return
        if self.worker is not None and self.worker.isRunning():
            self._closing = True
            self.message(tr("処理が終わり次第、画面を閉じます。"))
            event.ignore()
            return
        if not self._smoke and self.patient_id and self.scene is not None and not self._close_saved:
            self._closing = True
            self.save_work(closing=True)
            event.ignore()
            return
        self.write_status("closed")
        self.surface.shutdown()
        self.views.shutdown()
        event.accept()
