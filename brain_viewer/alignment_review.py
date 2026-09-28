"""Reversible rigid edits and image landmarks, always resampling the original CT."""
from .i18n import tr
from dataclasses import replace
import numpy as np
import nibabel as nib
from scipy.spatial.transform import Rotation
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QGridLayout, QHBoxLayout, QLabel,
    QPushButton, QDoubleSpinBox, QGroupBox)
from .imaging import InputError
from .registration import resample_ct


def rigid_delta(angles_degrees, translation_mm, center_ras):
    rotation = Rotation.from_euler("xyz", angles_degrees, degrees=True).as_matrix()
    center = np.asarray(center_ras)
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = center - rotation @ center + np.asarray(translation_mm)
    return matrix


def apply_alignment(scene, matrix, provenance=None):
    if scene.raw_ct is None:
        raise InputError(tr("この保存データには元CTがありません。作業フォルダから読み直してください。"))
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4,4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0,0,0,1]) or not np.allclose(matrix[:3,:3].T @ matrix[:3,:3], np.eye(3), atol=1e-5) or np.linalg.det(matrix[:3,:3]) < .99:
        raise InputError(tr("CT補正は剛体変換で指定してください。"))
    ct, valid = resample_ct(scene.raw_ct, scene.raw_ct_affine, scene, matrix)
    follow = matrix @ np.linalg.inv(scene.ct_to_mri)
    contacts = [replace(c, position=nib.affines.apply_affine(matrix, c.ct_position)
                        if c.ct_position is not None else nib.affines.apply_affine(follow, c.position)) for c in scene.contacts]
    quality = dict(scene.ct_quality)
    quality.setdefault("automatic_ct_to_mri_ras_mm", scene.ct_to_mri.tolist())
    quality["review_status"] = "manually_adjusted_requires_review"
    if "final_negative_mi" in quality:
        quality["automatic_final_negative_mi"] = quality.pop("final_negative_mi")
    if (provenance or {}).get("action") == "restore_automatic":
        quality["review_status"] = "automatic_alignment_requires_visual_review"
        if "automatic_final_negative_mi" in quality:
            quality["final_negative_mi"] = quality["automatic_final_negative_mi"]
    quality["manual_history"] = [*quality.get("manual_history", []), {"matrix":matrix.tolist(), **(provenance or {})}]
    electrode_quality = dict(scene.electrode_quality)
    if "raw_ct_to_reference_ras_mm" in electrode_quality:
        electrode_quality["reference_ct_to_mri_ras_mm"] = (matrix @ np.linalg.inv(electrode_quality["raw_ct_to_reference_ras_mm"])).tolist()
    return replace(scene, ct=ct, ct_valid=valid, ct_to_mri=matrix, ct_quality=quality,
                   contacts=contacts, electrode_quality=electrode_quality)


def landmark_residuals(matrix, pairs):
    if not pairs:
        return np.empty(0)
    source = np.array([p["ct_ras_mm"] for p in pairs])
    target = np.array([p["mri_ras_mm"] for p in pairs])
    return np.linalg.norm(nib.affines.apply_affine(matrix, source)-target, axis=1)


class AlignmentReview(QWidget):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.pending_mri = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        self.info = QLabel()
        self.info.setWordWrap(True)
        layout.addWidget(self.info)
        group = QGroupBox(tr("CTの微調整"))
        form = QGridLayout(group)
        form.addWidget(QLabel(tr("回転 °")),0,1)
        form.addWidget(QLabel(tr("移動 mm")),0,2)
        self.angles, self.shifts = [], []
        for row, text in enumerate((tr("R軸（矢状断）"), tr("A軸（冠状断）"), tr("S軸（水平断）")),1):
            form.addWidget(QLabel(text),row,0)
            for column, collection, limit in ((1,self.angles,45),(2,self.shifts,50)):
                spin = QDoubleSpinBox()
                spin.setDecimals(2); spin.setRange(-limit,limit); spin.setSingleStep(.1)
                form.addWidget(spin,row,column); collection.append(spin)
        note = QLabel(tr("現在のカーソルを中心に、現在のCT位置へ補正を加えます。"))
        note.setWordWrap(True); form.addWidget(note,4,0,1,3)
        apply = QPushButton(tr("補正を適用"))
        apply.clicked.connect(self.apply)
        form.addWidget(apply,5,0,1,3)
        reset = QPushButton(tr("自動位置合わせに戻す"))
        reset.clicked.connect(self.reset)
        form.addWidget(reset,6,0,1,3)
        layout.addWidget(group)
        landmarks = QGroupBox(tr("対応点でずれを確認"))
        rows = QVBoxLayout(landmarks)
        note = QLabel(tr("同じ解剖学的な点をMRI、CTの順に選びます。点は位置合わせの計算には使用しません。"))
        note.setWordWrap(True); rows.addWidget(note)
        self.mri_button = QPushButton(tr("1. MRIの点を記録"))
        self.mri_button.clicked.connect(self.capture_mri); rows.addWidget(self.mri_button)
        self.ct_button = QPushButton(tr("2. CTの対応点を記録"))
        self.ct_button.clicked.connect(self.capture_ct); rows.addWidget(self.ct_button)
        self.landmark_info = QLabel()
        self.landmark_info.setWordWrap(True); rows.addWidget(self.landmark_info)
        clear = QPushButton(tr("確認用の点をクリア"))
        clear.clicked.connect(self.clear); rows.addWidget(clear)
        layout.addWidget(landmarks)
        layout.addStretch()

    def refresh(self):
        scene = self.window.scene
        self.setEnabled(scene is not None and scene.ct is not None and scene.raw_ct is not None)
        if scene is None or scene.ct is None:
            self.info.setText(tr("CT未読込")); return
        quality = scene.ct_quality
        spread = quality.get("refined_seed_spread_p95_mm")
        self.info.setText(tr("画像から剛体位置合わせ\n解剖学的な精度は未検証") +
            (tr("\n初期角度によるばらつき（95%）：{value} mm").format(value=f"{spread:.2f}") if spread is not None else ""))
        errors = landmark_residuals(scene.ct_to_mri, quality.get("landmarks", []))
        self.ct_button.setEnabled(self.pending_mri is not None)
        self.landmark_info.setText(tr("確認用の対応点：0点") if not len(errors) else
            tr("対応点：{count}点\nRMS {rms} mm / 最大 {maximum} mm\n各点：{errors} mm").format(count=len(errors),
                rms=f"{np.sqrt(np.mean(errors**2)):.2f}",maximum=f"{errors.max():.2f}", errors=", ".join(f"{e:.2f}" for e in errors)))

    def execute(self, matrix, provenance):
        scene = self.window.scene
        state = self.window.view_state()
        def completed(result):
            result.view_state = state
            self.window.install_scene(result)
            for spin in self.angles+self.shifts: spin.setValue(0)
            self.pending_mri = None
            self.refresh()
        self.window.run_job(lambda _: apply_alignment(scene,matrix,provenance),completed,tr("元CTから補正後の画像を作成しています…"))

    def apply(self):
        scene = self.window.scene
        angles = [s.value() for s in self.angles]; shifts = [s.value() for s in self.shifts]
        if not any(angles+shifts): return
        center = scene.world(self.window.ijk)
        matrix = rigid_delta(angles,shifts,center) @ scene.ct_to_mri
        self.execute(matrix,{"action":"manual_rigid_delta", "center_ras_mm":center.tolist(),
                             "angles_xyz_degrees":angles,"translation_ras_mm":shifts})

    def reset(self):
        scene = self.window.scene
        matrix = np.asarray(scene.ct_quality.get("automatic_ct_to_mri_ras_mm",scene.ct_to_mri))
        self.execute(matrix,{"action":"restore_automatic"})

    def capture_mri(self):
        self.pending_mri = self.window.scene.world(self.window.ijk).tolist()
        for layer in self.window.scene.extra_mris:
            self.window.image_layers.rows[layer.uid].check.setChecked(False)
        self.window.ct_visible.setChecked(True)
        self.window.mri_visible.setChecked(False)
        self.window.ct_opacity.setValue(100)
        self.window.ct_mode.setCurrentIndex(self.window.ct_mode.findData("full"))
        self.window.ct_window.setValue(100); self.window.ct_level.setValue(40)
        self.refresh()

    def capture_ct(self):
        if self.pending_mri is None: return
        scene = self.window.scene
        raw = nib.affines.apply_affine(np.linalg.inv(scene.ct_to_mri),scene.world(self.window.ijk))
        pairs = list(scene.ct_quality.get("landmarks",[]))
        pairs.append({"mri_ras_mm":self.pending_mri,"ct_ras_mm":raw.tolist(),"role":"review_only"})
        scene.ct_quality["landmarks"] = pairs
        self.pending_mri = None
        self.window.mri_visible.setChecked(True)
        for layer in scene.extra_mris:
            self.window.image_layers.rows[layer.uid].check.setChecked(False)
        self.window.mri_opacity.setValue(100)
        self.window.ct_visible.setChecked(False)
        self.refresh()

    def clear(self):
        self.window.scene.ct_quality["landmarks"] = []
        self.pending_mri = None
        self.refresh()
