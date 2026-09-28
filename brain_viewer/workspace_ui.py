"""Patient selection and one image-import entry point for the desktop window."""
from pathlib import Path
from tempfile import TemporaryDirectory
from PySide6.QtCore import QTimer, QUrl, Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QLabel, QComboBox, QPushButton, QMenu, QGroupBox, QVBoxLayout,
                               QHBoxLayout, QFileDialog, QInputDialog, QCheckBox, QSizePolicy, QMessageBox)
from .i18n import tr
from .imaging import InputError, load_scene, load_mri, validate_freesurfer_folder
from .patients import PatientStore
from .sequences import SEQUENCES, find_image_sources, add_image, source_modality


class ImportFeedbackLabel(QLabel):
    """Keep the whole reason readable when the narrow sidebar changes language."""
    def setText(self, text):
        super().setText(text)
        self.fit_height()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_height()

    def fit_height(self):
        height = max(1, self.heightForWidth(max(1, self.width())))
        if self.minimumHeight() != height or self.maximumHeight() != height:
            self.setFixedHeight(height)


class WorkspaceUI:
    def setup_patients(self, row, root):
        self.project_root = Path(root)
        (root/'private_reports').mkdir(exist_ok=True)
        self._patient_temp = TemporaryDirectory(dir=root/'private_reports') if self._smoke or self._demo else None
        self.patient_store = PatientStore(Path(self._patient_temp.name)/'patients' if self._patient_temp else root/'patients')
        self.patient_id = None
        self._close_saved = False
        row.addWidget(QLabel(tr('患者')))
        self.patient_combo = QComboBox()
        self.patient_combo.setObjectName('patientNames')
        self.patient_combo.setMinimumWidth(175)
        self.patient_combo.setMaximumWidth(270)
        self.patient_combo.activated.connect(self.patient_selected)
        row.addWidget(self.patient_combo)
        self.new_patient_button = QPushButton(tr('新規患者'))
        self.new_patient_button.clicked.connect(self.new_patient)
        row.addWidget(self.new_patient_button)
        self.patient_menu_button = QPushButton(tr('患者の操作'))
        menu = QMenu(self.patient_menu_button)
        menu.addAction(tr('表示名を変更'), self.rename_patient)
        menu.addAction(tr('患者フォルダを開く'), self.open_patient_folder)
        menu.addAction(tr('保存履歴から開く'), self.choose_history)
        menu.addAction(tr('以前の作業を患者として取り込む'), self.choose_case)
        menu.addSeparator()
        self.delete_patient_action = menu.addAction(tr('患者を削除'), self.delete_patient)
        self.patient_menu_button.setMenu(menu)
        row.addWidget(self.patient_menu_button)
        self.refresh_patients()

    def refresh_patients(self):
        self.patient_combo.blockSignals(True)
        self.patient_combo.clear()
        self.patient_combo.addItem(tr('患者を選択'), None)
        for record in self.patient_store.list_patients():
            self.patient_combo.addItem(record['name'], record['id'])
        self.patient_combo.setCurrentIndex(max(0, self.patient_combo.findData(self.patient_id)))
        self.patient_combo.blockSignals(False)
        self.delete_patient_action.setEnabled(self.patient_id is not None and self.worker is None)

    def next_patient_name(self):
        names = {p['name'] for p in self.patient_store.list_patients()}
        number = 1
        while f'Case {number:03d}' in names: number += 1
        return f'Case {number:03d}'

    def new_patient(self):
        name, ok = QInputDialog.getText(self, tr('新規患者'), tr('患者の表示名・症例ID'), text=self.next_patient_name())
        if not ok: return
        try:
            record = self.patient_store.create(name)
            self.switch_patient(record['id'])
        except (OSError, ValueError):
            self._job_failed(tr('患者を作成できません。同じ表示名や保存先を確認してください。'))

    def rename_patient(self):
        if not self.patient_id: return
        record = self.patient_store.read(self.patient_id)
        name, ok = QInputDialog.getText(self, tr('表示名を変更'), tr('患者の表示名・症例ID'), text=record['name'])
        if not ok: return
        try:
            self.patient_store.rename(self.patient_id, name)
            self.refresh_patients()
        except (OSError, ValueError): self._job_failed(tr('患者を作成できません。同じ表示名や保存先を確認してください。'))

    def open_patient_folder(self):
        if self.patient_id:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.patient_store.folder(self.patient_id))))

    def confirm_patient_delete(self, name, folder):
        dialog = QMessageBox(self)
        dialog.setWindowTitle(tr('患者を削除'))
        dialog.setTextFormat(Qt.TextFormat.PlainText)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setText(tr('「{name}」を患者フォルダごと削除しますか？').format(name=name))
        dialog.setInformativeText(tr('この患者の保存履歴・電極・領域・解析結果・アプリ内の出力を、患者フォルダごとごみ箱へ移動します。\n読み込み元のMRI・CT・FreeSurferなど、患者フォルダ外のファイルは削除しません。'))
        dialog.setDetailedText(str(folder))
        remove = dialog.addButton(tr('削除する'), QMessageBox.ButtonRole.DestructiveRole)
        cancel = dialog.addButton(tr('キャンセル'), QMessageBox.ButtonRole.RejectRole)
        dialog.setDefaultButton(cancel)
        dialog.setEscapeButton(cancel)
        dialog.exec()
        accepted = dialog.clickedButton() is remove
        dialog.deleteLater()
        return accepted

    def delete_patient(self):
        if self.patient_id is None or self.worker is not None or self.results.exporting: return
        patient_id = self.patient_id
        try:
            record = self.patient_store.read(patient_id)
            folder = self.patient_store.folder(patient_id)
            if not self.confirm_patient_delete(record['name'], folder): return
        except (OSError, ValueError):
            self.patient_delete_failed(tr('患者フォルダを確認できません。開いているファイルやアクセス権を確認してください。'))
            return
        def completed(_):
            # Detach before clearing the view: closing/switching must never save
            # the deleted scene back into a new patient folder.
            self.patient_id = None
            self._after_job = None
            self._close_saved = False
            self.clear_scene()
            self.refresh_patients()
            self.side_tabs.setCurrentIndex(0)
            self.message(tr('患者フォルダをごみ箱へ移動しました。患者を選択するか、新規患者を作成してください。'))
            self.write_status('empty_patient', {'last_operation': 'patient_deleted'})
        self.run_job(lambda _: self.patient_store.delete(patient_id), completed,
                     tr('患者フォルダを削除しています…'), on_error=self.patient_delete_failed)

    def patient_delete_failed(self, reason):
        self._closing = False
        self._after_job = None
        self.refresh_patients()
        self.message(tr(reason))
        self.write_status('ready' if self.scene is not None else 'empty_patient',
                          {'last_operation': 'patient_delete_failed'})
        if not self._smoke:
            QMessageBox.warning(self, tr('患者を削除'), tr(reason))

    def patient_selected(self, index):
        target = self.patient_combo.itemData(index)
        self.refresh_patients()  # Never label the previous patient's images as the requested patient.
        if target and target != self.patient_id: self.switch_patient(target)

    def switch_patient(self, target, relative=None):
        if self.worker is not None: return
        old_id, old_scene = self.patient_id, self.scene
        old_state = self.view_state() if old_scene is not None else None
        def operation(progress):
            if old_id and old_scene is not None:
                progress('現在の患者の作業を保存しています…')
                self.patient_store.save(old_id, old_scene, old_state)
            progress('患者の作業を開いています…')
            scene = self.patient_store.load(target, relative)
            self.patient_store.remember(target)
            return scene
        def completed(scene):
            self.clear_scene()
            self.patient_id = target
            self.refresh_patients()
            if scene is None: self.clear_scene()
            else: self.install_scene(scene)
            self.side_tabs.setCurrentIndex(0)
        self.run_job(operation, completed, tr('患者の作業を切り替えています…'))

    def start_workspace(self, root, demo=False):
        if demo or self._smoke:
            self.load_default(root, demo)
            return
        patients = self.patient_store.list_patients()
        selected = self.patient_store.last_selected()
        if patients:
            self.switch_patient(selected or patients[0]['id'])
            return
        if (self.patient_store.root/'selection.json').is_file() or (
                root == self.project_root/'sample_images' and not root.exists()):
            # A clean source install has no samples; a cleared patient list stays empty.
            self.patient_id = None
            self.clear_scene()
            self.refresh_patients()
            self.message(tr('患者が登録されていません。「新規患者」から作成してください。'))
            return
        from .multimodal import load_workspace
        name = self.next_patient_name()
        def operation(progress):
            scene = load_workspace(root, self.project_root/'local_data'/'registration', progress)
            record = self.patient_store.create(name)
            progress('最初の患者フォルダに保存しています…')
            self.patient_store.save(record['id'], scene, {})
            self.patient_store.remember(record['id'])
            return scene, record['id']
        def completed(result):
            scene, self.patient_id = result
            self.refresh_patients()
            self.install_scene(scene)
            self.side_tabs.setCurrentIndex(0)
        self.run_job(operation, completed, tr('患者リストを準備しています…'))

    def choose_case(self):
        path, _ = QFileDialog.getOpenFileName(self, tr('以前の作業を患者として取り込む'), str(self.project_root/'cases'), 'scene.json (scene.json)')
        if not path: return
        name, ok = QInputDialog.getText(self, tr('新規患者'), tr('患者の表示名・症例ID'), text=self.next_patient_name())
        if ok: self.open_case(Path(path), name)

    def open_case(self, path, name=None):
        if self._smoke:
            self.run_job(lambda _: load_scene(Path(path)), self.install_scene, tr('保存した作業を読み込んでいます…'))
            return
        old_id, old_scene = self.patient_id, self.scene
        old_state = self.view_state() if old_scene else None
        def operation(progress):
            scene = load_scene(Path(path))
            if old_id and old_scene: self.patient_store.save(old_id, old_scene, old_state)
            record = self.patient_store.create(name or self.next_patient_name())
            self.patient_store.save(record['id'], scene, scene.view_state)
            self.patient_store.remember(record['id'])
            return scene, record['id']
        def completed(result):
            self.clear_scene()
            scene, self.patient_id = result
            self.refresh_patients()
            self.install_scene(scene)
        self.run_job(operation, completed, tr('以前の作業を患者フォルダへ取り込んでいます…'))

    def choose_history(self):
        if not self.patient_id: return
        history = self.patient_store.read(self.patient_id)['history']
        if not history: return
        labels = [f"{index+1}  /  {item['saved'].replace('T',' ')[:19]} UTC" for index, item in enumerate(history)]
        choice, ok = QInputDialog.getItem(self, tr('保存履歴から開く'), tr('開く保存時点'), labels, len(labels)-1, False)
        if ok: self.switch_patient(self.patient_id, history[labels.index(choice)]['scene'])

    def save_work(self, closing=False):
        if self.scene is None or self.worker is not None: return
        if not self.patient_id:
            record = self.patient_store.create(self.next_patient_name())
            self.patient_id = record['id']; self.refresh_patients()
        patient_id, scene, state = self.patient_id, self.scene, self.view_state()
        def completed(path):
            self.patient_store.remember(patient_id)
            self.message(tr('患者フォルダに保存しました。'))
            if closing: self._close_saved = True
        self.run_job(lambda _: self.patient_store.save(patient_id, scene, state), completed, tr('現在の患者の作業を保存しています…'))

    def export_directory(self):
        path = self.patient_store.folder(self.patient_id)/'exports' if self.patient_id else self.project_root/'exports'
        path.mkdir(parents=True, exist_ok=True)
        return path

    def build_import_panel(self):
        group = QGroupBox(tr('画像を追加'))
        layout = QVBoxLayout(group)
        self.sequence_combo = QComboBox()
        for key, name in SEQUENCES: self.sequence_combo.addItem(tr(name), key)
        self.sequence_combo.currentIndexChanged.connect(self.update_import_hint)
        layout.addWidget(self.sequence_combo)
        buttons = QHBoxLayout()
        self.import_folder_button = QPushButton(tr('フォルダを開く'))
        self.import_folder_button.setObjectName('primary')
        self.import_folder_button.clicked.connect(self.choose_image_folder)
        buttons.addWidget(self.import_folder_button)
        self.import_file_button = QPushButton(tr('NIfTIファイル'))
        self.import_file_button.clicked.connect(self.choose_image_file)
        buttons.addWidget(self.import_file_button)
        layout.addLayout(buttons)
        self.import_hint = QLabel(tr('① 基準MRIの撮像フォルダまたはNIfTI → ② 同じ患者のFreeSurferフォルダ（mri・surfを含む）の順に選択します。'))
        self.import_hint.setObjectName('muted'); self.import_hint.setWordWrap(True)
        layout.addWidget(self.import_hint)
        self.import_ct_contacts = QCheckBox(tr('読込後に電極候補も作成'))
        self.import_ct_contacts.setChecked(False)
        self.import_ct_contacts.setToolTip(tr('オフの場合はCTの重ね合わせまで行います。電極候補は後から「CTから電極候補を作成」で追加できます。'))
        self.import_ct_contacts.hide()
        layout.addWidget(self.import_ct_contacts)
        self.import_feedback = ImportFeedbackLabel()
        self.import_feedback.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.import_feedback.setWordWrap(True)
        self.import_feedback.setTextFormat(Qt.TextFormat.PlainText)
        self.import_feedback.setObjectName('importFeedback')
        self.import_feedback.hide()
        layout.addWidget(self.import_feedback)
        self.import_dicom_button = QPushButton(tr('同じフォルダの元DICOMを開く'))
        self.import_dicom_button.clicked.connect(self.retry_image_dicom)
        self.import_dicom_button.hide()
        layout.addWidget(self.import_dicom_button)
        self._import_recovery = None
        self._import_error_reason = None
        return group

    def clear_import_feedback(self):
        if hasattr(self,'diffusion'): self.diffusion.feedback.hide()
        if hasattr(self,'postop'): self.postop.feedback.hide()
        self.import_feedback.hide()
        self.import_dicom_button.hide()
        self._import_recovery = None
        self._import_error_reason = None

    def refresh_import_feedback(self):
        if self._import_error_reason is not None:
            text = tr('画像を読み込めませんでした。\n{reason}').format(reason=tr(self._import_error_reason))
            self.import_feedback.setText(text)
            if hasattr(self,'diffusion') and not self.diffusion.feedback.isHidden():
                self.diffusion.feedback.setText(text)
            if hasattr(self,'postop') and not self.postop.feedback.isHidden():
                self.postop.feedback.setText(text)

    def image_import_failed(self, source, sequence, reason):
        self._closing = False
        self._after_job = None
        self.clear_import_feedback()
        self._import_error_reason = reason
        self.refresh_import_feedback()
        self.import_feedback.show()
        if sequence == 'DTI':
            self.diffusion.feedback.setText(self.import_feedback.text())
            self.diffusion.feedback.show()
        if sequence.startswith('postop-'):
            self.postop.feedback.setText(self.import_feedback.text())
            self.postop.feedback.show()
        self.message(self.import_feedback.text().replace('\n', ' '))
        if isinstance(source, (str, Path)) and str(source).lower().endswith(('.nii', '.nii.gz')):
            self._import_recovery = (Path(source).parent, sequence)
            self.import_dicom_button.show()
        self.write_status('ready' if self.scene is not None else 'empty',
                          {'last_operation': 'image_import_failed', 'error': reason})

    def retry_image_dicom(self):
        if self._import_recovery is None or self.worker is not None: return
        folder, sequence = self._import_recovery
        self.open_image_folder(folder, sequence, dicom_only=True)

    def update_import_hint(self, *_):
        if not hasattr(self, 'import_hint'): return
        key = self.sequence_combo.currentData()
        message = '① 基準MRIの撮像フォルダまたはNIfTI → ② 同じ患者のFreeSurferフォルダ（mri・surfを含む）の順に選択します。' if key == 'reference' else (
            'DTIの元DICOM、または4D NIfTIと同名の.bval・.bvecを選択します。' if key == 'DTI' else
            '術後CTを基準MRIへ位置合わせします。既存CTは保存履歴に残します。' if key == 'CT' else
            '骨条件のCTを別の画像として追加します。術後CT・電極は保持します。' if key == 'CT-bone' else
            'PETの元DICOMまたはNIfTIを選択します。基準MRIへ位置合わせし、ヒートマップで重ねます。' if key == 'PET' else
            '追加画像を基準MRIへ位置合わせします。術後CT・電極は保持します。')
        self.import_hint.setText(tr(message))
        if hasattr(self, 'import_ct_contacts'):
            self.import_ct_contacts.setVisible(key == 'CT')

    def refresh_import_choices(self):
        has_scene = self.scene is not None
        for index in range(self.sequence_combo.count()):
            reference = self.sequence_combo.itemData(index) == 'reference'
            self.sequence_combo.model().item(index).setEnabled(not has_scene if reference else has_scene)
        if has_scene and self.sequence_combo.currentData() == 'reference': self.sequence_combo.setCurrentIndex(self.sequence_combo.findData('T2'))
        if not has_scene: self.sequence_combo.setCurrentIndex(0)
        self.image_layers.rows['mri'].setVisible(has_scene)
        self.image_layers.rows['ct'].setVisible(has_scene and self.scene.ct is not None)
        self.image_empty.setVisible(not has_scene)
        self.update_import_hint()

    def choose_image_file(self):
        sequence = self.sequence_combo.currentData()
        title = '1/2：基準MRIのNIfTIを選択' if sequence == 'reference' else '追加する画像を選択'
        path, _ = QFileDialog.getOpenFileName(self, tr(title), str(self.project_root/'sample_images'), 'NIfTI (*.nii *.nii.gz)')
        if path: self.prepare_image_import(Path(path), sequence)

    def choose_image_folder(self):
        sequence = self.sequence_combo.currentData()
        title = '1/2：基準MRIの撮像フォルダを選択' if sequence == 'reference' else 'MRI・CT・PETの撮像フォルダ'
        folder = QFileDialog.getExistingDirectory(self, tr(title), str(self.project_root/'sample_images'))
        if not folder: return
        self.open_image_folder(Path(folder), sequence)

    def open_image_folder(self, folder, sequence, *, dicom_only=False):
        if self.worker is not None: return
        self.clear_import_feedback()
        def found(sources):
            self.message(tr('画像候補を確認しました。読み込む画像を選択してください。'))
            index = 0
            rejected = getattr(sources, 'rejected', [])
            excluded = getattr(sources, 'excluded_localizers', 0)
            if len(sources) > 1 or rejected or excluded:
                labels = [f'{i+1}: {label}' for i, (label, _) in enumerate(sources)]
                notes = [tr('画像・DICOM系列'), tr('元DICOMを先頭に表示します。画像の位置・間隔を確認して選択してください。')]
                if excluded:
                    notes.append(tr('撮影位置決め画像 {count}枚を3D画像から除外しました。').format(count=excluded))
                if rejected:
                    notes.append(tr('座標などに問題のある候補 {count}件を除外しました。').format(count=len(rejected)))
                    notes.extend(tr(reason) for reason in dict.fromkeys(reason for _, reason in rejected))
                choice, ok = QInputDialog.getItem(self, tr('読み込む画像を選択'), '\n'.join(notes), labels, 0, False)
                if not ok:
                    self.message(tr('画像の選択をキャンセルしました。'))
                    return
                index = labels.index(choice)
            self._after_job = lambda: self.prepare_image_import(sources[index][1], sequence)
        from .diffusion_io import find_diffusion_sources
        self.run_job(lambda progress: find_diffusion_sources(folder,progress) if sequence=='DTI' else find_image_sources(folder, source_modality(sequence), progress, dicom_only=dicom_only),
                     found, tr('フォルダ内の画像を確認しています…'),
                     on_error=lambda reason: self.image_import_failed(folder, sequence, reason))

    def prepare_image_import(self, source, sequence):
        if sequence == 'reference':
            if self.scene is not None: return
            while True:
                fs = QFileDialog.getExistingDirectory(self, tr('2/2：対応するFreeSurferフォルダ（mri・surfを含む）'), str(self.project_root/'sample_images'))
                if not fs: return
                try:
                    fs = validate_freesurfer_folder(Path(fs))
                except InputError as exc:
                    self._job_failed(str(exc))
                    continue
                self.import_image(source, sequence, fs=fs)
                break
            return
        if self.scene is None: return
        name = sequence
        if sequence != 'CT':
            count = sum(layer.sequence == sequence for layer in self.scene.extra_mris)
            default_name = (tr('CT：骨条件') if sequence == 'CT-bone' else
                            tr('術後MRI')+' '+sequence.removeprefix('postop-') if sequence.startswith('postop-') else sequence)
            name, ok = QInputDialog.getText(self, tr('画像の表示名'), tr('一覧に表示する名前'), text=default_name+(f' {count+1}' if count else ''))
            if not ok or not name.strip(): return
            name = name.strip()[:100]
        self.import_image(source, sequence, name=name)

    def import_image(self, source, sequence, name='', fs=None, detect_contacts=None):
        if self.worker is not None: return
        self.clear_import_feedback()
        if not self.patient_id:
            record = self.patient_store.create(self.next_patient_name())
            self.patient_id = record['id']; self.refresh_patients()
        patient_id, scene = self.patient_id, self.scene
        state = self.view_state() if scene else {}
        cache = self.patient_store.folder(patient_id)/'cache'
        detect_contacts = self.import_ct_contacts.isChecked() if detect_contacts is None else detect_contacts
        correct_motion = self.diffusion.correct_motion.isChecked() if sequence == 'DTI' else False
        def operation(progress):
            if sequence == 'reference':
                result = load_mri(source, fs, progress)
            elif sequence == 'CT':
                from .multimodal import attach_ct
                result, _, _ = attach_ct(scene, source, cache/'registration', progress=progress)
            elif sequence == 'DTI':
                from .diffusion import add_diffusion
                result = add_diffusion(scene,source,name,cache/'diffusion',progress,correct_motion=correct_motion)
            else:
                result = add_image(scene, source, sequence, name, cache/'registration', progress)
            # Invalid input must not trigger large snapshots before it is rejected.
            if scene is not None:
                progress('変更前の作業を保存しています…')
                self.patient_store.save(patient_id, scene, state)
            result.view_state = dict(state)
            if sequence == 'CT':
                result.view_state = dict(state, image_layers={**state.get('image_layers', {}), 'ct': {'visible': True, 'opacity': .65}})
            if sequence == 'PET':
                uid = result.extra_mris[-1].uid
                result.view_state = dict(state,
                    extra_selected=uid,
                    extra_styles={**state.get('extra_styles', {}), uid:'heatmap'},
                    extra_palettes={**state.get('extra_palettes', {}), uid:'hot'},
                    image_layers={**state.get('image_layers', {}), uid:{'visible':True,'opacity':.6}})
            if sequence.startswith('postop-'):
                layer=result.extra_mris[-1]
                layer.quality=dict(layer.quality,postop_role='rigid')
                result.view_state=dict(state,extra_selected=layer.uid,
                    postop={'image':layer.uid,'stage':layer.uid,'display':'checker'},
                    extra_styles={**state.get('extra_styles',{}),layer.uid:'checker'},
                    image_layers={key:{'visible':key in ('mri',layer.uid),'opacity':1.}
                                  for key in ('mri','ct',*[item.uid for item in result.extra_mris])})
            if sequence == 'DTI':
                model=result.diffusions[-1]
                result.view_state=dict(state,
                    image_layers={**state.get('image_layers',{}),**{uid:{'visible':key=='fa','opacity':.6} for key,uid in model.layer_ids.items()}},
                    extra_styles={**state.get('extra_styles',{}),model.layer_ids['fa']:'direction'},
                    diffusion={'model':model.uid,'mode':1,'map':'color','motion':correct_motion})
            progress('追加した画像を患者フォルダに保存しています…')
            self.patient_store.save(patient_id, result, result.view_state)
            return result
        def completed(result):
            self.patient_store.remember(patient_id)
            self.install_scene(result)
            self.side_tabs.setCurrentIndex(0)
            if sequence not in ('reference', 'CT'):
                self.extra_combo.setCurrentIndex(self.extra_combo.findData(result.extra_mris[-1].uid))
            if sequence == 'DTI':
                self.workflow_tabs.setCurrentIndex(5)
                self.diffusion.show_map()
            if sequence.startswith('postop-'):
                self.workflow_tabs.setCurrentIndex(3)
                self.postop.apply_view()
            self.message(tr('画像を患者フォルダへ追加しました。重なりを確認してください。'))
            if sequence == 'CT' and detect_contacts:
                # Install and save CT before starting the optional, separate detector.
                self._after_job = self.detect_contacts
        self.run_job(operation, completed, tr('画像を読み込み、位置合わせしています…'),
                     on_error=lambda reason: self.image_import_failed(source, sequence, reason))
