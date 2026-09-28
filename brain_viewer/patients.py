"""Local patient folders. A completed snapshot is published atomically, never in-place."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import stat
from uuid import uuid4
from .imaging import InputError, save_scene, load_scene


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid4().hex + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def move_to_trash(folder):
    from PySide6.QtCore import QFile
    return QFile(str(folder)).moveToTrash()


class PatientStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.unreadable = 0

    def folder(self, patient_id):
        if not isinstance(patient_id, str) or not re.fullmatch(r'patient_[0-9a-f]{16}', patient_id):
            raise InputError('患者フォルダの識別子が不正です。')
        path = self.root / patient_id
        if path.resolve().parent != self.root:
            raise InputError('患者フォルダの保存先が不正です。')
        return path

    def read(self, patient_id):
        record = json.loads((self.folder(patient_id) / 'patient.json').read_text(encoding='utf-8'))
        if record.get('schema') != 'cortex-patient/1' or record.get('id') != patient_id:
            raise InputError('対応していない患者リストです。')
        if (not isinstance(record.get('name'),str) or not record['name'].strip()
                or not isinstance(record.get('created'),str) or not isinstance(record.get('history'),list)
                or any(not isinstance(item,dict) or not isinstance(item.get('scene'),str)
                       or not isinstance(item.get('saved'),str) for item in record['history'])):
            raise InputError('患者リストの保存情報が不正です。')
        return record

    def list_patients(self):
        result = []
        self.unreadable = 0
        if self.root.exists():
            for path in self.root.glob('patient_*/patient.json'):
                try:
                    result.append(self.read(path.parent.name))
                except (OSError, ValueError, KeyError):
                    self.unreadable += 1
        return sorted(result, key=lambda p: p['created'])

    def create(self, name):
        name = str(name).strip()
        if not name or len(name) > 100 or any(ord(c) < 32 for c in name):
            raise InputError('患者の表示名を1〜100文字で入力してください。')
        if any(p['name'].casefold() == name.casefold() for p in self.list_patients()):
            raise InputError('同じ表示名の患者が登録されています。別の表示名を指定してください。')
        patient_id = 'patient_' + uuid4().hex[:16]
        folder = self.folder(patient_id)
        folder.mkdir(parents=True)
        (folder / 'work').mkdir()
        (folder / 'exports').mkdir()
        record = {'schema': 'cortex-patient/1', 'id': patient_id, 'name': name,
                  'created': datetime.now(timezone.utc).isoformat(), 'latest_scene': None, 'history': []}
        atomic_json(folder / 'patient.json', record)
        return record

    def scene_path(self, patient_id, relative=None):
        folder = self.folder(patient_id)
        relative = relative or self.read(patient_id).get('latest_scene')
        if not relative:
            return None
        path = (folder / relative).resolve()
        if not path.is_relative_to(folder.resolve()) or path.name != 'scene.json':
            raise InputError('患者の作業ファイルの保存先が不正です。')
        return path

    def rename(self, patient_id, name):
        name = str(name).strip()
        if not name or len(name) > 100 or any(ord(c) < 32 for c in name):
            raise InputError('患者の表示名を1〜100文字で入力してください。')
        if any(p['id'] != patient_id and p['name'].casefold() == name.casefold() for p in self.list_patients()):
            raise InputError('同じ表示名の患者が登録されています。別の表示名を指定してください。')
        record = self.read(patient_id)
        record['name'] = name
        atomic_json(self.folder(patient_id)/'patient.json', record)

    def delete(self, patient_id):
        """Remove only this managed folder, using the OS trash; never input paths."""
        folder = self.folder(patient_id)
        # Reject aliases, including junctions to other patients or external data.
        def is_link(path):
            info = path.lstat()
            return (stat.S_ISLNK(info.st_mode) or
                    bool(getattr(info, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT))
        if folder.resolve(strict=True) != folder or is_link(folder):
            raise InputError('患者フォルダの保存先が不正です。')
        self.read(patient_id)
        def unreadable(error):
            raise InputError('患者フォルダを確認できません。開いているファイルやアクセス権を確認してください。') from error
        # Inspect names/attributes only, without following links or reading images.
        for current, directories, files in os.walk(folder, topdown=True, followlinks=False, onerror=unreadable):
            for name in directories + files:
                if is_link(Path(current)/name):
                    raise InputError('患者フォルダにリンクが含まれるため削除できません。元データを保護するため、リンクの配置を確認してください。')
        selection = self.root/'selection.json'
        previous = json.loads(selection.read_text(encoding='utf-8')) if selection.is_file() else None
        change_selection = previous is None or previous.get('patient_id') == patient_id
        if change_selection:
            # Mark an intentionally empty selection before the folder is moved.
            # A failed selection write must not delete the folder.
            atomic_json(selection, {'patient_id': None})
        try:
            moved = move_to_trash(folder)
            if not moved or folder.exists():
                raise InputError('患者フォルダを削除できませんでした。開いているファイルを閉じて、もう一度お試しください。')
        except Exception:
            if change_selection and folder.exists():
                atomic_json(selection, previous or {'patient_id': None})
            raise
        return patient_id

    def save(self, patient_id, scene, state):
        folder = self.folder(patient_id)
        self.read(patient_id)  # Validate before writing any images.
        snapshot = save_scene(scene, folder / 'work', state, asset_root=folder / 'assets')
        # Re-read immediately before publication; an interrupted save cannot change latest_scene.
        record = self.read(patient_id)
        relative = (snapshot / 'scene.json').relative_to(folder).as_posix()
        record['latest_scene'] = relative
        record['updated'] = datetime.now(timezone.utc).isoformat()
        record['history'].append({'scene': relative, 'saved': record['updated']})
        record['images'] = [{'sequence': 'MRI', 'role': 'reference'}]
        record['images'] += [{'sequence': layer.sequence, 'name': layer.name, 'id': layer.uid} for layer in scene.extra_mris]
        if scene.ct is not None:
            record['images'].append({'sequence': 'CT', 'role': 'post_implantation'})
        atomic_json(folder / 'patient.json', record)
        return snapshot

    def load(self, patient_id, relative=None):
        path = self.scene_path(patient_id, relative)
        return load_scene(path) if path else None

    def remember(self, patient_id):
        self.read(patient_id)
        atomic_json(self.root / 'selection.json', {'patient_id': patient_id})

    def last_selected(self):
        try:
            patient_id = json.loads((self.root / 'selection.json').read_text(encoding='utf-8'))['patient_id']
            self.read(patient_id)
            return patient_id
        except (OSError, ValueError, KeyError):
            return None
