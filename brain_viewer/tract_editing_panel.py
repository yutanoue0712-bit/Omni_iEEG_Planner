"""Preview, apply and undo exclusions for the selected tract bundle."""
import numpy as np
from PySide6.QtWidgets import QGroupBox,QVBoxLayout,QHBoxLayout,QComboBox,QLabel,QPushButton,QDoubleSpinBox
from .i18n import tr
from .tract_editing import select_slice,select_projection,exclude,restore_all,undo


class TractEditingPanel(QGroupBox):
    def __init__(self,panel):
        super().__init__(tr('選択した線維束を修正'))
        self.panel=panel;self.window=panel.window
        self.preview_bundle=None;self.pending_ids=np.array([],dtype=np.int64);self.selection={};self.current_uid=None
        layout=QVBoxLayout(self)
        self.mode=QComboBox()
        for text,key in [('線維を確認','navigate'),('2Dで囲んで除外候補を選択','2d'),('3Dで囲んで除外候補を選択','3d')]:
            self.mode.addItem(tr(text),key)
        self.mode.currentIndexChanged.connect(self.mode_changed);layout.addWidget(self.mode)
        row=QHBoxLayout();self.thickness_label=QLabel(tr('2Dの選択厚み mm'));row.addWidget(self.thickness_label)
        self.thickness=QDoubleSpinBox();self.thickness.setRange(.5,30);self.thickness.setValue(2);self.thickness.setSingleStep(.5)
        self.thickness.valueChanged.connect(lambda *_:self.clear_preview())
        row.addWidget(self.thickness);layout.addLayout(row)
        self.guide=QLabel();self.guide.setWordWrap(True);self.guide.setObjectName('muted');layout.addWidget(self.guide)
        self.info=QLabel();self.info.setWordWrap(True);layout.addWidget(self.info)
        self.apply_button=QPushButton(tr('候補の線維を除外'));self.apply_button.setObjectName('primary')
        self.apply_button.clicked.connect(self.apply);layout.addWidget(self.apply_button)
        self.cancel_button=QPushButton(tr('除外候補を取り消す'));self.cancel_button.clicked.connect(self.clear_preview)
        layout.addWidget(self.cancel_button)
        self.undo_button=QPushButton(tr('線維の編集を1つ戻す'));self.undo_button.clicked.connect(self.undo);layout.addWidget(self.undo_button)
        self.restore_button=QPushButton(tr('この束の除外をすべて戻す'));self.restore_button.clicked.connect(self.restore_all);layout.addWidget(self.restore_button)
        self.refresh()

    def reset(self):
        self.clear_preview(refresh=False);self.current_uid=None;self.mode.setCurrentIndex(0);self.refresh()

    def active(self):
        b=self.panel.bundle()
        return bool(b and b.visible and b.opacity>0 and b.count and self.window.workflow_tabs.currentIndex()==5)

    def mode_changed(self,*_):
        self.clear_preview(refresh=False)
        if self.mode.currentData()!='navigate':
            self.panel.draw_mode.setCurrentIndex(0)
        self.update_tools();self.refresh()
        self.panel.refresh_tract_views()

    def update_tools(self):
        self.window.segmentation.update_tools()
        views=getattr(self.window,'views',None)
        if views is not None:
            for surface in views.surface_panels():
                surface.widget.set_tract_lasso(self.active() and self.mode.currentData()=='3d')

    def bundle_changed(self):
        b=self.panel.bundle();uid=b.uid if b else None
        if uid!=self.current_uid:
            self.clear_preview(refresh=False);self.mode.setCurrentIndex(0);self.current_uid=uid
            if b:
                self.thickness.blockSignals(True);self.thickness.setValue(b.slab_mm);self.thickness.blockSignals(False)
        if not self.active():self.clear_preview(refresh=False);self.mode.setCurrentIndex(0)
        if b and not b.visible_2d and self.mode.currentData()=='2d':self.mode.setCurrentIndex(0)
        self.update_tools();self.refresh()

    def clear_preview(self,refresh=True):
        if self.preview_bundle is not None:
            self.preview_bundle._highlight_ids=[]
            self.preview_bundle._highlight_revision=getattr(self.preview_bundle,'_highlight_revision',0)+1
        self.preview_bundle=None;self.pending_ids=np.array([],dtype=np.int64);self.selection={}
        if refresh:
            self.panel.refresh_tract_views();self.refresh()

    def refresh(self):
        b=self.panel.bundle();enabled=self.active()
        self.mode.setEnabled(enabled)
        self.mode.model().item(self.mode.findData('2d')).setEnabled(bool(b and b.visible_2d))
        self.thickness.setVisible(self.mode.currentData()=='2d')
        self.thickness_label.setVisible(self.mode.currentData()=='2d')
        self.apply_button.setEnabled(bool(len(self.pending_ids)) and enabled)
        self.cancel_button.setEnabled(self.preview_bundle is not None)
        self.undo_button.setEnabled(bool(b and b.edits))
        self.restore_button.setEnabled(bool(b and b.count<b.total_count))
        if b:
            text=tr('残り {remaining} / 元 {total}本・除外 {removed}本').format(
                remaining=b.count,total=b.total_count,removed=b.total_count-b.count)
            if self.preview_bundle is b:
                text+='\n'+tr('赤い候補：{count}本。確認後に除外を確定します。').format(count=len(self.pending_ids))
            if not b.visible:text+='\n'+tr('束のチェックを入れると編集できます。')
        else:text=tr('修正する線維束を一覧から選択してください。')
        self.info.setText(text)
        self.guide.setText(tr('3D：左ドラッグで囲みます。囲んだ範囲の奥行き全体を通る線維が対象です。回転する場合は「線維を確認」に戻します。')
            if self.mode.currentData()=='3d' else
            tr('2D：断面上を左ドラッグで囲みます。指定した厚みを通る線維を1本単位で選びます。Escで描画を取消。'))

    def select_2d(self,axis,vertices):
        if not self.active() or self.mode.currentData()!='2d' or self.window.worker is not None:return
        b=self.panel.bundle();scene=self.window.scene;thickness=self.thickness.value()
        record={'kind':'slice_lasso','axis':int(axis),'thickness_mm':thickness,
                'vertices_ras_mm':scene.world(np.asarray(vertices)).tolist()}
        self.preview(lambda:select_slice(b,scene,axis,vertices,thickness),b,record)

    def select_3d(self,surface,vertices):
        if not self.active() or self.mode.currentData()!='3d' or self.window.worker is not None:return
        b=self.panel.bundle()
        matrix=surface.renderer.GetActiveCamera().GetCompositeProjectionTransformMatrix(
            surface.renderer.GetTiledAspectRatio(),-1,1)
        matrix=np.array([[matrix.GetElement(r,c) for c in range(4)] for r in range(4)])
        polygon=np.asarray(vertices,float)
        record={'kind':'projected_lasso','depth':'entire_visible_frustum',
                'vertices_ndc':polygon.tolist(),'ras_to_clip':matrix.tolist()}
        self.preview(lambda:select_projection(b,polygon,matrix),b,record)

    def preview(self,operation,bundle,record):
        self.clear_preview()
        def completed(ids):
            if self.panel.bundle() is not bundle:return
            self.preview_bundle=bundle;self.pending_ids=ids;self.selection=record
            bundle._highlight_ids=ids
            bundle._highlight_revision=getattr(bundle,'_highlight_revision',0)+1
            self.panel.refresh_tract_views();self.refresh()
        self.window.run_job(lambda _:operation(),completed,tr('囲んだ範囲を通る線維を確認しています…'))

    def apply(self):
        b=self.panel.bundle()
        if b is None or self.preview_bundle is not b or not len(self.pending_ids):return
        count=exclude(b,self.pending_ids,self.selection)
        self.clear_preview(refresh=False);self.panel.refresh_bundles(b.uid);self.panel.refresh_tract_views()
        self.window.message(tr('{count}本を除外しました。線維の編集を戻す操作で取り消せます。').format(count=count))

    def undo(self):
        b=self.panel.bundle()
        if b is None:return
        self.clear_preview(refresh=False);undo(b);self.panel.refresh_bundles(b.uid);self.panel.refresh_tract_views()

    def restore_all(self):
        b=self.panel.bundle()
        if b is None:return
        self.clear_preview(refresh=False);restore_all(b);self.panel.refresh_bundles(b.uid);self.panel.refresh_tract_views()
