"""User-provided reference labels. Width meaning and exact product codes are unspecified.

These labels never set detected positions, counts, pitch or extraction settings.
"""
from .imaging import InputError


REFERENCE_PRESETS = [
    {'id':f'adtech-reference-{count}-{width:g}', 'label':f'AD-Tech / {count}極 / {width:g} mm幅',
     'manufacturer':'AD-Tech','contact_count':count,'stated_width_mm':width,
     'width_definition':'unspecified','model_number':None,'source':'user_reference'}
    for count,width in ((4,2.5),(8,5.),(10,5.),(12,5.))
]


def set_reference_preset(scene,group,preset_id):
    if group not in {c.group for c in scene.contacts}:
        raise InputError('電極を選択してください。')
    preset=next((p for p in REFERENCE_PRESETS if p['id']==preset_id),None)
    if preset_id and preset is None: raise InputError('未対応の電極参考ラベルです。')
    metadata=scene.electrode_quality.setdefault('groups',{}).setdefault(group,{})
    if preset is None:
        metadata.pop('reference_preset',None); metadata.pop('reference_spec',None)
    else:
        metadata['reference_preset']=preset['id']
        metadata['reference_spec']=dict(preset)

