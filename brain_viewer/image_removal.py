"""Remove scene image registrations without touching source files or saved history."""
from .imaging import InputError


def removal_ids(scene, uid):
    if uid=='ct' and scene.ct is not None:
        return {'ct'}
    if uid not in {layer.uid for layer in scene.extra_mris}:
        raise InputError('削除する追加画像を選択してください。')
    removed={uid}
    # Correction candidates depend on their rigid source. Never leave orphaned stages.
    while True:
        children={layer.uid for layer in scene.extra_mris
                  if layer.quality.get('rigid_source_uid') in removed}
        if children.issubset(removed):return removed
        removed.update(children)


def remove_images(scene, uid):
    removed=removal_ids(scene,uid)
    scene.extra_mris=[layer for layer in scene.extra_mris if layer.uid not in removed]
    if 'ct' in removed:
        scene.ct=None; scene.ct_valid=None
        # Native CT evidence is still required for reviewed electrode coordinates.
        if not any(c.ct_position is not None for c in scene.contacts):
            scene.raw_ct=None; scene.raw_ct_affine=None; scene.ct_to_mri=None; scene.ct_quality={}
    for model in scene.diffusions:
        model.layer_ids={key:value for key,value in model.layer_ids.items() if value not in removed}
    return removed
