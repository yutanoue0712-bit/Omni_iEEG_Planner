"""Run ANTs in its own interpreter. Receives only local, task-owned image paths."""
import json
import os
from pathlib import Path
import shutil
import sys
import time


def run(folder):
    folder = Path(folder).resolve()
    options = json.loads((folder / 'options.json').read_text(encoding='utf-8'))
    # Set before importing either numerical runtime; never alter the GUI's runtime.
    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '4'
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['ANTS_RANDOM_SEED'] = '31415'
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['MPLCONFIGDIR'] = str(folder / 'matplotlib')

    def status(stage):
        path = folder / 'status.partial'
        path.write_text(json.dumps({'stage': stage}), encoding='utf-8')
        path.replace(folder / 'status.json')

    status('ANTsPyを準備しています…')
    import ants
    import numpy as np
    started = time.perf_counter()
    fixed = ants.image_read(str(folder / 'fixed.nii.gz'))
    moving = ants.image_read(str(folder / 'moving.nii.gz'))
    fixed_mask = ants.image_read(str(folder / 'fixed_mask.nii.gz'))
    moving_mask = ants.image_read(str(folder / 'moving_mask.nii.gz'))
    if options['n4']:
        status('MRIの濃淡むらを補正しています…')
        fixed = ants.n4_bias_field_correction(fixed, mask=fixed_mask, shrink_factor=2,
                    convergence={'iters': [30, 20], 'tol': 1e-6})
        moving = ants.n4_bias_field_correction(moving, mask=fixed_mask * moving_mask, shrink_factor=2,
                    convergence={'iters': [30, 20], 'tol': 1e-6})
    status('ANTsPyで残存脳の変形補正を計算しています…')
    result = ants.registration(fixed=fixed, moving=moving,
        type_of_transform='SyNOnly', initial_transform='Identity',
        mask=fixed_mask, moving_mask=moving_mask, mask_all_stages=True,
        syn_metric=options['metric'], syn_sampling=2 if options['metric']=='CC' else 40,
        reg_iterations=tuple(options['iterations']), grad_step=.15,
        flow_sigma=3., total_sigma=0., singleprecision=True,
        random_seed=31415, outprefix=str(folder / 'ants_'), verbose=False)
    status('ANTsPyの変形情報をまとめています…')
    if not np.isfinite(result['warpedmovout'].numpy()).all():
        raise RuntimeError('Non-finite registered image')
    # ANTs image resampling uses a fixed -> moving pull mapping, in physical LPS.
    composed = ants.apply_transforms(fixed, moving, result['fwdtransforms'],
                                    compose=str(folder / 'composed_'))
    shutil.copyfile(composed, folder / 'displacement.nii.gz')
    (folder / 'result.json').write_text(json.dumps({
        'ants_version': ants.__version__, 'elapsed_seconds': time.perf_counter()-started,
        'threads': 4, 'seed': 31415, 'grad_step': .15, 'flow_sigma': 3., 'total_sigma': 0.,
        'initial_transform': 'Identity', 'transform': 'SyNOnly', **options
    }), encoding='utf-8')


if __name__ == '__main__':
    run(sys.argv[1])
