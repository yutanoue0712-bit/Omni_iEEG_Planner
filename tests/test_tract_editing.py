import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import nibabel as nib
import numpy as np
import vtk
from brain_viewer.diffusion import TractBundle,validate_diffusion
from brain_viewer.diffusion_storage import export_tract
from brain_viewer.diffusion_rendering import tract_segments
from brain_viewer.tract_editing import (segments_hit_polygon,select_slice,select_projection,
                                      project_segments,exclude,restore_all,undo)
from brain_viewer.imaging import save_scene,load_scene,InputError
from test_diffusion import synthetic_model


def example_bundle():
    points=np.array([[-10,0,0],[10,0,0],[-10,0,4],[10,0,4],
                     [-10,6,0],[10,6,0],[0,0,-8],[0,0,8]],np.float32)
    return TractBundle('tract_editing','Test fibers','dti_synthetic',points,np.arange(0,10,2,dtype=np.int64),
                       {'fa_threshold':.2,'max_angle':35})


class TractEditingTests(unittest.TestCase):
    def test_polygon_crossings_concavity_boundary_and_point(self):
        polygon=np.array([[0,0],[4,0],[4,1],[1,1],[1,4],[0,4]])
        lines=np.array([[[-2,.5],[6,.5]],[[2,2],[3,3]],[[.5,.5],[.5,.5]],
                        [[0,-2],[0,6]],[[2,2],[6,2]]])
        np.testing.assert_array_equal(segments_hit_polygon(lines,polygon),[True,False,True,True,False])
        np.testing.assert_array_equal(segments_hit_polygon(lines,polygon[::-1]),[True,False,True,True,False])

    def test_2d_slab_thickness_crossing_and_cache_invalidation(self):
        bundle=example_bundle()
        # Non-unit reference lattice: polygon in voxel coordinates maps to patient mm.
        affine=np.diag([2.,3.,4.,1.]);affine[:3,3]=[10.,20.,30.]
        scene=SimpleNamespace(world=lambda pts:nib.affines.apply_affine(affine,pts))
        polygon_world=np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]])
        vertices=nib.affines.apply_affine(np.linalg.inv(affine),polygon_world)
        np.testing.assert_array_equal(select_slice(bundle,scene,2,vertices,2),[0,3])
        original=bundle.points.copy();exclude(bundle,[0],{'kind':'test'})
        np.testing.assert_array_equal(select_slice(bundle,scene,2,vertices,2),[3])
        self.assertEqual(len(tract_segments(bundle)),3)
        self.assertEqual(bundle.count,3)
        undo(bundle);self.assertEqual(len(tract_segments(bundle)),4)
        np.testing.assert_array_equal(bundle.points,original)

    def test_camera_projection_depth_near_far_and_both_outside(self):
        lines=np.array([[[-2,0,0],[2,0,0]],[[0,0,2],[.1,0,3]],[[0,0,-2],[0,0,2]]],float)
        projected,ids=project_segments(lines,np.eye(4))
        np.testing.assert_array_equal(ids,[0,2])
        np.testing.assert_allclose(projected[0],[[-1,0],[1,0]])
        bundle=TractBundle('tract_camera','test','dti_synthetic',lines.reshape(-1,3),np.array([0,2,4,6]))
        polygon=[[-.1,-.1],[.1,-.1],[.1,.1],[-.1,.1]]
        np.testing.assert_array_equal(select_projection(bundle,polygon,np.eye(4)),[0,2])

    def test_perspective_projection_matches_vtk_and_rejects_behind_camera(self):
        camera=vtk.vtkCamera();camera.SetPosition(0,0,10);camera.SetFocalPoint(0,0,0)
        camera.SetClippingRange(1,30);camera.SetViewAngle(60)
        m=camera.GetCompositeProjectionTransformMatrix(1,-1,1)
        matrix=np.array([[m.GetElement(r,c) for c in range(4)] for r in range(4)])
        points=np.array([[-1,0,0],[1,0,0],[-1,0,20],[1,0,20]],float)
        bundle=TractBundle('tract_perspective','test','dti_synthetic',points,np.array([0,2,4]))
        np.testing.assert_array_equal(select_projection(bundle,[[-.1,-.1],[.1,-.1],[.1,.1],[-.1,.1]],matrix),[0])

    def test_persistence_undo_restore_and_export_only_remaining_fibers(self):
        scene,_=synthetic_model();bundle=example_bundle();scene.tracts=[bundle]
        original=bundle.points.copy()
        exclude(bundle,[0,1,3],{'kind':'test'});self.assertEqual(bundle.count,1)
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'private_reports') as directory:
            folder=Path(directory);saved=save_scene(scene,folder/'work',{})
            restored=load_scene(saved/'scene.json').tracts[0]
            self.assertEqual(restored.count,1);np.testing.assert_array_equal(restored.points,original)
            for suffix in ('tck','trk','vtp'):
                path=folder/f'remaining.{suffix}';export_tract(restored,path,scene)
                if suffix=='vtp':
                    reader=vtk.vtkXMLPolyDataReader();reader.SetFileName(str(path));reader.Update()
                    self.assertEqual(reader.GetOutput().GetNumberOfLines(),1)
                    self.assertEqual(reader.GetOutput().GetNumberOfPoints(),2)
                else:
                    lines=list(nib.streamlines.load(path).streamlines)
                    self.assertEqual(len(lines),1);np.testing.assert_allclose(lines[0],original[4:6])
                metadata=json.loads(path.with_suffix(path.suffix+'.json').read_text(encoding='utf-8'))
                self.assertEqual(metadata['excluded_count'],3)
            undo(restored);self.assertEqual(restored.count,4)
            exclude(restored,[0,1],{'kind':'test'});restore_all(restored);self.assertEqual(restored.count,4)
            undo(restored);self.assertEqual(restored.count,2)
            exclude(restored,[2,3],{'kind':'test'})
            with self.assertRaises(InputError):export_tract(restored,folder/'empty.tck',scene)
            scene.tracts=[restored]
            validate_diffusion(scene)
            saved=save_scene(scene,folder/'empty_work',{})
            self.assertEqual(load_scene(saved/'scene.json').tracts[0].count,0)

    def test_invalid_exclusion_mask_or_history_is_rejected(self):
        scene,_=synthetic_model();bundle=example_bundle();scene.tracts=[bundle]
        bundle.excluded=np.zeros(3,bool)
        with self.assertRaises(InputError):validate_diffusion(scene)
        bundle.excluded=np.zeros(4,bool);bundle.edits=[{'action':'exclude','ids':[12]}]
        with self.assertRaises(InputError):validate_diffusion(scene)
