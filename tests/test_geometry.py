import json
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np

from brain_viewer.imaging import (
    InputError, Mesh, Scene, _check_surface_volume, display_position,
    fs_to_scanner, index_from_display, load_scene, plane_pixels, save_scene, validate_scene,
)


class GeometryTests(unittest.TestCase):
    def test_radiological_orientation_known_corners(self):
        # RAS voxel intensity identifies coordinates unambiguously.
        x, y, z = np.indices((4, 5, 6))
        data = 100*x + 10*y + z
        axial = plane_pixels(data, 2, 2)
        self.assertEqual(axial[0, 0], 342)  # Right + anterior at top left.
        self.assertEqual(axial[-1, -1], 2)
        coronal = plane_pixels(data, 1, 2)
        self.assertEqual(coronal[0, 0], 325)  # Right + superior.
        self.assertEqual(coronal[-1, -1], 20)
        sagittal = plane_pixels(data, 0, 1)
        self.assertEqual(sagittal[0, 0], 145)  # Anterior + superior.
        self.assertEqual(sagittal[-1, -1], 100)

    def test_display_point_selects_expected_voxel_and_clamps(self):
        shape = (17, 23, 31)
        index = np.array([3, 7, 19])
        np.testing.assert_array_equal(display_position(shape, 2, index), [13, 15])
        np.testing.assert_array_equal(index_from_display(shape, 2, [13, 15], [0, 0, 19]), index)
        np.testing.assert_array_equal(index_from_display(shape, 2, [-8, 40], index), [16, 0, 19])
        for axis in (0, 1, 2):
            np.testing.assert_array_equal(index_from_display(shape, axis, display_position(shape, axis, index), index), index)

    def test_tkras_uses_scanner_origin_rotation_and_voxel_size(self):
        # Deliberately non-default shape, anisotropic voxels and rotated scanner axes.
        affine = np.array([[0, -2, 0, 41], [1, 0, 0, -37], [0, 0, 3, 12], [0, 0, 0, 1]], dtype=float)
        orig = nib.MGHImage(np.zeros((11, 13, 17), np.float32), affine)
        voxel = np.array([2, 5, 7, 1])
        tkras = orig.header.get_vox2ras_tkr() @ voxel
        scanner = fs_to_scanner(orig) @ tkras
        np.testing.assert_allclose(scanner, [31, -35, 33, 1], atol=1e-5)
        self.assertFalse(np.allclose(scanner[:3], tkras[:3]))

    def test_mismatched_surface_metadata_is_rejected(self):
        affine = np.diag([1., 1., 1., 1.])
        orig = nib.MGHImage(np.zeros((8, 10, 12), np.float32), affine)
        metadata = {"valid": "1  # volume info valid", "volume": [8, 10, 12], "voxelsize": [1, 1, 1],
                    "xras": [1, 0, 0], "yras": [0, 1, 0], "zras": [0, 0, 1], "cras": [4, 5, 6]}
        _check_surface_volume(metadata, orig)
        metadata["cras"] = [14, 5, 6]
        with self.assertRaises(InputError):
            _check_surface_volume(metadata, orig)

    def test_patient_world_to_oblique_native_index(self):
        scene = small_scene()
        scene.source_affine = np.array([[0, -2, 0, 41], [1, 0, 0, -37], [0, 0, 3, 12], [0, 0, 0, 1]], dtype=float)
        np.testing.assert_allclose(scene.native_index([31, -35, 33]), [2, 5, 7])
        np.testing.assert_allclose(scene.world([2, 5, 7]), [-5, 5, 21])

    def test_invalid_affine_and_mesh_rejected(self):
        scene = small_scene()
        scene.affine[0, 1] = .1
        with self.assertRaises(InputError):
            validate_scene(scene)
        scene = small_scene()
        scene.surfaces["pial_lh"].faces[0, 0] = 99
        with self.assertRaises(InputError):
            validate_scene(scene)


def small_scene():
    affine = np.diag([1., 2., 3., 1.])
    affine[:3, 3] = [-7, -5, 0]
    vertices = np.array([[1, 2, 3], [4, 2, 3], [1, 4, 3]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    return Scene(np.arange(8*10*12, dtype=np.float32).reshape(8, 10, 12), affine,
                 affine.copy(), (8, 10, 12), {"pial_lh": Mesh(vertices, faces), "pial_rh": Mesh(vertices.copy(), faces.copy())})


class BundleTests(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parents[1] / "private_reports"
        parent.mkdir(exist_ok=True)
        self.folder = tempfile.TemporaryDirectory(dir=parent)
        assert Path(self.folder.name).resolve().is_relative_to(parent.resolve())
        self.addCleanup(self.folder.cleanup)

    def test_roundtrip_preserves_geometry_and_is_self_contained(self):
        scene = small_scene()
        state = {"ijk": [3, 5, 7], "opacity": 45}
        saved = save_scene(scene, Path(self.folder.name), state)
        restored = load_scene(saved / "scene.json")
        np.testing.assert_array_equal(restored.data, scene.data)
        np.testing.assert_allclose(restored.affine, scene.affine)
        np.testing.assert_array_equal(restored.surfaces["pial_lh"].vertices, scene.surfaces["pial_lh"].vertices)
        self.assertEqual(restored.view_state, state)
        manifest = json.loads((saved / "scene.json").read_text())
        self.assertEqual(manifest["volume_role"], "resampled_display_MRI")
        self.assertEqual(manifest["coordinate_system"], "scanner_RAS_mm")

    def test_path_outside_saved_case_is_rejected(self):
        saved = save_scene(small_scene(), Path(self.folder.name), {})
        path = saved / "scene.json"
        obj = json.loads(path.read_text())
        obj["volume"] = "../../outside.nii.gz"
        path.write_text(json.dumps(obj))
        with self.assertRaises(InputError):
            load_scene(path)

    def test_new_save_never_overwrites_existing_bundle(self):
        scene = small_scene()
        first = save_scene(scene, Path(self.folder.name), {"ijk": [1, 2, 3]})
        second = save_scene(scene, Path(self.folder.name), {"ijk": [4, 5, 6]})
        self.assertNotEqual(first, second)
        self.assertEqual(load_scene(first / "scene.json").view_state["ijk"], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
