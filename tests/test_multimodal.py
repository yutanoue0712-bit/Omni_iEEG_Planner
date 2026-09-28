import json
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
from scipy.io import savemat

from brain_viewer.brainstorm_import import read_contacts, scs_m_to_ras_mm
from brain_viewer.imaging import Contact, InputError, Mesh, Scene, load_scene, save_scene, validate_scene
from brain_viewer.registration import ct_to_mri_matrix, resample_ct, sampling_transform, to_sitk
from brain_viewer.rendering import composite_plane


def small_scene():
    affine = np.eye(4)
    affine[:3, 3] = [6, 23, 23]
    vertices = np.array([[12, 28, 26], [15, 33, 27], [15, 29, 34]], np.float32)
    mesh = Mesh(vertices, np.array([[0, 1, 2]], np.int32))
    return Scene(np.ones((15, 15, 15), np.float32)*100, affine, affine.copy(), (15, 15, 15),
                 {"pial_lh": mesh, "pial_rh": mesh})


def bs_metadata():
    affine = np.diag([2., 3., 4., 1.])
    affine[:3, 3] = [-10, 20, -30]
    init = np.empty((1, 2), dtype=object)
    init[0] = ["vox2ras", affine]
    return {"InitTransf": init, "Voxsize": np.array([2., 3., 4.]),
            "SCS": {"R": np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]]), "T": np.array([10., -20, 30])}}


class MultimodalGeometryTests(unittest.TestCase):
    def test_brainstorm_metres_one_based_voxels_and_rotation(self):
        world = nib.affines.apply_affine(scs_m_to_ras_mm(bs_metadata()), [-.008, -.01, .058])
        np.testing.assert_allclose(world, [-2, 35, -6], atol=1e-10)

    def test_missing_brainstorm_basis_is_rejected(self):
        with self.assertRaises(InputError):
            scs_m_to_ras_mm({})

    def test_ct_transform_direction_ras_lps_and_resampling_landmark(self):
        ct_affine = np.diag([-2., 1.5, 3., 1.])
        ct_affine[:3, 3] = [18, -12, -9]
        data = np.zeros((9, 10, 11), dtype=np.float32)
        data[4, 6, 3] = 800
        transform = np.array([[0., -1, 0, 10], [1, 0, 0, 20], [0, 0, 1, 30], [0, 0, 0, 1]])
        image = to_sitk(data, ct_affine)
        np.testing.assert_allclose(image.TransformIndexToPhysicalPoint((4, 6, 3)), [-10, 3, 0])
        np.testing.assert_allclose(ct_to_mri_matrix(sampling_transform(transform)), transform, atol=1e-10)
        scene = small_scene()
        output, valid = resample_ct(data, ct_affine, scene, transform)
        # Raw RAS (10,-3,0) -> MRI RAS (13,30,30) -> display voxel (7,7,7).
        self.assertAlmostEqual(float(output[7, 7, 7]), 800, places=4)
        self.assertTrue(valid[7, 7, 7])
        self.assertEqual(np.unravel_index(np.argmax(output), output.shape), (7, 7, 7))

    def test_contact_follows_ct_without_snapping_or_refitting(self):
        old = np.eye(4); old[:3, 3] = [4, 5, 6]
        new = np.array([[0., -1, 0, -3], [1, 0, 0, 2], [0, 0, 1, 7], [0, 0, 0, 1]])
        reference_point = nib.affines.apply_affine(old, [10, 20, 30])
        followed = nib.affines.apply_affine(new @ np.linalg.inv(old), reference_point)
        np.testing.assert_allclose(followed, [-23, 12, 37])
        np.testing.assert_allclose(nib.affines.apply_affine(np.linalg.inv(new), followed), [10, 20, 30])

    def test_compositor_fov_mask_and_label_ids(self):
        scene = small_scene()
        scene.ct = np.full(scene.data.shape, 2000., np.float32)
        scene.ct_valid = np.zeros(scene.data.shape, bool)
        scene.ct_valid[4, 5, 6] = True
        scene.ct_to_mri = np.eye(4)
        scene.label_volume = np.zeros(scene.data.shape, np.uint16)
        scene.label_volume[7, 5, 6] = 1030
        scene.label_names = {1030: "ctx-lh-superiortemporal"}
        plain = composite_plane(scene, 2, 6, 0, 200)
        overlay = composite_plane(scene, 2, 6, 0, 200, {"visible": True, "opacity": 1, "window": 2000, "level": 1000})
        np.testing.assert_array_equal(plain[0, 0], overlay[0, 0])
        self.assertGreater(overlay[4, 5, 0], overlay[4, 5, 2])
        self.assertEqual(scene.annotation([7, 5, 6]), (1030, "ctx-lh-superiortemporal"))
        self.assertEqual(scene.annotation([-1, 0, 0])[0], 0)


class InputAndPersistenceTests(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parents[1] / "private_reports"
        parent.mkdir(exist_ok=True)
        self.folder = tempfile.TemporaryDirectory(dir=parent)
        self.path = Path(self.folder.name)
        assert self.path.resolve().is_relative_to(parent.resolve())
        self.addCleanup(self.folder.cleanup)

    def test_import_keeps_physical_contacts_excludes_bipolar(self):
        mri = self.path / "mri.mat"
        savemat(mri, bs_metadata())
        channel = self.path / "channel.mat"
        contact = {"Name": "A1", "Group": "A", "Type": "SEEG", "Loc": np.array([-.008, -.01, .058])}
        bipolar = dict(contact, Name="A1-A2")
        savemat(channel, {"Channel": [contact, bipolar], "IntraElectrodes": {"Name": "A", "Color": np.array([230, 100, 75], np.uint8)}})
        result = read_contacts(channel, mri)
        self.assertEqual(len(result.contacts), 1)
        self.assertEqual(result.contacts[0].name, "A1")
        np.testing.assert_allclose(result.contacts[0].color, np.array([230, 100, 75])/255)
        np.testing.assert_allclose(result.contacts[0].position, [-2, 35, -6])

    def test_bundle_preserves_ct_annotation_and_both_contact_positions(self):
        scene = small_scene()
        scene.ct = np.full(scene.data.shape, 777., np.float32)
        scene.ct_valid = np.ones(scene.data.shape, bool)
        scene.ct_to_mri = np.eye(4)
        scene.label_volume = np.full(scene.data.shape, 1030, np.uint16)
        scene.label_names = {1030: "ctx-lh-superiortemporal"}
        scene.contacts = [Contact("A1", "A", np.array([13., 30, 30]), tuple(np.array([1., .5, .2], np.float32)), np.array([12., 29, 31]))]
        scene.electrode_quality = {"placement": "test"}
        folder = save_scene(scene, self.path, {"ct": {"visible": True}})
        restored = load_scene(folder / "scene.json")
        np.testing.assert_array_equal(restored.ct, scene.ct)
        np.testing.assert_array_equal(restored.ct_valid, scene.ct_valid)
        np.testing.assert_array_equal(restored.label_volume, scene.label_volume)
        np.testing.assert_allclose(restored.contacts[0].source_position, [12, 29, 31])
        self.assertEqual(restored.label_names, scene.label_names)
        manifest = json.loads((folder / "scene.json").read_text())
        self.assertEqual(manifest["schema"], "cortex-viewer/9")
        manifest["ct"] = "../../outside.nii.gz"
        (folder / "scene.json").write_text(json.dumps(manifest))
        with self.assertRaises(InputError):
            load_scene(folder / "scene.json")

    def test_v1_bundle_remains_readable(self):
        folder = save_scene(small_scene(), self.path, {})
        path = folder / "scene.json"
        manifest = json.loads(path.read_text())
        manifest["schema"] = "cortex-viewer/1"
        path.write_text(json.dumps(manifest))
        self.assertIsNone(load_scene(path).ct)


if __name__ == "__main__":
    unittest.main()
