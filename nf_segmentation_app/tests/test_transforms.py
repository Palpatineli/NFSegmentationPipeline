import unittest
import torch
import numpy as np
from monai.data import MetaTensor
from lib.transforms.transforms import ReorientToOriginald, AssembleAnatomyMask, AssembleAnatomyMaskd
from monai.utils import InterpolateMode

class TestReorientToOriginald(unittest.TestCase):
    """
    Unit test class for testing the ReorientToOriginald transform.
    """

    def setUp(self):
        """
        Set up a default instance of ReorientToOriginald with mock data for testing.
        """
        self.transform = ReorientToOriginald(
            keys=["image"],
            ref_image="ref_image",
            has_channel=False,
            invert_orient=True,
            mode=InterpolateMode.NEAREST
        )

        # Sample MetaTensor with affine and metadata
        self.affine = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.input_tensor = MetaTensor(torch.rand(1, 128, 128, 128), affine=self.affine)
        self.input_tensor.meta["original_channel_dim"] = 0
        self.input_tensor.meta["channel_dim"] = 0
        self.input_tensor.meta["pixdim"] = np.array([1.0, 1.0, 1.0])
        self.input_tensor.meta["original_affine"] = self.affine

    def test_initialization(self):
        """
        Test that the transform initializes with the correct attributes.
        """
        self.assertEqual(self.transform.ref_image, "ref_image")
        self.assertFalse(self.transform.has_channel)
        self.assertTrue(self.transform.invert_orient)
        self.assertEqual(self.transform.mode, (InterpolateMode.NEAREST,))

    def test_missing_metadata(self):
        """
        Test that the transform works if metadata is missing in the reference image.
        """
        ref_image = MetaTensor(torch.rand(1, 128, 128, 128), affine=self.affine)  # No original_affine metadata
        data = {"image": self.input_tensor, "ref_image": ref_image}

        result = self.transform(data)
        np.testing.assert_array_equal(result["image"].numpy(), self.input_tensor.numpy())

    def test_valid_reorientation(self):
        """
        Test that the transform applies inverse orientation if metadata is present.
        """
        ref_image = MetaTensor(torch.rand(1, 128, 128, 128), affine=self.affine)
        ref_image.meta["original_affine"] = torch.tensor([
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        data = {"image": self.input_tensor, "ref_image": ref_image}

        result = self.transform(data)
        np.testing.assert_array_equal(result["image_meta_dict"]["affine"], ref_image.meta["original_affine"])

    def test_meta_update_with_affine(self):
        """
        Test that the transform updates the metadata with 'original_affine'.
        """
        ref_image = MetaTensor(torch.rand(1, 128, 128, 128), affine=self.affine)
        ref_image.meta["original_affine"] = self.affine
        data = {"image": self.input_tensor, "ref_image": ref_image}

        result = self.transform(data)
        self.assertIn("image_meta_dict", result)
        self.assertIn("affine", result["image_meta_dict"])
        np.testing.assert_array_equal(result["image_meta_dict"]["affine"], self.affine)


class TestAssembleAnatomyMask(unittest.TestCase):
    """
    Unit test class for testing the AssembleAnatomyMask transform.
    """

    def setUp(self):
        """
        Set up a default instance of AssembleAnatomyMask for testing.
        """
        self.transform = AssembleAnatomyMask(has_channel=False)

    def test_initialization(self):
        """
        Test the initialization and default attributes of AssembleAnatomyMask.
        """
        self.assertFalse(self.transform.has_channel)
        self.assertEqual(self.transform.dilate_iter_spine, 7)
        self.assertEqual(self.transform.dilate_iter_lung, 5)

    def test_invalid_input(self):
        """
        Test that invalid input types raise a NotImplementedError.
        """
        with self.assertRaises(NotImplementedError):
            self.transform("invalid_input")

    def test_label_merging(self):
        """
        Test that label merging works correctly according to the rules.
        """
        raw_mask = np.zeros((128, 128, 128), dtype=np.int32)
        raw_mask[50:60, 50:60, 50:60] = 2  # Left kidney
        raw_mask[70:80, 70:80, 70:80] = 3  # Right kidney
        raw_mask[30:40, 30:40, 30:40] = 12  # Heart

        output_mask = self.transform(raw_mask)
        self.assertTrue(np.all(output_mask[50:60, 50:60, 50:60] == 2))
        self.assertTrue(np.all(output_mask[70:80, 70:80, 70:80] == 2))
        self.assertTrue(np.all(output_mask[30:40, 30:40, 30:40] == 5))

    def test_spine_dilation(self):
        """
        Test that the spine is dilated and a high-risk zone label is added.
        """
        raw_mask = np.zeros((128, 128, 128), dtype=np.int32)
        raw_mask[50:55, 50:55, 50:55] = 25  # Spine label

        output_mask = self.transform(raw_mask)
        self.assertTrue(np.any(output_mask == 12))

    def test_lung_dilation(self):
        """
        Test that the lungs are dilated and a high-risk zone label is added.
        """
        raw_mask = np.zeros((128, 128, 128), dtype=np.int32)
        raw_mask[50:55, 50:55, 50:55] = 11  # Right lung label

        output_mask = self.transform(raw_mask)
        self.assertTrue(np.any(output_mask == 12))

    def test_invalid_labels_in_input(self):
        """
        Test that an error is raised if the input mask contains labels outside of MRSEGMENTATOR_LABELS.
        """
        raw_mask = np.zeros((128, 128, 128), dtype=np.int32)
        raw_mask[50:60, 50:60, 50:60] = 99  # Invalid label

        with self.assertRaises(ValueError) as context:
            self.transform(raw_mask)
        self.assertIn("Input mask contains labels not defined in MRSEGMENTATOR_LABELS", str(context.exception))


class TestAssembleAnatomyMaskd(unittest.TestCase):
    """
    Unit test class for testing the AssembleAnatomyMaskd transform.
    """

    def setUp(self):
        """
        Set up a default instance of AssembleAnatomyMaskd with mock data for testing.
        """
        self.transform = AssembleAnatomyMaskd(
            keys=["segmentation"],
            has_channel=True,
            dilate_structure_size=3,
            dilate_iter_spine=7,
            dilate_iter_lung=5,
            dimension=3,
            allow_missing_keys=False
        )

        affine = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.input_tensor = MetaTensor(torch.zeros((1, 128, 128, 128), dtype=torch.uint8), affine=affine)
        self.input_tensor.meta["original_affine"] = affine
        self.input_tensor.meta["pixdim"] = np.array([1.0, 1.0, 1.0])

    def test_initialization(self):
        """
        Test that the class is initialized with the correct attributes.
        """
        self.assertEqual(self.transform.assembler.dilate_iter_spine, 7)
        self.assertEqual(self.transform.assembler.dilate_iter_lung, 5)
        self.assertTrue(self.transform.assembler.has_channel)

    def test_valid_input(self):
        """
        Test that the transformation works with valid input data.
        """
        data = {"segmentation": self.input_tensor}

        result = self.transform(data)
        self.assertIn("segmentation", result)
        self.assertIsInstance(result["segmentation"], MetaTensor)
        self.assertEqual(result["segmentation"].shape, (1, 128, 128, 128))
        self.assertEqual(result["segmentation"].dtype, torch.uint8)

if __name__ == '__main__':
    unittest.main()
