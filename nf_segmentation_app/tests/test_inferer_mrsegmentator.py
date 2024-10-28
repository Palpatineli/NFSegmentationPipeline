import unittest
from unittest.mock import MagicMock, patch
import torch
import numpy as np
from monai.data import MetaTensor


class TestInfererMRSegmentator(unittest.TestCase):
    """
    Unit test class for testing the inference behavior of InfererMRSegmentator.
    """

    def setUp(self):
        """
        Set up default parameters before running each test. This includes initializing the 
        model path, labels, target spacing, orientation, and dilation settings.
        """
        self.path = "/path/to/model"
        self.checkpoint_file_name = "checkpoint.pth"
        self.labels = {"organ1": 1, "organ2": 2}
        self.dimension = 3
        self.target_spacing = (1.5, 1.5, 1.5)
        self.orientation = "SPL"
        self.folds = [0, 1, 2]
        self.dilate_structure_size = 3
        self.dilate_iter_spine = 5
        self.dilate_iter_lung = 7

    def _initialize_inferer(self, path=None):
        """
        Helper method to initialize the InfererMRSegmentator object.
        
        Args:
            path (str): Model path to initialize with. Default is self.path.

        Returns:
            InfererMRSegmentator: Initialized configuration object.
        """
        from lib.infers.inferer_mrsegmentator import InfererMRSegmentator

        config = InfererMRSegmentator(
            path=path if path is None else self.path,
            checkpoint_file_name=self.checkpoint_file_name,
            labels=self.labels,
            target_spacing=self.target_spacing,
            orientation=self.orientation,
            folds=self.folds,
            dilate_structure_size=self.dilate_structure_size,
            dilate_iter_spine=self.dilate_iter_spine,
            dilate_iter_lung=self.dilate_iter_lung
        )
        return config

    def test_initialization(self):
        """
        Test if the InfererMRSegmentator class is initialized with the correct values.
        """
        config = self._initialize_inferer(self.path)

        self.assertEqual(config.target_spacing, (1.5, 1.5, 1.5))
        self.assertEqual(config.dilate_structure_size, 3)
        self.assertEqual(config.orientation, "SPL")

    def test_model_path_none(self):
        """
        Test that a ValueError is raised when the model path is None.
        """
        with self.assertRaises(ValueError) as context:
            self._initialize_inferer(path=None)

        # Validate the error message
        self.assertTrue("Model path cannot be None" in str(context.exception))

    def test_required_input_key(self):
        """
        Test that run_inferer raises a KeyError when the required input key is missing.
        """
        config = self._initialize_inferer(self.path)

        config.network = MagicMock()  # Mock the network

        with patch.object(config.network, 'initialize_from_trained_model_folder', return_value=MagicMock()):
            data = {}  # Missing "image" key

            # Ensure that a KeyError is raised when the input is missing
            with self.assertRaises(KeyError) as context:
                config.run_inferer(data)
            self.assertTrue("image" in str(context.exception))

    def test_run_inferer_valid_output(self):
        """
        Test that `run_inferer` produces valid output and the output shape matches the expected shape.
        """
        config = self._initialize_inferer(self.path)

        # Mock self.network and simulate segmentation output
        config.network = MagicMock()
        mock_segmentation = np.random.rand(1, 128, 128, 128)  # Simulated segmentation output
        config.network.predict_from_list_of_npy_arrays.return_value = mock_segmentation

        with patch.object(config.network, 'initialize_from_trained_model_folder', return_value=MagicMock()):
            # Simulate input MetaTensor data
            affine = torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ])
            input_tensor = MetaTensor(torch.rand(1, 128, 128, 128), affine=affine)
            input_tensor.meta["original_channel_dim"] = 0
            input_tensor.meta["channel_dim"] = 0
            input_tensor.meta["pixdim"] = np.array([1.0, 1.0, 1.0])
            input_tensor.meta["original_affine"] = affine

            data = {"image": input_tensor}

            # Run the inference
            result = config.run_inferer(data)

            # Check that the output exists and has metadata
            self.assertIn(config.output_label_key, result)
            self.assertIsNotNone(result[config.output_label_key])
            self.assertTrue(torch.is_tensor(result[config.output_label_key]))
            self.assertEqual(result[config.output_label_key].shape, (1, 128, 128, 128))

if __name__ == '__main__':
    unittest.main()
