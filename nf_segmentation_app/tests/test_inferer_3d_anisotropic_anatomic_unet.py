import unittest
from unittest.mock import MagicMock, patch
import torch
from monai.data import MetaTensor
import numpy as np


class TestInferer3DAnisotropicAnatomicUnet(unittest.TestCase):
    """
    Unit test class for testing the inference behavior of Inferer3DAnisotropicAnatomicUnet.
    """

    def setUp(self):
        """
        Set up the test data before running each test. This includes initializing paths, 
        labels, and other configuration parameters for the inferer.
        """
        self.path = "/path/to/model"
        self.labels = {"neurofibroma": 1, "background": 0}
        self.spatial_size = (10, 640, 256)
        self.target_spacing = (7.8, 0.625, 0.625)
        self.overlap = 0.25
        self.number_intensity_ch = 2
        self.sw_batch_size = 4
        self.number_anatomical_structures = 12

    def _initialize_inferer(self, path):
        """
        Helper method to initialize the Inferer3DAnisotropicAnatomicUnet object.
        
        Args:
            path (str): Model path to initialize with. Default is self.path.

        Returns:
            Inferer3DAnisotropicAnatomicUnet: Initialized configuration object.
        """
        from lib.infers.inferer_3d_anisotropic_anatomic_unet import Inferer3DAnisotropicAnatomicUnet

        config = Inferer3DAnisotropicAnatomicUnet(
            path=path if path is None else self.path,
            network=None,
            labels=self.labels,
            spatial_size=self.spatial_size,
            target_spacing=self.target_spacing,
            overlap=self.overlap,
            number_intensity_ch=self.number_intensity_ch,
            sw_batch_size=self.sw_batch_size,
            number_anatomical_structures=self.number_anatomical_structures
        )
        return config

    def test_model_path_none(self):
        """
        Test that a ValueError is raised when the model path is None.
        """
        with self.assertRaises(ValueError) as context:
            self._initialize_inferer(path=None)

        # Validate the error message
        self.assertTrue('Model path cannot be None' in str(context.exception))

    def test_initialization(self):
        """
        Test if the class is initialized with the correct values.
        """
        config = self._initialize_inferer(self.path)

        # Validate initialization values
        self.assertEqual(config.spatial_size, (10, 640, 256))
        self.assertEqual(config.target_spacing, (7.8, 0.625, 0.625))
        self.assertEqual(config.sw_batch_size, 4)
        self.assertEqual(config.overlap, 0.25)
        self.assertEqual(config.number_anatomical_structures, 12)
        self.assertEqual(config.path_folds, [self.path])

    def test_run_inferer_missing_input_key(self):
        """
        Test that `run_inferer` raises an error if the required input key is missing.
        """
        config = self._initialize_inferer(self.path)

        # Patch the network and set input data without the required "image_joint" key
        with patch.object(config, '_get_network', return_value=MagicMock()):
            data = {}  # Missing the "image_joint" key

            # Ensure that a KeyError is raised when the input is missing
            with self.assertRaises(KeyError) as context:
                config.run_inferer(data)

            self.assertTrue('image_joint' in str(context.exception))

    @patch('lib.infers.inferer_3d_anisotropic_anatomic_unet.SlidingWindowInferer')
    def test_run_inferer_output_exists_and_correct(self, mock_sliding_window_inferer):
        """
        Test that `run_inferer` produces output and the output shape matches the input shape.
        """
        config = self._initialize_inferer(self.path)

        # Mock the SlidingWindowInferer to return the input data directly
        mock_inferer = MagicMock(side_effect=lambda inputs, _: inputs)
        mock_sliding_window_inferer.return_value = mock_inferer

        # Patch the network and simulate input data
        mock_network = MagicMock()
        with patch.object(config, '_get_network', return_value=mock_network):
            # Simulate input MetaTensor data
            affine = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0, 0.0],
                                   [0.0, 0.0, 1.0, 0.0],
                                   [0.0, 0.0, 0.0, 1.0]])
            input_tensor = MetaTensor(torch.rand(2, 10, 640, 256), affine=affine)
            input_tensor.meta["original_channel_dim"] = 0
            input_tensor.meta["channel_dim"] = 0
            input_tensor.meta["pixdim"] = np.array([1.0, 1.0, 1.0])
            input_tensor.meta["original_affine"] = affine

            # Simulated input data dictionary
            data = {"image_joint": input_tensor}

            # Run the inference
            result = config.run_inferer(data)

            # Check that output key exists and is not None
            self.assertIn(config.output_label_key, result)
            self.assertIsNotNone(result[config.output_label_key])

            # Ensure the output shape matches the input shape
            self.assertEqual(result[config.output_label_key].shape, (10, 640, 256))

            # Ensure the output is a tensor
            self.assertTrue(torch.is_tensor(result[config.output_label_key]))

if __name__ == '__main__':
    unittest.main()
