import unittest
from unittest.mock import patch, MagicMock
import torch


class TestInferer3DAnisotropicUnet(unittest.TestCase):
    """
    Unit test class for testing the inference behavior of Inferer3DAnisotropicUnet.
    """

    def setUp(self):
        """
        Set up test data before running each test. This includes initializing paths, 
        labels, and other configuration parameters for the inferer.
        """
        self.path = "/path/to/model"
        self.labels = {"neurofibroma": 1, "background": 0}
        self.spatial_size = (10, 640, 256)
        self.target_spacing = (7.8, 0.625, 0.625)
        self.overlap = 0.25
        self.number_intensity_ch = 1
        self.sw_batch_size = 4

    def _initialize_inferer(self, path):
        """
        Helper method to initialize the Inferer3DAnisotropicUnet object.
        
        Args:
            path (str): Model path to initialize with. Default is self.path.

        Returns:
            Inferer3DAnisotropicUnet: Initialized configuration object.
        """
        from lib.infers.inferer_3d_anisotropic_unet import Inferer3DAnisotropicUnet

        config = Inferer3DAnisotropicUnet(
            path=path if path is None else self.path,
            network=None,
            labels=self.labels,
            spatial_size=self.spatial_size,
            target_spacing=self.target_spacing,
            overlap=self.overlap,
            number_intensity_ch=self.number_intensity_ch,
            sw_batch_size=self.sw_batch_size
        )
        return config

    def test_model_path_none(self):
        """
        Test that a ValueError is raised when the model path is None.
        """
        with self.assertRaises(ValueError) as context:
            self._initialize_inferer(path=None)
        
        # Validate the error message
        self.assertTrue("Model path cannot be None" in str(context.exception))

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

    def test_run_inferer_missing_input_key(self):
        """
        Test that `run_inferer` raises an error if the required input key is missing.
        """
        config = self._initialize_inferer(self.path)

        # Patch the network and set input data without the required "image" key
        with patch.object(config, '_get_network', return_value=MagicMock()):
            data = {}  # Missing the "image" key

            # Ensure that a KeyError is raised when the input is missing
            with self.assertRaises(KeyError) as context:
                config.run_inferer(data)

            self.assertTrue("image" in str(context.exception))

    @patch('lib.infers.inferer_3d_anisotropic_unet.SlidingWindowInferer')
    def test_run_inferer_output_exists_and_correct(self, mock_sliding_window_inferer):
        """
        Test that `run_inferer` produces output and the output shape matches the input shape.
        """
        config = self._initialize_inferer(self.path)

        # Mock the SlidingWindowInferer to return the input data directly
        mock_inferer = MagicMock(side_effect=lambda inputs, _: inputs)
        mock_sliding_window_inferer.return_value = mock_inferer

        mock_network = MagicMock()
        with patch.object(config, '_get_network', return_value=mock_network):
            # Simulate input data
            input_tensor = torch.rand(1, 1, 64, 128, 128)  # Single-channel input
            data = {"image": input_tensor}

            # Run the inference
            result = config.run_inferer(data)

            # Check that output key exists and is not None
            self.assertIn(config.output_label_key, result)
            self.assertIsNotNone(result[config.output_label_key])

            # Ensure the output shape matches the input shape
            self.assertEqual(result[config.output_label_key].shape, input_tensor.shape)

            # Ensure the output is a tensor
            self.assertTrue(torch.is_tensor(result[config.output_label_key]))

if __name__ == '__main__':
    unittest.main()
