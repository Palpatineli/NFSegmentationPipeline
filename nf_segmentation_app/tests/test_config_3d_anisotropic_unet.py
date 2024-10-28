import unittest
import os


class TestConfig3DAnisotropicUnet(unittest.TestCase):
    """
    Unit test class for testing Config3DAnisotropicUnet configurations and behavior.
    """

    def setUp(self):
        """
        Set up test data before running each test. It includes the model directory, model file name, 
        and necessary configuration for testing.
        """
        self.model_dir = "/path/to/model_directory"
        self.model_file = "3d_anisotropic_unet.pth"
        self.model_path = os.path.join(self.model_dir, self.model_file)
        self.conf = {"preload": "false"}
        self.name = "TestConfig"
        from lib.configs.config_3d_anisotropic_unet import Config3DAnisotropicUnet
        self.ConfigClass = Config3DAnisotropicUnet

    def _init_config(self, resample_only_in_2d=None):
        """
        Helper function to initialize the Config3DAnisotropicUnet object.

        Args:
            resample_only_in_2d (bool): Whether to resample only in 2D or not.
        """
        config = self.ConfigClass()
        config.init(
            name=self.name,
            model_dir=self.model_dir,
            conf=self.conf,
            planner=None,
            resample_only_in_2d=resample_only_in_2d
        )
        return config

    def test_missing_model_file(self):
        """
        Test case to simulate the scenario where the model file is missing. 
        It verifies that the system raises a FileNotFoundError when attempting to infer with a missing model file.
        """
        # Remove the model file to simulate a missing file scenario
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

        # Initialize the Config3DAnisotropicUnet with the missing model file
        config = self._init_config()

        # Verify that the initialization fails due to the missing model file
        with self.assertRaises(FileNotFoundError):
            config.infer()

    def test_resample_in_2d_cases(self):
        """
        Test case to check if the target spacing and other parameters are correctly set based on 
        the 'resample_only_in_2d' flag. Multiple test cases are covered using subTest.
        """
        test_cases = [
            (False, (7.8, 0.625, 0.625)),
            (True, (-1, 0.625, 0.625)),
        ]

        for resample_only_in_2d, expected_target_spacing in test_cases:
            with self.subTest(resample_only_in_2d=resample_only_in_2d):
                config = self._init_config(resample_only_in_2d)

                # Validate the target_spacing
                self.assertEqual(config.target_spacing, expected_target_spacing)

                # Validate other key attributes
                self.assertEqual(config.dimension, 3)
                self.assertEqual(config.labels["neurofibroma"], 1)
                self.assertEqual(config.labels["background"], 0)
                self.assertEqual(config.sw_batch_size, 4)
                self.assertEqual(config.number_intensity_ch, 1)

if __name__ == '__main__':
    unittest.main()
