import unittest
import os


class TestConfig3DAnisotropicAnatomicUnet(unittest.TestCase):
    """
    Unit test class for testing Config3DAnisotropicAnatomicUnet configurations and behavior.
    """

    def setUp(self):
        """
        Set up test data before running each test. It includes the model directory, model file name, 
        and necessary configuration for testing.
        """
        self.model_dir = "/path/to/model_directory"
        self.model_file = "3d_anisotropic_anatomic_unet_fold_0.pth"
        self.model_path = os.path.join(self.model_dir, self.model_file)
        self.conf = {"preload": "false"}
        self.name = "TestConfig"
        from lib.configs.config_3d_anisotropic_anatomic_unet import Config3DAnisotropicAnatomicUnet
        self.ConfigClass = Config3DAnisotropicAnatomicUnet

    def _init_config(self, resample_only_in_2d=None):
        """
        Helper function to initialize the Config3DAnisotropicAnatomicUnet object.

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

        # Initialize the Config3DAnisotropicAnatomicUnet with the missing model file
        config = self._init_config()

        # Verify that the initialization fails due to the missing model file
        with self.assertRaises(FileNotFoundError):
            config.infer()

    def test_resample_in_2d_cases(self):
        """
        Test case to check if the target spacing and other parameters are correctly set based on 
        the 'resample_only_in_2d' flag. Multiple test cases are covered using subTest.
        """
        # Define test cases for resampling in 2D and expected target spacing
        test_cases = [
            (False, (7.8, 0.625, 0.625)),
            (True, (-1, 0.625, 0.625)),
        ]

        # Loop through the test cases using subTest to check different scenarios
        for resample_only_in_2d, expected_target_spacing in test_cases:
            with self.subTest(resample_only_in_2d=resample_only_in_2d):
                config = self._init_config(resample_only_in_2d)

                # Verify that the configuration settings are as expected
                self.assertEqual(config.target_spacing, expected_target_spacing)
                self.assertEqual(config.dimension, 3)
                self.assertEqual(config.labels["neurofibroma"], 1)
                self.assertEqual(config.labels["background"], 0)
                self.assertEqual(config.number_anatomical_structures, 12)

if __name__ == '__main__':
    unittest.main()
