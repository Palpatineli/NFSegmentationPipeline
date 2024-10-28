import unittest
import os


class TestConfigTumorCandidateClassification(unittest.TestCase):
    """
    Unit test class for testing the configuration and behavior of ConfigTumorCandidateClassification.
    """

    def setUp(self):
        """
        Set up test data before running each test. It includes the model directory, configuration name,
        and default anatomical regions.
        """
        self.model_dir = "/path/to/model_directory"
        self.name = "TestConfig"
        self.conf = {"preload": "false"}
        self.default_anatomical_regions = ["head", "chest", "abdomen", "legs"]
        from lib.configs.config_tumor_candidate_classification import ConfigTumorCandidateClassification
        self.ConfigClass = ConfigTumorCandidateClassification  # Avoid repeated imports in each test

    def _initialize_config(self, resample_only_in_2d=None):
        """
        Helper function to initialize the ConfigTumorCandidateClassification object.

        Args:
            resample_only_in_2d (bool, optional): Whether to resample only in 2D or not.

        Returns:
            config (ConfigTumorCandidateClassification): Initialized configuration object.
        """
        config = self.ConfigClass()
        config.init(name=self.name, model_dir=self.model_dir, conf=self.conf, planner=None, resample_only_in_2d=resample_only_in_2d)
        return config

    def test_default_initialization(self):
        """
        Test if the configuration initializes with default settings, including resample_only_in_2d.
        """
        config = self._initialize_config()

        # Validate default values including resample_only_in_2d
        self.assertFalse(config.resample_only_in_2d)
        self.assertEqual(config.target_spacing, (0.625, 0.625, 7.8))
        self.assertEqual(config.downsampled_spacing, (1.5, 1.5, 7.8))

    def test_missing_files(self):
        """
        Simulate a missing file scenario by removing model and radiomic feature files, 
        and verify if a FileNotFoundError is raised.
        """
        config = self._initialize_config()

        # Remove model and radiomic feature files to simulate the missing file scenario
        for path in config.model_path:
            if os.path.exists(path):
                os.remove(path)

        if os.path.exists(config.radiomic_extractor_config_path):
            os.remove(config.radiomic_extractor_config_path)

        for path in config.radiomic_feature_list_path:
            if os.path.exists(path):
                os.remove(path)

        # Verify that a FileNotFoundError is raised due to missing files
        with self.assertRaises(FileNotFoundError):
            config.infer()

    def test_resample_in_2d_cases(self):
        """
        Test different resample_only_in_2d cases to validate that target_spacing and related 
        attributes are set correctly.
        """
        # Define test cases for resample_only_in_2d and expected target_spacing
        test_cases = [
            (False, (0.625, 0.625, 7.8)),
            (True, (0.625, 0.625, -1)),
        ]

        for resample_only_in_2d, expected_target_spacing in test_cases:
            with self.subTest(resample_only_in_2d=resample_only_in_2d):
                config = self._initialize_config(resample_only_in_2d)

                # Validate target_spacing based on resample_only_in_2d
                self.assertEqual(config.target_spacing, expected_target_spacing)

                # Validate other key attributes
                self.assertEqual(config.dimension, 3)
                self.assertEqual(config.labels["neurofibroma"], 1)
                self.assertEqual(config.labels["background"], 0)

                # Validate anatomical regions and labels
                self.assertEqual(config.anatomical_regions, self.default_anatomical_regions)
                self.assertEqual(config.anatomical_labels["urinary_bladder"], 1)
                self.assertEqual(config.anatomical_labels["high_risk_zone"], 12)

if __name__ == '__main__':
    unittest.main()
