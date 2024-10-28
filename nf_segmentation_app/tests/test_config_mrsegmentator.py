import unittest
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


class TestConfigMRSegmentator(unittest.TestCase):
    """
    Unit test class for testing the configuration and behavior of ConfigMRSegmentator.
    """

    def setUp(self):
        """
        Set up test data before running each test. It includes the model directory, model configuration,
        target spacing, and labels for segmentation.
        """
        self.model_dir = "/path/to/model_directory"
        self.conf = {"preload": "false"}
        self.name = "TestConfig"
        self.default_target_spacing = (1.5, 1.5, 1.5)
        self.labels = {
            "background": 0, "urinary_bladder": 1, "kidneys": 2, "stomach": 3,
            "liver": 4, "heart": 5, "hips": 6, "femurs": 7, "muscles": 8, "sacrum": 9,
            "lungs": 10, "spine": 11, "high_risk_zone": 12
        }
        from lib.configs.config_mrsegmentator import ConfigMRSegmentator
        self.ConfigClass = ConfigMRSegmentator  # Avoid re-importing in every test

    def _initialize_config(self):
        """
        Helper function to initialize the ConfigMRSegmentator object.

        Returns:
            config (ConfigMRSegmentator): Initialized configuration object.
        """
        config = self.ConfigClass()
        config.init(name=self.name, model_dir=self.model_dir, conf=self.conf, planner=None)
        return config

    def test_initialization(self):
        """
        Test if the configuration initializes properly and sets key attributes correctly.
        """
        config = self._initialize_config()

        # Assert that the key attributes are correctly set
        self.assertEqual(config.name, self.name)
        self.assertEqual(config.dimension, 3)
        self.assertEqual(config.target_spacing, self.default_target_spacing)
        self.assertEqual(config.labels, self.labels)
        self.assertIsInstance(config.network, nnUNetPredictor)  # Ensure nnUNetPredictor is instantiated properly

    def test_infer_creation(self):
        """
        Test the creation of the inference task and its parameters.
        """
        config = self._initialize_config()

        # Generate the inference task
        infer_task = config.infer()

        # Assert that the inference task is created with correct parameters
        self.assertIn(self.name, infer_task)
        self.assertEqual(infer_task[self.name].labels, self.labels)
        self.assertEqual(infer_task[self.name].target_spacing, self.default_target_spacing)
        self.assertEqual(infer_task[self.name].folds, [0, 1, 2, 3, 4])

if __name__ == '__main__':
    unittest.main()
