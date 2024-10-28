import unittest


class TestConfigProbabilityThresholding(unittest.TestCase):
    """
    Unit test class for testing the configuration and behavior of ConfigProbabilityThresholding.
    """

    def setUp(self):
        """
        Set up test data before running each test. It includes the model directory, 
        configuration name, and default settings.
        """
        self.model_dir = "/path/to/model_directory"
        self.name = "TestConfig"
        self.conf = {"preload": "false"}
        from lib.configs.config_probability_thresholding import ConfigProbabilityThresholding
        self.ConfigClass = ConfigProbabilityThresholding  # Avoid repeated imports in each test

    def _initialize_config(self):
        """
        Helper function to initialize the ConfigProbabilityThresholding object.

        Returns:
            config (ConfigProbabilityThresholding): Initialized configuration object.
        """
        config = self.ConfigClass()
        config.init(name=self.name, model_dir=self.model_dir, conf=self.conf, planner=None)
        return config

    def test_default_initialization(self):
        """
        Test if the configuration initializes with the default threshold and other expected settings.
        """
        config = self._initialize_config()

        # Validate default values
        self.assertEqual(config.threshold, 0.5)  # Default threshold
        self.assertEqual(config.dimension, 3)  # 3D segmentation
        self.assertEqual(config.labels["neurofibroma"], 1)
        self.assertEqual(config.labels["background"], 0)
        self.assertIsNone(config.network)  # No AI network used for thresholding

    def test_infer_creation(self):
        """
        Test the creation of the inference task and validate its configuration.
        """
        config = self._initialize_config()

        # Generate the inference task
        infer_task = config.infer()

        # Assert that the inference task is created with the correct parameters
        self.assertIn(self.name, infer_task)
        self.assertEqual(infer_task[self.name].threshold, 0.5)  # Default threshold
        self.assertEqual(infer_task[self.name].labels["neurofibroma"], 1)
        self.assertEqual(infer_task[self.name].dimension, 3)
        self.assertIsNone(infer_task[self.name].network)  # No network used for thresholding

if __name__ == '__main__':
    unittest.main()
