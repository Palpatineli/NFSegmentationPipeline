import unittest
import numpy as np
from monai.data import MetaTensor


class TestInfererProbabilityThresholdingInputValidation(unittest.TestCase):
    """
    Unit test class for testing input validation in InfererProbabilityThresholding.
    """

    def setUp(self):
        """
        Set up default parameters and a default image for testing. This includes creating 
        an instance of MetaTensor with simulated image data and affine transformations.
        """
        self.path = "/some/path"
        self.labels = {"foreground": 1, "background": 0}
        self.threshold = 0.5

        self.default_affine = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        image = MetaTensor(np.random.rand(1, 128, 128, 128), affine=self.default_affine)
        image.meta["original_channel_dim"] = 0
        image.meta["channel_dim"] = 0
        image.meta["pixdim"] = np.array([1.0, 1.0, 1.0])
        image.meta["original_affine"] = self.default_affine
        self.default_image = image

    def test_initialization(self):
        """
        Test that the InfererProbabilityThresholding is initialized with the correct default values.
        """
        from lib.infers.inferer_probability_thresholding import InfererProbabilityThresholding
        inferer = InfererProbabilityThresholding(
            path=self.path,
            labels=self.labels,
            threshold=self.threshold
        )

        # Validate initialization values
        self.assertEqual(inferer.threshold, 0.5)
        self.assertEqual(inferer.dimension, 3)
        self.assertEqual(inferer.labels, {"foreground": 1, "background": 0})

    def test_call_method_missing_proba(self):
        """
        Test that the `__call__` method raises an error if 'proba' is missing in the input data.
        """
        from lib.infers.inferer_probability_thresholding import InfererProbabilityThresholding
        inferer = InfererProbabilityThresholding(
            path=self.path,
            labels=self.labels,
            threshold=self.threshold
        )

        # Simulate input data without the 'proba' key
        request = {
            "image": self.default_image  # Simulate valid image input
        }

        # Validate that a ValueError is raised when 'proba' is missing
        with self.assertRaises(ValueError):
            inferer(request)

    def test_call_method_proba_values_out_of_range(self):
        """
        Test that the `__call__` method raises an error if 'proba' contains values outside the range 0-255.
        """
        from lib.infers.inferer_probability_thresholding import InfererProbabilityThresholding
        inferer = InfererProbabilityThresholding(
            path=self.path,
            labels=self.labels,
            threshold=self.threshold
        )

        # Simulate 'proba' data with values outside the valid range (e.g., negative or above 255)
        invalid_proba = MetaTensor(np.random.randint(-10, 300, (1, 128, 128, 128)), affine=self.default_affine)
        invalid_proba.meta["original_channel_dim"] = 0
        invalid_proba.meta["channel_dim"] = 0
        invalid_proba.meta["pixdim"] = np.array([1.0, 1.0, 1.0])
        invalid_proba.meta["original_affine"] = self.default_affine

        request = {
            "proba": invalid_proba,  # Invalid 'proba'
            "image": self.default_image
        }

        # Validate that a ValueError is raised due to out-of-range 'proba' values
        with self.assertRaises(ValueError):
            inferer(request)

if __name__ == '__main__':
    unittest.main()
