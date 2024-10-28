import unittest
from lib.infers.inferer_single_stage_pipeline import InfererSingleStagePipeline
from lib.infers.inferer_multi_stage_pipeline import InfererMultiStagePipeline
from unittest.mock import MagicMock
from monailabel.interfaces.tasks.infer_v2 import InferTask

class TestPipelineInitialization(unittest.TestCase):
    """
    Unit test class for testing the initialization of single-stage and multi-stage pipelines.
    """

    def test_single_stage_pipeline_initialization(self):
        """
        Test that the single-stage pipeline is initialized correctly with the provided tasks.
        """
        # Mock segmentation and thresholding tasks
        task_segmentation = MagicMock()
        task_thresholding = MagicMock()

        # Initialize the single-stage pipeline
        inferer = InfererSingleStagePipeline(
            task_segmentation=task_segmentation,
            task_thresholding=task_thresholding
        )

        # Validate initialization of the pipeline with the correct tasks
        self.assertEqual(inferer.task_segmentation, task_segmentation)
        self.assertEqual(inferer.task_thresholding, task_thresholding)

    def test_multi_stage_pipeline_initialization(self):
        """
        Test that the multi-stage pipeline is initialized correctly with the provided anatomy, 
        neurofibroma segmentation, and thresholding tasks.
        """
        # Mock segmentation tasks for anatomy, neurofibroma, and thresholding
        task_anatomy_segmentation = MagicMock(spec=InferTask)
        task_neurofibroma_segmentation = MagicMock(spec=InferTask)
        task_thresholding = MagicMock(spec=InferTask)

        # Set up mock labels and dimension for the neurofibroma segmentation task
        task_neurofibroma_segmentation.labels = {"neurofibroma": 1, "background": 0}
        task_neurofibroma_segmentation.dimension = 3

        # Initialize the multi-stage pipeline
        inferer = InfererMultiStagePipeline(
            task_anatomy_segmentation=task_anatomy_segmentation,
            task_neurofibroma_segmentation=task_neurofibroma_segmentation,
            task_thresholding=task_thresholding
        )

        # Validate that the pipeline's tasks are set correctly
        self.assertEqual(inferer.task_anatomy_segmentation, task_anatomy_segmentation)
        self.assertEqual(inferer.task_neurofibroma_segmentation, task_neurofibroma_segmentation)
        self.assertEqual(inferer.task_thresholding, task_thresholding)

        # Validate inherited attributes from the neurofibroma segmentation task
        self.assertEqual(inferer.labels, {"neurofibroma": 1, "background": 0})
        self.assertEqual(inferer.dimension, 3)
        self.assertEqual(
            inferer.description,
            "Combines anatomy segmentation, neurofibroma segmentation, and thresholding into a multi-stage pipeline"
        )

if __name__ == '__main__':
    unittest.main()
