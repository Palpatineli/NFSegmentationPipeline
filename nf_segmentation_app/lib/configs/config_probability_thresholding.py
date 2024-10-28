import logging
from typing import Any, Dict, Optional, Union

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType
from monailabel.interfaces.tasks.train import TrainTask
from lib.infers.inferer_probability_thresholding import InfererProbabilityThresholding

# Initialize the logger for the module
logger = logging.getLogger(__name__)


class ConfigProbabilityThresholding(TaskConfig):
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        """
        Initialize the configuration for probability thresholding.

        Args:
            name (str): The name of the configuration.
            model_dir (str): Directory where model files are stored.
            conf (Dict[str, str]): Configuration dictionary.
            planner (Any): Planner object to be used for task planning.
            **kwargs: Additional configuration parameters, including 'threshold'.
        """
        super().init(name, model_dir, conf, planner, **kwargs)
        
        # Epistemic uncertainty settings (disabled by default)
        self.epistemic_enabled = None
        self.epistemic_samples = None
        
        # Set the probability threshold, with a default value of 0.5
        self.threshold = kwargs.get('threshold', 0.5)
        
        # Set configuration parameters
        self.dimension = 3
        self.labels = {
            "neurofibroma": 1,
            "background": 0,
        }
        self.path = self.model_dir
        self.network = None  # No network since this is a threshold-based approach

        # Logging initialization details
        logger.info(f"Initialized ConfigProbabilityThresholding with threshold: {self.threshold}")

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        """
        Create the inference task for probability thresholding.
        
        Returns:
            Union[InferTask, Dict[str, InferTask]]: An inference task or a dictionary of tasks.
        """
        # Logging inference configuration
        logger.debug(f"Creating InfererProbabilityThresholding with threshold {self.threshold}")

        # Return an inference task with the set configurations
        return {
            self.name: InfererProbabilityThresholding(
                path=self.path,
                network=self.network,
                threshold=self.threshold,
                labels=self.labels,
                dimension=self.dimension,
                type=InferType.SEGMENTATION,
            )
        }

    def trainer(self) -> Optional[TrainTask]:
        """
        Since the task does not involve training, this function returns None.

        Returns:
            Optional[TrainTask]: None, as no training is required for probability thresholding.
        """
        logger.info("No training task required for probability thresholding.")
        return None
