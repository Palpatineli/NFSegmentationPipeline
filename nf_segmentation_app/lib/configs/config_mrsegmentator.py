import logging
import torch
from typing import Any, Dict, Optional, Union
from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.utils.others.generic import strtobool

from lib.infers.inferer_mrsegmentator import InfererMRSegmentator
from mrsegmentator import config, utils
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# Disable nnUNet path warnings from mrsegmentator config
config.disable_nnunet_path_warnings()

# Initialize logger for this module
logger = logging.getLogger(__name__)


class ConfigMRSegmentator(TaskConfig):
    """
    Configuration class for MRSegmentator using nnUNet architecture for multi-organ segmentation.
    """

    def init(
        self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs
    ):
        """
        Initialize the configuration for MRSegmentator.

        Args:
            name (str): Name of the task configuration.
            model_dir (str): Directory where the model is stored.
            conf (Dict[str, str]): Configuration dictionary.
            planner (Any): Planner object for model planning.
            **kwargs: Additional arguments for customization.
        """
        super().init(name, model_dir, conf, planner, **kwargs)

        # Epistemic uncertainty settings (disabled by default)
        self.epistemic_enabled = None
        self.epistemic_samples = None
        
        # Setup the MRSegmentator and get the model weights directory
        config.setup_mrseg()
        self.path = config.get_weights_dir()
        
        
        # Define labels for the organs to be segmented
        self.labels = {
            "background": 0, "urinary_bladder": 1, "kidneys": 2, "stomach": 3,
            "liver": 4, "heart": 5, "hips": 6, "femurs": 7, "muscles": 8, "sacrum": 9,
            "lungs": 10, "spine": 11, "high_risk_zone": 12
        }
        
        # Set key configuration parameters
        self.dimension = 3  # 3D model
        self.target_spacing = (1.5, 1.5, 1.5)  # Target voxel spacing for the model

        # Instantiate nnUNetPredictor with configuration
        # self.folds = [0, 1, 2, 3, 4]  # Use all folds for inference
        self.folds = [0, 1]
        
        self.network = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            device=torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu"),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True,
        )
    
    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        """
        Create and return the inference task using the MRSegmentator.

        Returns:
            Union[InferTask, Dict[str, InferTask]]: The inference task for segmentation.
        """
        return {
            self.name: InfererMRSegmentator(
                path=self.path,
                network=self.network,
                labels=self.labels,
                preload=strtobool(self.conf.get("preload", "false")),
                dimension=self.dimension,
                target_spacing=self.target_spacing,
                folds=self.folds,
                type=InferType.SEGMENTATION,
            )
        }

    def trainer(self) -> Optional[TrainTask]:
        """
        Since training is not required for this configuration, it returns None.
        """
        return None
