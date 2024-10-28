import logging
import os
from typing import Any, Dict, Optional, Union
from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType
from monailabel.interfaces.tasks.train import TrainTask
from lib.infers.inferer_tumor_candidate_classification import InfererTumorCandidateClassification

# Initialize the logger for this module
logger = logging.getLogger(__name__)


class ConfigTumorCandidateClassification(TaskConfig):
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        """
        Initialize the configuration for tumor candidate classification.

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
        
        # Define model subfolder
        model_subfolder = "tumor_candidate_classification"
        
        # Set thresholds used in tumor candidate extraction and classification
        self.confidence_threshold = kwargs.get('confidence_threshold', 0.5)
        self.classification_threshold = kwargs.get('classification_threshold', 0.5)
        # Smaller tumors are considered False Positives
        self.size_min_threshold = kwargs.get('size_min_threshold', 10) 
         # Larger tumors are considered True Positives
        self.size_max_threshold = kwargs.get('size_max_threshold', 500000)
        
        # Set anatomical regions and labels for tumor candidate classification 
        self.anatomical_regions = ["head", "chest", "abdomen", "legs"]       
        self.anatomical_labels = {
            "background": 0, "urinary_bladder": 1, "kidneys": 2, "stomach": 3,
            "liver": 4, "heart": 5, "hips": 6, "femurs": 7, "muscles": 8, "sacrum": 9,
            "lungs": 10, "spine": 11, "high_risk_zone": 12
            }
        
        # Set paths to model and radiomic feature files for each region
        self.model_path = [
            os.path.join(self.model_dir, model_subfolder, f"model_{region}.joblib")
            for region in self.anatomical_regions
        ]
        self.radiomic_extractor_config_path = os.path.join(
            self.model_dir, model_subfolder, "radiomic_extractor_config.yml"
        )
        self.radiomic_feature_list_path = [
            os.path.join(self.model_dir, model_subfolder, f"radiomic_feature_list_{region}.joblib")
            for region in self.anatomical_regions
        ]
        
        # Set geometric settings
        self.dimension = 3
        self.resample_only_in_2d = kwargs.get("resample_only_in_2d", False)
        self.target_spacing = (0.625, 0.625, 7.8) if not self.resample_only_in_2d else (0.625, 0.625, -1)
        self.downsampled_spacing= (1.5, 1.5, 7.8) if not self.resample_only_in_2d else (1.5, 1.5, -1)
        
        # Set segmentation labels
        self.labels = {
            "neurofibroma": 1,
            "background": 0,
        }
        
        self.path = self.model_dir
        self.network = None  # No network since this method uses 4 random forest classifiers
        
        # Logging initialization details
        logger.info(f"Initialized ConfigTumorCandidateClassification with confidence threshold: {self.confidence_threshold}")
        
    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        """
        Create the inference task for tumor candidate classification.
        
        Returns:
            Union[InferTask, Dict[str, InferTask]]: An inference task or a dictionary of tasks.
        """
        # Logging inference configuration
        logger.debug(f"Creating InfererTumorCandidateClassification with confidence threshold {self.confidence_threshold}")
        
        # Check if model files exist
        if not all(os.path.exists(p) for p in self.model_path):
            raise FileNotFoundError(f"Missing model files at {', '.join(self.model_path)}")
        
        if not all(os.path.exists(p) for p in self.radiomic_feature_list_path):
            raise FileNotFoundError(f"Missing radiomic feature list file at {', '.join(self.radiomic_feature_list_path)}")
        
        if not os.path.exists(self.radiomic_extractor_config_path):
            raise FileNotFoundError(f"Missing radiomic extractor configuration file at {self.radiomic_extractor_config_path}")
        
        # Return an inference task with the set configurations
        return {
            self.name: InfererTumorCandidateClassification(
                path=self.path,
                network=self.network,
                type=InferType.SEGMENTATION,
                labels=self.labels,
                anatomical_labels=self.anatomical_labels,
                dimension=self.dimension,
                model_path=self.model_path,
                radiomic_extractor_config_path=self.radiomic_extractor_config_path,
                radiomic_feature_list_path=self.radiomic_feature_list_path,
                confidence_threshold=self.confidence_threshold,
                classification_threshold=self.classification_threshold,
                size_min_threshold=self.size_min_threshold,
                size_max_threshold=self.size_max_threshold,
                anatomical_regions=self.anatomical_regions,
                target_spacing=self.target_spacing,
                downsampled_spacing=self.downsampled_spacing
            )
        }

    def trainer(self) -> Optional[TrainTask]:
        """
        Since the task does not involve training, this function returns None.

        Returns:
            Optional[TrainTask]: None, as no training is required for tumor candidate classification.
        """
        logger.info("No training task required for tumor candidate classification.")
        return None
        