import logging
import os
import copy
from typing import Dict

import monailabel
from monailabel.interfaces.app import MONAILabelApp
from monailabel.utils.others.class_utils import get_class_names
from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.exception import MONAILabelError, MONAILabelException
from monailabel.utils.others.generic import handle_torch_linalg_multithread
from monailabel.interfaces.datastore import Datastore, DefaultLabelTag
from monailabel.config import settings

import lib.configs
from lib.infers.inferer_single_stage_pipeline import InfererSingleStagePipeline
from lib.infers.inferer_multi_stage_pipeline import InfererMultiStagePipeline

logger = logging.getLogger(__name__)


class NFSegmentationApp(MONAILabelApp):
    """
    A MONAILabel application for neurofibroma segmentation using a combination of models 
    for segmentation and thresholding with a single-stage pipeline approach.
    """
    def __init__(self, app_dir: str, studies: str, conf: Dict[str, str]):
        """
        Initialize the NFSegmentationApp with the model directory, studies, and configuration.

        Args:
            app_dir (str): The application directory path.
            studies (str): Path to the directory containing studies.
            conf (Dict[str, str]): The configuration dictionary.
        """
        self.model_dir = os.path.join(app_dir, "model")
        
        # Get the configuration of the batch size used for inference
        self.batch_size = int(conf.get("batch_size", "4"))
        self.resample_only_in_2d = conf.get("resample_only_in_2d", "False").lower() == "true"

        # Get all available model configurations
        configs = {}
        for c in get_class_names(lib.configs, "TaskConfig"):
            name = c.split(".")[-2].lower()
            configs[name] = c

        configs = {k: v for k, v in sorted(configs.items())}
        self.planner = None  
        self.models: Dict[str, TaskConfig] = {}
        # Predefined thresholds for the thresholding model
        thresholds = {"low": 0.25, "medium": 0.5, "high": 0.75}
        

        # Initialize models and configure them with thresholds
        for k, v in configs.items():
            if k == "config_probability_thresholding":
                for name, threshold in thresholds.items():
                    k_updated = f"{k}_{name}"
                    logger.info(f"+++ Adding Model: {k_updated} => {v}")
                    self.models[k_updated] = eval(f"{v}()")
                    self.models[k_updated].init(
                        name=k_updated, model_dir=self.model_dir, conf=conf, 
                        planner=self.planner, threshold=threshold
                    )
            else:
                logger.info(f"+++ Adding Model: {k} => {v}")
                self.models[k] = eval(f"{v}()")
                self.models[k].init(k, self.model_dir, conf, self.planner, 
                                    batch_size=self.batch_size, 
                                    resample_only_in_2d=self.resample_only_in_2d)
        
        logger.info(f"+++ Using Models: {list(self.models.keys())}")

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name=f"MONAILabel - NF Segmentation ({monailabel.__version__})",
            description="Pipeline for performing neurofibroma segmentation on T2-weighted WB-MRI scans",
            version=monailabel.__version__,
        )
        
    def init_datastore(self) -> Datastore:
        datastore = super().init_datastore()
        return datastore

    def init_infers(self) -> Dict[str, InferTask]:
        """
        Initialize and return a dictionary of inference tasks.

        Returns:
            Dict[str, InferTask]: A dictionary containing all initialized inference tasks.
        """
        components: Dict[str, InferTask] = {}
        infers: Dict[str, InferTask] = {}

        # Initialize inferers for each model configuration
        for n, task_config in self.models.items():
            c = task_config.infer()
            c = c if isinstance(c, dict) else {n: c}
            for k, v in c.items():
                logger.info(f"+++ Adding Components: {k} => {v}")
                components[k] = v
        
        # Initialize pipelines based on existing inferers
        infers.update(
            {
                "Single-Stage_NF_Segmentation": InfererSingleStagePipeline(
                    task_segmentation=components["config_3d_anisotropic_unet"],
                    task_thresholding=components["config_probability_thresholding_medium"],
                    description="Single-Stage Pipeline for neurofibroma segmentation",
                ),
                "Multi-Stage_NF_Segmentation_(with_Anatomy)": InfererMultiStagePipeline(
                    task_anatomy_segmentation=components["config_mrsegmentator"],
                    task_neurofibroma_segmentation=components["config_3d_anisotropic_anatomic_unet"],
                    task_thresholding=components["config_probability_thresholding_medium"],
                    description="Multi-Stage Pipeline for neurofibroma segmentation",    
                ),
                
                "Post-Processing:_Low_Confidence_Filter_(0.25)": components["config_probability_thresholding_low"],
                "Post-Processing:_Medium_Confidence_Filter_(0.50)": components["config_probability_thresholding_medium"],
                "Post-Processing:_High_Confidence_Filter_(0.75)": components["config_probability_thresholding_high"],
                
                "Post-Processing:_Tumor_Candidate_Classification_(Needs_Anatomy)": components["config_tumor_candidate_classification"],
                
                "Anatomy_Segmentation": components["config_mrsegmentator"],
                
            }
        )
        
        logger.info(infers)
        return infers

    @staticmethod
    def get_file_path(datastore, image_id, tag):
        """
        Get the file path for a specific label or tag.
        """
        output_id = datastore.get_label_by_image_id(image_id, tag)
        output_path = datastore.get_label_uri(output_id, label_tag=tag)
        return output_path

    @staticmethod
    def save_output(datastore, image_id, model, result_json, result_file_name, tag):
        """
        Save the output result of the inference.

        Args:
            datastore (Datastore): The datastore object.
            image_id (str): The ID of the image.
            model (str): The model used for inference.
            result_json (dict): The result metadata.
            result_file_name (str): The path to the result file.
            tag (str): The tag to be associated with the result.

        Returns:
            str: The ID of the saved label.
        """
        output_id = None
        if result_file_name and os.path.exists(result_file_name):
            output_id = datastore.save_label(
                image_id, result_file_name, tag, {"model": model, "params": result_json}
            )
        else:
            raise MONAILabelException(
                MONAILabelError.INFERENCE_ERROR,
                "No output file provided for saving the result",
            )
        return output_id

    def infer(self, request, datastore=None):
        """
        Perform the inference task based on the request.

        Args:
            request (dict): The inference request containing model and image information.
            datastore (Datastore, optional): The datastore object.

        Returns:
            dict: The inference result including label, tag, and file information.
        """
        model = request.get("model")
        if not model:
            raise MONAILabelException(
                MONAILabelError.INVALID_INPUT,
                "Model is not provided for Inference Task",
            )

        task = self._infers.get(model)
        if not task:
            raise MONAILabelException(
                MONAILabelError.INVALID_INPUT,
                f"Inference Task is not Initialized. There is no model '{model}' available",
            )
        
        # Prepare the request and fetch the required inputs
        request = copy.deepcopy(request)
        request["description"] = task.description
        image_id = request["image"]
        
        if isinstance(image_id, str):
            datastore = datastore or self.datastore()
            if os.path.exists(image_id):
                request["save_label"] = False
            else:
                request["image"] = datastore.get_image_uri(request["image"])

            if os.path.isdir(request["image"]):
                logger.info("Input is a Directory; Consider it as DICOM")

            logger.debug(f"Image => {request['image']}")
        else:
            request["save_label"] = False

        # Ensure all required inputs are available
        request["label"] = self.get_file_path(datastore, image_id, tag="final")
        request["proba"] = self.get_file_path(datastore, image_id, tag="proba")
        request["anatomy"] = self.get_file_path(datastore, image_id, tag="anatomy")

        for required_input in task.required_inputs:
            if request.get(required_input, "") == "":
                raise MONAILabelException(
                    MONAILabelError.INVALID_INPUT, 
                    f"Missing required input '{required_input}' for Inference Task."
                )
        
        # Run inference task using thread pool or directly
        if self._infers_threadpool:

            def run_infer_in_thread(t, r):
                handle_torch_linalg_multithread(r)
                return t(r)

            f = self._infers_threadpool.submit(run_infer_in_thread, t=task, r=request)
            result_file_name_dict, result_json = f.result(
                request.get("timeout", settings.MONAI_LABEL_INFER_TIMEOUT)
            )
        else:
            result_file_name_dict, result_json = task(request)

        # Save the results and return the output
        label_id, output_file_name = None, None
        output_id_dict = {}
        for name, result_file_name in result_file_name_dict.items():
            if result_file_name:
                output_id = self.save_output(datastore, image_id, model, result_json, result_file_name, name)
                output_id_dict[name] = output_id
                if name == "final":
                    label_id = output_id
                    output_file_name = result_file_name
        
        # If no final segmentation file found, return anatomies if available
        if (label_id is None) and (output_file_name is None) and ("anatomy" in result_file_name_dict.keys()):
            label_id = output_id_dict["anatomy"]
            output_file_name = result_file_name_dict["anatomy"]

        return {
            "label": label_id,
            "tag": DefaultLabelTag.ORIGINAL,
            "file": output_file_name,
            "params": result_json,
        }
