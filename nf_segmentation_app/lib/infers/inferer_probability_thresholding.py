import logging
import time
import copy
from typing import Callable, Dict, Sequence, Tuple, Union, Any

from monai.transforms import LoadImaged, AsDiscreted, Lambdad
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.interfaces.utils.transform import dump_data
from monailabel.transform.writer import Writer

# Initialize logger for this module
logger = logging.getLogger(__name__)


class InfererProbabilityThresholding(BasicInferTask):
    def __init__(
        self,
        path=None,
        network=None,
        type=InferType.SEGMENTATION,
        labels=None,
        dimension=3,
        threshold=0.5,
        description="Thresholding of the probability map",
        **kwargs
    ):
        """
        Initialization of the probability thresholding inference task.

        Args:
            path (str): Path to the model or resources.
            network (Any): The network used for inference (None in this case).
            type (InferType): The type of task (e.g., SEGMENTATION).
            labels (dict): A dictionary of label mappings (foreground/background).
            dimension (int): The dimension of the data (e.g., 3 for 3D segmentation).
            threshold (float): Threshold for binarizing the probability map.
            description (str): Description of the task.
            **kwargs: Additional configuration options.
        """
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            input_key="proba",  # Key used for probability map input
            output_label_key="proba",
            output_json_key="result",
            load_strict=False,
            **kwargs,
        )
        # Threshold for binarization
        self.threshold = threshold

    @property
    def required_inputs(self):
        """
        Return the required input keys for inference.
        """
        return ["proba"]

    def pre_transforms(self, data=None):
        """
        Reading data and preprocessing it 
        only if this inferer is used in a standalone mode.

        Args:
            data (dict): Input data dictionary.

        Returns:
            Sequence[Callable]: A sequence of preprocessing transforms.
        """
        if data and isinstance(data.get("proba"), str):
            t = [
                LoadImaged(keys="proba", reader="ITKReader"),  # Load image data
                Lambdad(keys="proba", func=lambda x: x / 255),  # Normalize probability map
            ]
        else:
            t = []
        return t
            
    def inverse_transforms(self, data=None) -> Union[None, Sequence[Callable]]:
        """
        Run all applicable pre-transforms which has inverse method.
        """
        return []

    def post_transforms(self, data=None) -> Sequence[Callable]:
        """
        Apply thresholding.

        Args:
            data (dict): Input data dictionary.

        Returns:
            Sequence[Callable]: A sequence of postprocessing transforms.
        """
        # Threshold the probability map to get binary segmentation
        return [
            AsDiscreted(keys="proba", threshold=self.threshold),
        ]

    def __call__(self, request) -> Union[Dict, Tuple[str, Dict[str, Any]]]:
        """
        Execute the inference task.

        Args:
            request (dict): The request payload for inference.

        Returns:
            Tuple[str, Dict]: The result file name and associated metadata.
        """
        begin = time.time()
        req = copy.deepcopy(self._config)  # Deep copy of the configuration
        req.update(request)  # Update with request parameters

        # Set logging level based on the request
        logger.setLevel(req.get("logging", "INFO").upper())

        # Handling image input path
        if req.get("image") and isinstance(req.get("image"), str):
            logger.info(f"Infer Request (final): {req}")
            data = copy.deepcopy(req)
            data.update({"image_path": req.get("image")})
        else:
            dump_data(req, logger.level)
            data = req

        # Pre-transforms
        start = time.time()
        pre_transforms = self.pre_transforms(data)
        data = self.run_pre_transforms(data, pre_transforms)
        latency_pre = time.time() - start

        # Post-transforms
        start = time.time()
        data = self.run_post_transforms(data, self.post_transforms(data))
        latency_post = time.time() - start

        # Return directly in pipeline mode
        if data.get("pipeline_mode", False):
            return {"pred": data["proba"]}, {}

        # Prepare final output metadata
        data.update({
            "final": data["proba"],
            "result_extension": ".nii.gz",  # Save result as NIfTI format
        })

        # Writing output
        start = time.time()
        result_file_name, result_json = Writer(
            label="final", ref_image="proba", key_extension="result_extension"
        )(data)
        latency_write = time.time() - start

        result_file_name_dict = {"final": result_file_name, "proba": None}

        # Total latency
        latency_total = time.time() - begin
        logger.info(
            f"++ Latencies => Total: {latency_total:.4f}; "
            f"Pre: {latency_pre:.4f}; Post: {latency_post:.4f}; Write: {latency_write:.4f}"
        )

        # Updating result JSON with label names and latencies
        result_json["label_names"] = self.labels
        result_json["latencies"] = {
            "pre": round(latency_pre, 2),
            "post": round(latency_post, 2),
            "write": round(latency_write, 2),
            "total": round(latency_total, 2),
            "transform": data.get("latencies"),
        }

        # Log the result file and metadata
        if result_file_name:
            logger.info(f"Result File: {result_file_name}")
        logger.info(f"Result Json Keys: {list(result_json.keys())}")
        
        return result_file_name_dict, result_json
