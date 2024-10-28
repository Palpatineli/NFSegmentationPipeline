import logging
import time
import copy
from typing import Callable, Sequence

from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.interfaces.tasks.infer_v2 import InferType, InferTask
from monailabel.utils.others.generic import name_to_device
from monailabel.transform.writer import Writer

# Initialize logger for the module
logger = logging.getLogger(__name__)


class InfererMultiStagePipeline(BasicInferTask):
    def __init__(
        self,
        task_anatomy_segmentation: InferTask,
        task_neurofibroma_segmentation: InferTask,
        task_thresholding: InferTask,
        type=InferType.SEGMENTATION,
        description="Combines anatomy segmentation, neurofibroma segmentation, and thresholding into a multi-stage pipeline",
        **kwargs,
    ):
        """
        Initialize the multi-stage pipeline combining anatomy segmentation, neurofibroma segmentation, and thresholding.

        Args:
            task_anatomy_segmentation (InferTask): Task for anatomy segmentation.
            task_neurofibroma_segmentation (InferTask): Task for neurofibroma segmentation.
            task_thresholding (InferTask): Task for thresholding neurofibroma segmentation probability map.
            type (InferType): Type of inference task (default is SEGMENTATION).
            description (str): Description of the pipeline.
            **kwargs: Additional keyword arguments.
        """
        self.task_anatomy_segmentation = task_anatomy_segmentation
        self.task_neurofibroma_segmentation = task_neurofibroma_segmentation
        self.task_thresholding = task_thresholding

        super().__init__(
            path=None,
            network=None,
            type=type,
            labels=task_neurofibroma_segmentation.labels,
            dimension=task_neurofibroma_segmentation.dimension,
            description=description,
            load_strict=False,
            **kwargs,
        )
    
    @property
    def required_inputs(self):
        """
        Define the required input keys for this pipeline.

        Returns:
            List[str]: A list of required input keys.
        """
        return ["image"]
    
    def pre_transforms(self, data=None) -> Sequence[Callable]:
        """
        No pre-transforms are needed in this pipeline.
        """
        return []
    
    def post_transforms(self, data=None) -> Sequence[Callable]:
        """
        No posttransforms are needed in this pipeline.
        """
        return []
    
    def is_valid(self) -> bool:
        return True
    
    def _latencies(self, result_meta, latencies=None):
        """
        Update and accumulate latencies from different stages of the pipeline.

        Args:
            result_meta (dict): Metadata containing latencies.
            latencies (dict): Latency values to update (optional).

        Returns:
            dict: Updated latency dictionary.
        """
        if latencies is None:
            latencies = {"pre": 0, "infer": 0, "invert": 0, "post": 0, "write": 0, "total": 0}

        for key in latencies:
            latencies[key] += result_meta.get("latencies", {}).get(key, 0)
        return latencies
    
    def segment_anatomy(self, request):
        """
        Execute the anatomy segmentation stage of the pipeline.

        Args:
            request (dict): Input request data.

        Returns:
            tuple: Anatomy segmentation result, metadata, and updated latencies.
        """
        req = copy.deepcopy(request)
        req.update({"pipeline_mode": True})
        data, meta = self.task_anatomy_segmentation(req)
        return data, meta, self._latencies(meta)
    
    def segment_neurofibroma(self, request, anatomy):
        """
        Execute the neurofibroma segmentation stage of the pipeline.

        Args:
            request (dict): Input request data.
            anatomy (ndarray): Anatomy segmentation output.

        Returns:
            tuple: Neurofibroma segmentation result, metadata, and updated latencies.
        """
        req = copy.deepcopy(request)
        req.update({"anatomy": anatomy, "pipeline_mode": True})
        data, meta = self.task_neurofibroma_segmentation(req)
        return data, meta, self._latencies(meta)
    
    def threshold_neurofibroma(self, request, proba):
        """
        Execute the thresholding task on the probability map.

        Args:
            request (dict): Input request data.
            proba (ndarray): Probability map from segmentation task.

        Returns:
            Tuple[dict, dict, dict]: Thresholding result, metadata, and latencies.
        """
        req = copy.deepcopy(request)
        req.update({"proba": proba, "pipeline_mode": True})
        data, meta = self.task_thresholding(req)
        return data, meta, self._latencies(meta)
    
    def __call__(self, request):
        """
        Execute the entire multi-stage pipeline.

        Args:
            request (dict): Input request data.

        Returns:
            tuple: Result files and corresponding metadata.
        """
        start = time.time()

        # Set the image path and device for the pipeline
        request.update({"image_path": request.get("image")})
        request["device"] = name_to_device(request.get("device", "cuda"))
        
        # Run anatomy segmentation task
        data_1, _, latency_1 = self.segment_anatomy(request)
        anatomy = data_1["anatomy"]
        anatomy_meta = data_1["anatomy_meta_dict"]
        
        # Run neurofibroma segmentation task
        data_2, _, latency_2 = self.segment_neurofibroma(request, anatomy)
        proba = data_2["proba"]
        proba_meta = data_2["proba_meta_dict"]
                                
        # Run thresholding task
        data_3, _, latency_3 = self.threshold_neurofibroma(request, proba)
        result_mask = data_3["pred"]
        
        # Prepare data for writing results
        data = copy.deepcopy(request)
        data.update({
            "final": result_mask,
            "result_extension": ".nii.gz",
            "result_meta_dict": proba_meta,
            
            "anatomy": anatomy,
            "anatomy_extension": ".nii.gz",
            
            "proba": proba * 255,  # Convert probabilities to 0-255 range for saving
            "proba_extension": ".nii.gz",
            "proba_dtype": "uint8",
        })
        
        # Write the results to files
        begin = time.time()
        result_file_pred, _ = Writer(
            label="final", ref_image="result", key_extension="result_extension")(data)
        result_file_proba, _ = Writer(
            label="proba", ref_image="result", key_extension="proba_extension", key_dtype="proba_dtype"
        )(data)
        result_file_anatomy, _ = Writer(
            label="anatomy", ref_image="result", key_extension="anatomy_extension",)(data)
        latency_write = round(time.time() - begin, 2)
        
        # Calculate total latency
        total_latency = round(time.time() - start, 2)

        # Create result metadata including latencies
        result_json = {
            "label_names": self.task_neurofibroma_segmentation.labels,
            "latencies": {
                "anatomy_neurofibroma": latency_1,
                "segment_neurofibroma": latency_2,
                "threshold_neurofibroma": latency_3,
                "write": latency_write,
                "total": total_latency,
            },
        }
        
        # Log the final result
        logger.info(f"Result Mask: {result_mask.shape}; total_latency: {total_latency}")

        return {"final": result_file_pred, "proba": result_file_proba, "anatomy": result_file_anatomy}, result_json
        