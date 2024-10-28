import logging
import time
import copy
import torch
import numpy as np
from typing import Callable, Dict, Sequence, Tuple, Union, List, Any

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor 
from monailabel.interfaces.utils.transform import dump_data
from monai.data.meta_tensor import MetaTensor
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.transform.writer import Writer
from monailabel.transform.post import Restored
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd
)
from lib.transforms.transforms import ReorientToOriginald, AssembleAnatomyMaskd

# Initialize logger for this module
logger = logging.getLogger(__name__)


class InfererMRSegmentator(BasicInferTask):
    """
    Inference class for MRSegmentator using 
    """
    
    def __init__(
        self,
        path: str,
        checkpoint_file_name: str,
        network: nnUNetPredictor = None,
        type: InferType = InferType.SEGMENTATION,
        labels: Dict[str, int] = None,
        dimension: int = 3,
        target_spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5),
        orientation: str = "SPL",
        folds: Union[List[int], Tuple[int, ...]] = [0, 1, 2, 3, 4],
        dilate_structure_size: int = 3, 
        dilate_iter_spine: int = 5,
        dilate_iter_lung: int = 7,
        description: str = "Anatomy segmentation using MRSegmentator",
        **kwargs
    ):
        """
        Initialize the inference task for MRSegmentator with nnUNetPredictor for multi-organ segmentation.

        Args:
            path (str): Path to the model weights.
            network (nnUNetPredictor): nnUNetPredictor instance for inference.
            type (InferType): Task type, default is SEGMENTATION.
            labels (Dict[str, int]): Dictionary mapping labels.
            dimension (int): Data dimensionality.
            target_spacing (Tuple[float]): Target voxel spacing for the model.
            orientation (str): Reorientation axcodes (default: "SPL").
            folds (List[int]): Cross-validation folds for nnUNet.
            dilate_structure_size (int): Size for dilating structures in postprocessing.
            dilate_iter_spine (int): Number of iterations for dilating spine structures.
            dilate_iter_lung (int): Number of iterations for dilating lung structures.
            description (str): Task description.
            **kwargs: Additional parameters for the inference task.
        """
        if path is None:
            raise ValueError("Model path cannot be None")
        
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            input_key="image",
            output_label_key="pred",
            output_json_key="result",
            load_strict=False,
            **kwargs,
        )
        self.checkpoint_file_name = checkpoint_file_name
        self.target_spacing = target_spacing
        self.orientation = orientation
        self.folds = folds
        self.dilate_structure_size = dilate_structure_size
        self.dilate_iter_spine = dilate_iter_spine
        self.dilate_iter_lung = dilate_iter_lung
    
    @property
    def required_inputs(self):
        """
        Define the required input keys for this inference task.

        Returns:
            List[str]: A list of required input keys.
        """
        return ["image"]
    
    def pre_transforms(self, data=None) -> Sequence[Callable]:
        """
        Define the preprocessing transformations for the input data.

        Args:
            data (dict): Input data dictionary.

        Returns:
            Sequence[Callable]: A list of preprocessing transformations.
        """
        transforms = [
            LoadImaged(keys="image", meta_keys="meta_data_image", reader="ITKReader"),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys="image", axcodes=self.orientation),
            Spacingd(keys="image", pixdim=self.target_spacing, mode="bilinear"),
        ]
        return transforms

    @staticmethod
    def get_meta_from_affine(affine_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract spacing, origin, and direction from affine matrix.

        Args:
            affine_matrix (np.ndarray): Affine matrix of the image.

        Returns:
            Tuple: Spacing, origin, and direction information.
        """
        direction_matrix = affine_matrix[:3, :3]
        spacing = np.linalg.norm(direction_matrix, axis=0)
        origin = affine_matrix[:3, 3]
        direction = (direction_matrix / spacing).flatten()
        return spacing, origin, direction

    def run_inferer(self, data: Dict[str, Any], device: str = "cuda"):
        """
        Run the segmentation inference using nnUNet.

        Args:
            data (dict): Input data containing image information.
            device (str): Device on which the inference should run (default: "cuda").

        Returns:
            dict: The input data dictionary updated with predictions.
        """
        # Initialize nnUNetPredictor from model folder
        self.network.initialize_from_trained_model_folder(
            self.path,
            use_folds=self.folds,
            checkpoint_name=self.checkpoint_file_name
        )
        
        if self.network is None:
            raise RuntimeError("Network is not created, probably due to the non-existing checkpoint files.")
        if self.input_key not in data:
            raise KeyError(f"Required input key '{self.input_key}' is missing from the data.")
        
        # Extract image data and affine matrix
        image_data = data[self.input_key].numpy().astype(np.float32)
        affine_matrix = data[self.input_key].affine
        spacing, origin, direction = self.get_meta_from_affine(affine_matrix)
        
        # Prepare meta information for segmentation
        meta_dict = {
            "sitk_stuff": {
                "spacing": spacing,
                "origin": origin,
                "direction": direction,
                "orientation": self.orientation,
            },
            "spacing": list(spacing)[::-1],
        }
        
        # Perform segmentation
        segmentations = self.network.predict_from_list_of_npy_arrays(
            image_or_list_of_images=image_data,
            segs_from_prev_stage_or_list_of_segs_from_prev_stage=None,
            properties_or_list_of_properties=meta_dict,
            truncated_ofname=None,
            save_probabilities=False,
        )
        segmentations = torch.from_numpy(np.expand_dims(segmentations[0], axis=0)).type(torch.uint8)

        # Copy metadata from input to segmentation if MetaTensor
        if isinstance(data["image"], MetaTensor):
            segmentations = MetaTensor(segmentations).copy_meta_from(data["image"], copy_attr=False)
        
        data[self.output_label_key] = segmentations
        return data
    
    def inverse_transforms(self, data=None) -> Union[None, Sequence[Callable]]:
        """
        No inverse transforms required for this task.
        """
        return None
    
    def post_transforms(self, data=None) -> Sequence[Callable]:
        """
        Define the postprocessing transformations after inference.

        Args:
            data (dict): Input data dictionary.

        Returns:
            Sequence[Callable]: A list of postprocessing transformations.
        """
        return [
            AssembleAnatomyMaskd(keys="pred", 
                                 dilate_structure_size=self.dilate_structure_size, 
                                 dilate_iter_spine=self.dilate_iter_spine,
                                 dilate_iter_lung=self.dilate_iter_lung
                                 ),
            ReorientToOriginald(keys="pred", ref_image="image"),
            Restored(keys="pred", ref_image="image"),
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
        
        # Inference
        start = time.time()
        data = self.run_inferer(data)
        latency_inferer = time.time() - start
        
        # Inverse transforms
        start = time.time()
        data = self.run_invert_transforms(data, pre_transforms, self.inverse_transforms(data))
        latency_invert = time.time() - start
                
        # Post-transforms
        start = time.time()
        data = self.run_post_transforms(data, self.post_transforms(data))
        latency_post = time.time() - start
        
        # Update the affine matrix of the output data
        data["pred"].meta["affine"] = data["pred"].meta["original_affine"]
                
        # Return directly in pipeline mode 
        if data.get("pipeline_mode", False):
            return {"anatomy": data["pred"], "anatomy_meta_dict": data["pred_meta_dict"]}, {}
        
        
        # Otherwise return the anatomy segmentation to 3D Slicer and visualize it
        # Prepare final output metadata
        data.update({
            "final": data["pred"],
            "result_extension": ".nii.gz",  # Save result as NIfTI format
        })

        # Writing output
        start = time.time()
        result_file_name, result_json = Writer(
            label="final", ref_image="pred", key_extension="result_extension"
        )(data)
        latency_write = time.time() - start

        result_file_name_dict = {"final": None, "pred": None, "anatomy": result_file_name}
        latency_total = time.time() - begin
        
        logger.info(
            "++ Latencies => Total: {:.4f}; "
            "Pre: {:.4f}; Inferer: {:.4f}; Invert: {:.4f}; Post: {:.4f}; Write: {:.4f}".format(
                latency_total,
                latency_pre,
                latency_inferer,
                latency_invert,
                latency_post,
                latency_write,
            )
        )

        result_json["label_names"] = self.labels
        result_json["latencies"] = {
            "pre": round(latency_pre, 2),
            "infer": round(latency_inferer, 2),
            "invert": round(latency_invert, 2),
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
