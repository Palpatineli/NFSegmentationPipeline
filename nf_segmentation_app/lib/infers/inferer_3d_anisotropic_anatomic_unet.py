import logging
from typing import Callable, Dict, Sequence, Tuple, Union, Any

from monai.inferers import Inferer, SlidingWindowInferer
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    EnsureTyped,
    Activationsd,
    ToNumpyd,
    Lambdad,
    ScaleIntensityRanged,
    ConcatItemsd
)
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.transform.post import Restored

# Initialize logger for this module
logger = logging.getLogger(__name__)


class Inferer3DAnisotropicAnatomicUnet(BasicInferTask):
    def __init__(
        self,
        path,
        network=None,
        type=InferType.SEGMENTATION,
        labels=None,
        dimension=3,
        spatial_size=(10, 640, 256),
        target_spacing=(7.8, 0.625, 0.625),
        overlap=0.25,
        number_intensity_ch=2,
        sw_batch_size=8,
        number_anatomical_structures=12,
        description="3D Anisotropic Anatomy-Informed U-Net for multi-stage NF segmentation",
        **kwargs
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            input_key="image",  # Key for input image data
            output_label_key="pred",  # Key for prediction output
            output_json_key="result",  # Key for JSON result output
            load_strict=False,  # Do not enforce strict loading
            **kwargs,
        )

        # Assign class attributes for inference settings
        self.spatial_size = spatial_size
        self.target_spacing = target_spacing
        self.overlap = overlap
        self.number_intensity_ch = number_intensity_ch
        self.sw_batch_size = sw_batch_size
        self.number_anatomical_structures = number_anatomical_structures
        
        
    @property
    def required_inputs(self):
        """
        Define the required input keys for this inference task.

        Returns:
            List[str]: A list of required input keys.
        """
        return ["image", "anatomy"]
    
    def pre_transforms(self, data=None):
        """
        We assume that the anatomy data is provided as a separate input key
        """
        transforms = [
            LoadImaged(keys="image", reader="ITKReader"),  # Load the image data
            EnsureTyped(keys=["image", "anatomy"], device=data.get("device") if data else None),
            EnsureChannelFirstd(keys=["image", "anatomy"]),  # Ensure the channels are first
            Orientationd(keys=["image", "anatomy"], axcodes="ASR"),  # TODO: ASR or ARS???
            Spacingd(keys=["image", "anatomy"], pixdim=self.target_spacing, mode=["bilinear", "nearest"]),  # Resample with target spacing
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),  # Normalize intensities
            ScaleIntensityRanged(keys="anatomy", 
                                 a_min=0, 
                                 a_max=self.number_anatomical_structures, 
                                 b_min=0.0, 
                                 b_max=1.0),
            ConcatItemsd(keys=["image", "anatomy"], 
                         name="image_joint",
                         axis=0),  # Concatenate the image and anatomy data along the channel axis
        ]
        # Cache the transforms if caching is enabled
        self.add_cache_transform(transforms, data)

        # Ensure data is moved to the correct device if applicable
        transforms.append(EnsureTyped(keys="image_joint", device=data.get("device") if data else None))
        return transforms
    
    def inferer(self, data=None) -> Inferer:
        """
        Define the inference method using sliding window inference.

        Args:
            data (dict): Input data dictionary.

        Returns:
            Inferer: SlidingWindowInferer object with configured parameters.
        """
        return SlidingWindowInferer(
            roi_size=self.spatial_size,
            sw_batch_size=self.sw_batch_size,
            overlap=self.overlap,
        )
    
    def inverse_transforms(self, data=None) -> Union[None, Sequence[Callable]]:
        """
        Run all applicable pre-transforms which has inverse method.
        """
        return []
    
    def post_transforms(self, data=None) -> Sequence[Callable]:
        """
        Define the postprocessing transformations after inference.

        Args:
            data (dict): Input data dictionary.

        Returns:
            Sequence[Callable]: A list of postprocessing transformations.
        """
        # Postprocessing transformations including softmax activation and restoring the original image orientation
        return [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            Activationsd(keys="pred", softmax=True),  # Apply softmax activation to the prediction
            Lambdad(keys="pred", func=lambda x: x[1]),  # Extract the first channel (foreground)
            ToNumpyd(keys="pred"),  # Convert the prediction to a NumPy array
            Restored(keys="pred", ref_image="image"),  # Restore the spatial orientation
        ]
    
    def run_inferer(self, data: Dict[str, Any], convert_to_batch=True, device="cuda"):
        pass
    
    def __call__(self, request) -> Union[Dict, Tuple[str, Dict[str, Any]]]:
        pass
    
    def writer(self, data: Dict[str, Any], extension=None, dtype=None) -> Tuple[Any]:
        """
        Write the output data into the required format.

        Args:
            data (dict): Output data dictionary.
            extension (str): File extension for saving the result (optional).
            dtype (Any): Data type for saving (optional).

        Returns:
            Tuple[Any]: The output data and metadata for probability map.
        """
        # Return the prediction and its metadata in a dictionary
        return {"proba": data["pred"], "proba_meta_dict": data["pred_meta_dict"]}, {}
    
    