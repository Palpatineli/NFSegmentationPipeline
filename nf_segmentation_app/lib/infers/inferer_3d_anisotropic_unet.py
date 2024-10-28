import logging
from typing import Callable, Dict, Sequence, Tuple, Union, Any

from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.transform.post import Restored
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
)

# Initialize logger for this module
logger = logging.getLogger(__name__)


class Inferer3DAnisotropicUnet(BasicInferTask):
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
        number_intensity_ch=1,
        sw_batch_size=4,
        description="3D Anisotropic U-Net for single-stage NF segmentation",
        **kwargs
    ):
        """
        Initialization of the 3D Anisotropic U-Net Inference Task.

        Args:
            path (str): Path to model or resources.
            network (torch.nn.Module, optional): The neural network model to use for inference. 
                                                 Defaults to None if loaded from the path.
            type (InferType, optional): The type of inference task, e.g., SEGMENTATION. 
                                        Defaults to InferType.SEGMENTATION.
            labels (dict, optional): Dictionary of label names and corresponding values. 
                                     Defaults to None.
            dimension (int, optional): The dimension of the input data (e.g., 2D or 3D). 
                                       Defaults to 3 (3D).
            spatial_size (tuple, optional): The spatial size of the sliding window for inference. 
                                            Defaults to (10, 640, 256).
            target_spacing (tuple, optional): The target voxel spacing for resampling the input data. 
                                              Defaults to (7.8, 0.625, 0.625).
            overlap (float, optional): Overlap percentage for the sliding window inference. 
                                       Defaults to 0.25.
            number_intensity_ch (int, optional): Number of intensity channels in the input data. 
                                                 Defaults to 1.
            sw_batch_size (int, optional): Batch size for sliding window inference. Defaults to 4.
            description (str): Description of the inference task.
            **kwargs: Additional arguments for task configuration.
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
        self.spatial_size = spatial_size
        self.target_spacing = target_spacing
        self.overlap = overlap
        self.number_intensity_ch = number_intensity_ch
        self.sw_batch_size = sw_batch_size

    @property
    def required_inputs(self):
        """
        Define the required input keys for this inference task.

        Returns:
            List[str]: A list of required input keys.
        """
        return ["image"]

    def pre_transforms(self, data=None):
        """
        Define the preprocessing transformations for the input data.

        Args:
            data (dict): Input data dictionary.

        Returns:
            Sequence[Callable]: A list of preprocessing transformations.
        """
        transforms = [
            LoadImaged(keys="image", reader="ITKReader"),
            EnsureTyped(keys="image", device=data.get("device") if data else None),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys="image", axcodes="ASR"),
            Spacingd(keys="image", pixdim=self.target_spacing, mode="bilinear"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
        self.add_cache_transform(transforms, data)
        transforms.append(EnsureTyped(keys="image", device=data.get("device") if data else None))
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
        return [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            Activationsd(keys="pred", softmax=True),
            Lambdad(keys="pred", func=lambda x: x[1]),
            ToNumpyd(keys="pred"),
            Restored(keys="pred", ref_image="image"),
        ]

    def writer(self, data: Dict[str, Any], extension=None, dtype=None) -> Tuple[Any]:
        """
        Modified writer method to fit the interface of the multi-stage segmentation pipeline.
        Returns the prediction and its metadata in a dictionary.

        Args:
            data (dict): Output data dictionary.
            extension (str): File extension for saving the result (optional).
            dtype (Any): Data type for saving (optional).

        Returns:
            Tuple[Any]: The output data and metadata for probability map.
        """
        # Return the prediction and its metadata in a dictionary
        return {"proba": data["pred"], "proba_meta_dict": data["pred_meta_dict"]}, {}
