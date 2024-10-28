import logging
from typing import Callable, Dict, Sequence, Tuple, Union, Any
import torch

from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.transform.post import Restored
from monai.inferers import Inferer, SlidingWindowInferer
from monai.data import decollate_batch
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    EnsureChannelFirst,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    EnsureTyped,
    Activations,
    ToNumpyd,
    Lambda,
    ScaleIntensityRanged,
    ConcatItemsd,
    MeanEnsemble
)
from lib.transforms.transforms import ReorientToOriginald

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
        sw_batch_size=4,
        number_anatomical_structures=12,
        description="3D Anisotropic Anatomy-Informed U-Net for multi-stage NF segmentation",
        **kwargs
    ):
        """
        Initialize the 3D Anisotropic Anatomy-Informed U-Net inference task.

        Args:
            path (str or list): Path to the model weights. If a list of paths is provided, 
                                ensemble predictions from different folds are used.
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
                                                 Defaults to 2, since the model uses an image and an antomy mask.
            sw_batch_size (int, optional): Batch size for sliding window inference. Defaults to 4.
            number_anatomical_structures (int, optional): Number of anatomical structures used in 
                                                          segmentation. Defaults to 12.
            description (str, optional): Description of the inference task.
            **kwargs: Additional keyword arguments for task configuration.

        Raises:
            ValueError: If `path` is not provided.
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
            input_key="image_joint",
            output_label_key="pred",
            output_json_key="result",
            load_strict=False,
            **kwargs,
        )

        # Assign class attributes for inference settings
        self.spatial_size = spatial_size
        self.target_spacing = target_spacing
        self.overlap = overlap
        self.number_intensity_ch = number_intensity_ch
        self.sw_batch_size = sw_batch_size
        self.number_anatomical_structures = number_anatomical_structures
        self.path_folds = path if isinstance(path, list) else [path, ]  # Path to folds directory
        
        
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
        Define the pre-processing transformations before inference.
        We assume that the anatomy data is provided as a separate input key.

        Args:
            data (dict): Input data dictionary.

        Returns:
            List[Callable]: A list of pre-processing transformations.
        """
        transforms = [
            LoadImaged(keys="image", reader="ITKReader"),
            EnsureTyped(keys=["image", "anatomy"], device=data.get("device") if data else None),
            EnsureChannelFirstd(keys=["image", "anatomy"]), 
            Orientationd(keys=["image", "anatomy"], axcodes="ASR"), 
            Spacingd(keys=["image", "anatomy"], pixdim=self.target_spacing, mode=["bilinear", "nearest"]),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ScaleIntensityRanged(keys="anatomy", a_min=0, 
                                 a_max=self.number_anatomical_structures, 
                                 b_min=0.0, b_max=1.0),
            ConcatItemsd(keys=["image", "anatomy"], name="image_joint", dim=0), 
        ]
        self.add_cache_transform(transforms, data)
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
            ReorientToOriginald(keys="pred", ref_image="image"), 
            ToNumpyd(keys="pred"), 
            Restored(keys="pred", ref_image="image"),
        ]
    
    def run_inferer(self, data: Dict[str, Any], convert_to_batch=True, device="cuda"):
        """
        Perform the inference using an ensemble of models and get the probability map.

        Args:
            data (dict): Input data dictionary.
            convert_to_batch (bool): Whether to convert the input to batch format.
            device (str): The device to perform inference on.

        Returns:
            dict: The data dictionary with added prediction results.
        """
        inferer = self.inferer(data)
        logger.info(f"Inferer:: {device} => {inferer.__class__.__name__} => {inferer.__dict__}")
        
        # Check if the required input key is present in the data
        if self.input_key not in data:
            raise KeyError(f"Required input key '{self.input_key}' is missing from the data.")
                
        # Get the input data
        inputs = data[self.input_key]
        if inputs is None:
            raise ValueError(f"Input for key '{self.input_key}' cannot be None.")
        
        inputs = inputs if torch.is_tensor(inputs) else torch.from_numpy(inputs)
        inputs = inputs[None] if convert_to_batch else inputs
        inputs = inputs.to(torch.device(device))
        
        # Logic for getting predictions from all folds of the model
        outputs_list = []
        for i, path_fold in enumerate(self.path_folds):
            self.path = [path_fold, ]
            network = self._get_network(device, data)
            
            with torch.no_grad():
                outputs = inferer(inputs, network)

            if device.startswith("cuda"):
                torch.cuda.empty_cache()
                
            if convert_to_batch:
                if isinstance(outputs, dict):
                    outputs_d = decollate_batch(outputs)
                    outputs = outputs_d[0]
                else:
                    outputs = outputs[0]
            
            # Apply Softmax activation and get "probability map" for tumors
            outputs = Activations(softmax=True)(outputs)
            outputs = Lambda(lambda x: x[1])(outputs)
            outputs_list.append(outputs)
            
            logger.info(f"Finished inference for fold: {i}")
            
        # Apply ensembling of predictions by averaging
        output_ensemble = MeanEnsemble()(outputs_list)
        output_ensemble = EnsureChannelFirst()(output_ensemble)
        data[self.output_label_key] = output_ensemble
        return data
            
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
        return {"proba": data["pred"], "proba_meta_dict": data["pred_meta_dict"]}, {}
