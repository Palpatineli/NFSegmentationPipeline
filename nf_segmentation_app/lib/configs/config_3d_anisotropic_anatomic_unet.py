import logging
import os
import torch
from typing import Any, Dict, Optional, Union
from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.utils.others.generic import strtobool
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from lib.infers.inferer_3d_anisotropic_anatomic_unet import Inferer3DAnisotropicAnatomicUnet

# Initialize logger for this module
logger = logging.getLogger(__name__)


class Config3DAnisotropicAnatomicUnet(TaskConfig):
    def init(
        self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs
    ):
        super().init(name, model_dir, conf, planner, **kwargs)
        
        # Epistemic uncertainty settings (disabled by default)
        self.epistemic_enabled = None
        self.epistemic_samples = None
        
        # Three folds of the model
        self.path = [os.path.join(self.model_dir, 
                                  f"single_model_pipeline/3d_anisotropic_unet_{i}.pth") for i in range(3)]
        
        # Define labels for segmentation (foreground and background)
        self.labels = {
            "foreground": 1,
            "background": 0,
        }
        
        # Set the dimensionality and key configuration parameters
        self.dimension = 3
        self.spatial_size = (10, 640, 256)
        self.target_spacing = (7.8, 0.625, 0.625)
        self.overlap = 0.25
        self.number_intensity_ch = 2
        self.sw_batch_size = 8
        self.number_anatomical_structures = 12
        
        # Initialize the 3D Anisotropic Anatomy-Informed UNet architecture
        self.network = PlainConvUNet(
            num_classes=2,
            input_channels=2,
            n_stages=7,
            features_per_stage=[32, 64, 128, 256, 320, 320, 320],
            conv_op=torch.nn.Conv3d,
            kernel_sizes=[
                (1, 3, 3), (1, 3, 3), (1, 3, 3),
                (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3),
            ],
            strides=[
                (1, 1, 1), (1, 2, 2), (1, 2, 2),
                (1, 2, 2), (2, 2, 2), (1, 2, 2), (1, 2, 2),
            ],
            n_conv_per_stage=2,
            n_conv_per_stage_decoder=2,
            conv_bias=True,
            norm_op=torch.nn.InstanceNorm3d,
            norm_op_kwargs={
                "eps": 1e-05,
                "momentum": 0.1,
                "affine": True,
                "track_running_stats": False,
            },
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=torch.nn.LeakyReLU,
            nonlin_kwargs={"negative_slope": 0.01, "inplace": True},
            deep_supervision=False,
            nonlin_first=False,
        )
    
    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        """
        Create and return an inference task using the 3D Anisotropic UNet.

        Returns:
            Union[InferTask, Dict[str, InferTask]]: The inference task for the model.
        """
        return {
            self.name: Inferer3DAnisotropicAnatomicUnet(
                path=self.path,
                network=self.network,
                labels=self.labels,
                preload=strtobool(self.conf.get("preload", "false")),
                dimension=self.dimension,
                spatial_size=self.spatial_size,
                target_spacing=self.target_spacing,
                overlap=self.overlap,
                number_intensity_ch=self.number_intensity_ch,
                sw_batch_size=self.sw_batch_size,
                number_anatomical_structures=self.number_anatomical_structures,
                type=InferType.SEGMENTATION,
            )
        }

    def trainer(self) -> Optional[TrainTask]:
        """
        Since training is not required for this configuration, it returns None.
        """
        return None
