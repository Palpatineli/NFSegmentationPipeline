from monai.inferers import Inferer, SlidingWindowInferer
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    CropForegroundd,
    EnsureTyped,
    Activationsd,
    ToNumpyd,
    Lambdad,
)

from monailabel.tasks.infer.basic_infer import BasicInferTask, CallBackTypes
from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.transform.post import Restored

from typing import Callable, Dict, Sequence, Tuple, Union, Any
import logging

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
        sw_batch_size=8,
        description="3D Anisotropic U-Net for single-stage NF segmentation",
        **kwargs
    ):
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
        self.load_strict = False

    @property
    def required_inputs(self):
        return [
            "image",
        ]

    def pre_transforms(self, data=None):
        t = [
            LoadImaged(keys="image", reader="ITKReader"),
            EnsureTyped(keys="image", device=data.get("device") if data else None),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys="image", axcodes="ASR"),
            Spacingd(keys="image", pixdim=self.target_spacing, mode="bilinear"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CropForegroundd(
                keys="image", source_key="image", k_divisible=self.target_spacing
            ),
        ]
        self.add_cache_transform(t, data)
        t.append(EnsureTyped(keys="image", device=data.get("device") if data else None))
        return t

    def inferer(self, data=None) -> Inferer:
        return SlidingWindowInferer(
            roi_size=self.spatial_size,
            sw_batch_size=self.sw_batch_size,
            overlap=self.overlap,
        )

    def inverse_transforms(self, data=None) -> Union[None, Sequence[Callable]]:
        return []  # Self-determine from the list of pre-transforms provided

    def post_transforms(self, data=None) -> Sequence[Callable]:
        # Add transform to extract the 1st channel
        t = [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            Activationsd(keys="pred", softmax=True),
            Lambdad(keys="pred", func=lambda x: x[1]),
            ToNumpyd(keys="pred"),
            Restored(keys="pred", ref_image="image"),
        ]
        return t

    def writer(self, data: Dict[str, Any], extension=None, dtype=None) -> Tuple[Any]:
        return {"proba": data["pred"], "proba_meta_dict": data["pred_meta_dict"]}, {}
