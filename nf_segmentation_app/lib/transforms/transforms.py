import logging
from typing import Optional, Sequence, Union, Mapping, Hashable

import nibabel as nib
import numpy as np
import torch
from monai.config import KeysCollection
from monai.data import MetaTensor
from monai.transforms import MapTransform, Orientation, Transform
from monai.utils import InterpolateMode, ensure_tuple_rep, TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
from collections import OrderedDict
from scipy.ndimage import binary_dilation

logger = logging.getLogger(__name__)


class ReorientToOriginald(MapTransform):
    """
    A MONAI MapTransform that reorients an image back to its original orientation using metadata
    from a reference image. This is useful for restoring images that have been resampled or reoriented
    during preprocessing steps.

    Args:
        keys (KeysCollection): Keys of the items to be transformed.
        ref_image (str): The key for the reference image used to restore the original orientation.
        has_channel (bool): Whether the image has a channel dimension (default: True).
        invert_orient (bool): Whether to invert the orientation (default: False).
        mode (str): Interpolation mode for reorientation (default: 'nearest').
        config_labels (Optional[dict]): Optional dictionary to map config labels (default: None).
        align_corners (Optional[Union[Sequence[bool], bool]]): Alignment option for interpolation.
        meta_key_postfix (str): The postfix used for the metadata key (default: 'meta_dict').
    """
    
    def __init__(
        self,
        keys: KeysCollection,
        ref_image: str,
        has_channel: bool = True,
        invert_orient: bool = False,
        mode: str = InterpolateMode.NEAREST,
        config_labels=None,
        align_corners: Union[Sequence[Optional[bool]], Optional[bool]] = None,
        meta_key_postfix: str = "meta_dict",
    ):
        super().__init__(keys)
        self.ref_image = ref_image
        self.has_channel = has_channel
        self.invert_orient = invert_orient
        self.config_labels = config_labels
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.meta_key_postfix = meta_key_postfix

    def __call__(self, data):
        """
        Reorient the image to its original orientation using the affine transformation stored
        in the reference image's metadata.

        Args:
            data (dict): A dictionary containing the image and metadata.

        Returns:
            dict: The input dictionary with the reoriented image and updated metadata.
        """
        d = dict(data)

        # Extract the metadata from the reference image
        meta_dict = (
            d[self.ref_image].meta
            if d.get(self.ref_image) is not None and isinstance(d[self.ref_image], MetaTensor)
            else d.get(f"{self.ref_image}_{self.meta_key_postfix}", {})
        )

        # Loop through each key (image) to apply the inverse transformation
        for idx, key in enumerate(self.keys):
            result = d[key]
            
            # Retrieve the original affine matrix for the inverse transformation
            orig_affine = meta_dict.get("original_affine", None)
            if orig_affine is not None:
                orig_axcodes = nib.orientations.aff2axcodes(orig_affine)
                inverse_transform = Orientation(axcodes=orig_axcodes)

                # Apply inverse reorientation
                with inverse_transform.trace_transform(False):
                    result = inverse_transform(result)
            else:
                logger.info("Failed to invert orientation - 'original_affine' not found in image metadata.")

            d[key] = result

            # Update the metadata with the affine of the original image
            meta = d.get(f"{key}_{self.meta_key_postfix}")
            if meta is None:
                meta = dict()
                d[f"{key}_{self.meta_key_postfix}"] = meta
            meta["affine"] = meta_dict.get("original_affine")
            print("INSODE RESTORED ORIENTATION")
            print(meta)
            print(d[key].meta)
        return d


class AssembleAnatomyMask(Transform):
    """
    Assembles the anatomy segmentation mask by merging specific labels according to predefined rules.

    Args:
        has_channel (bool): Whether the input mask has a channel dimension (default: True).
        dilate_structure_size (int): Size of the dilation structure for binary dilation (default: 3).
        dilate_iter_spine (int): Number of iterations for dilating the spine region (default: 7).
        dilate_iter_lung (int): Number of iterations for dilating the lung region (default: 5).
        dimension (int): Dimensionality of the input data (default: 3).
    """
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]
    
    MRSEGMENTATOR_LABELS = {
        "background": 0, "spleen": 1, "right_kidney": 2, "left_kidney": 3, 
            "gallbladder": 4, "liver": 5, "stomach": 6, "pancreas": 7, 
            "right_adrenal_gland": 8, "left_adrenal_gland": 9, "left_lung": 10, 
            "right_lung": 11, "heart": 12, "aorta": 13, "inferior_vena_cava": 14, 
            "portal_vein_and_splenic_vein": 15, "left_iliac_artery": 16, 
            "right_iliac_artery": 17, "left_iliac_vena": 18, "right_iliac_vena": 19, 
            "esophagus": 20, "small_bowel": 21, "duodenum": 22, "colon": 23, 
            "urinary_bladder": 24, "spine": 25, "sacrum": 26, "left_hip": 27, 
            "right_hip": 28, "left_femur": 29, "right_femur": 30, 
            "left_autochthonous_muscle": 31, "right_autochthonous_muscle": 32, 
            "left_iliopsoas_muscle": 33, "right_iliopsoas_muscle": 34, 
            "left_gluteus_maximus": 35, "right_gluteus_maximus": 36, 
            "left_gluteus_medius": 37, "right_gluteus_medius": 38, 
            "left_gluteus_minimus": 39, "right_gluteus_minimus": 40
            }
    
    ASSEMBLE_RULES = OrderedDict({
        25: 11, # Spine
        (10, 11): 10, # Merge left and right lungs
        26: 9, # Sacrum
        tuple(range(31, 41)): 8, # Merge all available muscles
        (29, 30): 7, # Merge left and right femurs
        (27, 28): 6, # Merge left and right hips
        12: 5, # Heart
        (4, 5): 4, # Merge gallbladder and liver
        6: 3, # Stomach
        (2, 3): 2, # Merge left and right kidneys
        24: 1, # Urinary bladder
        })
    
    def __init__(self,  
                 has_channel: bool = True,
                 dilate_structure_size: int = 3,
                 dilate_iter_spine: int = 7,
                 dilate_iter_lung: int = 5,
                 dimension: int = 3
                 ) -> None:
        self.has_channel = has_channel
        self.dilate_structure = np.ones((dilate_structure_size,) * dimension)
        self.dilate_iter_spine = dilate_iter_spine
        self.dilate_iter_lung = dilate_iter_lung
    
    def __call__(self, raw_mask: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Assemble the anatomy segmentation mask by applying predefined merging rules
        and dilating regions for high-risk neurofibroma zones.

        Args:
            raw_mask (NdarrayOrTensor): Input raw mask to be processed.

        Returns:
            NdarrayOrTensor: The assembled and processed output mask.
        """
        if not isinstance(raw_mask, (np.ndarray, torch.Tensor)):
            raise NotImplementedError(f"{self.__class__} can not handle data of type {type(img)}.")
        
        if self.has_channel:
            raw_mask = raw_mask[0]  # Assuming the first channel is the desired mask
        
        output_mask = np.zeros_like(raw_mask)
        
        for key, value in self.ASSEMBLE_RULES.items():
            # Copy and merge labels from the raw mask to the output mask
            if isinstance(key, tuple): # Merge multiple labels into one
                for k in key:
                    output_mask[raw_mask == k] = value
            else: # Re-assign a single label
                output_mask[raw_mask == key] = value
            
            # Dilate specific regions for high-risk neurofibroma zones
            if value == 11: # Spine
                region = output_mask == 11
                dilated = binary_dilation(region, self.dilate_structure, iterations=self.dilate_iter_spine)
                output_mask[dilated & ~region] = 12 # Add a new label for high-risk meurofibroma zone
            if value == 10: # Lungs
                region = output_mask == 10
                dilated = binary_dilation(region, self.dilate_structure, iterations=self.dilate_iter_lung)
                output_mask[dilated & ~region] = 12 # Add a new label for high-risk meurofibroma zone
        
        if self.has_channel:
            output_mask = np.expand_dims(output_mask, axis=0)  # Add a new channel for the output mask
        return output_mask

class AssembleAnatomyMaskd(MapTransform):
    """
    MONAI MapTransform that assembles the anatomy mask based on predefined merging rules
    and applies postprocessing such as dilation for certain anatomical regions.

    Args:
        keys (KeysCollection): Keys of the items to be transformed.
        has_channel (bool): Whether the input mask has a channel dimension (default: True).
        dilate_structure_size (int): Size of the structure element for binary dilation (default: 3).
        dilate_iter_spine (int): Number of iterations for spine region dilation (default: 7).
        dilate_iter_lung (int): Number of iterations for lung region dilation (default: 5).
        dimension (int): Dimensionality of the input data (default: 3).
        allow_missing_keys (bool): Whether to allow missing keys (default: False).
    """
    def __init__(self, 
                 keys: KeysCollection, 
                 has_channel: bool = True,
                 dilate_structure_size: int = 3,
                 dilate_iter_spine: int = 7,
                 dilate_iter_lung: int = 5,
                 dimension: int = 3,
                 allow_missing_keys: bool = False,
                 ):
        super().__init__(keys, allow_missing_keys)
        self.assembler = AssembleAnatomyMask(has_channel,
                                             dilate_structure_size, 
                                             dilate_iter_spine, 
                                             dilate_iter_lung, 
                                             dimension)
    
    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        """
        Apply the transformation to assemble the anatomy mask for each key.

        Args:
            data (Mapping[Hashable, NdarrayOrTensor]): The input data containing the segmentation masks.

        Returns:
            dict[Hashable, NdarrayOrTensor]: The transformed data with assembled anatomy masks.
        """
        d = dict(data)
        for key in self.key_iterator(d):
            # Completely new numpy array that needs to be converted into MataTensor
            anatomy_segmentation_mask = torch.from_numpy(self.assembler(d[key])).type(torch.uint8)
            d[key] = MetaTensor(anatomy_segmentation_mask).copy_meta_from(d[key], copy_attr=False)
            print(key)
            print("ASSEMBLING - META DATA")
            print(d[key].meta)
        return d
