import logging
from typing import Optional, Sequence, Union

import nibabel as nib
from monai.config import KeysCollection
from monai.data import MetaTensor
from monai.transforms import MapTransform, Orientation
from monai.utils import InterpolateMode, ensure_tuple_rep

logger = logging.getLogger(__name__)


class ReorientToOriginal(MapTransform):
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

        return d
