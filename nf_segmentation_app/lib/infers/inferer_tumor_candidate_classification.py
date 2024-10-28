import logging
from typing import Callable, Dict, Sequence, Tuple, Union, Any
from scipy.ndimage import label as perform_connected_components_analysis, center_of_mass, zoom
from radiomics import featureextractor
import concurrent.futures
import SimpleITK as sitk
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
import copy
import torch
import joblib

from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.interfaces.utils.transform import dump_data
from monailabel.transform.writer import Writer
from monailabel.transform.post import Restored
from monai.data import MetaTensor
from monai.transforms import (
    LoadImaged,
    AsDiscreted,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
    ToNumpyd,
    Lambdad
)
from lib.transforms.transforms import ReorientToOriginald

# Initialize logger for this module
logger = logging.getLogger(__name__)
# Disable intermediate radiomics notifications
logging.getLogger('radiomics').setLevel(logging.ERROR)


class InfererTumorCandidateClassification(BasicInferTask):
    def __init__(
        self,
        path=None,
        network=None,
        type=InferType.SEGMENTATION,
        labels=None,
        anatomical_labels=None,
        dimension=3,
        model_path=None,
        radiomic_extractor_config_path=None,
        radiomic_feature_list_path=None,
        confidence_threshold=0.5,
        classification_threshold=0.5,
        size_min_threshold=10,
        size_max_threshold=500000,
        anatomical_regions=["head", "chest", "abdomen", "legs"],
        target_spacing=(0.625, 0.625, 7.8),
        downsampled_spacing=(1.5, 1.5, 7.8),
        description="Tumor Candidate Classification based on radiomics",
        **kwargs
    ):
        """
        Inference task for tumor candidate classification based on radiomics.        
        This class uses connected components analysis, radiomic feature extraction,
        and a random forest to identify tumor candidates.
        
        Minimum size thresholds prevents radiomic feature extractor from crushing due to small imuput objects.
        Maximum size thresholds prevents too long computation time of radiomic features. 
        We assume that all tumor candidates smaller than the minimum size are noise and should be ignored.
        We also assume that all tumor candidates larger than the maximum size are correctly detected tumors.
        Everything in between is subjected to classification.

        Args:
            path (str): Path to the model directory.
            network (Any): Network used for inference (not required in this task).
            type (InferType): The type of task (SEGMENTATION).
            labels (dict): Dictionary mapping of tumor labels.
            anatomical_labels (dict): Dictionary mapping of anatomical region labels.
            dimension (int): The dimension of the data (default: 3).
            model_path (list): Paths to the models for each anatomical region.
            radiomic_extractor_config_path (str): Path to the radiomic feature extractor config.
            radiomic_feature_list_path (list): Paths to the radiomic feature lists for each anatomical region.
            confidence_threshold (float): Threshold for binary tumor segmentation (default: 0.5).
            classification_threshold (float): Threshold for tumor candidate classification (default: 0.5).
            size_min_threshold (int): Minimum size threshold for tumor candidates (default: 10 pixels).
            size_max_threshold (int): Maximum size threshold for tumor candidates (default: 500000 pixels).
            anatomical_regions (list): List of anatomical regions to classify tumor candidates (default: head, chest, abdomen, legs).
            target_spacing (tuple): Target spacing for image resampling (default: (0.625, 0.625, 7.8)).
            downsampled_spacing (tuple): Downsampled spacing for image resampling (default: (1.5, 1.5, 7.8)).
            description (str): Description of the inference task.
        """
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            input_key="proba",  # Key used for probability map input
            output_label_key="pred",
            output_json_key="result",
            load_strict=False,
            **kwargs,
        )
        
        if model_path is None or radiomic_extractor_config_path is None or radiomic_feature_list_path is None:
            raise ValueError("model_path, radiomic_extractor_config_path, and radiomic_feature_list_path cannot be None")

        self.anatomical_labels = anatomical_labels
        self.model_path = model_path
        self.radiomic_extractor_config_path = radiomic_extractor_config_path
        self.radiomic_feature_list_path = radiomic_feature_list_path
        self.confidence_threshold = confidence_threshold
        self.classification_threshold = classification_threshold
        self.size_min_threshold = size_min_threshold
        self.size_max_threshold = size_max_threshold
        self.anatomical_regions = anatomical_regions
        self.target_spacing = target_spacing
        self.downsampled_spacing = downsampled_spacing
        
    @property
    def required_inputs(self):
        """
        Define the required input keys for this inference task.

        Returns:
            List[str]: A list of required input keys.
        """
        return ["image", "anatomy", "proba"]
    
    def pre_transforms(self, data=None):
        """
        Define the preprocessing transformations.

        Args:
            data (dict): Input data dictionary.

        Returns:
            List[Callable]: Preprocessing transformations.
        """
        transforms = [
            LoadImaged(keys=["image", "anatomy", "proba"], reader="ITKReader"),
            EnsureChannelFirstd(keys=["image", "anatomy", "proba"]),
            Orientationd(keys=["image", "anatomy", "proba"], axcodes="RSA"),
            Lambdad(keys="proba", func=lambda x: x / 255),
            Spacingd(keys=["image", "anatomy", "proba"], 
                     pixdim=self.target_spacing, 
                     mode=["bilinear", "nearest", "bilinear"]),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            AsDiscreted(keys="proba", threshold=self.confidence_threshold),
        ]
        self.add_cache_transform(transforms, data)
        return transforms
    
    def inverse_transforms(self, data=None) -> Union[None, Sequence[Callable]]:
        return None
    
    def post_transforms(self, data=None) -> Sequence[Callable]:
        """
        Define the postprocessing transformations.

        Args:
            data (dict): Input data dictionary.

        Returns:
            List[Callable]: Postprocessing transformations.
        """
        return [
            ReorientToOriginald(keys="pred", ref_image="image"),
            ToNumpyd(keys="pred"),
            Restored(keys="pred", ref_image="image"),
        ]
    
    @staticmethod
    def _find_first_dict_in_list(lst):
        for item in lst:
            if isinstance(item, dict):
                return item
        return None  # Return None if no dictionary is found
    
    @staticmethod
    def _fill_with_none(expected_keys):
        """Return a dictionary with None values for each key."""
        return {key: None for key in expected_keys}
    
    @staticmethod
    def _find_exteme_3d_points(anatomy_mask, label):
        """Find the minimum and maximum points of 
        a specific anatomical structure in an anatomy mask."""
        selected_anatomy_mask = (anatomy_mask == label)
        indices = np.argwhere(selected_anatomy_mask)
        if len(indices) == 0:
            return None
        axes = [i for i in range(3)]
        max_min_mode = ["max", "min"]
        extreme_points = {}
        for axis in axes:
            for mode in max_min_mode:
                if mode == "min":
                    extreme_points[f"axis_{axis}_{mode}"] = indices[np.argmin(indices[:, axis])]
                else:
                    extreme_points[f"axis_{axis}_{mode}"] = indices[np.argmax(indices[:, axis])]
        return extreme_points
    
    @staticmethod
    def _define_anatomical_region(y_coord, boundaries):
        """Define the anatomical region based on y-coordinate according
        to anatomical landmarks."""
        # Aligned with the order of a: "Head", "Chest", "Abdomen", "Legs"
        if y_coord < boundaries['abdomen_start']:
            return 3  # Legs
        elif y_coord < boundaries['chest_start']:
            return 2  # Abdomen
        elif y_coord < boundaries['head_start']:
            return 1  # Chest
        else:
            return 0  # Head
        
    def _define_tumor_localization(self, anatomy_np, raw_semantic_np, raw_instance_np, raw_num_instances):
        """Define tumor localization based on anatomical regions."""
        # Get anatomical landmarks
        anatomical_landmarks = {
            "lungs": self._find_exteme_3d_points(anatomy_np, self.anatomical_labels["lungs"]),
            "hips": self._find_exteme_3d_points(anatomy_np, self.anatomical_labels["hips"])
        } 
        anatomical_regions_boundaries = {
            "head_start": anatomical_landmarks["lungs"]["axis_1_max"][1],
            "chest_start": anatomical_landmarks["lungs"]["axis_1_min"][1],
            "abdomen_start": anatomical_landmarks["hips"]["axis_1_min"][1],
        }
        
        # Define localization of tumor candidates within a body
        tumor_candidates_centers = np.array(
            center_of_mass(raw_semantic_np, raw_instance_np, range(1, raw_num_instances + 1)))
        
        anatomical_regions = [
            self._define_anatomical_region(center[1], anatomical_regions_boundaries) 
            for center in tumor_candidates_centers
        ]
        return {"anatomical_region": np.array(anatomical_regions)}
    
    @staticmethod
    def _get_meta_from_affine(affine_matrix):
        """Extract spacing, origin, and direction from affine matrix."""
        direction_matrix = affine_matrix[:3, :3]
        spacing = np.linalg.norm(direction_matrix, axis=0)
        origin = affine_matrix[:3, 3]
        direction = (direction_matrix / spacing).flatten()
        return spacing, origin.numpy(), direction.numpy() 
    
    @staticmethod
    def _convert_numpy_to_sitk(data_np, spacing, origin, direction):
        """Convert numpy array to SimpleITK image with given spacing, origin, and direction
        to process data with PyRadiomics later."""
        # Need to transpose the numpy array to match SimpleITK's axes order
        data_sitk = sitk.GetImageFromArray(np.transpose(data_np, (2, 1, 0)))
        data_sitk.SetSpacing(list(spacing))
        data_sitk.SetOrigin(list(origin))
        data_sitk.SetDirection(list(direction))
        return data_sitk
    
    def _downsample_data(self, image_np, raw_instance_np, spacing):
        """Downsample the input data to reduce the time of radiomic features calculation."""
        zoom_factors = tuple(spacing[i] / self.downsampled_spacing[i] for i in range(self.dimension))
        raw_instance_np_down = zoom(raw_instance_np, zoom_factors, order=0)
        image_np_down = zoom(image_np, zoom_factors, order=1)
        return image_np_down, raw_instance_np_down
    
    def _extract_features_for_tumor_instance(self, image_sitk, tumor_instance_mask_sitk, extractor):
        """Call the Pyradiomics feature extractor on the tumor candidate instance."""
        tumor_instance_features = extractor.execute(image_sitk, tumor_instance_mask_sitk) # Requires SimpleITK
        features_dict = {
            key: tumor_instance_features[key] for key in tumor_instance_features.keys() 
            if key.startswith('original') or key.startswith('wavelet')
            }             
        return features_dict
    
    # Helper function to extract radiomic features for each tumor candidate
    def _extract_radiomic_features(self, raw_instance_np_down, image_np_down, extractor, origin, direction, raw_num_instances):
        """Iterate over each tumor candidate and extract radiomic features."""
        features_list = [None] * raw_num_instances
        futures = []
        tumor_candidates_sizes = []

        image_sitk = self._convert_numpy_to_sitk(image_np_down, self.downsampled_spacing, origin, direction)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            for instance_id in range(1, raw_num_instances + 1):
                tumor_instance_mask = (raw_instance_np_down == instance_id)
                tumor_instance_size = np.sum(tumor_instance_mask)
                tumor_candidates_sizes.append(tumor_instance_size)

                if tumor_instance_size > self.size_max_threshold:
                    features_list[instance_id - 1] = True
                    continue

                if tumor_instance_size < self.size_min_threshold:
                    features_list[instance_id - 1] = False
                    continue

                tumor_instance_mask_sitk = self._convert_numpy_to_sitk(tumor_instance_mask.astype(np.uint8), 
                                                                      self.downsampled_spacing, origin, direction)
                futures.append((instance_id, executor.submit(self._extract_features_for_tumor_instance, 
                                                             image_sitk, tumor_instance_mask_sitk, extractor)))

        for instance_id, future in tqdm(futures, total=len(futures)):
            features_list[instance_id - 1] = future.result()

        return features_list, tumor_candidates_sizes
    
    def _create_tumors_dataframe(self, tumor_candidates_localization, features_list, tumor_candidates_sizes):
        """Create a pandas DataFrame with tumor candidates metadata and radiomic features."""
        expected_keys = self._find_first_dict_in_list(features_list).keys()
        features_list = [
            features if isinstance(features, dict) else self._fill_with_none(expected_keys) 
            for features in features_list
        ]
        tumors_meta_dict = {
            **tumor_candidates_localization,
            "size": tumor_candidates_sizes,
        }
        return pd.concat([pd.DataFrame(tumors_meta_dict), pd.DataFrame(features_list)], axis=1), expected_keys
    
    def _classify_tumors(self, tumors_dataframe, expected_keys):
        """Classify tumors based on their radiomic features and anatomical region."""
        for anatomical_idx, anatomical_region in enumerate(self.anatomical_regions):
            logger.info(f"Processing anatomical region: {anatomical_region}...")

            tumors_regional_dataframe = tumors_dataframe[
                tumors_dataframe["anatomical_region"] == anatomical_idx].dropna(subset=expected_keys)

            if len(tumors_regional_dataframe) == 0:
                continue

            random_forest_classifier = joblib.load(self.model_path[anatomical_idx])
            features_to_use = joblib.load(self.radiomic_feature_list_path[anatomical_idx])

            tumors_regional_dataframe["probability"] = random_forest_classifier.predict_proba(
                tumors_regional_dataframe[features_to_use])[:, 1]

            tumors_regional_dataframe["prediction"] = tumors_regional_dataframe["probability"] > self.classification_threshold
            tumors_dataframe.loc[tumors_regional_dataframe.index, "prediction"] = tumors_regional_dataframe["prediction"]

        # Assign predictions for small and large tumors
        # Large tumor candidates are considered as True Positives
        tumors_dataframe.loc[tumors_dataframe["size"] >= self.size_max_threshold, "prediction"] = True 
        # Small tumor candidates are considered as False Positives
        tumors_dataframe.loc[tumors_dataframe["size"] <= self.size_min_threshold, "prediction"] = False
    
    @staticmethod
    def _filter_instances(instance_mask, predictions):
        """Filter instances based on their predictions."""
        filtered_mask = np.zeros_like(instance_mask)
        for idx, prediction in enumerate(predictions):
            if prediction:
                filtered_mask[instance_mask == idx+1] = 1
        return filtered_mask
    
    def _prepare_output(self, filtered_instance_np, image_metatensor):
        """Prepare the output with filtered tumor candidates."""
        filtered_instance_np = torch.unsqueeze(torch.from_numpy(filtered_instance_np).type(torch.uint8), dim=0)
        return MetaTensor(filtered_instance_np).copy_meta_from(image_metatensor, copy_attr=False)

        
    def run_inferer(self, data: Dict[str, Any], convert_to_batch=True, device="cuda"):
        """
        Execute the tumor candidate classification inference task.
        """
        logger.info(f"Running Tumor Candidate Classification...")
        
        # Stage 0. Get the data
        logger.info(f"Stage 0: Getting the data...")
        image_metatensor = data["image"]
        anatomy_metatensor = data["anatomy"]
        raw_semantic_metatensor = data["proba"]
        
        # Get meta data from affine
        affine_matrix = image_metatensor.affine
        spacing, origin, direction = self._get_meta_from_affine(affine_matrix)
        
         # Adjust downsampled spacing if needed
         # If downsampled spacing along one of axes is not positive, 
         # it means that this axis is not downsampled
        self.downsampled_spacing = tuple(
            self.downsampled_spacing[i] if self.downsampled_spacing[i] > 0 else spacing[i] for i in range(self.dimension)
            )
        
        # Convert data to numpy
        image_np = torch.squeeze(image_metatensor, dim=0).cpu().numpy()
        anatomy_np = torch.squeeze(anatomy_metatensor, dim=0).cpu().numpy()
        raw_semantic_np = torch.squeeze(raw_semantic_metatensor, dim=0).cpu().numpy().astype(np.uint8)
        
        # Stage 1: Get tumor candidates
        logger.info(f"Stage 1: Getting tumor candidates...")
        raw_instance_np, raw_num_instances = perform_connected_components_analysis(raw_semantic_np)
        tumor_candidates_localization = self._define_tumor_localization(
            anatomy_np, raw_semantic_np, raw_instance_np, raw_num_instances) 
        logger.info(f"Total number of tumor candidates: {raw_num_instances}")
        
        # Dowmsample data to speed up processing
        image_np_down, raw_instance_np_down = self._downsample_data(image_np, raw_instance_np, spacing)        
        
        # Stage 2: Extract radiomic features
        logger.info(f"Stage 2: Extracting radiomic features...")
        extractor = featureextractor.RadiomicsFeatureExtractor(self.radiomic_extractor_config_path)
        features_list, tumor_candidates_sizes = self._extract_radiomic_features(
            raw_instance_np_down, image_np_down, extractor, origin, direction, raw_num_instances
        )
        logger.info(f"Finished radiomic features extraction...")
                
        # Stage 3: Classify tumor candidates
        logger.info(f"Stage 3: Classifying tumor candidates...")
        tumors_dataframe, expected_keys = self._create_tumors_dataframe(
            tumor_candidates_localization, features_list, tumor_candidates_sizes
        )
        self._classify_tumors(tumors_dataframe, expected_keys)
        logger.info(f"Finished tumor candidate classification...")
           
        # Stage 4: Filtering tumor candidates based on classification results
        logger.info(f"Stage 4: Filtering the segmentation mask and forming an output...")
        filtered_instance_np = self._filter_instances(raw_instance_np, 
                                                     tumors_dataframe["prediction"].astype(np.uint8))
        
        # Form an output image
        filtered_instance_np = torch.unsqueeze(torch.from_numpy(filtered_instance_np).type(torch.uint8), dim=0)
        filtered_instance_metatensor = MetaTensor(filtered_instance_np).copy_meta_from(image_metatensor, copy_attr=False)
        
        # Perform required data management operations
        data[self.output_label_key] = filtered_instance_metatensor
        return data
    
    def __call__(self, request) -> Union[Dict, Tuple[str, Dict[str, Any]]]:
        """
        Execute the inference task.

        Args:
            request (dict): The request payload for inference.

        Returns:
            Tuple[str, Dict]: The result file name and associated metadata.
        """
        begin = time.time()
        req = copy.deepcopy(self._config)
        req.update(request)

        logger.setLevel(req.get("logging", "INFO").upper())
        if req.get("image") is not None and isinstance(req.get("image"), str):
            logger.info(f"Infer Request (final): {req}")
            data = copy.deepcopy(req)
            data.update({"image_path": req.get("image")})
        else:
            dump_data(req, logger.level)
            data = req

        start = time.time()
        pre_transforms = self.pre_transforms(data)
        data = self.run_pre_transforms(data, pre_transforms)
        latency_pre = time.time() - start

        start = time.time()
        data = self.run_inferer(data)
        latency_inferer = time.time() - start

        start = time.time()
        data = self.run_invert_transforms(data, pre_transforms, self.inverse_transforms(data))
        latency_invert = time.time() - start

        start = time.time()
        data = self.run_post_transforms(data, self.post_transforms(data))
        latency_post = time.time() - start
        
        # Return directly in pipeline mode
        if data.get("pipeline_mode", False):
            return {"pred": data["pred"]}, {}

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

        result_file_name_dict = {"final": result_file_name, "proba": None}
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
 