import logging
import time
import copy
import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk
import concurrent.futures
from tqdm import tqdm
from scipy.ndimage import label as perform_connected_components_analysis
from scipy.ndimage import center_of_mass, zoom
from typing import Callable, Dict, Sequence, Tuple, Union, Any
from radiomics import featureextractor
import joblib

from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.interfaces.utils.transform import dump_data
from monailabel.transform.writer import Writer
from monailabel.transform.post import Restored
from monai.data import MetaTensor
from monai.transforms import LoadImaged, AsDiscreted, Lambdad
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    ToNumpyd,
    Lambdad
)
from lib.transforms.transforms import ReorientToOriginald

# Initialize logger for this module
logger = logging.getLogger(__name__)


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
        An inference task for tumor candidate classification based on radiomics.
        
        This class uses connected components analysis, radiomic feature extraction, and a random forest 
        to identify tumor candidates.
        
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
        transforms = [
            LoadImaged(keys=["image", "anatomy", "proba"], reader="ITKReader"),
            EnsureChannelFirstd(keys=["image", "anatomy", "proba"]),
            Orientationd(keys=["image", "anatomy", "proba"], axcodes="RSA"),
            Lambdad(keys="proba", func=lambda x: x / 255),
            Spacingd(keys=["image", "anatomy", "proba"], 
                     pixdim=self.target_spacing, 
                     mode=["bilinear", "nearest", "bilinear"]),  # Resample with target spacing
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            AsDiscreted(keys="proba", threshold=self.confidence_threshold),
        ]
        # Cache the transforms if caching is enabled
        self.add_cache_transform(transforms, data)
        return transforms
    
    def inverse_transforms(self, data=None) -> Union[None, Sequence[Callable]]:
        return None
    
    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            ReorientToOriginald(keys="pred", ref_image="image"),  # Reorient to original orientation
            ToNumpyd(keys="pred"),  # Convert the prediction to a NumPy array
            Restored(keys="pred", ref_image="image"),  # Restore the spatial orientation
        ]
    
    @staticmethod
    def find_first_dict_in_list(lst):
        for item in lst:
            if isinstance(item, dict):
                return item
        return None  # Return None if no dictionary is found
    
    @staticmethod
    def fill_with_none(expected_keys):
        return {key: None for key in expected_keys}
    
    @staticmethod
    def find_exteme_3d_points(anatomy_mask, label):
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
    def define_anatomical_region(y_coord, boundaries):
        # Aligned with the order of a: "Head", "Chest", "Abdomen", "Legs"
        if y_coord < boundaries['abdomen_start']:
            return 3  # Legs
        elif y_coord < boundaries['chest_start']:
            return 2  # Abdomen
        elif y_coord < boundaries['head_start']:
            return 1  # Chest
        else:
            return 0  # Head
        
    def define_tumor_localization(self, anatomy_np, raw_semantic_np, raw_instance_np, raw_num_instances):
        # Get anatomical landmarks
        anatomical_landmarks = {
            "lungs": self.find_exteme_3d_points(anatomy_np, self.anatomical_labels["lungs"]),
            "hips": self.find_exteme_3d_points(anatomy_np, self.anatomical_labels["hips"])
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
            self.define_anatomical_region(center[1], anatomical_regions_boundaries) 
            for center in tumor_candidates_centers
        ]
        return {"anatomical_region": np.array(anatomical_regions)}
    
    
    @staticmethod
    def get_meta_from_affine(affine_matrix):
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
        return spacing, origin.numpy(), direction.numpy() 
    
    
    def extract_features_for_tumor_instance(self, image_sitk, tumor_instance_mask_sitk, extractor):
        tumor_instance_features = extractor.execute(image_sitk, tumor_instance_mask_sitk) # Requires SimpleITK
        features_dict = {
            key: tumor_instance_features[key] for key in tumor_instance_features.keys() 
            if key.startswith('original') or key.startswith('wavelet')
            }             
        return features_dict
    
    @staticmethod
    def convert_numpy_to_sitk(data_np, spacing, origin, direction):
        data_sitk = sitk.GetImageFromArray(data_np)
        data_sitk.SetSpacing(list(spacing))
        data_sitk.SetOrigin(list(origin))
        data_sitk.SetDirection(list(direction))
        return data_sitk
    
    @staticmethod
    def filter_instances(instance_mask, predictions):
        filtered_mask = np.zeros_like(instance_mask)
        for idx, prediction in enumerate(predictions):
            if prediction:
                filtered_mask[instance_mask == idx+1] = 1
        return filtered_mask
        
    def run_inferer(self, data: Dict[str, Any], convert_to_batch=True, device="cuda"):
        logger.info(f"Running Tumor Candidate Classification...")
        
        # Stage 0. Get the data
        image_metatensor = data["image"]
        anatomy_metatensor = data["anatomy"]
        raw_semantic_metatensor = data["proba"]
        
        # Get meta data
        affine_matrix = data["image"].affine
        spacing, origin, direction = self.get_meta_from_affine(affine_matrix)
                
        # Convert data to numpy
        image_np = torch.squeeze(image_metatensor, dim=0).cpu().numpy()
        anatomy_np = torch.squeeze(anatomy_metatensor, dim=0).cpu().numpy()
        raw_semantic_np = torch.squeeze(raw_semantic_metatensor, dim=0).cpu().numpy().astype(np.uint8)
        
        # Stage 1: Get tumor candidates
        # Perform connected components analysis
        logger.info(f"Performing connected components analysis...")
        raw_instance_np, raw_num_instances = perform_connected_components_analysis(raw_semantic_np)
        tumor_candidates_localization = self.define_tumor_localization(
            anatomy_np, raw_semantic_np, raw_instance_np, raw_num_instances) 
        logger.info(f"Total number of tumor candidates: {raw_num_instances}")
        
        # Dowmsample data to speed up processing
        zoom_factors = (self.target_spacing[0]/self.downsampled_spacing[0], 
                        self.target_spacing[1]/self.downsampled_spacing[1], 
                        self.target_spacing[2]/self.downsampled_spacing[2])
        raw_instance_np_down = zoom(raw_instance_np, zoom_factors, order=0)
        image_np_down = zoom(image_np, zoom_factors, order=1)
        tumor_candidates_sizes = np.bincount(raw_instance_np_down.ravel())[1:]
        
        # Stage 2: Extract radiomic features
        # Load radiomic feature extractor configuration
        logger.info(f"Extracting radiomic features...")
        extractor = featureextractor.RadiomicsFeatureExtractor(self.radiomic_extractor_config_path)
        
        # Convert Numpy arrays to SimpleITK image, since it is required for radiomic feature extraction
        image_sitk = self.convert_numpy_to_sitk(image_np_down, self.downsampled_spacing, origin, direction)
        
        # Prepare empty list to store features for each tumor candidate
        features_list = [None] * raw_num_instances
        futures = []
        
        # Extract radiomic features for each tumor candidate in parallel using a ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for instance_id in range(1, raw_num_instances + 1):
                
                # Get tumor instance mask and size
                tumor_instance_mask = (raw_instance_np_down == instance_id)
                tumor_instance_size = np.sum(tumor_instance_mask)
                
                # Skip tumor instances with size below the minimum or above the maximum threshold
                if tumor_instance_size > self.size_max_threshold:
                    features_list[instance_id - 1] = True
                    continue
                
                if tumor_instance_size < self.size_min_threshold:
                    features_list[instance_id - 1] = False
                    continue
                
                # Convert tumor instance mask to SimpleITK image and set metadata, 
                # since it is required for radiomic feature extraction
                tumor_instance_mask_sitk = self.convert_numpy_to_sitk(tumor_instance_mask.astype(np.uint8), 
                                                                      self.downsampled_spacing, origin, direction)
                
                futures.append((instance_id, 
                        executor.submit(self.extract_features_for_tumor_instance, 
                                        image_sitk,
                                        tumor_instance_mask_sitk,
                                        extractor)))
        
        for instance_id, future in tqdm(futures, total=len(futures)):
            features_list[instance_id - 1] = future.result()
                
        # Form the final feature list        
        expected_keys = self.find_first_dict_in_list(features_list).keys()
        features_list = [
            features if isinstance(features, dict) else self.fill_with_none(expected_keys) 
            for features in features_list 
            ]
                
        # Form a dataframe describing all tumor candidates
        tumors_meta_dict = {
            **tumor_candidates_localization,  
            "size": tumor_candidates_sizes,
        }
        tumors_dataframe = pd.concat([pd.DataFrame(tumors_meta_dict), pd.DataFrame(features_list)], axis=1)
        
        logger.info(f"Finished radiomic features extraction...")
                
        # Stage 3: Classify tumor candidates
        # Iterate over the body parts
        logger.info(f"Classifying tumor candidates...")
        for anatomical_idx, anatomical_region in enumerate(self.anatomical_regions):
            logger.info(f"Processing anatomical region: {anatomical_region}...")
            
            # Extract features for the current body part
            tumors_regional_dataframe = tumors_dataframe[
                tumors_dataframe["anatomical_region"] == anatomical_idx].dropna(subset=expected_keys)
            
            # Check if there are tumor candidates in the current body region
            if len(tumors_regional_dataframe) == 0:
                continue
            
            # Load respective model and list of features
            random_forest_classifier = joblib.load(self.model_path[anatomical_idx])
            features_to_use = joblib.load(self.radiomic_feature_list_path[anatomical_idx])
            
            # Predict probability with a random forest
            tumors_regional_dataframe["probability"] = random_forest_classifier.predict_proba(
                tumors_regional_dataframe[features_to_use])[:, 1]
                        
            tumors_regional_dataframe["prediction"] = False
            tumors_regional_dataframe.loc[
                tumors_regional_dataframe["probability"] > self.classification_threshold, "prediction"] = True
            
            # Insert the predictions into the main dataframe
            tumors_dataframe.loc[
                tumors_regional_dataframe.index, "prediction"] = tumors_regional_dataframe.loc[:, "prediction"]
         
        # Assign prediction to small and large tumor candidates
        tumors_dataframe.loc[tumors_dataframe["size"] >= self.size_max_threshold, "prediction"] = True
        tumors_dataframe.loc[tumors_dataframe["size"] <= self.size_min_threshold, "prediction"] = False
        
        # Stage 4: Filtering tumor candidates based on classification results
        # Filter the instance mask
        filtered_instance_np = self.filter_instances(raw_instance_np, 
                                                     tumors_dataframe["prediction"].astype(np.uint8))
        
        # Form an output image
        filtered_instance_np = torch.unsqueeze(
            torch.from_numpy(filtered_instance_np).type(torch.uint8), dim=0)
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
 