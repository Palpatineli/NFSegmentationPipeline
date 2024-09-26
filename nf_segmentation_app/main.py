import monailabel
from monailabel.interfaces.app import MONAILabelApp
from monailabel.utils.others.class_utils import get_class_names
from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.infer_v2 import InferTask

import lib.configs
from lib.infers.inferer_3d_anisotropic_unet import Inferer3DAnisotropicUnet
from lib.infers.inferer_probability_thresholding import InfererProbabilityThresholding
from lib.infers.inferer_single_stage_pipeline import InfererSingleStagePipeline
import logging
import os
from typing import List, Dict

import copy
from monailabel.interfaces.exception import MONAILabelError, MONAILabelException
from monailabel.utils.others.generic import (
    file_checksum,
    handle_torch_linalg_multithread,
    is_openslide_supported,
    name_to_device,
    strtobool,
)
from monailabel.interfaces.datastore import Datastore, DefaultLabelTag
from monailabel.config import settings


logger = logging.getLogger(__name__)


class NFSegmentationApp(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):
        self.model_dir = os.path.join(app_dir, "model")

        # Get all available model configurations
        configs = {}
        for c in get_class_names(lib.configs, "TaskConfig"):
            name = c.split(".")[-2].lower()
            configs[name] = c

        configs = {k: v for k, v in sorted(configs.items())}
        self.planner = None  # ToDo: Dropped self.planner
        self.models: Dict[str, TaskConfig] = {}
        for k, v in configs.items():
            logger.info(f"+++ Adding Model: {k} => {v}")
            self.models[k] = eval(f"{v}()")
            self.models[k].init(
                k, self.model_dir, conf, self.planner
            )  # ToDo: Dropped self.planner
        logger.info(f"+++ Using Models: {list(self.models.keys())}")

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name=f"MONAILabel - NF Segmentation ({monailabel.__version__})",
            description="Pipeline for performing neurofibroma segmentation on T2-weighted WB-MRI scans",
            version=monailabel.__version__,
        )

    def init_datastore(self) -> Datastore:
        datastore = super().init_datastore()
        return datastore

    def init_infers(self) -> Dict[str, InferTask]:
        components: Dict[str, InferTask] = {}
        infers: Dict[str, InferTask] = {}

        # Initialize inferers for each model configuration
        for n, task_config in self.models.items():
            c = task_config.infer()
            c = c if isinstance(c, dict) else {n: c}
            for k, v in c.items():
                logger.info(f"+++ Adding Components: {k} => {v}")
                components[k] = v

        # Initialize pipelines based on existing inferers
        components["config_default_threshold"] = InfererProbabilityThresholding(
            threshold=0.5
        )
        # components["config_low_threshold"] = InfererProbabilityThresholding(threshold=0.25)
        # components["config_high_threshold"] = InfererProbabilityThresholding(threshold=0.75)
        # infers.update({"Post_processing_threshold": components["config_low_threshold"]})

        infers.update(
            {
                "Segmentation_with_single_stage_pipeline": InfererSingleStagePipeline(
                    task_segmentation=components["config_3d_anisotropic_unet"],
                    task_thresholding=components["config_default_threshold"],
                    description="Single-Stage Pipeline for neurofibroma segmentation",
                ),
            }
        )
        logger.info(infers)
        return infers

    @staticmethod
    def get_file_path(datastore, image_id, tag):
        output_id = datastore.get_label_by_image_id(image_id, tag)
        output_path = datastore.get_label_uri(output_id, label_tag=tag)
        return output_path

    @staticmethod
    def save_output(datastore, image_id, model, result_json, result_file_name, tag):
        output_id = None
        if result_file_name and os.path.exists(result_file_name):
            output_id = datastore.save_label(
                image_id, result_file_name, tag, {"model": model, "params": result_json}
            )
        else:
            raise MONAILabelException(
                MONAILabelError.INFERENCE_ERROR,
                "No output file provided for saving the result",
            )
        return output_id

    def infer(self, request, datastore=None):
        model = request.get("model")
        if not model:
            raise MONAILabelException(
                MONAILabelError.INVALID_INPUT,
                "Model is not provided for Inference Task",
            )

        task = self._infers.get(model)
        if not task:
            raise MONAILabelException(
                MONAILabelError.INVALID_INPUT,
                f"Inference Task is not Initialized. There is no model '{model}' available",
            )
        required_inputs = task.required_inputs

        request = copy.deepcopy(request)
        request["description"] = task.description

        image_id = request["image"]
        if isinstance(image_id, str):
            datastore = datastore if datastore else self.datastore()
            if os.path.exists(image_id):
                request["save_label"] = False
            else:
                request["image"] = datastore.get_image_uri(request["image"])

            if os.path.isdir(request["image"]):
                logger.info("Input is a Directory; Consider it as DICOM")

            logger.debug(f"Image => {request['image']}")
        else:
            request["save_label"] = False

        # Checking for the required inputs
        request["label"] = self.get_file_path(datastore, image_id, tag="final")
        request["proba"] = self.get_file_path(datastore, image_id, tag="proba")
        request["anatomy"] = self.get_file_path(datastore, image_id, tag="anatomy")

        for (
            require_input
        ) in required_inputs:  # ToDo: Make sure that this check works correctly
            if request[require_input] == "":
                raise MONAILabelException(
                    MONAILabelError.INVALID_INPUT,
                    f"Missing required input '{require_input}' for Inference Task. Run Segmentation first.",
                )

        if self._infers_threadpool:

            def run_infer_in_thread(t, r):
                handle_torch_linalg_multithread(r)
                return t(r)

            f = self._infers_threadpool.submit(run_infer_in_thread, t=task, r=request)
            result_file_name_dict, result_json = f.result(
                request.get("timeout", settings.MONAI_LABEL_INFER_TIMEOUT)
            )
        else:
            result_file_name_dict, result_json = task(request)

        label_id = None
        for name, result_file_name in result_file_name_dict.items():
            if result_file_name:
                output_id = self.save_output(
                    datastore, image_id, model, result_json, result_file_name, name
                )
                if name == "final":
                    label_id = output_id
            else:
                continue

        return {
            "label": label_id,
            "tag": DefaultLabelTag.ORIGINAL,
            "file": result_file_name,
            "params": result_json,
        }

    # def infer(self, request, datastore=None):
    #     image_id = request["image"]

    #     if isinstance(image_id, str):
    #         datastore = datastore if datastore else self.datastore()

    #     label_id = datastore.get_label_by_image_id(image_id, tag="final")
    #     label = datastore.get_label_uri(label_id, label_tag="final")
    #     request["label"] = label
    #     return super().infer(request, datastore)
