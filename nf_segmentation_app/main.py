import monailabel
from monailabel.interfaces.app import MONAILabelApp
from monailabel.utils.others.class_utils import get_class_names
from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.infer_v2 import InferTask

import lib.configs
import logging
import os
from typing import List, Dict

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
        infers: Dict[str, InferTask] = {}

        # Initialize inferers for each model configuration
        for n, task_config in self.models.items():
            c = task_config.infer()
            c = c if isinstance(c, dict) else {n: c}
            for k, v in c.items():
                logger.info(f"+++ Adding Inferer: {k} => {v}")
                infers[k] = v

        # Initialize pipelines based on existing inferers
        # ToDo: Implement pipeline initialization based on existing inferers
        return infers

    def infer(self, request, datastore=None):
        image_id = request["image"]

        if isinstance(image_id, str):
            datastore = datastore if datastore else self.datastore()

        label_id = datastore.get_label_by_image_id(image_id, tag="final")
        label = datastore.get_label_uri(label_id, label_tag="final")
        request["label"] = label
        return super().infer(request, datastore)
