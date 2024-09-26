from monai.transforms import LoadImaged, AsDiscreted
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.interfaces.utils.transform import dump_data

from typing import Callable, Dict, Sequence, Tuple, Union, Any
import logging
import time
import copy

logger = logging.getLogger(__name__)


class InfererProbabilityThresholding(BasicInferTask):
    def __init__(
        self,
        threshold=0.5,
        type=InferType.SEGMENTATION,
        labels=None,
        dimension=3,
        description="Thresholding probability map with varied threshold",
        **kwargs,
    ):
        super().__init__(
            path=None,
            network=None,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            input_key="proba",
            output_label_key="proba",
            output_json_key="result",
            load_strict=False,
            **kwargs,
        )
        self.threshold = threshold

    @property
    def required_inputs(self):
        return [
            "proba",
        ]

    def pre_transforms(self, data=None):
        if data and isinstance(data.get("proba"), str):
            t = [
                LoadImaged(keys="proba", reader="ITKReader"),
            ]
        else:
            t = []
        return t

    def post_transforms(self, data=None) -> Sequence[Callable]:
        # Add transform to extract the 1st channel
        t = [
            AsDiscreted(keys="proba", threshold=self.threshold),
        ]
        return t

    def __call__(self, request) -> Union[Dict, Tuple[str, Dict[str, Any]]]:
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
        data = self.run_post_transforms(data, self.post_transforms(data))
        latency_post = time.time() - start

        if self.skip_writer:
            return dict(data)

        start = time.time()
        result_file_name, result_json = self.writer(data)
        latency_write = time.time() - start

        latency_total = time.time() - begin
        logger.info(
            "++ Latencies => Total: {:.4f}; "
            "Pre: {:.4f}; Post: {:.4f}; Write: {:.4f}".format(
                latency_total,
                latency_pre,
                latency_post,
                latency_write,
            )
        )

        result_json["label_names"] = self.labels
        result_json["latencies"] = {
            "pre": round(latency_pre, 2),
            "post": round(latency_post, 2),
            "write": round(latency_write, 2),
            "total": round(latency_total, 2),
            "transform": data.get("latencies"),
        }

        if result_file_name is not None and isinstance(result_file_name, str):
            logger.info(f"Result File: {result_file_name}")
        logger.info(f"Result Json Keys: {list(result_json.keys())}")
        return result_file_name, result_json

    def writer(self, data: Dict[str, Any], extension=None, dtype=None) -> Tuple[Any]:
        if data.get("pipeline_mode", False):
            return {"pred": data["proba"]}, {}
        return super().writer(data, extension=".nii.gz")
