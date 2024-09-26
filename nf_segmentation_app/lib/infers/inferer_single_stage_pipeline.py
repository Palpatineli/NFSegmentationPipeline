from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType
from monailabel.utils.others.generic import name_to_device
from monailabel.transform.writer import Writer

from typing import Callable, Sequence
import logging
import time
import copy

logger = logging.getLogger(__name__)


class InfererSingleStagePipeline(BasicInferTask):
    def __init__(
        self,
        task_segmentation: InferTask,
        task_thresholding: InferTask,
        type=InferType.SEGMENTATION,
        description="Combines segmentation and thresholding into a single stage pipeline",
        **kwargs,
    ):
        self.task_segmentation = task_segmentation
        self.task_thresholding = task_thresholding

        super().__init__(
            path=None,
            network=None,
            type=type,
            labels=task_segmentation.labels,
            dimension=task_segmentation.dimension,
            description=description,
            load_strict=False,
            **kwargs,
        )

    @property
    def required_inputs(self):
        return [
            "image",
        ]

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        return []

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return []

    def is_valid(self) -> bool:
        return True

    def _latencies(self, r, e=None):
        if not e:
            e = {"pre": 0, "infer": 0, "invert": 0, "post": 0, "write": 0, "total": 0}

        for key in e:
            e[key] = e[key] + r.get("latencies", {}).get(key, 0)
        return e

    def segment_nf(self, request):
        req = copy.deepcopy(request)
        req.update({"pipeline_mode": True})
        data, meta = self.task_segmentation(req)
        return data, meta, self._latencies(meta)

    def threshold_nf(self, request, proba):
        req = copy.deepcopy(request)
        req.update({"proba": proba, "pipeline_mode": True})
        data, meta = self.task_thresholding(req)
        return data, meta, self._latencies(meta)

    def __call__(self, request):
        start = time.time()

        request.update({"image_path": request.get("image")})

        device = name_to_device(request.get("device", "cuda"))
        request["device"] = device

        data_1, _, latency_1 = self.segment_nf(request)
        proba = data_1["proba"]
        proba_meta = data_1["proba_meta_dict"]

        data_2, _, latency_2 = self.threshold_nf(request, proba)
        result_mask = data_2["pred"]

        data = copy.deepcopy(request)
        data.update(
            {
                "final": result_mask,
                "result_extension": ".nii.gz",
                "result_meta_dict": proba_meta,
                "proba": proba,
                "proba_extension": ".nii.gz",
                "proba_dtype": "float32",
            }
        )

        begin = time.time()
        result_file_pred, _ = Writer(
            label="final", ref_image="result", key_extension="result_extension"
        )(data)
        result_file_proba, _ = Writer(
            label="proba",
            ref_image="result",
            key_extension="proba_extension",
            key_dtype="proba_dtype",
        )(data)
        result_file_name_dict = {"final": result_file_pred, "proba": result_file_proba}
        latency_write = round(time.time() - begin, 2)

        total_latency = round(time.time() - start, 2)
        result_json = {
            "label_names": self.task_segmentation.labels,
            "latencies": {
                "segment_nf": latency_1,
                "threshold_nf": latency_2,
                "write": latency_write,
                "total": total_latency,
            },
        }
        logger.info(f"Result Mask: {result_mask.shape}; total_latency: {total_latency}")
        return result_file_name_dict, result_json
