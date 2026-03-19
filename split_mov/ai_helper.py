from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import DetectionConfig


@dataclass(slots=True)
class AIResult:
    loading_probability: float
    detail: str


class OptionalAIHelper:
    """ONNXモデルがある場合のみ補助判定を行う。"""

    def __init__(self, config: DetectionConfig) -> None:
        self.enabled = bool(config.ai_enabled)
        self.input_size = int(config.ai_input_size)
        self._session: Any | None = None
        self._input_name: str | None = None

        if not self.enabled or not config.ai_model_path:
            self.enabled = False
            return

        model_path = Path(config.ai_model_path)
        if not model_path.exists():
            self.enabled = False
            return

        try:
            import onnxruntime as ort
        except Exception:
            self.enabled = False
            return

        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name

    def is_available(self) -> bool:
        return self.enabled and self._session is not None and self._input_name is not None

    def score_range(self, video_path: str, start_sec: float, end_sec: float) -> AIResult | None:
        if not self.is_available():
            return None

        mid = (start_sec + end_sec) / 2.0
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, mid) * 1000.0)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        x = image.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]

        outputs = self._session.run(None, {self._input_name: x})
        raw = outputs[0]
        arr = np.array(raw).astype(np.float32).squeeze()

        if arr.ndim == 0:
            prob = float(np.clip(arr, 0.0, 1.0))
        elif arr.size == 1:
            prob = float(np.clip(arr[0], 0.0, 1.0))
        else:
            # [loading, normal] の2クラス想定。違う場合は先頭を loading とみなす。
            if arr.size >= 2:
                logits = arr[:2]
                e = np.exp(logits - np.max(logits))
                prob = float(e[0] / np.sum(e))
            else:
                prob = float(np.clip(arr.flat[0], 0.0, 1.0))

        return AIResult(loading_probability=prob, detail="onnx")
