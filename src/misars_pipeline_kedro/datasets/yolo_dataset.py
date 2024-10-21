from ultralytics import YOLO
from kedro.io import AbstractDataset

class YOLODataset(AbstractDataset):
    def __init__(self, filepath: str):
        self._filepath = filepath

    def _load(self) -> YOLO:
        return YOLO(self._filepath)

    def _save(self, model: YOLO) -> None:
        model.save(self._filepath)

    def _describe(self) -> dict:
        return {"filepath": self._filepath, "model_type": "YOLO"}