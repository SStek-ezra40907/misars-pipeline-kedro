# src/kedro_datasets/onnx_datasets.py

from kedro.io import AbstractDataset
import onnx


class ONNXDataset(AbstractDataset):
    """Kedro Dataset for loading ONNX models."""

    def __init__(self, filepath: str):
        self.filepath = filepath

    def _load(self) -> onnx.ModelProto:
        """Load the ONNX model."""
        return onnx.load(self.filepath)

    def _save(self, data: onnx.ModelProto) -> None:
        """Save the ONNX model."""
        onnx.save(data, self.filepath)

    def _describe(self) -> dict:
        return dict(filepath=self.filepath)
