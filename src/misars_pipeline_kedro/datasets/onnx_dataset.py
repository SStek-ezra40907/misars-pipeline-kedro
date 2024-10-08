# from pathlib import PurePosixPath
# from typing import Any, Dict
#
# import fsspec
# import numpy as np
# import onnx
# from pathlib import Path
# from onnx import numpy_helper
#
# from kedro.io import AbstractDataset
# from kedro.io.core import get_filepath_str, get_protocol_and_path
#
# class ONNXDataSet(AbstractDataset):
#     """Kedro dataset for loading and saving ONNX models."""
#
#     def __init__(self, filepath: str):
#         self._filepath = Path(filepath)
#
#     def _load(self) -> np.ndarray:
#         """Load ONNX model from the file and convert to a numpy array if possible."""
#         # Load the ONNX model
#         model = onnx.load(self._filepath)
#
#         # Get the model's graph initializers (weights and biases)
#         initializers = model.graph.initializer
#
#         # Convert the initializers to numpy arrays
#         weights = []
#         for initializer in initializers:
#             np_array = onnx.numpy_helper.to_array(initializer)
#             weights.append(np_array)
#
#         # Optionally, you can return the arrays directly or perform further processing
#         return np.array(weights, dtype=object)  # Use object dtype if shapes are inconsistent
#
#     def _save(self, data):
#         """Save ONNX model to the file."""
#         onnx.save(data, self._filepath)
#
#     def _describe(self):
#         return dict(filepath=str(self._filepath))
#
# # class ONNXDataset(AbstractDataset[np.ndarray, np.ndarray]):
# #     """``ONNXDataSet`` loads ONNX model data from a given filepath and extracts tensors as `numpy` arrays.
# #
# #     Example:
# #     ::
# #
# #         >>> ONNXDataset(filepath='/model/file/path.onnx')
# #     """
# #
# #     def __init__(self, filepath: str):
# #         """Creates a new instance of ONNXDataSet to load ONNX model tensors.
# #
# #         Args:
# #             filepath: The location of the ONNX file to load.
# #         """
# #         protocol, path = get_protocol_and_path(filepath)
# #         self._protocol = protocol
# #         self._filepath = PurePosixPath(path)
# #         self._fs = fsspec.filesystem(self._protocol)
# #
# #     def _load(self) -> np.ndarray:
# #         """Loads data from the ONNX model file and returns the tensors as numpy arrays.
# #
# #         Returns:
# #             A list of numpy arrays containing the model's tensor data.
# #         """
# #         load_path = get_filepath_str(self._filepath, self._protocol)
# #         with self._fs.open(load_path, 'rb') as f:
# #             model = onnx.load(f)
# #
# #         # Extract tensors (initializers) from the model
# #         tensors = []
# #         for initializer in model.graph.initializer:
# #             tensor = numpy_helper.to_array(initializer)
# #             tensors.append(tensor)
# #
# #         return np.array(tensors)
# #
# #     def _save(self, data: np.ndarray) -> None:
# #         """Saves ONNX model data to the specified filepath.
# #
# #         You may add saving logic here if needed."""
# #         pass
# #
# #     def _describe(self) -> Dict[str, Any]:
# #         """Returns a dict that describes the attributes of the dataset."""
# #         return {"filepath": str(self._filepath)}