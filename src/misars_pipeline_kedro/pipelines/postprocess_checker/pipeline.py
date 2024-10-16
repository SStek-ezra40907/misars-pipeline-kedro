"""
This is a boilerplate pipeline 'postprocess_checker'
generated using Kedro 0.19.8
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import get_onnx_input_shape, get_unconverted_onnx_models


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=get_unconverted_onnx_models,
            inputs=["onnx_models","modified_unet_models_list"],
            outputs="unconverted_onnx_models",
        ),
        node(
            func=get_onnx_input_shape,
            inputs=["onnx_models","unet_models_list"],
            outputs="modified_unet_models_list",
            name="get_onnx_input_shape_node",
        ),
    ])
