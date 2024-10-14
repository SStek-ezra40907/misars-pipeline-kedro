"""
This is a boilerplate pipeline 'postprocess_checker'
generated using Kedro 0.19.8
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import get_onnx_input_shape


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=get_onnx_input_shape,
            inputs=["onnx_models","unet_models_list"],
            outputs="modified_unet_models_list",
            name="load_unet_onnx_models_node",
        ),
    ])
