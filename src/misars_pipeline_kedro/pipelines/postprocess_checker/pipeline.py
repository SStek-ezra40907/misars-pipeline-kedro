"""
This is a boilerplate pipeline 'postprocess_checker'
generated using Kedro 0.19.8
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import get_onnx_input_shape, get_unconverted_onnx_models, get_pt_input_shape_by_name, merge_dataframes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=merge_dataframes,
            inputs=["modified_yolo_models_list", "modified_unet_models_list"],
            outputs="merged_models_list",
            name="merge_dataframes_node",
        ),
        node(
            func=get_unconverted_onnx_models,
            inputs=["onnx_models","merged_models_list"],
            outputs="unconverted_onnx_models",
            name="get_unconverted_onnx_models_node",
        ),
        node(
            func=get_onnx_input_shape,
            inputs=["onnx_models","unet_models_list"],
            outputs="modified_unet_models_list",
            name="get_onnx_input_shape_node",
        ),
        node(
            func=get_pt_input_shape_by_name,
            inputs=["yolo_models","yolo_models_list"],
            outputs="modified_yolo_models_list",
            name="get_yolo_input_shape_node",
        ),
    ])
