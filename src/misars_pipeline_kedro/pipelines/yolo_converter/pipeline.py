"""
This is a boilerplate pipeline 'yolo_converter'
generated using Kedro 0.19.8
"""

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import export_yolo_to_onnx


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=export_yolo_to_onnx,
            inputs=["yolo_models", "modified_yolo_models_list","params:export_options"],
            outputs="onnx_models",
            name="export_yolo_to_onnx_node",
        )
    ])
