"""
This is a boilerplate pipeline 'yolo_converter'
generated using Kedro 0.19.8
"""

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import convert_onnx_to_nhwc


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # node(
        #     func=get_pending_models,
        #     inputs=["preprocessed_models","onnx_models"],
        #     outputs="pending_models",
        #     name="get_converted_models_node"
        # ),
        node(
            func=convert_onnx_to_nhwc,
            inputs=["unconverted_onnx_models", "modified_yolo_models_list"],
            outputs="converted_models",
            name="convert_onnx_to_nhwc_node"
        )
    ])
