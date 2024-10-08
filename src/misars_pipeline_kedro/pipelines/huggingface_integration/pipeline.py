"""
This is a boilerplate pipeline 'huggingface_integration'
generated using Kedro 0.19.8
"""

from kedro.pipeline import Pipeline, pipeline,node
from .nodes import get_repo_models_list, filter_yolo_models, filter_models, filter_unet_models, \
    download_huggingface_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=get_repo_models_list,
            inputs=["params:repo_options"],
            outputs="repo_models_list",
            name="get_repo_models_node"
        ),
        node(
            func=download_huggingface_model,
            inputs=["repo_models_list","params:repo_options"],
            outputs=None,
            name="download_all_huggingface_model_node"
        ),
        node(
            func=filter_models,
            inputs=["repo_models_list"],
            outputs="filter_models_list",
            name="filter_models_node"
        ),
        node(
            func=filter_yolo_models,
            inputs=["filter_models_list"],
            outputs="yolo_models_list",
            name="filter_yolo_models_node"
        ),
        node(
            func=filter_unet_models,
            inputs=["filter_models_list"],
            outputs="unet_models_list",
            name="filter_unet_models_node"
        ),
    ])
