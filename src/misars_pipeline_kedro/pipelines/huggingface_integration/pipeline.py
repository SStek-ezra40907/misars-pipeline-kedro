"""
This is a boilerplate pipeline 'huggingface_integration'
generated using Kedro 0.19.8
"""

from kedro.pipeline import Pipeline, pipeline,node
from .nodes import get_repo_models_list, filter_yolo_models, categorize_and_aggregate_models, filter_unet_models, \
    download_huggingface_model, get_download_status


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
            outputs="downloaded_models_list",
            name="download_all_huggingface_model_node"
        ),
        node(
            func=get_download_status,
            inputs=["repo_models_list","downloaded_models_list"],
            outputs="models_list",
            name="get_download_status_node"
        ),
        node(
            func=categorize_and_aggregate_models,
            inputs=["models_list"],
            outputs="aggregated_models",
            name="categorize_and_aggregate_models_node"
        ),
        node(
            func=filter_yolo_models,
            inputs=["aggregated_models"],
            outputs="yolo_models_list",
            name="filter_yolo_models_node"
        ),
        node(
            func=filter_unet_models,
            inputs=["aggregated_models"],
            outputs="unet_models_list",
            name="filter_unet_models_node"
        ),
    ])
