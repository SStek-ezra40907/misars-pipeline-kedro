import requests
import pandas as pd
import os
from pathlib import Path
from typing import Dict
import logging
from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings
from huggingface_hub import (
    hf_hub_download,
    HfApi,
    login
)

logger = logging.getLogger(__name__)

# 設定專案路徑與配置路徑
# project_path = "C:/Users/Ezra4/PycharmProjects/misars_pipeline_kedro"
project_path = Path.cwd()
conf_path = str(Path(project_path) / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
credentials = conf_loader["credentials"]


def get_repo_models_list(parameters: Dict) -> pd.DataFrame:
    """
        Retrieves a list of models from a Hugging Face repository.

        Args:
            parameters (Dict): A dictionary containing the repository ID and Hugging Face token.

        Returns:
            pd.DataFrame: A DataFrame containing information about the models, including the model name and file extension.
    """
    repo_id = parameters["repo_id"]
    hf_token = credentials["huggingface_token"]

    headers = {
        "Authorization": f"Bearer {hf_token}"
    }

    url = repo_id

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Failed to fetch models: {response.status_code} - {response.text}")
        return pd.DataFrame()  # 返回空 DataFrame

    response_json = response.json()
    siblings = response_json['siblings']
    models_info = [sibling['rfilename'] for sibling in siblings]

    models_data = []
    pt_onnx_files = list(filter(lambda x: x.endswith('.pt') or x.endswith('.onnx'), models_info))
    pt_onnx_files = sorted(pt_onnx_files)
    for model in pt_onnx_files:
        name, ext = os.path.splitext(model)
        models_data.append({
            'model_name': name,
            'ext': ext
        })
    return pd.DataFrame(models_data)


def categorize_and_aggregate_models(models_data: pd.DataFrame) -> pd.DataFrame:
    models_data['has_pt'] = models_data['ext'].str.contains('pt')
    models_data['has_onnx'] = models_data['ext'].str.contains('onnx')
    models_data = models_data.drop(columns=['ext'])
    models_data['model_type'] = models_data['model_name'].apply(set_model_type)

    models_data = models_data.groupby('model_name').agg({
        'has_pt': 'max',
        'has_onnx': 'max',
        'model_type': 'first'
    }).reset_index()

    return models_data


def set_model_type(model_name):
    if 'yolo' in model_name.lower():
        return 'yolo'
    elif 'unet' in model_name.lower() or 'flexible' in model_name.lower():
        return 'unet'
    else:
        return 'unknown'


def filter_yolo_models(models_data: pd.DataFrame) -> pd.DataFrame:
    yolo_models = (models_data[models_data['model_type'].str
    .contains('yolo', case=False)]
    ).drop(columns='model_type')
    return yolo_models


def filter_unet_models(models_data: pd.DataFrame) -> pd.DataFrame:
    unet_models = (models_data[models_data['model_type'].str
    .contains('unet', case=False)]
    ).drop(columns=['model_type', 'has_pt', 'has_onnx'])
    return unet_models


def download_huggingface_model(data: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    import os
    target_dir = os.path.join(project_path, parameters["download_models_path"])
    repo_id = parameters['repo_name']
    hf_token = credentials["huggingface_token"]
    os.makedirs(target_dir, exist_ok=True)

    success_indices = []

    for index, row in data.iterrows():
        model_name = row['model_name']
        file_ext = row['ext']
        target_file_path = os.path.join(target_dir, f"{model_name}{file_ext}")

        try:
            hf_hub_download(repo_id=repo_id, filename=f"{model_name}{file_ext}", token=hf_token, local_dir=target_dir,
                            cache_dir=None)
            print(f"Downloaded: {target_file_path}")
            success_indices.append(index)  # 下載成功的模型記錄下來
        except Exception as e:
            print(f"Failed to download {model_name}: {str(e)}")

    return data.loc[success_indices].reset_index(drop=True)


def get_model_list(models_dir):
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    return model_files


def get_download_status(repo_models: pd.DataFrame, downloaded_models: pd.DataFrame) -> pd.DataFrame:
    repo_models['combined'] = repo_models['model_name'] + repo_models['ext']
    downloaded_models['combined'] = downloaded_models['model_name'] + downloaded_models['ext']

    repo_models['downloaded'] = repo_models['combined'].isin(downloaded_models['combined'])
    repo_models = repo_models.drop(columns='combined')

    return repo_models


def push_model_to_huggingface(converted_models: pd.DataFrame, parameters: Dict):
    repo_id = parameters['repo_name']
    hf_token = credentials["huggingface_token"]

    login(hf_token)
    api = HfApi()
    model_save_path = parameters['model_save_path']
    print(f"filepath")
    for model_id, _ in converted_models.items():
        model_name = model_id
        model_path = os.path.join(model_save_path, f"{model_name}.onnx")
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=f"{model_name}.onnx",
            repo_id=repo_id,
            repo_type="model"
        )
        logger.info(f"Model {model_name} successfully pushed to {repo_id}")
