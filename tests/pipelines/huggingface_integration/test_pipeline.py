import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.misars_pipeline_kedro.pipelines.huggingface_integration.nodes import (
    get_repo_models_list,
    get_model_list,
    filter_models
)
from huggingface_hub import HfApi, login
from src.misars_pipeline_kedro.pipelines.huggingface_integration.nodes import credentials

from unittest import mock

@pytest.fixture
def dummy_parameters():
    parameters = {
        "repo_id": "https://huggingface.co/api/models/smartsurgery/urology-models",
        "min_downloads": 1000,
        "model_save_dir": "data/01_models_info/huggingface_models",
        "output_csv_path": "data/02_downloaded_models/downloaded_models.csv"

    }
    return parameters


@pytest.fixture
def dummy_credentials():
    return {
        "huggingface_token": "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"
    }


def test_get_repo_models(dummy_parameters, dummy_credentials):
    with patch('src.misars_pipeline_kedro.pipelines.huggingface_integration.nodes.credentials', dummy_credentials):
        # Act
        result = get_repo_models_list(dummy_parameters)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 14


def test_get_model_list():
    models_dir = "data/06_models"

    # Mock os.listdir to simulate files in the directory
    with mock.patch("os.listdir") as mocked_listdir:
        mocked_listdir.return_value = ["model_a.pkl", "model_b.pkl"]

        model_files = get_model_list(models_dir)
        assert len(model_files) == 2
        assert "model_a.pkl" in model_files
        assert "model_b.pkl" in model_files


@pytest.fixture
def sample_models_data():
    data = {
        "model_name": [
            "Urology_1-2-7val640rezize_4.36.0",
            "Urology_yolov11x-seg_3-13-16-17val640rezize_1_4.40.0",
            "Urology_yolov9c-seg_3-13-16-17val640rezize-yoloTuring_4.38.1",
            "Urology_yolov9c-seg_3-13-16-17val640rezize_4.38.0",
            "Urology_yolov9c-seg_3-13-16-17val640rezize_4.38.0",
            "urology_yolov8n_3000random640resize_20240811_4.34",
        ],
        "ext": [
            ".pt",
            ".pt",
            ".pt",
            ".pt",
            ".onnx",
            ".pt"
        ]
    }
    return pd.DataFrame(data)

def test_filter_models(sample_models_data):
    # Act
    filtered_models = filter_models(sample_models_data)

    # Assert
    assert isinstance(filtered_models, pd.DataFrame)
    assert len(filtered_models) == 5  # Expecting 5 unique model_name after grouping

    # Check that model types are correctly assigned
    assert all(filtered_models['model_type'].isin(['yolo', 'unet', 'unknown']))

    # Check if the has_pt and has_onnx columns are correct
    assert all(filtered_models['has_pt'] == True)  # All should have .pt files
    assert any(filtered_models['has_onnx'] == True)  # At least one should have .onnx files
