import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.misars_pipeline_kedro.pipelines.huggingface_integration.nodes import (
    get_repo_models_list,
    get_model_list,
    categorize_and_aggregate_models,
    get_download_status,
    download_huggingface_model
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
    filtered_models = categorize_and_aggregate_models(sample_models_data)

    # Assert
    assert isinstance(filtered_models, pd.DataFrame)
    assert len(filtered_models) == 5  # Expecting 5 unique model_name after grouping

    # Check that model types are correctly assigned
    assert all(filtered_models['model_type'].isin(['yolo', 'unet', 'unknown']))

    # Check if the has_pt and has_onnx columns are correct
    assert all(filtered_models['has_pt'] == True)  # All should have .pt files
    assert any(filtered_models['has_onnx'] == True)  # At least one should have .onnx files


@pytest.fixture
def repo_models_data():
    return pd.DataFrame({
        'model_name': [
            'Urology_1-2-7val640rezize_4.36.0',
            'Urology_yolov11x-seg_3-13-16-17val640rezize_1_4.40.0',
            'urology_RegUNet_3000random320resize_DiceFocalLoss_4.34'
        ],
        'ext': ['.pt', '.pt', '.onnx']
    })


@pytest.fixture
def downloaded_models_data():
    return pd.DataFrame({
        'model_name': [
            'Urology_1-2-7val640rezize_4.36.0',
            'urology_RegUNet_3000random320resize_DiceFocalLoss_4.34'
        ],
        'ext': ['.pt', '.onnx']
    })


# Test the get_download_status function
def test_get_download_status(repo_models_data, downloaded_models_data):
    # Call the function being tested
    result = get_download_status(repo_models_data, downloaded_models_data)

    # Check if the result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check if the 'downloaded' column is present
    assert 'downloaded' in result.columns

    # Check if downloaded models are correctly marked
    expected_downloaded_status = [True, False, True]
    assert list(result['downloaded']) == expected_downloaded_status

    # Check if the number of rows in the result is correct
    assert len(result) == 3


@pytest.fixture
def mock_data():
    # Mock a DataFrame containing model names and extensions
    data = {
        "model_name": ["model_a", "model_b", "model_c"],
        "ext": [".pt", ".onnx", ".pt"]
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_parameters():
    # Mock parameters
    return {
        "download_models_path": "mock_target_dir",
        "repo_name": "mock_repo"
    }


@pytest.fixture
def mock_credentials():
    return {
        "huggingface_token": "mock_token"
    }


@patch("os.makedirs")
@patch("src.misars_pipeline_kedro.pipelines.huggingface_integration.nodes.hf_hub_download")
def test_download_huggingface_model_success(mock_hf_hub_download, mock_makedirs, mock_data, mock_parameters,
                                            mock_credentials):
    # Mock successful Hugging Face download
    mock_hf_hub_download.return_value = None  # Simulate no exception raised
    with patch('src.misars_pipeline_kedro.pipelines.huggingface_integration.nodes.credentials', mock_credentials):
        result = download_huggingface_model(mock_data.copy(), mock_parameters)

        # Ensure all models were processed
        assert len(result) == len(mock_data)
        mock_hf_hub_download.assert_called()
        assert mock_hf_hub_download.call_count == len(mock_data)


@patch("os.makedirs")
@patch("src.misars_pipeline_kedro.pipelines.huggingface_integration.nodes.hf_hub_download")
def test_download_huggingface_model_failure(mock_hf_hub_download, mock_makedirs, mock_data, mock_parameters,
                                            mock_credentials):
    # Mock Hugging Face download with an exception for one model
    def mock_download_side_effect(*args, **kwargs):
        if "model_b" in kwargs.get('filename', ''):
            raise Exception("Download failed")

    mock_hf_hub_download.side_effect = mock_download_side_effect

    with patch('src.misars_pipeline_kedro.pipelines.huggingface_integration.nodes.credentials', mock_credentials):
        result = download_huggingface_model(mock_data.copy(), mock_parameters)

        # Ensure the failed model is removed
        assert len(result) == len(mock_data) - 1  # model_b should be removed
        assert "model_b" not in result['model_name'].values

        # Ensure the download function was called correctly
        assert mock_hf_hub_download.call_count == len(mock_data)
