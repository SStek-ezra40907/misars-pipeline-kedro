import pytest
import pandas as pd
import onnx
import onnx_graphsurgeon as gs
from unittest.mock import patch, MagicMock
from src.misars_pipeline_kedro.pipelines.onnx_converter.nodes import convert_onnx_to_nhwc


@pytest.fixture
def sample_onnx_models():
    return {
        'model_a': MagicMock(return_value=onnx.ModelProto()),
        'model_b': MagicMock(return_value=onnx.ModelProto())
    }


@pytest.fixture
def sample_onnx_model_list():
    return pd.DataFrame({
        'model_name': ['model_a', 'model_b'],
        'input_shape_h': [224, 256],
        'input_shape_w': [224, 256],
        'input_shape_c': [3, 3]
    })


@patch('onnx_graphsurgeon.import_onnx')
@patch('onnx_graphsurgeon.export_onnx')
def test_convert_onnx_to_nhwc(mock_export_onnx, mock_import_onnx, sample_onnx_models, sample_onnx_model_list):
    # Setup
    mock_graph = MagicMock()
    mock_input = MagicMock(spec=gs.Variable)
    mock_input.dtype = "float32"
    mock_input.shape = [1, 3, 224, 224]
    mock_input.name = "INPUT__0"
    mock_graph.inputs = [mock_input]
    mock_graph.nodes = []
    mock_import_onnx.return_value = mock_graph
    mock_export_onnx.return_value = MagicMock(spec=onnx.ModelProto)

    # Execute
    result = convert_onnx_to_nhwc(sample_onnx_models, sample_onnx_model_list)

    # Assert
    assert len(result) == 2
    assert all(f"{model_name}_nhwc" in result for model_name in ['model_a', 'model_b'])

    # Check if the original input was renamed
    assert mock_graph.inputs[0].name == "INPUT__0"

    # Check if a new input was created
    assert any(input.name == "INPUT__0" for input in mock_graph.inputs)

    # Check if a transpose node was added
    assert any(node.op == "Transpose" for node in mock_graph.nodes)
    assert any(node.name == "transpose_input" for node in mock_graph.nodes)

    # Check if the graph was cleaned up and sorted
    assert mock_graph.cleanup.call_count == 2
    mock_graph.cleanup.return_value = mock_graph

    # Check if the converted models were exported
    assert mock_export_onnx.call_count == 2

    # Check if the returned models are of the correct type
    assert all(isinstance(model, MagicMock) and model._spec_class == onnx.ModelProto for model in result.values())
