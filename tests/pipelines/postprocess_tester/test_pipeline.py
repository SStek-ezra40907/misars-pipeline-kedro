import pytest
from src.misars_pipeline_kedro.pipelines.postprocess_checker.nodes import *
import onnx

@pytest.mark.parametrize("model_shape, expected_layout", [
    ([("input", [1, 3, 224, 224])], "NCHW"),
    ([("input", [1, 224, 224, 3])], "NHWC"),
    ([("input", [1, 224, 224, 4])], "UNKNOWN"),
    ([("input", [1, 4, 224, 224])], "UNKNOWN"),
    ([("input", [1, 3])], "UNKNOWN"),
])
def test_get_layout_from_shape(model_shape, expected_layout):
    result = get_layout_from_shape(model_shape)
    assert result == expected_layout

def test_get_model_io_shapes():
    # Create a mock ONNX model
    input_tensor = onnx.helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [1, 3, 224, 224])
    output_tensor = onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [1, 1000])
    node_def = onnx.helper.make_node('Relu', inputs=['input'], outputs=['output'])
    graph_def = onnx.helper.make_graph([node_def], 'test_graph', [input_tensor], [output_tensor])
    model = onnx.helper.make_model(graph_def)

    # Call the function
    input_shapes, output_shapes = get_model_io_shapes(model)

    # Assert the results
    assert input_shapes == [('input', [1, 3, 224, 224])]
    assert output_shapes == [('output', [1, 1000])]