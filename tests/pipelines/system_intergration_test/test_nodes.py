from unittest.mock import Mock, patch
import pandas as pd
from src.misars_pipeline_kedro.pipelines.yolo_converter.nodes import *
import pytest
# from src.model_deployment_SIT.pipelines.system_intergration_test.nodes import (
#     check_onnx_version,
#     convert_to_onnx,
#     validate_onnx_model,
#     trigger_testing_pipeline,
#     handle_non_compliance,
# )
#
#
# def test_check_onnx_version():
#     model = "dummy_model.pt"
#     onnx_version_exists = check_onnx_version(model)
#
#     assert onnx_version_exists == True, "ONNX version check failed."
#
# def test_convert_to_onnx():
#     model = "dummy_model.pt"
#     onnx_model = convert_to_onnx(model)
#
#     assert onnx_model == "model.onnx", "Model conversion to ONNX failed."
#
# def test_validate_onnx_model():
#     onnx_model = "model.onnx"
#     validation_passed = validate_onnx_model(onnx_model)
#
#     assert validation_passed == True, "ONNX model validation failed."
#
# def test_trigger_testing_pipeline():
#     onnx_model = "model.onnx"
#     result = trigger_testing_pipeline(onnx_model)
#
#     assert result == "Test Passed", "Trigger testing failed."
#
# def test_handle_non_compliance():
#     onnx_model = "model.onnx"
#     result = handle_non_compliance(onnx_model)
#
def test_export_yolo_to_onnx():
    yolo_models = {
        'model1': lambda: Mock(model=Mock(fuse=lambda: Mock(eval=lambda: Mock()))),
        'model2': lambda: Mock(model=Mock(fuse=lambda: Mock(eval=lambda: Mock())))
    }
    pt_model_list = pd.DataFrame({
        'model_name': ['model1', 'model2'],
        'input_shape_h': [640, 320],
        'input_shape_w': [640, 320],
        'input_shape_c': [3, 3]
    })
    parameters = {
        'device': 'cpu',
        'opset_version': 11,
        'sim': True
    }

    result = export_yolo_to_onnx(yolo_models, pt_model_list, parameters)

    assert isinstance(result, dict)
    assert len(result) == 2
    assert 'model1' in result
    assert 'model2' in result
    assert isinstance(result['model1'], onnx.ModelProto)
    assert isinstance(result['model2'], onnx.ModelProto)

def test_export_yolo_to_onnx_skip_specific_model():
    yolo_models = {
        'Urology_yolov11x-seg_3-13-16-17val640rezize_1_4.40.0': lambda: Mock(),
        'model2': lambda: Mock(model=Mock(fuse=lambda: Mock(eval=lambda: Mock())))
    }
    pt_model_list = pd.DataFrame({
        'model_name': ['Urology_yolov11x-seg_3-13-16-17val640rezize_1_4.40.0', 'model2'],
        'input_shape_h': [640, 320],
        'input_shape_w': [640, 320],
        'input_shape_c': [3, 3]
    })
    parameters = {
        'device': 'cpu',
        'opset_version': 11,
        'sim': True
    }

    result = export_yolo_to_onnx(yolo_models, pt_model_list, parameters)

    assert isinstance(result, dict)
    assert len(result) == 1
    assert 'Urology_yolov11x-seg_3-13-16-17val640rezize_1_4.40.0' not in result
    assert 'model2' in result
    assert isinstance(result['model2'], onnx.ModelProto)

def test_export_yolo_to_onnx_simplification_failure():
    yolo_models = {
        'model1': lambda: Mock(model=Mock(fuse=lambda: Mock(eval=lambda: Mock())))
    }
    pt_model_list = pd.DataFrame({
        'model_name': ['model1'],
        'input_shape_h': [640],
        'input_shape_w': [640],
        'input_shape_c': [3]
    })
    parameters = {
        'device': 'cpu',
        'opset_version': 11,
        'sim': True
    }

    with patch('onnxsim.simplify', side_effect=Exception('Simplification error')):
        result = export_yolo_to_onnx(yolo_models, pt_model_list, parameters)

    assert isinstance(result, dict)
    assert len(result) == 1
    assert 'model1' in result
    assert isinstance(result['model1'], onnx.ModelProto)

def test_export_yolo_to_onnx_no_simplification():
    yolo_models = {
        'model1': lambda: Mock(model=Mock(fuse=lambda: Mock(eval=lambda: Mock())))
    }
    pt_model_list = pd.DataFrame({
        'model_name': ['model1'],
        'input_shape_h': [640],
        'input_shape_w': [640],
        'input_shape_c': [3]
    })
    parameters = {
        'device': 'cpu',
        'opset_version': 11,
        'sim': False
    }

    result = export_yolo_to_onnx(yolo_models, pt_model_list, parameters)

    assert isinstance(result, dict)
    assert len(result) == 1
    assert 'model1' in result
    assert isinstance(result['model1'], onnx.ModelProto)

def test_export_yolo_to_onnx_different_input_shapes():
    yolo_models = {
        'model1': lambda: Mock(model=Mock(fuse=lambda: Mock(eval=lambda: Mock()))),
        'model2': lambda: Mock(model=Mock(fuse=lambda: Mock(eval=lambda: Mock())))
    }
    pt_model_list = pd.DataFrame({
        'model_name': ['model1', 'model2'],
        'input_shape_h': [640, 320],
        'input_shape_w': [640, 480],
        'input_shape_c': [3, 1]
    })
    parameters = {
        'device': 'cpu',
        'opset_version': 11,
        'sim': True
    }

    result = export_yolo_to_onnx(yolo_models, pt_model_list, parameters)

    assert isinstance(result, dict)
    assert len(result) == 2
    assert 'model1' in result
    assert 'model2' in result
    assert isinstance(result['model1'], onnx.ModelProto)
    assert isinstance(result['model2'], onnx.ModelProto)
#     assert result == "Handled non-compliance", "Non-compliance handling failed."