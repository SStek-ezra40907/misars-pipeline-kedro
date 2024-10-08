# import pytest
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
#     assert result == "Handled non-compliance", "Non-compliance handling failed."