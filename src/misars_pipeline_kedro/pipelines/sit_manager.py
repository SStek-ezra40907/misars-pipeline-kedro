from kedro.pipeline import node, Pipeline

def check_onnx_version(model):
    # 檢查模型是否存在 ONNX 版本
    # 這裡加入檢查的邏輯，返回 ONNX 模型或 None
    pass

def convert_to_onnx(model):
    # 將 .pt 模型轉換為 ONNX 模型
    pass

def validate_onnx_model(onnx_model):
    # 驗證 ONNX 模型的規格
    pass

def trigger_testing_pipeline(onnx_model):
    # 觸發測試流程
    pass

def handle_non_compliance(onnx_model):
    # 處理不符合規格的情況
    pass

def create_sit_manager_pipeline():
    return Pipeline(
        [
            # node(check_onnx_version, inputs="model", outputs="onnx_model"),
            # node(
            #     convert_to_onnx,
            #     inputs=["model", "onnx_model"],
            #     outputs=None,
            #     name="convert_node",
            #     tags={"run_if": "onnx_model is None"},
            # ),
            # node(
            #     validate_onnx_model,
            #     inputs="onnx_model",
            #     outputs="validation_result",
            # ),
            # node(
            #     trigger_testing_pipeline,
            #     inputs=["onnx_model", "validation_result"],
            #     outputs=None,
            #     name="testing_node",
            #     tags={"run_if": "validation_result is True"},
            # ),
            # node(
            #     handle_non_compliance,
            #     inputs="onnx_model",
            #     outputs=None,
            #     name="non_compliance_node",
            #     tags={"run_if": "validation_result is False"},
            # ),
        ]
    )