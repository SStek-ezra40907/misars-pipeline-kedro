from kedro.pipeline import node, Pipeline


def create_pipeline():
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

