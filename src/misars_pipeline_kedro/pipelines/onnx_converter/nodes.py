import pandas as pd
import onnx_graphsurgeon as gs


def convert_onnx_to_nhwc(onnx_models: pd.DataFrame, onnx_model_list: pd.DataFrame):
    converted_models = {}
    for model_name, model_load_func in onnx_models.items():
        model = model_load_func()

        graph = gs.import_onnx(model)

        # Update graph input name
        graph.inputs[0].name += "_old"

        # Insert a transpose node
        nhwc_to_nchw_in = gs.Node("Transpose", name="transpose_input", attrs={"perm": [0, 3, 1, 2]})
        nhwc_to_nchw_in.outputs = graph.inputs
        model_info = onnx_model_list.loc[onnx_model_list['model_name'] == model_name].iloc[0]
        h = int(model_info["input_shape_h"])
        w = int(model_info["input_shape_w"])
        c = int(model_info["input_shape_c"])

        # Create new input with NHWC shape
        new_input = gs.Variable("INPUT__0", dtype=graph.inputs[0].dtype, shape=[1, h, w, c])
        graph.inputs = [new_input]
        nhwc_to_nchw_in.inputs = graph.inputs

        # Add the transpose node to the graph
        graph.nodes.extend([nhwc_to_nchw_in])

        # Clean up and sort the graph
        graph.cleanup().toposort()

        # Export the modified graph
        converted_model = gs.export_onnx(graph)
        converted_models[f"{model_name}_nhwc"] = converted_model

    return converted_models

# def get_pending_models(preprocessed_models: pd.DataFrame, onnx_models: pd.DataFrame):
#     preprocessed_keys = set(key.replace('_nhwc', '') for key in preprocessed_models.keys())
#     pending_model_keys = set(onnx_models.keys()) - preprocessed_keys
#     pending_models = onnx_models[onnx_models.index.isin(pending_model_keys)]
#     return pending_models

# def get_converted_models(preprocessed_models: pd.DataFrame, onnx_models: pd.DataFrame):
#     preprocessed_models = {}
#     onnx_models = set(onnx_models.keys())- set(preprocessed_models.keys())
#     print(f"converted_models: {preprocessed_models}")
#     for model_name, converted_model in preprocessed_models.items():
#         preprocessed_models[model_name] = converted_model
#     return preprocessed_models
