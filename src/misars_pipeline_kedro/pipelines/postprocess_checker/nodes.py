import pandas as pd

import onnx_graphsurgeon as gs


def filter_onnx_model(model_list: pd.DataFrame) -> pd.DataFrame:
    model_list = model_list.loc[model_list['ext'] == '.onnx']
    return model_list


def get_onnx_input_shape(onnx_models: pd.DataFrame, onnx_model_list: pd.DataFrame):
    for partition_id, partition_load_func in onnx_models.items():
        loaded_model = partition_load_func()
        input_shape, _ = get_model_io_shapes(loaded_model)
        input_layout = get_layout_from_shape(input_shape)
        input_shape = input_shape[0]
        h, w, c = get_hw_c_from_shape(input_shape, input_layout)
        onnx_model_list.loc[onnx_model_list['model_name'] == partition_id, 'input_layout'] = input_layout
        onnx_model_list.loc[onnx_model_list['model_name'] == partition_id, 'input_shape_h'] = h
        onnx_model_list.loc[onnx_model_list['model_name'] == partition_id, 'input_shape_w'] = w
        onnx_model_list.loc[onnx_model_list['model_name'] == partition_id, 'input_shape_c'] = c
    onnx_model_list["input_shape_h"] = onnx_model_list["input_shape_h"].astype(int)
    onnx_model_list["input_shape_w"] = onnx_model_list["input_shape_w"].astype(int)
    onnx_model_list["input_shape_c"] = onnx_model_list["input_shape_c"].astype(int)
    print(f"onnx_model_list: {onnx_model_list}")
    return onnx_model_list


def filter_nchw_model(onnx_model_list: pd.DataFrame):
    nchw_model_list = onnx_model_list.loc[onnx_model_list['input_layout'] == 'NCHW']
    return nchw_model_list


def filter_nhwc_model(onnx_model_list: pd.DataFrame):
    nhwc_model_list = onnx_model_list.loc[onnx_model_list['input_layout'] == 'NHWC']
    return nhwc_model_list


def convert_onnx_to_nhwc(onnx_models: pd.DataFrame, onnx_model_list: pd.DataFrame):
    converted_models = {}
    for model_info in onnx_model_list:
        model_id, model_load_func = onnx_models[model_info].items()
        model = model_load_func()

        graph = gs.import_onnx(model)

        # Update graph input name
        graph.inputs[0].name += "_old"

        # Insert a transpose node
        nhwc_to_nchw_in = gs.Node("Transpose", name="transpose_input", attrs={"perm": [0, 3, 1, 2]})
        nhwc_to_nchw_in.outputs = graph.inputs

        h = model_info["input_shape_h"]
        w = model_info["input_shape_w"]
        c = model_info["input_shape_c"]

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
        converted_models[model_id] = converted_model

    return converted_models


def get_model_io_shapes(model):
    # Get input shapes
    input_shapes = []
    for input_tensor in model.graph.input:
        input_shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        input_shapes.append((input_tensor.name, input_shape))

    # Get output shapes
    output_shapes = []
    for output_tensor in model.graph.output:
        output_shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
        output_shapes.append((output_tensor.name, output_shape))

    return input_shapes, output_shapes


def get_layout_from_shape(model_shape):
    """
    Determine the data layout (NCHW or NHWC) based on the input shape list.

    Args:
        model_shape (list): A list representing the model input shape.

    Returns:
        str: Layout type ('NCHW', 'NHWC', or 'UNKNOWN').
    """

    name, shape = model_shape[0]

    return (
        "NCHW" if len(shape) >= 4 and shape[1] == 3 else
        "NHWC" if len(shape) >= 4 and shape[-1] == 3 else
        "UNKNOWN"
    )


def get_hw_c_from_shape(input_shape, input_layout) -> tuple[int, int, int]:
    """
    Extract Height, Width, and Channels from the input shape based on the given layout.

    Args:
        input_shape (list): A list representing the shape, e.g., [1, 3, 640, 640].
        input_layout (str): The data layout, either 'NCHW' or 'NHWC'.

    Returns:
        tuple[int, int, int]: A tuple containing (Height, Width, Channels).

    Raises:
        ValueError: If an unsupported input layout is provided.
    """
    name, shape = input_shape

    if input_layout == 'NCHW':
        # NCHW format: [Batch, Channels, Height, Width]
        N, C, H, W = shape
    elif input_layout == 'NHWC':
        # NHWC format: [Batch, Height, Width, Channels]
        N, H, W, C = shape
    else:
        raise ValueError("Unsupported input layout. Use 'NCHW' or 'NHWC'.")

    return int(H), int(W), int(C)
