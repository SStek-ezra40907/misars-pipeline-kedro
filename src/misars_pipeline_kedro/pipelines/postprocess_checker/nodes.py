import pandas as pd
import re


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


def get_pt_input_shape_by_name(pt_models: pd.DataFrame, pt_model_list: pd.DataFrame):
    pattern = r"(\d{3})re[sz]{1}ize"
    pt_model_list['matches'] = pt_model_list['model_name'].str.extract(pattern).astype(int)
    pt_model_list['input_shape_h'] = pt_model_list['matches']
    pt_model_list['input_shape_w'] = pt_model_list['matches']
    pt_model_list['input_shape_c'] = 3
    pt_model_list['input_layout'] = 'NCHW'
    pt_model_list.drop(columns=['matches'], inplace=True)
    return pt_model_list


def filter_nchw_model(onnx_model_list: pd.DataFrame):
    nchw_model_list = onnx_model_list.loc[onnx_model_list['input_layout'] == 'NCHW']
    return nchw_model_list


def filter_nhwc_model(onnx_model_list: pd.DataFrame):
    nhwc_model_list = onnx_model_list.loc[onnx_model_list['input_layout'] == 'NHWC']
    return nhwc_model_list


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


def get_unconverted_onnx_models(onnx_models: dict, merged_model_list: pd.DataFrame):
    onnx_model_names = set(onnx_models.keys())
    converted_model_names = set(merged_model_list['model_name'])

    unconverted_models = {}
    for model_name in onnx_model_names:
        if model_name.endswith('_nhwc'):
            continue  # 跳過已經帶有 _nhwc 後綴的模型
        if f"{model_name}_nhwc" in converted_model_names:
            continue  # 如果存在對應的 _nhwc 版本，則視為已轉換
        unconverted_models[model_name] = onnx_models[model_name]

    print(f"unconverted_models: {unconverted_models}")
    return unconverted_models

# def get_unconverted_pt_models(pt_models: dict, pt_model_list: pd.DataFrame):
#     pt_model_names = set(model_name.replace('_nhwc', '') for model_name in pt_models.keys())
#     converted_model_names = set(model_name.replace('_nhwc', '') for model_name in pt_model_list['model_name'])
#     print(f"converted_model_names: {converted_model_names}")
#     unconverted_model_keys = pt_model_names - converted_model_names
#     print(f"unconverted_models: {unconverted_model_keys}")
#     unconverted_models = {k: v for k, v in pt_models.items() if k in unconverted_model_keys}
#     return unconverted_models

def merge_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Merges two DataFrames with the same format.

    Args:
        df1 (pd.DataFrame): The first DataFrame
        df2 (pd.DataFrame): The second DataFrame

    Returns:
        pd.DataFrame: The merged DataFrame
    """
    merged_df = pd.concat([df1, df2], ignore_index=True)
    return merged_df