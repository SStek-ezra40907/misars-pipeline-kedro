import pandas as pd
from typing import Dict
import torch
import onnx
from io import BytesIO
import onnxsim
from .models.common import optim


def export_yolo_to_onnx(yolo_models: pd.DataFrame,
                        pt_model_list: pd.DataFrame,
                        parameters: Dict
                        ) -> pd.DataFrame:
    # input_shape: tuple,
    # device: str,
    # opset: int,
    # sim: bool) -> str:
    onnx_models = {}
    device = parameters["device"]
    opset = parameters["opset_version"]
    sim = parameters["sim"]
    for index, row in pt_model_list.iterrows():
        has_onnx = row['has_onnx']
        model_name = row['model_name']
        input_shape_h = row['input_shape_h']
        input_shape_w = row['input_shape_w']
        input_shape_c = row['input_shape_c']
        input_shape = (1, input_shape_c, input_shape_h, input_shape_w)
        if model_name == 'Urology_yolov11x-seg_3-13-16-17val640rezize_1_4.40.0':
            continue
        if has_onnx:
            continue
        load_func = yolo_models[model_name]
        YOLOv8 = load_func()
        model = YOLOv8.model.fuse().eval()

        for m in model.modules():
            optim(m)
            m.to(device)
        model.to(device)

        fake_input = torch.randn(input_shape).to(device)
        for _ in range(2):
            model(fake_input)

        with BytesIO() as f:
            torch.onnx.export(model,
                              fake_input,
                              f,
                              opset_version=opset,
                              input_names=['images'],
                              output_names=['outputs', 'proto'])
            f.seek(0)
            onnx_model = onnx.load(f)

        onnx.checker.check_model(onnx_model)
        if sim:
            try:
                onnx_model, check = onnxsim.simplify(onnx_model)
                assert check, 'assert check failed'
            except Exception as e:
                print(f'Simplifier failure: {e}')
        onnx_models[model_name] = onnx_model

        # onnx_models[model_name].save(onnx_model)

    return onnx_models
