# convert_to_onnx.py
import torch
import torch.onnx
from pytorch_model import Classifier, BasicBlock
import os

# Constants
MODEL_PATH = "weights/pytorch_model_weights.pth"
ONNX_PATH = "model_output/model.onnx"
DUMMY_INPUT_SHAPE = (1, 3, 224, 224)


def load_model():
    model = Classifier(BasicBlock, [2, 2, 2, 2])
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def convert_to_onnx(model):
    dummy_input = torch.randn(*DUMMY_INPUT_SHAPE)
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=["input"],
        output_names=["output"],
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Exported model to {ONNX_PATH}")


if __name__ == "__main__":
    os.makedirs("model_output", exist_ok=True)
    model = load_model()
    convert_to_onnx(model)
