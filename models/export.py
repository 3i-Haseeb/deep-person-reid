import torch
from torch.autograd import Variable

import torchreid

import onnx

torchreid.models.show_avai_models()

model = torchreid.models.build_model(name="osnet_x0_25", num_classes=1000)

torchreid.utils.load_pretrained_weights(
    model, "./osnet_x0_25_msmt17_combineall_256x128.pth"
)


input_name = ["input"]
output_name = ["output"]
input = Variable(torch.randn(1, 3, 256, 128))
torch.onnx.export(
    model,
    input,
    "model.onnx",
    input_names=input_name,
    output_names=output_name,
    verbose=True,
    export_params=True,
)


onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)