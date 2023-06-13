import torch
from torch.autograd import Variable

import torchreid

import onnx

torchreid.models.show_avai_models()

model = torchreid.models.build_model(name="osnet_ain_x1_0", num_classes=1000)

torchreid.utils.load_pretrained_weights(model, "./osnet_ain_x1_0_imagenet.pth")


input_name = ["input"]
output_name = ["output"]
input = Variable(torch.randn(1, 3, 256, 128))
torch.onnx.export(
    model,
    input,
    "osnet_ain_x1_0.onnx",
    input_names=input_name,
    output_names=output_name,
    verbose=True,
    export_params=True,
)


onnx_model = onnx.load("osnet_ain_x1_0.onnx")
onnx.checker.check_model(onnx_model)
