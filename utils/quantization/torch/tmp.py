import copy

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.ao.quantization import (
    DeQuantStub,
    QConfigMapping,
    QuantStub,
    get_default_qconfig,
)
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from torch.quantization import quantize_fx

import torchreid

# Build and load the model
loaded_model = torchreid.models.build_model(
    name="osnet_x0_25",
    num_classes=1000,
    loss="softmax",
    pretrained=False,
)
torchreid.utils.load_pretrained_weights(
    loaded_model, "../weights/model.pth.tar"
)
# model_tmp = copy.deepcopy(model_float)


# Create a new model that wraps the loaded model with QuantStub and DequantStub
# class WrappedModel(nn.Module):
#     def __init__(self, model):
#         super(WrappedModel, self).__init__()
#         self.quant = QuantStub()
#         self.model = model
#         self.dequant = DeQuantStub()

#     def forward(self, x):
#         x = self.quant(x)
#         x = self.model(x)
#         x = self.dequant(x)
#         return x


# # Wrap the loaded model with QuantStub and DequantStub
# wrapped_model = WrappedModel(loaded_model)
# model_float = copy.deepcopy(wrapped_model)
# model_float.eval()

# Prepare calibration data
normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
dataset = torchvision.datasets.ImageFolder(
    "./test_images/",
    transform=T.Compose(
        [
            T.Resize((256, 128)),
            T.ToTensor(),
            normalize,
        ]
    ),
)
sampler = torch.utils.data.SequentialSampler(dataset)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=64, sampler=sampler
)

backend = "fbgemm"

m = copy.deepcopy(loaded_model)
m.eval()

m = nn.Sequential(
    QuantStub(),
    m,
    DeQuantStub(),
)

m.qconfig = torch.quantization.get_default_qconfig(backend)
torch.quantization.prepare(m, inplace=True)


def calibrate(model, data_loader):
    model.eval()
    with torch.inference_mode():
        for image, target in data_loader:
            model(image)


calibrate(m, data_loader)


"""Convert"""
torch.quantization.convert(m, inplace=True)

torch.save(m.state_dict(), "../weights/model_quant.pth")


# # Post training static quantization
# model_float.qconfig = torch.ao.quantization.get_default_qconfig("x86")
# # model_float_fused = torch.ao.quantization.fuse_modules(
# #     model_float, [["torch.nn.Conv2d", "torch.nn.BatchNorm2d", "torch.nn.ReLU"]]
# # )
# model_float_prepared = torch.ao.quantization.prepare(model_float)

# input_float = torch.randn(4, 3, 256, 128)
# model_float(input_float)

# model_int8 = torch.ao.quantization.convert(model_float_prepared)
# print(model_int8)

# qconfig = get_default_qconfig("x86")
# qconfig_mapping = QConfigMapping().set_global(qconfig)


# def calibrate(model, data_loader):
#     model.eval()
#     with torch.no_grad():
#         for image, target in data_loader:
#             model(image)


# prepared_model = prepare_fx(
#     model_float.eval(), qconfig_mapping, example_inputs
# )  # fuse modules and insert observers
# model_quant = convert_fx(prepared_model)
# print(model_quant)
