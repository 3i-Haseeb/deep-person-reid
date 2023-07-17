# import copy

import torch

# import torchvision
# import torchvision.transforms as T
import torch.nn.utils.prune as prune

import torchreid

# Build and load the model
model = torchreid.models.build_model(
    name="osnet_x0_25",
    num_classes=1000,
    loss="softmax",
    pretrained=False,
)
torchreid.utils.load_pretrained_weights(model, "../../weights/model.pth.tar")

# module = model.conv1.conv
# print(list(module.named_parameters()))
# print(list(module.named_buffers()))

# prune.random_unstructured(module, name="weight", amount=0.4)
# print(list(module.named_buffers()))
# print(list(module.named_buffers()))
# print(module._forward_pre_hooks)

modules = [name for name, module in model.named_modules()]
# print(modules)

for module in modules:
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name="weight", amount=0.3)
        prune.l1_unstructured(module, name="bias", amount=0.3)
    # prune 40% of connections in all linear layers
    elif isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name="weight", amount=0.5)
        prune.l1_unstructured(module, name="bias", amount=0.5)

# print(dict(model.named_buffers()).keys())  # to verify that all masks exist

print(model)
