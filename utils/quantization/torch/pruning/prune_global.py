import copy

import torch
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
num_params, flops = torchreid.utils.compute_model_complexity(
    model, (1, 3, 256, 128)
)
print(num_params, flops)
# print(model.state_dict().keys())


modules_weights = [
    module
    for module in filter(
        lambda m: type(m)
        in [torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d],
        model.modules(),
    )
]
modules_bias = [
    module
    for module in filter(
        lambda m: type(m) in [torch.nn.Linear, torch.nn.BatchNorm2d],
        model.modules(),
    )
]
# print(module_names)

# * Prune the model
weights_to_prune = [
    (module, "weight") for module in modules_weights[:-1]
]  # Excluding last classifier layer
bias_to_prune = [
    (module, "bias") for module in modules_bias[:-1]
]  # Excluding last classifier layer
parameters_to_prune = weights_to_prune + bias_to_prune
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.3,
)

# * Remove pruning re-parametrization
# # TEST
# # Before removing
# module = model.conv1.conv
# print(list(module.named_parameters()))
# print(list(module.named_buffers()))
# # After removing
# prune.remove(module, "weight")
# print(list(module.named_parameters()))


for module in modules_weights[:-1]:
    prune.remove(module, "weight")
for module in modules_bias[:-1]:
    prune.remove(module, "bias")

torch.save(model.state_dict(), "../../weights/model_pruned.pth.tar")
# num_params, flops = torchreid.utils.compute_model_complexity(
#     model, (1, 3, 256, 128)
# )
# print(num_params, flops)
