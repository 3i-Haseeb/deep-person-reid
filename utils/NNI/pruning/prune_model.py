import torch
from nni.compression.pytorch.pruning import L1NormPruner
from nni.compression.pytorch.speedup import ModelSpeedup

import torchreid

# Build and load the model
model = torchreid.models.build_model(
    name="osnet_x0_25",
    num_classes=1000,
    loss="softmax",
    pretrained=False,
)
torchreid.utils.load_pretrained_weights(model, "../weights/model.pth.tar")

config_list = [
    {"sparsity_per_layer": 0.5, "op_types": ["Linear", "Conv2d"]},
    {"exclude": True, "op_names": ["classifier"]},
]

pruner = L1NormPruner(model, config_list)
# print(model)

# compress the model and generate the masks
_, masks = pruner.compress()

# for name, mask in masks.items():
#     print(
#         name,
#         " sparsity : ",
#         "{:.2}".format(mask["weight"].sum() / mask["weight"].numel()),
#     )

# need to unwrap the model, if the model is wrapped before speedup
pruner._unwrap_model()

ModelSpeedup(model, torch.rand(1, 3, 256, 128), masks).speedup_model()
print(model)
