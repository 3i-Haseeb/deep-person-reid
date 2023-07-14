import copy

import torch
import torchvision
import torchvision.transforms as T
from nni.algorithms.compression.pytorch.quantization import ObserverQuantizer

import torchreid

# Build and load the model
model_float = torchreid.models.build_model(
    name="osnet_x0_25",
    num_classes=1000,
    loss="softmax",
    pretrained=False,
)
torchreid.utils.load_pretrained_weights(model_float, "../weights/model.pth.tar")


def get_module_names(module, parent_name="", module_names=None):
    if module_names is None:
        module_names = []

    # Check if the module is a Conv, FC, or ReLU module
    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.ReLU)):
        module_names.append(parent_name)

    # Recursive traversal of child modules
    for name, child_module in module.named_children():
        if parent_name == "":
            child_name = name
        else:
            child_name = f"{parent_name}.{name}"
        get_module_names(child_module, child_name, module_names)

    return module_names


# Get the names of conv, fc, and relu modules
module_names = get_module_names(model_float)

# Print the list of module names
# print(module_names)
# print(model_float)

model_tmp = copy.deepcopy(model_float)

# Prepare calibration data
normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
dataset = torchvision.datasets.ImageFolder(
    "../calibration_dataset/",
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
    dataset, batch_size=150, sampler=sampler
)


def calibration(model, data_loader):
    model.eval()
    with torch.no_grad():
        for image, target in data_loader:
            model(image)


configure_list = [
    {
        "quant_bits": {"weight": 8, "input": 8},
        "quant_types": ["weight", "input"],
        "op_types": ["Conv2d", "Linear"],
        "op_names": [
            "conv1.conv",
            "conv1.relu",
            "conv2.0.conv1.conv",
            "conv2.0.conv1.relu",
            "conv2.0.conv2a.conv1",
            "conv2.0.conv2a.conv2",
            "conv2.0.conv2a.relu",
            "conv2.0.conv2b.0.conv1",
            "conv2.0.conv2b.0.conv2",
            "conv2.0.conv2b.0.relu",
            "conv2.0.conv2b.1.conv1",
            "conv2.0.conv2b.1.conv2",
            "conv2.0.conv2b.1.relu",
            "conv2.0.conv2c.0.conv1",
            "conv2.0.conv2c.0.conv2",
            "conv2.0.conv2c.0.relu",
            "conv2.0.conv2c.1.conv1",
            "conv2.0.conv2c.1.conv2",
            "conv2.0.conv2c.1.relu",
            "conv2.0.conv2c.2.conv1",
            "conv2.0.conv2c.2.conv2",
            "conv2.0.conv2c.2.relu",
            "conv2.0.conv2d.0.conv1",
            "conv2.0.conv2d.0.conv2",
            "conv2.0.conv2d.0.relu",
            "conv2.0.conv2d.1.conv1",
            "conv2.0.conv2d.1.conv2",
            "conv2.0.conv2d.1.relu",
            "conv2.0.conv2d.2.conv1",
            "conv2.0.conv2d.2.conv2",
            "conv2.0.conv2d.2.relu",
            "conv2.0.conv2d.3.conv1",
            "conv2.0.conv2d.3.conv2",
            "conv2.0.conv2d.3.relu",
            "conv2.0.gate.fc1",
            "conv2.0.gate.relu",
            "conv2.0.gate.fc2",
            "conv2.0.conv3.conv",
            "conv2.0.downsample.conv",
            "conv2.1.conv1.conv",
            "conv2.1.conv1.relu",
            "conv2.1.conv2a.conv1",
            "conv2.1.conv2a.conv2",
            "conv2.1.conv2a.relu",
            "conv2.1.conv2b.0.conv1",
            "conv2.1.conv2b.0.conv2",
            "conv2.1.conv2b.0.relu",
            "conv2.1.conv2b.1.conv1",
            "conv2.1.conv2b.1.conv2",
            "conv2.1.conv2b.1.relu",
            "conv2.1.conv2c.0.conv1",
            "conv2.1.conv2c.0.conv2",
            "conv2.1.conv2c.0.relu",
            "conv2.1.conv2c.1.conv1",
            "conv2.1.conv2c.1.conv2",
            "conv2.1.conv2c.1.relu",
            "conv2.1.conv2c.2.conv1",
            "conv2.1.conv2c.2.conv2",
            "conv2.1.conv2c.2.relu",
            "conv2.1.conv2d.0.conv1",
            "conv2.1.conv2d.0.conv2",
            "conv2.1.conv2d.0.relu",
            "conv2.1.conv2d.1.conv1",
            "conv2.1.conv2d.1.conv2",
            "conv2.1.conv2d.1.relu",
            "conv2.1.conv2d.2.conv1",
            "conv2.1.conv2d.2.conv2",
            "conv2.1.conv2d.2.relu",
            "conv2.1.conv2d.3.conv1",
            "conv2.1.conv2d.3.conv2",
            "conv2.1.conv2d.3.relu",
            "conv2.1.gate.fc1",
            "conv2.1.gate.relu",
            "conv2.1.gate.fc2",
            "conv2.1.conv3.conv",
            "conv2.2.0.conv",
            "conv2.2.0.relu",
            "conv3.0.conv1.conv",
            "conv3.0.conv1.relu",
            "conv3.0.conv2a.conv1",
            "conv3.0.conv2a.conv2",
            "conv3.0.conv2a.relu",
            "conv3.0.conv2b.0.conv1",
            "conv3.0.conv2b.0.conv2",
            "conv3.0.conv2b.0.relu",
            "conv3.0.conv2b.1.conv1",
            "conv3.0.conv2b.1.conv2",
            "conv3.0.conv2b.1.relu",
            "conv3.0.conv2c.0.conv1",
            "conv3.0.conv2c.0.conv2",
            "conv3.0.conv2c.0.relu",
            "conv3.0.conv2c.1.conv1",
            "conv3.0.conv2c.1.conv2",
            "conv3.0.conv2c.1.relu",
            "conv3.0.conv2c.2.conv1",
            "conv3.0.conv2c.2.conv2",
            "conv3.0.conv2c.2.relu",
            "conv3.0.conv2d.0.conv1",
            "conv3.0.conv2d.0.conv2",
            "conv3.0.conv2d.0.relu",
            "conv3.0.conv2d.1.conv1",
            "conv3.0.conv2d.1.conv2",
            "conv3.0.conv2d.1.relu",
            "conv3.0.conv2d.2.conv1",
            "conv3.0.conv2d.2.conv2",
            "conv3.0.conv2d.2.relu",
            "conv3.0.conv2d.3.conv1",
            "conv3.0.conv2d.3.conv2",
            "conv3.0.conv2d.3.relu",
            "conv3.0.gate.fc1",
            "conv3.0.gate.relu",
            "conv3.0.gate.fc2",
            "conv3.0.conv3.conv",
            "conv3.0.downsample.conv",
            "conv3.1.conv1.conv",
            "conv3.1.conv1.relu",
            "conv3.1.conv2a.conv1",
            "conv3.1.conv2a.conv2",
            "conv3.1.conv2a.relu",
            "conv3.1.conv2b.0.conv1",
            "conv3.1.conv2b.0.conv2",
            "conv3.1.conv2b.0.relu",
            "conv3.1.conv2b.1.conv1",
            "conv3.1.conv2b.1.conv2",
            "conv3.1.conv2b.1.relu",
            "conv3.1.conv2c.0.conv1",
            "conv3.1.conv2c.0.conv2",
            "conv3.1.conv2c.0.relu",
            "conv3.1.conv2c.1.conv1",
            "conv3.1.conv2c.1.conv2",
            "conv3.1.conv2c.1.relu",
            "conv3.1.conv2c.2.conv1",
            "conv3.1.conv2c.2.conv2",
            "conv3.1.conv2c.2.relu",
            "conv3.1.conv2d.0.conv1",
            "conv3.1.conv2d.0.conv2",
            "conv3.1.conv2d.0.relu",
            "conv3.1.conv2d.1.conv1",
            "conv3.1.conv2d.1.conv2",
            "conv3.1.conv2d.1.relu",
            "conv3.1.conv2d.2.conv1",
            "conv3.1.conv2d.2.conv2",
            "conv3.1.conv2d.2.relu",
            "conv3.1.conv2d.3.conv1",
            "conv3.1.conv2d.3.conv2",
            "conv3.1.conv2d.3.relu",
            "conv3.1.gate.fc1",
            "conv3.1.gate.relu",
            "conv3.1.gate.fc2",
            "conv3.1.conv3.conv",
            "conv3.2.0.conv",
            "conv3.2.0.relu",
            "conv4.0.conv1.conv",
            "conv4.0.conv1.relu",
            "conv4.0.conv2a.conv1",
            "conv4.0.conv2a.conv2",
            "conv4.0.conv2a.relu",
            "conv4.0.conv2b.0.conv1",
            "conv4.0.conv2b.0.conv2",
            "conv4.0.conv2b.0.relu",
            "conv4.0.conv2b.1.conv1",
            "conv4.0.conv2b.1.conv2",
            "conv4.0.conv2b.1.relu",
            "conv4.0.conv2c.0.conv1",
            "conv4.0.conv2c.0.conv2",
            "conv4.0.conv2c.0.relu",
            "conv4.0.conv2c.1.conv1",
            "conv4.0.conv2c.1.conv2",
            "conv4.0.conv2c.1.relu",
            "conv4.0.conv2c.2.conv1",
            "conv4.0.conv2c.2.conv2",
            "conv4.0.conv2c.2.relu",
            "conv4.0.conv2d.0.conv1",
            "conv4.0.conv2d.0.conv2",
            "conv4.0.conv2d.0.relu",
            "conv4.0.conv2d.1.conv1",
            "conv4.0.conv2d.1.conv2",
            "conv4.0.conv2d.1.relu",
            "conv4.0.conv2d.2.conv1",
            "conv4.0.conv2d.2.conv2",
            "conv4.0.conv2d.2.relu",
            "conv4.0.conv2d.3.conv1",
            "conv4.0.conv2d.3.conv2",
            "conv4.0.conv2d.3.relu",
            "conv4.0.gate.fc1",
            "conv4.0.gate.relu",
            "conv4.0.gate.fc2",
            "conv4.0.conv3.conv",
            "conv4.0.downsample.conv",
            "conv4.1.conv1.conv",
            "conv4.1.conv1.relu",
            "conv4.1.conv2a.conv1",
            "conv4.1.conv2a.conv2",
            "conv4.1.conv2a.relu",
            "conv4.1.conv2b.0.conv1",
            "conv4.1.conv2b.0.conv2",
            "conv4.1.conv2b.0.relu",
            "conv4.1.conv2b.1.conv1",
            "conv4.1.conv2b.1.conv2",
            "conv4.1.conv2b.1.relu",
            "conv4.1.conv2c.0.conv1",
            "conv4.1.conv2c.0.conv2",
            "conv4.1.conv2c.0.relu",
            "conv4.1.conv2c.1.conv1",
            "conv4.1.conv2c.1.conv2",
            "conv4.1.conv2c.1.relu",
            "conv4.1.conv2c.2.conv1",
            "conv4.1.conv2c.2.conv2",
            "conv4.1.conv2c.2.relu",
            "conv4.1.conv2d.0.conv1",
            "conv4.1.conv2d.0.conv2",
            "conv4.1.conv2d.0.relu",
            "conv4.1.conv2d.1.conv1",
            "conv4.1.conv2d.1.conv2",
            "conv4.1.conv2d.1.relu",
            "conv4.1.conv2d.2.conv1",
            "conv4.1.conv2d.2.conv2",
            "conv4.1.conv2d.2.relu",
            "conv4.1.conv2d.3.conv1",
            "conv4.1.conv2d.3.conv2",
            "conv4.1.conv2d.3.relu",
            "conv4.1.gate.fc1",
            "conv4.1.gate.relu",
            "conv4.1.gate.fc2",
            "conv4.1.conv3.conv",
            "conv5.conv",
            "conv5.relu",
            "fc.0",
            "fc.2",
            "classifier",
        ],
    },
]
optimizer = torch.optim.SGD(model_tmp.parameters(), lr=0.01, momentum=0.5)

quantizer = ObserverQuantizer(model_tmp.eval(), configure_list, optimizer)
calibration(model_tmp, data_loader)
quantizer.compress()

model_path = "./model_quant.pth"
calibration_path = "./model_calibration.pth"
calibration_config = quantizer.export_model(model_path, calibration_path)
print("calibration_config: ", calibration_config)
