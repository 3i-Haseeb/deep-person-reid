import torch
import torchvision.transforms as T
from PIL import Image

import os
import onnx
import time
import onnxruntime as ort

# Define model names
model = "model.onnx"
model_quant = "model_quant.onnx"
models = [model, model_quant]

# Define config
test_dir = "./test_images"
pixel_mean = [0.485, 0.456, 0.406]
pixel_std = [0.229, 0.224, 0.225]
transforms = []
transforms += [T.Resize((256, 128))]
transforms += [T.ToTensor()]
transforms += [T.Normalize(mean=pixel_mean, std=pixel_std)]
preprocess = T.Compose(transforms)

# Load models
for i, model in enumerate(models):
    onnx_model = onnx.load(f"./{model}")
    onnx.checker.check_model(model)
    ort_sess = ort.InferenceSession(f"./{model}")

    time_list = []

    for path in os.listdir(test_dir):
        fullpath = os.path.join(test_dir, path)
        image = Image.open(fullpath).convert("RGB")

        img = preprocess(image)
        img = torch.unsqueeze(img, 0)

        start = time.perf_counter()
        outputs = ort_sess.run(None, {"input": img.numpy()})
        end = time.perf_counter()

        print(f"Inference time for image {path} = {end-start}")
        time_list.append(end - start)

    print(f"Average inference time = {sum(time_list) / len(time_list)}")
