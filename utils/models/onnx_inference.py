import torch
import torchvision.transforms as T
from PIL import Image

import onnx
import time
import onnxruntime as ort

model_name = "model.onnx"

onnx_model = onnx.load(f"./{model_name}")
onnx.checker.check_model(onnx_model)

ort_sess = ort.InferenceSession(f"./{model_name}")

image = Image.open(
    "../quantization/test_images/0007_c2s3_070952_01.jpg"
).convert("RGB")

pixel_mean = [0.485, 0.456, 0.406]
pixel_std = [0.229, 0.224, 0.225]
transforms = []
transforms += [T.Resize((256, 128))]
transforms += [T.ToTensor()]
transforms += [T.Normalize(mean=pixel_mean, std=pixel_std)]
preprocess = T.Compose(transforms)

img = preprocess(image)
img = torch.unsqueeze(img, 0)
print(img.shape)

start = time.perf_counter()
outputs = ort_sess.run(None, {"input": img.numpy()})
end = time.perf_counter()

print(outputs)
print(f"Inference time = {end-start}")
