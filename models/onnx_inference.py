import torch
import torchvision.transforms as T
from PIL import Image

import onnx
import onnxruntime as ort

onnx_model = onnx.load("./model.onnx")
onnx.checker.check_model(onnx_model)

ort_sess = ort.InferenceSession("./model.onnx")

image = Image.open("../../test-images/1.jpeg").convert("RGB")

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

outputs = ort_sess.run(None, {"input": img.numpy()})

print(outputs)
