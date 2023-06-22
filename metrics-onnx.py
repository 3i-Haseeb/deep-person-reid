import numpy as np
from numpy.linalg import norm
import torch
import torchvision.transforms as T
from PIL import Image

import glob
import onnx
import onnxruntime as ort


def cosine_similarity(arr1, arr2):
    dist = np.dot(arr1, arr2) / (norm(arr1) * norm(arr2))
    return dist


if __name__ == "__main__":
    onnx_model = onnx.load("./models/model.onnx")
    onnx.checker.check_model(onnx_model)

    ort_sess = ort.InferenceSession("./models/model.onnx")
    images = sorted(glob.glob("../test-images/*"))

    pixel_mean = [0.485, 0.456, 0.406]
    pixel_std = [0.229, 0.224, 0.225]
    transforms = []
    transforms += [T.Resize((256, 128))]
    transforms += [T.ToTensor()]
    transforms += [T.Normalize(mean=pixel_mean, std=pixel_std)]
    preprocess = T.Compose(transforms)

    features = []

    for image in images:
        img = Image.open(image).convert("RGB")
        img = preprocess(img)
        img = torch.unsqueeze(img, 0)

        res = ort_sess.run(None, {"input": img.numpy()})
        res = np.array(res).squeeze(axis=0).squeeze(axis=0)

        features.append(res)

    features = np.array(features)
    for i in range(0, 6):
        dist_diff = cosine_similarity(features[4], features[i])
        print(f"Distance: {dist_diff}")
