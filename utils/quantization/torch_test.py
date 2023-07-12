import os
import time

import numpy as np

from torchreid.utils import FeatureExtractor

model = "model.pth.tar"
# model_quant = "model_quant.onnx"
models = [model]
test_dir = "./test_images/"
image_list = os.listdir(test_dir)

for model in models:
    extractor = FeatureExtractor(
        model_name="osnet_x0_25",
        model_path=f"./{model}",
        device="cpu",
    )

    time_list = []

    for path in image_list:
        img = []
        fullpath = os.path.join(test_dir, path)
        img.append(fullpath)

        start = time.perf_counter()
        features = extractor(img)
        end = time.perf_counter()

        print(f"Inference time for image {path} = {end-start}")
        time_list.append(end - start)

    print(f"Average inference time = {sum(time_list) / len(time_list)}")
