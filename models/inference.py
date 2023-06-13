import numpy as np

from torchreid.utils import FeatureExtractor

if __name__ == "__main__":
    extractor = FeatureExtractor(
        model_name="osnet_x0_25",
        model_path="./osnet_x0_25_msmt17_combineall_256x128.pth",
        device="cpu",
    )

    image_list = [
        "../../test-images/1.jpeg",
    ]

    features = extractor(image_list).numpy()
    print(features)
