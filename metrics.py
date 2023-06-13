import numpy as np
from numpy.linalg import norm

from torchreid.utils import FeatureExtractor

# Define distance metrics


def euclidean_distance(arr1, arr2):
    dist = np.linalg.norm(arr1 - arr2)
    return dist


def manhattan_distance(arr1, arr2):
    dist = np.sum(np.abs(arr1 - arr2))
    return dist


def cosine_similarity(arr1, arr2):
    dist = np.dot(arr1, arr2) / (norm(arr1) * norm(arr2))
    return dist


if __name__ == "__main__":
    extractor = FeatureExtractor(
        model_name="osnet_ain_x1_0",
        model_path="./models/osnet_ain_x1_0_imagenet.pth",
        device="cpu",
    )

    image_list = [
        "../test-images/1.jpeg",
        "../test-images/2.jpeg",
        "../test-images/3.jpeg",
        "../test-images/4.jpeg",
        "../test-images/5.jpeg",
        "../test-images/6.jpeg",
    ]

    features = extractor(image_list).numpy()
    # print(features)

    # dist_same = euclidean_distance(features[0], features[0])
    # print(f"Same person: {dist_same}")

    for i in range(0, 6):
        dist_diff = cosine_similarity(features[0], features[i])
        print(f"Distance: {dist_diff}")
