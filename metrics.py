import numpy as np
from numpy.linalg import norm

from torchreid.utils import FeatureExtractor

from scipy.spatial import distance

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


def pearson_distance(arr1, arr2):
    dist = np.corrcoef(arr1, arr2)
    return dist[0][1]


def hamming_distance(arr1, arr2):
    dist = distance.hamming(arr1, arr2)
    return dist


def minkowski_distance(arr1, arr2, p=2):
    dist = distance.minkowski(arr1, arr2)
    return dist


def chebyshev_distance(arr1, arr2):
    dist = distance.chebyshev(arr1, arr2)
    return dist


def correlation(arr1, arr2):
    dist = distance.correlation(arr1, arr2)
    return dist


if __name__ == "__main__":
    extractor = FeatureExtractor(
        model_name="osnet_x0_25",
        model_path="./models/model.pth.tar",
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
