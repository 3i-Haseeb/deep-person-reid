from torchreid.utils import FeatureExtractor

if __name__ == "__main__":
    extractor = FeatureExtractor(
        model_name="osnet_x0_25",
        model_path="../quantization/weights/model.pth.tar",
        device="cpu",
    )

    image_list = [
        "../quantization/onnx/test_images/0007_c2s3_070952_01.jpg",
    ]

    features = extractor(image_list).numpy()
    print(features)
