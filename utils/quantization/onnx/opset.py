import onnx

model_path = "../weights/model-infer.onnx"
onnx_model = onnx.load(model_path)
print(onnx_model)
