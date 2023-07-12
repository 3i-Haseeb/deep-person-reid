from onnx_opcounter import calculate_params

import onnx

model = onnx.load_model("./model-infer.onnx")
params = calculate_params(model)

print("Number of params:", params)
