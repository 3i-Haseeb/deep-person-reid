import onnx
from onnx_opcounter import calculate_params

model = onnx.load_model("./model_quant.onnx")
params = calculate_params(model)

print("Number of params:", params)
