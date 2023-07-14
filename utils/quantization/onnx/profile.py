import onnx_tool

modelpath = "../weights/model_quant.onnx"
onnx_tool.model_profile(modelpath)
