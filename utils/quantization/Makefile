preprocess:
	python -m onnxruntime.quantization.preprocess --input ./model.onnx --output ./model-infer.onnx

quantize:
	python run.py --input_model model-infer.onnx --output_model model_quant.onnx --calibrate_dataset ./test_images/