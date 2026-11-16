# Loads every converted model and runs it once on random input, printing input/output shapes.
import sys, glob, numpy as np, onnxruntime as ort
for f in sorted(glob.glob(sys.argv[1] + '/*.onnx')):
    s = ort.InferenceSession(f, providers=['CPUExecutionProvider'])
    i = s.get_inputs()[0]
    y = s.run(None, {i.name: np.random.rand(*i.shape).astype(np.float32)})[0]
    print(f, i.shape, '->', y.shape, float(y.min()), float(y.max()))
