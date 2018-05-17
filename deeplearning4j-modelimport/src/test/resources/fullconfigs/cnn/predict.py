from keras.models import load_model
import numpy as np

np.random.seed(1337)

model = load_model("cnn_batch_norm.h5")
input = np.random.random((5, 10, 10, 3))

output = model.predict(input)

assert abs(-0.0520611  - output[0][0]) < 0.000001
assert abs(0.04986075  - output[1][0]) < 0.000001
assert abs(0.16297522  - output[2][0]) < 0.000001
assert abs(0.15389983  - output[3][0]) < 0.000001
assert abs(0.15537278  - output[4][0]) < 0.000001

np.save(arr=input, file="input.npy")
np.save(arr=output, file="predictions.npy")
