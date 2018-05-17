from keras.models import model_from_json
import pandas as pd
import numpy as np

# unit_forget_bias: Boolean. If True, add 1 to the bias of the forget gate at initialization. Setting it to true will also force  bias_initializer="zeros". This is recommended in Jozefowicz et al.

# TODO: stateful: Boolean (default False). If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.
# TODO: return_state: Boolean. Whether to return the last state in addition to the output.
# TODO: go_backwards: Boolean (default False). If True, process the input sequence backwards and return the reversed sequence.

# Keras: IFCO
# DL4J:  IFOG

model = None
with open("lstm_th_keras_2_config.json", "r") as f:
    model = model_from_json("\n".join(f.readlines()))

model.load_weights("lstm_th_keras_2_weights.h5")

preds = []

for i in range(0, 283):
    df = pd.read_csv("data/sequences/{}.csv".format(i), sep=";", header=None)
    preds.append(model.predict(np.array([df.values[:, :12]])))

predictions = np.array(preds).flatten()
np.save(arr=predictions, file="predictions.npy")