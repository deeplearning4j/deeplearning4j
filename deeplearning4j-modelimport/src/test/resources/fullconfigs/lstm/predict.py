from keras.models import model_from_json
import pandas as pd
import numpy as np

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
