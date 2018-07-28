# Getting started with importing Keras Sequential models

## Persisting your Keras model

Let's say you start with defining a simple MLP using Keras:

```python
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='sgd', metrics=['accuracy'])
```

In Keras there are several ways to save a model. You can store the whole model 
(model definition, weights and training configuration) as HDF5 file, just the
model configuration (as JSON or YAML file) or just the weights (as HDF5 file). 
Here's how you do each:

```python
model.save('full_model.h5')  # save everything in HDF5 format

model_json = model.to_json()  # save just the config. replace with "to_yaml" for YAML serialization
with open("model_config.json", "w") as f:
    f.write(model_json)

model.save_weights('model_weights.h5') # save just the weights.
```

If you decide to save the full model, you will have access to the training configuration of
the model, otherwise you don't. So if you want to further train your model in DL4J after import,
keep that in mind and use `model.save(...)` to persist your model.

## Loading your Keras model

Let's start with the recommended way, loading the full model back into DL4J (we assume it's
on your class path):

```java
String fullModel = new ClassPathResource("full_model.h5").getFile().getPath();
MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(fullModel);

```

In case you didn't compile your Keras model, it will not come with a training configuration.
In that case you need to explicitly tell model import to ignore training configuration by 
setting the `enforceTrainingConfig` flag to false like this:

```java
MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(fullModel, false);

```

To load just the model configuration from JSON, you use `KerasModelImport` as follows:

```java
String modelJson = new ClassPathResource("model_config.json").getFile().getPath();
MultiLayerNetworkConfiguration modelConfig = KerasModelImport.importKerasSequentialConfiguration(modelJson)
```

If additionally you also want to load the model weights with the configuration, here's what you do:

```java
String modelWeights = new ClassPathResource("model_weights.h5").getFile().getPath();
MultiLayerNetwork network = KerasModelImport.importKerasSequentialModelAndWeights(modelJson, modelWeights)
```

In the latter two cases no training configuration will be read.


