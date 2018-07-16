# Getting started with Keras model import

Below is a [video tutorial](https://www.youtube.com/embed/bI1aR1Tj2DM) demonstrating 
working code to load a Keras model into Deeplearning4j and validating the working network. 
Instructor Tom Hanlon provides an overview of a simple classifier over Iris data built 
in Keras with a Theano backend, and exported and loaded into Deeplearning4j:

<iframe width="560" height="315" src="https://www.youtube.com/embed/bI1aR1Tj2DM" frameborder="0" allowfullscreen></iframe>

If you have trouble viewing the video, please click here to [view it on YouTube](https://www.youtube.com/embed/bI1aR1Tj2DM).


## <a name="configs">Loading only model configurations</a>

To only load the model architecture or configuration, DL4J supports the following two methods.

* Sequential Model Configuration import, saved in Keras with model.to_json()

```
MultiLayerNetworkConfiguration modelConfig = KerasModelImport.importKerasSequentialConfiguration("PATH TO YOUR JSON FILE)

```

* ComputationGraph Configuration import, saved in Keras with model.to_json()

```
ComputationGraphConfiguration computationGraphConfig = KerasModelImport.importKerasModelConfiguration("PATH TO YOUR JSON FILE)

```

## <a name="configs">Loading model configuration and saved weights</a>

In this case you would save both the JSON config and the weights from the trained model in Keras. The weights are saved in an H5 formatted file. In Keras you can save the weights and the model configuration into a single H5 file, or you can save the config in a separate file.

* Sequential Model single file

```
MultiLayerNetwork network = KerasModelImport.importKerasSequentialModelAndWeights("PATH TO YOUR H5 FILE")

```

The network would be ready to use for inference by passing it input data, formatted, transformed, and normalized in the same manner that the original data was and calling network.output.

* Sequential Model one file for config one file for weights.


```
MultiLayerNetwork network = KerasModelImport.importKerasSequentialModelAndWeights("PATH TO YOUR JSON FILE","PATH TO YOUR H5 FILE")

```



You can learn more about saving Keras models on the Keras 
[FAQ Page](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model).

- [Guide to importing Sequential models](./keras-sequential-guide)
- [Guide to importing functional API models](./keras-model-guide)