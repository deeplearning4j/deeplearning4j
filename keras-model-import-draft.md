<!--
Here is a template for creating a feature page. The goal is to apply it to both new and existing features, fitting any extant documentation into this mold.

Restrict content to the feature at hand; broader, more conceptual discussions have their own place.

### NB. Whenever you use a term that has documentation associated with it elsewhere, please link to that documentation.
-->

# Keras Model Import

## Contents

* <a href="#description">Description</a>
* <a href="#examples">Examples and Use Cases</a>
* <a href="#setup">Setup</a>
    * <a href="#prereqs">Prerequisites</a>
    * <a href="#step-by-step">Step-by-step</a>
      * <a href="#step-1">Step One</a>
      * <a href="#step-2">Step Two</a>
* <a href="#troubleshooting">Troubleshooting</a>
* <a href="#further-reading">Further reading</a>

## <a name="description">Description</a>

The deeplearning4j-modelimport module provides routines for importing neural network models originally configured and trained using [Keras](https://keras.io/), a popular Python deep learning library that provides abstraction layers on top of Deeplearning4j, [Theano](http://deeplearning.net/software/theano/) and [TensorFlow](https://www.tensorflow.org)backends. 
You can learn more about saving Keras models on the Keras [FAQ Page](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model). 

### Keras Model Import Video

We have a [screencast](https://www.youtube.com/watch?v=Cran8wsZLN4) that demonstrates model import. 

***NOTE***
We also have begun work on having DeepLearning4J function directly as a Keras backend. See [DL4J Python API](https://deeplearning4j.org/keras)

## <a name="examples">Examples and Use Cases</a>
 

Code examples identified according to their real-world applications (e.g., image recognition, prediction, etc.).
Document each example as a **subheading**, included in the main table of contents.

## <a name="setup">Setup</a>

To use Keras Model Import version 0.8.0 or higher is recommended.

Add the following to your project's pom.xml

```
<dependency>
<groupId>org.deeplearning4j</groupId>
<artifactId>deeplearning4j-modelimport</artifactId>
<version>${dl4j.version}</version>
</dependency>
```

### <a name="prereqs">Prerequisites</a>

Mara, not sure this one has Prereqs and setup, we should make this optional, or there is no harm being explicit. 

I could say. 

Using Keras Model Import Requires

* updating your pom as above
* Version of DL4j 0.7.2 or above, 0.8.0 or above is preferred
* A saved Keras Model or Configuration

### <a name="step-by-step">Step By Step Instructions</a>

***Note to Mara, although written as step by step, as you read you will 
see that there are actually multiple paths here, config to config 2 types, network to network 2 types, 2 ways each. 
It still works IMO, just noting the slight difference. 

1. Save a Keras Model
Keras Model Import allows the loading of a Saved Model Configuration or the combination of the configuration and the weights. 

To save the configuration in Keras you would use code like this. 
```
json_string = model.to_json()
text_file = open("/tmp/iris_model_json", "w")
text_file.write(json_string)
text_file.close()
```

To save the weights. 

```
model.save_weights('/tmp/iris_model_weights')
```

To save the weights and the config in a single H5 archive. 

```
model.save('/tmp/full_iris_model')
```

2. Create a Model Configuration in DL4J by reading the Saved Keras File.

This is what you would use if you wanted to load a model configuration from Keras 
but perform all the training in DeepLearning4J

Note that the Keras Sequential Model is imported into a DeepLEarning4J MultiayerNetwork, and the Keras 
Functional API is imported into a ComputationGraph. 

* Sequential Model Configuration import, saved in Keras with
model.to_json()

```
MultiLayerNetworkConfiguration modelConfig = KerasModelIm
port.importKerasSequentialConfiguration("PATH TO YOUR JSON FI
LE)
```
* ComputationGraph Configuration import, saved in Keras with
model.to_json()

```
ComputationGraphConfiguration computationGraphConfig = KerasM
odelImport.importKerasModelConfiguration("PATH TO YOUR JSON F
ILE)
```

3. Create a Model by reading saved configuration and weights. 

This creates and loads the trained weights so the model can be used
immediately for inference, or possibly for further training. 

* Sequential Model single file

```
MultiLayerNetwork network = KerasModelImport.importKerasSeque
ntialModelAndWeights("PATH TO YOUR H5 FILE")

```

* Sequential Model one file for config one file for weights

```
MultiLayerNetwork network = KerasModelImport.importKerasSeque
ntialModelAndWeights("PATH TO YOUR JSON FILE","PATH TO YOUR H
5 FILE")
```

4. Enforce Training Config option

Some Keras Models may import with enough compatability to run inference but for various reasons
the model may not be suitable for further training. If enforceTrainingConfig is set to true then training
compatibility issues will create an error condition and the model will not import. If enforceTrainingConfig is 
set to false then warnings will be issued but the model will import. 

5. Test your model

You should always verify that model works as expected.

The [JavaDoc](https://deeplearning4j.org/doc/org/deeplearning4j/nn/modelimport/keras/KerasModelImport.html) for Keras Model Import.


## Troubleshooting

Q: My Model fails to import? 
A: Are you using the correct Network type, ComputationGraph vs MultiLayerNetwork. 




## Further reading

[Keras Website](https://keras.io/) has documentation and examples for Keras. 

[Keras Model Zoo](https://github.com/albertomontesg/keras-model-zoo) has pretrained models of popular Neural Networks. 



