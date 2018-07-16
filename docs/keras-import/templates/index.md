# Deeplearing4j: Keras model import

[Keras model import](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras) 
provides routines for importing neural network models originally configured and trained 
using [Keras](https://keras.io/), a popular Python deep learning library. 

Once you have imported your model into DL4J, our full production stack is at your disposal.
We support import of all Keras model types, most layers and practically all utility functionality. 
Please check [here](./supported-features) for a complete list of supported Keras features.


## Getting started: Import a Keras model in 60 seconds

To import a Keras model, you need to create and [serialize](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model)
one first. Here's a simple example that you can use. The model is a simple MLP that takes 
mini-batches of vectors of length 100, has two Dense layers and predicts a total of 10 
categories. After defining the model, we serialize it in HDF5 format.

```python
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='sgd', metrics=['accuracy'])

model.save('simple_mlp.h5')
```

If you put this model file (`simple_mlp.h5`) into the base of your resource folder of your 
project, you can load the Keras model as DL4J `MultiLayerNetwork` as follows

```java
final String SIMPLE_MLP = new ClassPathResource("simple_mlp.h5").getFile().getPath();
MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(SIMPLE_MLP);
```

That's it! The `KerasModelImport` is your main entry point to model import and class takes
care of mapping Keras to DL4J concepts internally. As user you just have to provide your model
file, see our [Getting started guide](./getting-started) for more details and options to load
Keras models into DL4J.

You can now use your imported model for inference (here with dummy data for simplicity)
```java
INDArray input = Nd4j.create(256, 100);
INDArray output = model.output(input);
```

Here's how you do training in DL4J for your imported model:

```java
model.fit(input, output);
``` 

The full example just shown can be found in our [DL4J examples](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/modelimport/keras/basic/SimpleSequentialMlpImport.java).


## Project setup

To use Keras model import in your existing project, all you need to do is add the following 
dependency to your pom.xml.

```
<dependency>
    <groupId>org.deeplearning4j</groupId>
    <artifactId>deeplearning4j-modelimport</artifactId>
    <version>1.0.0-beta</version> // This version should match that of your other DL4J project dependencies.
</dependency>
```

If you need a project to get started in the first place, consider cloning 
[DL4J examples](https://github.com/deeplearning4j/dl4j-examples) and follow
the instructions in the repository to build the project.

## Why Keras model import?

Keras is a popular and user-friendly deep learning library written in Python.
The intuitive API of Keras makes defining and running your deep learning
models in Python easy. Keras allows you to choose which lower-level
library it runs on, but provides a unified API for each such backend. Currently,
Keras supports Tensorflow, CNTK and Theano backends, but Skymind is 
working on an [ND4J backend](https://github.com/deeplearning4j/keras/tree/inference_only/nd4j_examples)
for Keras as well.

There is often a gap between the production system of a company and the 
experimental setup of its data scientists. Keras model import 
allows data scientists to write their models in Python, but still 
seamlessly integrates with the production stack.

Keras model import  is targeted at users mainly familiar with writing 
their models in Python with Keras. With model import you can bring your 
Python models to production by allowing users to import their models 
into the DL4J ecosphere for either further training or evaluation purposes.

You should use this module when the experimentation phase of your 
project is completed and you need to ship your models to production. [Skymind](https://skymind.ai) 
commercial support for Keras implementations in enterprise.