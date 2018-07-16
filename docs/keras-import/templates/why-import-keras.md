# Why use Keras model import?

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