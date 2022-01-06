# Model Hub Zoo Download Implementation

## Status
**Discussion**

Proposed by: Adam Gibson (1st Jan 2022)


## Context

Following on from work in [downloading models](0011%20-%20Model%20Hub-Zoo%20Download.md)
We need to be able to interop in different ecosystems. This ADR will address the
specs for implementing interop with the following ecosystems:
1. [Onnx](https://github.com/onnx/models/)
2. [Tensorflow](https://www.tensorflow.org/hub)
3. [Huggingface](https://huggingface.co/spaces)
4. [Pytorch model zoo](https://pytorch.org/serve/model_zoo.html)
5. [Keras applications](https://keras.io/api/applications/)

## Proposal

This proposal will be broken up in to separate sections detailing
the work and implementation needed to implement the loading of models
from each of these ecosystems.

Each section will cover how to implement the download and loading of model
download workflow described in [downloading models](0011%20-%20Model%20Hub-Zoo%20Download.md)

We will also cover how we will handle staging of models for each framework.


### Onnx

Onnx is pretty straightforward as a github repo download.
These models do not have any special structure beyond the zip file.
Our downloader will focus on the already uncompressed models
for ease of simplicity.

### Tensorflow

Tensorflow will use the tf hub web service. Our access will be focused
on using the uncompressed pb models + handling other conversion code
for freezing models to be reused.

For our purposes a staged model will be a frozen model
that can be directly imported.


### Pytorch

Pytorch will need to be converted to onnx. Pytorch serving uses
the [model archive tool](https://github.com/pytorch/serve/tree/master/model-archiver/) 
for handling model storage.

Unfortunately, this requires a bit of to integrate with.
Pytorch serve archives can vary in format. We will typically 
want to extract the model from it to manipulate it.

Separately, pytorch has various model zoos both official and community provided:
1. [Example of community provided](https://github.com/rwightman/pytorch-image-models)
2. [Torchvision model zoo](https://pytorch.org/vision/stable/models.html)
3. [Pytorch hub](https://pytorch.org/hub/)

At the end, we will want to convert the model to onnx. 
This will be considered a staged model that is consumable
by the framework.




### Huggingface

Huggingface spaces uses git repositories to store models.
URLs are accessible using the [huggingface hub SDK](https://huggingface.co/docs/hub/how-to-downstream)

Huggingface hub supports 3 frameworks: pytorch, tensorflow, JAX

Our initial support will only focus on pytorch and tensorflow.
JAX will come at a later date when we have implemented JAX
for the model import framework.

When loading a model, we will need to know which model type we are running
so we can convert it to onnx. We will know this by letting a user specify
the model type when they go to download it.


For each of tensorflow and pytorch we will be storing models under their respective
frameworks reusing the staging techniques from the tensorflow and onnx frameworks.

Huggingface paths should be repositories + the framework_name specifier.
We use AutoModel(pytorch) and AutoTFModel(tensorflow) for converting models
to onnx and saved_model -> pb respectively.

### Keras

Keras applications are simple archives that contain .h5 files.
We will use the keras applications library to download and cache the models.




## Consequences

### Advantages

* Greatly strengthens our ability to test and execute models
needed for different use cases
* Allows us to be flexible in what we can enable users to do with
our framework as a starting point
* Further, builds out the work from downloading models and greatly increases
the testing allowed for our model import framework



### Disadvantages
* Different work is needed for each ecosystem
* APIs may change and need to be updated
* Just pre-processing models does not mean they are guaranteed to be imported. Additional
  work will need to be done on model import to allow models to execute. This comes with additional validation work.
* Not a comprehensive solution, users will still need to know things like the inputs and outputs
and may still need to refer to the underlying docs for a given model to use effectively.
* A user may still need to understand how to pre-process different kinds of models