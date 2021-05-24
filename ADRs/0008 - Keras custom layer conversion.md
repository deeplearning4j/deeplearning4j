# Keras custom layer conversion

## Status
Accepted

Proposed by: Adam Gibson (24-5-2021)

Discussed with: no one

## Context
Currently, deeplearning4j's keras import requires manual registration of layers.
Usually these layers involve writing samediff operations to reflect whatever the logic
is in the keras custom layer.
A keras custom layer for most cases is either a lambda or a series of
tensorflow operations. Reasonably these sub graphs as "layers"
should be parseable.

A well defined spec exists [here](https://keras.io/guides/serialization_and_saving/#custom-objects)
and [here for custom layers](https://keras.io/guides/making_new_layers_and_models_via_subclassing/)
These standard concepts can easily be mapped to equivalent calls in either dl4j or samediff.

## Decision

Given the python code of a method and the get_config of a method, we can build a 
catalog of types used and parameters associated with the custom layer to build a dictionary.
For each wrapped layer, we would have to map equivalent python calls to the equivalent keras import calls
or samediff function calls for samediff lambda layers.


A user's python path would be the easiest thing for them to pass to us. We can then scan that for any relevant
custom layers that also maybe transitive dependencies of code they've introduced to us.


## Discussion

Parsing python code:
There are 3 ways to approach this:

1. Antlr grammar + extraction of certain metadata needed for automatic mapping based
on class definitions
   
2. Python execution using python4j: Use inspect from python to get all the relevant attributes on the python 
path a user could need. This has the benefit of a user being able to pass a python path for us to scan and register all
   needed layers.
   
3. Manual parsing similar to the [import IR](./Import_IR.md) and [./Mapping_IR.md] where we manually
scan a set of python files passed to us.
   
saudet:
the Keras -> PyTorch space it looks like reflection is the way to go: https://github.com/gmalivenko/pytorch2keras (edited)
Uses ONNX: https://github.com/gmalivenko/pytorch2keras/blob/master/pytorch2keras/converter.py#L13
Other framework specific converter approaches: https://github.com/ysh329/deep-learning-model-convertor

## Consequences
### Advantages

### Disadvantages


