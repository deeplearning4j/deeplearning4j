# SDValue

## Status
**Discussion**

Proposed by: Adam Gibson (8th Mar 2022)


## Context

Onnx and Tensorflow both support miscellaneous data structures to support the 
building and execution of data flow graphs.

Programming primitives such as lists, maps and optional types
can be valuable in building data flow graphs allowing the developer to express a 
wide variety of operations encoded within a neural network.

Tensorflow uses [RaggedTensors](https://www.tensorflow.org/guide/ragged_tensor).
Onnx uses [sequences](https://github.com/onnx/onnx/blob/main/docs/IR.md)
Of note is onnx also supports optionals and maps.


These primitives can actually be passed in as inputs and returned as outputs
from a graph as well.

In order to simplify the usage of these primitives a value type is used
to indicate what type of value it is being passed in (maps, lists, tensors,..)


SDValue is the Samediff interpretation of this and allows [TensorArrays](./nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl/shape/tensorops/TensorArray.java) to be passed around. 
as well as other types to be used within execution of a graph.



## Proposal

Create an SDValue class and associated enum abstraction for passing around and manipulating
variables of different types.

The following types will be supported:
1. LIST
2. DICT
3. TENSOR


An SDValue is passed around similar to placeholders and are utilized within InferenceSession to 
execute operations within a graph.

An SDValue's real underlying type will map to the following:
1. LIST: INDArray[]
2. DICT: Map<String,INDArray>
3. TENSOR: INDArray

Method calls that will use this off of samediff and InferenceSession will be:
```java
sd.output(Map<String,SDValue> values,...);

```

Each of these are named values with a value type. The scope of this for execution just augmenti 
arrays of arrays or TensorArrays.

The below describes how these operations will work:


### Operations
The following operations on sequences are supported:
1. addItemToSequence(String varName,INDArray item,int atIndex): add an array at a particular index
2. removeItemFromSequence(String varName,int indexOfItem): remove an item at a particular index, note that empty variables are removed
3. SDVariable sequence(String name,INDArray[] arrays): create the sequence with the given variable name
4. setItemForSequenceAtIndex(String varName,INDArray item,int index): set the item  at the particular index
5. sequenceLength(String varName): returns the length of a sequence


Note that all indices above also support negative indexing allowing you to index from the end of the list similar
to python.

### Model import
For model import, this also extends the work in [0004-Mapping_IR.md](0004-Mapping_IR.md)
to support sequences/lists of tensors. This means IRTensor and similar concepts will also support sequences.

## Consequences

### Advantages

* Allows flexible import of models that use sequences to pass around ndarrays
* Allows flexible management of ndarrays using a variable name as a group.


### Disadvantages
* Not robust enough to execute ops on
* Still need to implement Ragged Tensors at some point separately.