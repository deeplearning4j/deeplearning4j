# Java 9+ Support

## Status
**Discussion**

Proposed by: Adam Gibson (8th Mar 2022)


## Context

Onnx and Tensorflow both support a concept of sequences of ndarrays.
Tensorflow uses [RaggedTensors](https://www.tensorflow.org/guide/ragged_tensor).
Onnx uses [sequences](https://github.com/onnx/onnx/blob/main/docs/IR.md)
Both of these essentially group ndarrays as one variable.

SameDiff supports a sequence [variable type](https://github.com/eclipse/deeplearning4j/blob/4766032444de8e0c2c3389270576bb6e7c466211/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/VariableType.java#L28)
which allows for inputs and outputs of sequence types.
This will play nice with model import and allow us to implement a few select ops.

The scope of this type is for utility purposes only and will not go as far as tensorflow's ragged tensor
for supporting full operations. Implementing a full RaggedTensor type isn't really worth the work 
and can be worked around (at a performance cost) with sparse matrices
masked by zeros with the shape being the largest type in the list.


## Proposal

1. Support an SDVariableType that supports groups of ndarrays.
2. Add a new mechanism to samediff  that supports named sequences where groups of ndarrays can be looked up by name.
3. Supports saving and loading groups of ndarrays using the following flatbuffers structures:
```
table SequenceItem {
   name:string;
   associatedVariable:[FlatArray];
}

table SequenceItemRoot {
   sequenceItems:[SequenceItem];
}
```

These items will directly translate to a map of string (variable name) to a list of ndarrays.
A SameDiff graph will use and manipulate these variables as reading and writing to an array.

For op execution, errors will be thrown. Sequence type variables throw an error.
An alternative could be unpacking ndarrays where 1 variable can act as multiple to an op that requires
more than 1 input. This is fairly error prone and not worth producing workarounds for.


###Operations
The following operations on sequences are supported:
1. addItemToSequence(String varName,INDArray item,int atIndex): add an array at a particular index
2. removeItemFromSequence(String varName,int indexOfItem): remove an item at a particular index, note that empty variables are removed
3. SDVariable sequence(String name,INDArray[] arrays): create the sequence with the given variable name
4. setItemForSequenceAtIndex(String varName,INDArray item,int index): set the item  at the particular index
5. sequenceLength(String varName): returns the length of a sequence


Note that all indices above also support negative indexing allowing you to index from the end of the list similar
to python.

###Model import
For model import, we will also extended the work in [0004-Mapping_IR.md](0004-Mapping_IR.md)
to support sequences/lists of tensors. This means IRTensor and similar concepts will also support sequences.

## Consequences

### Advantages

* Allows flexible import of models that use sequences to pass around ndarrays
* Allows flexible management of ndarrays using a variable name as a group.


### Disadvantages
* Not robust enough to execute ops on
* Still need to implement Ragged Tensors at some point separately.