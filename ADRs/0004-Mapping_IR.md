# Import IR

## Status
Proposed

Proposed by: Adam Gibson (28-09-2020)

Discussed with: N/A

## Context
 
 Generally, every neural network file format defines a sequence of operations
 to execute mathematical operations that comprises a neural network.
 
 Each element in the sequence is a node that contains information such as the
 desired operation, and a set of attributes that represent parameters
 in to the mathematical function to execute.
 
In order to write import/export for different frameworks, we need to adapt 
an attribute based format from various popular deep learning frameworks.
Nd4j  has a different list based format for operation execution arguments.
In the [previous ADR](./Import_IR.md), we added an IR which makes it easier to 
interop with other frameworks. 

In this ADR, this work is extended to add a file format for 
describing lists of operations as MappingRules which allow transformations
from one framework to another.

These transformations manipulate protobuf as input and output Nd4j's
new OpDescriptor format as output.


##Related work

See [the import IR](./0003-Import_IR.md)

## Decision

We implement a mapping process framework that defines transforms on an input file format.
A MappingProcess defines a list of MappingRules which represent a sequence of transformations
on each attribute of an op definition.

To assist in mapping, a mapping context with needed information like rule arguments
for transformation, current node, and whole graph are used as input.

The input is a protobuf file for a specific framework and the output is an op descriptor
described [here](./0003-Import_IR.md).

A MappingRule converts 1 or more attributes in to 1 more or arg definitions. A potential definition
can be found in Appendix E.

Attributes are named values supporting a wide variety of types from floats/doubles
to lists of the same primitive types. See Appendix C for a theoretical definition.

Arg Definitions are the arguments for an OpDescriptor described in [the import IR ADR.](./0003-Import_IR.md)
See Appendix D for a potential definition of arg definitions.

All of this together describes how to implement a framework agnostic
interface to convert between a target deep learning framework and the nd4j format.


## Implementation details

In order to implement proper mapping functionality, a common interface is implemented.
Below are the needed common types for mapping:

1. IRNodeDef: A node definition in a graph
2. IRTensor: A tensor type for mapping
3. IROpList: A list of operations
4. IRAttrDef: An attribute definition
5. IRAttrValue: An attribute value
6. IROpDef: An op definition for the IR
7. IRDataType: A data type
8. IRGraph: A graph abstraction

Each one of these types is a wrapper around a specific framework's input types
of the equivalent concepts.

Each of these wrappers knows how to convert the specific concepts
in to the nd4j equivalents for interpretation by a mapper which applies
the mapping rules for a particular framework.

Doing this will allow us to share logic between mappers and making 1 implementation of 
mapping possible by calling associated getter methods for concepts like data types and nodes.

## Serialization 

In order to persist rules using protobuf, all rules will know how to serialize themselves.
A simple serialize() and load() methods are implemented which covers conversion using
interface methods up to the user to implement which describes how to persist the protobuf 
representation. This applies to any of the relevant functionality such as rules and processes.



## Custom types

Some types will  not map 1 to 1 or are directly applicable to nd4j.
In order to combat this, when an unknown type is discovered during mapping,
adapter functions for specific types must be specified.

Supported types include:

1. Long/Int
2. Double/Float
3. String
4. Boolean
5. Bytes
6. NDArrays


An example:

A Dim in tensorflow can be mapped to a long in nd4j.

Shape Information can be a  list of longs or multiple lists depending on the 
context.

## Consequences
### Advantages
* Allows a language neutral way of describing a set of transforms necessary
for mapping an set of operations found in a graph from one framework to the nd4j format.

* Allows a straightforward way of writing an interpreter as well as mappers
for different frameworks in nd4j in a standardized way.

* Replaces the old import and makes maintenance of imports/mappers more straightforward.

### Disadvantages

* More complexity in the code base instead of a more straightforward java implementation.

* Risks introducing new errors due to a rewrite


## Appendix A: Contrasting MappingRules with another implementation

We map names and types to equivalent concepts in each framework.
Onnx tensorflow does this with an [attribute converter](https://github.com/onnx/onnx-tensorflow/blob/08e41de7b127a53d072a54730e4784fe50f8c7c3/onnx_tf/common/attr_converter.py)

This is done by a handler (one for each op).
More can be found [here](https://github.com/onnx/onnx-tensorflow/tree/master/onnx_tf/handlers/backend)


## Appendix B: Challenges when mapping nd4j ops

The above formats are vastly different. Onnx and tensorflow
are purely attribute based. Nd4j is index based.
This challenge is addressed by the IR by adding names to each property.


In order to actually map these properties, we need to define rules for doing so.
Examples of why these mapping rules are needed below:

1. Different conventions for the same concept. One example that stands out from conv
is padding. Padding can be represented as a string or have a boolean that says what a string equals.
In nd4j, we represent this as a boolean: isSameMode. We need to do a conversion inline in order
to invoke nd4j correctly.

2. Another issue is implicit concepts. Commonly, convolution requires you to configure a layout
of NWHC (Batch size, Height, Width, Channels) 
or NCHW (Batch size, Channels,Height, Width). Tensorflow allows you to specify it,
nd4j also allows you to specify it. Onnx does not.
 
 A more in depth conversation on this specific issue relating to the 
 2 frameworks can be found [here](https://github.com/onnx/onnx-tensorflow/issues/31)
In order to address these challenges, we introduce a MappingRule allowing
us to define a series of steps to map the input format to the nd4j format
in a language neutral way via a protobuf declaration.


## Appendix C: A theoretical attribute definition
```kotlin
enum class AttributeValueType {
    FLOAT,
    LIST_FLOAT,
    BYTE,
    LIST_BYTE,
    INT,
    LIST_INT,
    BOOL,
    LIST_BOOL,
    STRING,
    LIST_STRING
}

interface IRAttribute<ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE> {

    fun name(): String

    fun floatValue(): Double

    fun listFloatValue(): List<Float>

    fun byteValue(): Byte

    fun listByteValue(): List<Byte>

    fun intValue(): Long

    fun listIntValue(): List<Long>

    fun boolValue(): Boolean

    fun listBoolValue(): List<Boolean>

    fun attributeValueType(): AttributeValueType

    fun internalAttributeDef(): ATTRIBUTE_TYPE

    fun internalAttributeValue(): ATTRIBUTE_VALUE_TYPE
}

```

## Appendix D: A theoretical kotlin definition of argument descriptors and op descriptors can be found below:
```kotlin
interface IRArgDef<T,DATA_TYPE> {
    fun name(): String

    fun description(): String

    fun dataType(): IRDataType<DATA_TYPE>

    fun internalValue(): T

    fun indexOf(): Integer
}

interface IROpDef<T,ARG_DEF_TYPE,DATA_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE> {
    fun opName(): String

    fun internalValue(): T

    fun inputArgs(): List<IRArgDef<ARG_DEF_TYPE,DATA_TYPE>>

    fun outputArgs(): List<IRArgDef<ARG_DEF_TYPE,DATA_TYPE>>

    fun attributes(): List<IRAttribute<ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE>>

}
```


##Appendix E: A theoretical kotlin definition of Mapping Rules, MappingProcess and ArgDef can be found below:
```kotlin
interface MappingProcess<T,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE> {
    fun opName(): String

    fun frameworkVersion(): String

    fun inputFramework(): String

    fun rules(): List<MappingRule<ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE>>


    fun applyProcess(inputNode: IRNode<T,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>): OpDeclarationDescriptor

    fun applyProcessReverse(input: OpDeclarationDescriptor): IRNode<T,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>

    fun createDescriptor(argDescriptors: List<OpNamespace.ArgDescriptor>): OpDeclarationDescriptor
}

interface MappingRule<ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE> {
    fun name(): String

    /**
     * Convert 1 or more attributes in to a list of {@link ArgDescriptor}
     */
    fun convert(inputs: List<IRAttribute<ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE>> ): List<OpNamespace.ArgDescriptor>

    fun convertReverse(input: List<OpNamespace.ArgDescriptor>): List<IRAttribute<ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE>>

}

```