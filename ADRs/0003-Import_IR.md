# Import IR

## Status

Implemented

Proposed by: Adam Gibson (28-09-2020)

Discussed with: Paul Dubs

## Context

Currently, there is a gap in the way samediff/nd4j operations are implemented
vs. how other frameworks represent their models.

Keras, Tensorflow, and Pytorch use an attribute based format with names. Interop
between Onnx ,Tensorflow, and Keras tends to follow the following formula:

1. Map names to equivalent names in the other framework for each operation
   configuration. Names being both op names and associated attributes of the
   operations such as in Conv2D where you have strides, kernel sizes.
2. Map input/output tensors to the equivalent tensor type in each framework.
3. Setup the complete graph in the equivalent framework. Sometimes the
   framework's concepts don't map 1 to 1. They should output equivalent results
   regardless though.  In order to do this, sometimes the framework needs to
   add/remove operations in order to produce equivalent output in a different
   graph. The [tensorflow onnx import](https://github.com/onnx/tensorflow-onnx#how-tf2onnx-works)
   is a good example of this.

Samediff/nd4j have their internal op representations as a set of ordered
arguments for execution in the form of:

1. t arguments: floating point arguments (float, double,..)
2. integer arguments: integer arguments (long, integer)
3. boolean argument: boolean arguments
4. data type arguments: data types for input/output
5. input arguments: ndarrays for input
6. output arguments: often optional (dynamically created) output ndarray
   arguments. If the user wants to pass in outputs to control memory, they are
   allowed to do so.
7. axis arguments: Integer arguments that represent the dimension(s) for an
   operation to be executed on.

[Reference implementation](https://github.com/KonduitAI/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/DynamicCustomOp.java#L58)

This maps well enough for execution, but not for file formats.

## Related Work
This may encourage future work to be done to the
[samediff file format](https://github.com/KonduitAI/deeplearning4j/blob/master/nd4j/ADRs/0001-SameDiff_File_Format.md).
Implementation of serialization of file format via flatbuffers can be found
[here](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/SameDiff.java#L4748)
Of note here for prior work is the
[current code generation]
(https://github.com/KonduitAI/dl4j-dev-tools/blob/master/codegen/src/main/ops/org/nd4j/codegen/ops/CNN.kt#L28)

The definitions for the kotlin dsl can be found
[here](https://github.com/KonduitAI/dl4j-dev-tools/blob/master/codegen/src/main/kotlin/org/nd4j/codegen/dsl/OpBuilder.kt)


While it does have the intended description,
it’s kotlin specific and is only available for a very small subset
of the ops where pre-created objects were created
for specific operations. The goal of this ADR is to expand upon
that and make it language agnostic by providing this information in a
 neutral file format that has code generation with it.

Current code generation efforts can be augmented using this file format.
More on this decision making can be found [here](https://github.com/KonduitAI/dl4j-dev-tools/blob/master/codegen/adr/0007-configuration_objects.md)



## Proposal

We expose a symbol based mapping in libnd4j in protobuf format, similar to how
other frameworks are doing it, as a bridge/intermediary format.

This makes it easier to implement interop with the other frameworks, because it
adds the necessary information that is needed to be able to define a direct
mapping.

This could be a future file format depending on how the framework evolves. For
now, this is considered a work around for making writing import code easier/more
portable.

Similar to [ONNX](https://onnx.ai/) and  [Tensorflow](https://tensorflow.org/)
we use protobuf to express an attribute based file format and map
samediff/nd4j operations to this format.

We use a translation layer that handles mapping from attributes to the ordered
arguments approach reflected in samediff/nd4j.

For each operation, we define a mapping process to/from this attribute format to the
order based execution format.

A separate but similar set of rules are used for mapping ndarrays.

This attribute based format is an Intermediary Representation that we then
"compile" to the equivalent calls in libnd4j.


The format definitions for the IR can be found [here](./src/main/proto/nd4j/nd4j.proto) 

## Consequences

Migration to an attribute based import format makes working with other deep
learning frameworks easier in the future.


### Drawbacks

1. Yet another file format.
2. Risk migrating to new file format in the future.
3. A lot of up front manual work to index set of current operations.
4. Backwards compatibility: yet another thing to maintain. We wrote converters
   for any forward compatibility. We address this by specifying an opset schema
   scheme similar to onnx.

### Advantages

1. Easy to maintain.
2. Backwards compatible.
3. Easily interops with existing other deep learning frameworks.
4. No additional dependencies from what's already normal.
5. Protobuf allows easy code generation for other languages.
6. Industry standard conventions being used over proprietary tooling reducing
   friction for adoption for people coming from other frameworks
7. Straightforward mapping of arguments for import  
8. Provide an easy bridge to existing libnd4j  
9. Allow automation of op descriptors  in any language that would understand how
   to pass data to the  c++ library.


## Appendix A: Comparison with other Frameworks, implicit vs. explicit

We can find the existing attributes from the conventions of the
libnd4j code base. The libnd4j [conv1d.cpp](https://github.com/KonduitAI/deeplearning4j/blob/master/libnd4j/include/ops/declarable/generic/nn/convo/conv1d.cpp#L104)
file contains the following declaration:

```
auto inputShapeInfo   = inputShape->at(0);
auto weightsShapeInfo = inputShape->at(1);
Nd4jLong const* biasShapeInfo    = block.width() > 2 ? inputShape->at(2) : nullptr;

int kW = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<int>(shape::sizeAt(weightsShapeInfo, 0)); // filter(kernel) width
int sW = INT_ARG(1);                                                        // strides width
int pW = INT_ARG(2);                                                        // paddings width
int dW = INT_ARG(3);                                                        // dilations width
int paddingMode = INT_ARG(4);                                               // 0-VALID, 1-SAME
int isNCW  = block.getIArguments()->size() > 5 ? !INT_ARG(5) : 1;           // INT_ARG(4): 1-NWC, 0-NCW
int wFormat = block.getIArguments()->size() > 6 ? INT_ARG(6) : 0;           // 0 - [kW, iC, oC], 1 - [oC, iC, kW], 2 - [oC, kW, iC]
```

We can see that there are macros in the libnd4j code base, which reflect how
each argument is accessed. Each list of arguments has an expected order, that we
need to explicitly map to a parseable structure.

In comparison, the
[onnx Convolution operator](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv)
has *explicit* attributes of various types such as lists of ints and named
tensors.

As shown above, these concepts exist internally in the operations and layers
themselves in nd4j/samediff, but they are not exposed directly to the user.


A theoretical op descriptor from libnd4j is as follows:
```java
    private String name;
    private int nIn,nOut,tArgs,iArgs;
    private boolean inplaceAble;
    private List<String> inArgNames;
    private List<String> outArgNames;
    private List<String> tArgNames;
    private List<String> iArgNames;
    private List<String> bArgNames;
    private OpDeclarationType opDeclarationType;

    public enum OpDeclarationType {
        CUSTOM_OP_IMPL,
        BOOLEAN_OP_IMPL,
        LIST_OP_IMPL,
        LOGIC_OP_IMPL,
        OP_IMPL,
        DIVERGENT_OP_IMPL,
        CONFIGURABLE_OP_IMPL,
        REDUCTION_OP_IMPL,
        BROADCASTABLE_OP_IMPL,
        BROADCASTABLE_BOOL_OP_IMPL
    }
```

It contains all the op declarations and fields associated with a descriptor.

In the libnd4j code base, we represent the op descriptor types above
*implicitly* through validation as well as the different macros present in the
code base representing what an op execution looks like.

Validation for what can be present in the various names can be found
[here](https://github.com/KonduitAI/deeplearning4j/blob/master/libnd4j/include/ops/declarable/impl/DeclarableOp.cpp#L734-L765)

The set of macro declarations in libnd4j can be found
[here](https://github.com/eclipse/deeplearning4j/blob/master/libnd4j/include/system/op_boilerplate.h)


## Appendix B: Format Comparison to other frameworks

An add op in tensorflow looks like:

```
op {
  name: "Add"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_BFLOAT16
        type: DT_HALF
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_UINT8
        type: DT_INT8
        type: DT_INT16
        type: DT_INT32
        type: DT_INT64
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_STRING
      }
    }
  }
}
```

Onnx’s add can be found here
https://github.com/onnx/onnx/blob/master/docs/Operators.md#Add

Onnx and tensorflow are purely attribute based formats.
