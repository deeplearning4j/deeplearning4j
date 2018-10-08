---
title: How to add new operations to SameDiff
short_title: Adding Ops
description: How to add differential functions and other ops to SameDiff graph.
category: SameDiff
weight: 2
---

## How to add new operations to SameDiff

### A quick SameDiff overview

To get started with SameDiff, familiarize yourself with the `autodiff` module of the ND4J API located [here on GitHub.](https://github.com/deeplearning4j/nd4j/tree/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff)

For better or worse, SameDiff code is organized in just a few key places. For basic usage and testing of SameDiff the following modules are key. We'll discuss some of them in more detail in just a bit.

- `functions`: This module has the basic building blocks to build SameDiff variables and graphs.
- `execution`: has everything related to SameDiff graph execution.
- `gradcheck`: Utility functionality for checking SameDiff gradients, similar in structure to the respective tool in DL4J.
- `loss`: Loss functions for SameDiff
- `samediff`: Main SameDiff module to define, set up and run SameDiff operations and graphs.

### Differential functions in the `functions` module

See the `functions` module on [GitHub.](https://github.com/deeplearning4j/nd4j/tree/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/functions)

The central abstraction of the `functions` module is `DifferentialFunction`, which underlies pretty much everything in SameDiff. Mathematically, what we're doing in SameDiff is build a directed acyclic graph whose nodes are differential functions, for which we can compute gradients. In that regard, `DifferentialFunction` makes up a SameDiff graph on a fundamental level.

Note that each `DifferentialFunction` comes with a `SameDiff` instance. We'll discuss `SameDiff` and this relationship later on. Also, while there's only few key abstractions, they're essentially used everywhere, so it's almost impossible to discuss SameDiff concepts separately. Eventually we'll get around to each part.

#### Properties and mappings

Each differential function comes with _properties_. In the simplest case, a differential function just has a name. Depending on the operation in question, you'll usually have many more properties (think strides or kernel sizes in convolutions). When we import computation graphs from other projects (TensorFlow, ONNX, etc.) these properties need to be mapped to the conventions we're using internally. The methods `attributeAdaptersForFunction`, `mappingsForFunction`, `propertiesForFunction` and `resolvePropertiesFromSameDiffBeforeExecution` are what you want to look at to get started.

Once properties are defined and properly mapped, you call `initFromTensorFlow`  and `initFromOnnx` for TensorFlow and ONNX import, respectively. More on this later, when we discuss building SameDiff operations.

#### Inputs and outputs

A differential function is executed on a list of inputs, using function properties, and produces one or more output variables. You have access to many helper functions to set or access these variables:

- `args()`: returns all input variables.
- `arg()`: returns the first input variable (the only one for unary operations).
- `larg()` and `rarg()`: return the first and second (read "left" and "right") argument for binary operations
- `outputVariables()`: returns a list of all output variables. Depending on the operation, this may be computed dynamically. As we'll see later on, to get the result for ops with a single output, we'll call `.outputVariables()[0]`.

Handling output variables is tricky and one of the pitfalls in using and extending SameDiff. For instance, implementing `calculateOutputShape` for a differential function might be necessary, but if implemented incorrectly can lead to hard-to-debug failures. (Note that SameDiff will eventually call op execution in `libnd4j` and dynamic custom ops either infer output shapes or need to be provided with the correct output shape.)

#### Automatic differentiation

Automatic differentiation for a differential functions is implemented in a single method: `doDiff`. Each operation has to provide an implementation of `doDiff`. If you're implementing a SameDiff operation for a `libnd4j` op `x` and you're lucky to find `x_bp` (as in "back-propagation") you can use that and your `doDiff` implementation comes essentially for free.

You'll also see a `diff` implementation that's used internally and calls `doDiff`.

#### Differential function factory

Importantly, each differential function has access to a _factory_, an instance of `DifferentialFunctionFactory`, by calling `f()`. More precisely, this will return the factory of the SameDiff instance the differential function has:

```java
public DifferentialFunctionFactory f() {
    return sameDiff.f();
}
```

This is used in many places and gives you access to all differential functions currently registered in SameDiff. Think of this factory as a provider of operations. Here's an example of exposing `sum` to the `DifferentialFunctionFactory`:

```java
public SDVariable sum(...) {
    return new Sum(...).outputVariables()[0];
}
```

We leave out the function arguments on purpose here. Note that all we do is redirect to the `Sum` operation defined elsewhere in ND4J and then return the first output variable (of type `SDVariable`, discussed in a second). Disregarding the implementation details for now, what this allows you to do is call `f().sum(...)` from anywhere you have access to a differential function factory. For instance, when implementing a SameDiff op `x` and you already have `x_bp` in your function factory, you can override `doDiff` for `x`

```java
@Override
public List<SDVariable> doDiff(List<SDVariable> grad) {
    ...
    return Arrays.asList(f().x_bp(...));
}
```


### Building and executing graphs in `samediff`

See the `samediff` module on [GitHub.](https://github.com/deeplearning4j/nd4j/tree/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff)

Not surprisingly, this is where the magic happens. This module has the core structures that SameDiff operates with. First, let's have a look at the variables that make up SameDiff operations.

#### SameDiff variables

`SDVariable` (read SameDiff variable) extends `DifferentialFunction` and is to SameDiff what `INDArray` is to good old ND4J. In particular, SameDiff graphs operate on these variables and each individual operation takes in and spits out a list of `SDVariable`. An `SDVariable` comes with a name, is equipped with a `SameDiff` instance, has shape information and knows how to initialize itself with an ND4J `WeightInitScheme`. You'll also find a few helpers to set and get these properties.  

One of the few things an `SDVariable` can do that a `DifferentialFunction` can't it evaluate its result and return an underlying `INDArray` by calling `eval()`. This will run SameDiff internally and retrieve the result. A similar getter is `getArr()` which you can call at any point to get the current value of this variable. This functionality is used extensively in testing, to assert proper results. An `SDVariable` also has access to its current gradient through `gradient()`. Upon initialization there won't be any gradient, it will usually be computed at a later point.

Apart from these methods, `SDVariable` also carries methods for concrete ops (and is in that regard a little similar to `DifferentialFunctionFactory`). For instance, defining `add` as follows:

```java
public SDVariable add(double sameDiffVariable) {
    return add(sameDiff.generateNewVarName(new AddOp().opName(),0),sameDiffVariable);
}
```

allows you to call `c = a.add(b)` on two SameDiff variables, the result of which can be accessed by `c.eval()`.


#### SameDiff

The `SameDiff` class is the main workhorse of the module and brings together most of the concepts discussed so far. A little unfortunately, the inverse is also true and `SameDiff` instances are part of all other SameDiff module abstractions in some way or the other (which is why you've seen it many times already). Generally speaking, `SameDiff` is the main entry point for automatic differentiation and you use it to define a symbolic graph that carries operations on `SDVariable`s. Once built, a SameDiff graph can be run in a few ways, for instance `exec()` and `execAndEndResult()`.

Convince yourself that invoking `SameDiff()` sets up a [million things!]( https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/SameDiff.java#L817-L846) Essentially, `SameDiff` will collect and give you access (in terms of both getters and setters) to

- All differential functions for the graph, with all their properties, which can be accessed in various ways (e.g. name or id).
- All inputs and output information for said functions.
- All function properties and how to map them, `propertiesToResolve` and `propertiesForFunction` are of particular note.

`SameDiff` is also the place where you expose new operations to the SameDiff module. Essentially, you write a little wrapper for the respective operation in the `DifferentialFunctionFactory` instance `f()`. Here's an example for cross products:

```java
public SDVariable cross(SDVariable a, SDVariable b) {
    return cross(null, a, b);
}

public SDVariable cross(String name, SDVariable a, SDVariable b) {
    SDVariable ret = f().cross(a, b);
    return updateVariableNameAndReference(ret, name);
}
```

#### SameDiff execution examples and tests

At this point it might be a good idea to check out and run a few examples. SameDiff tests are a good source for that. Here's an example of how to multiply two SameDiff variables

```java
SameDiff sd = SameDiff.create();

INDArray inArr = Nd4j.linspace(1, n, n).reshape(inOrder, d0, d1, d2);
INDArray inMul2Exp = inArr.mul(2);

SDVariable in = sd.var("in", inArr);
SDVariable inMul2 = in.mul(2.0);

sd.exec();
```

This example is taken from [SameDiffTests](https://github.com/deeplearning4j/nd4j/blob/4c00b19ad4972399264233b6f0b0f5a22493235b/nd4j-backends/nd4j-tests/src/test/java/org/nd4j/autodiff/samediff/SameDiffTests.java), one of the main test sources, in which you also find a few complete end-to-end examples.

The second place you find tests is in [gradcheck](https://github.com/deeplearning4j/nd4j/tree/4c00b19ad4972399264233b6f0b0f5a22493235b/nd4j-backends/nd4j-tests/src/test/java/org/nd4j/autodiff/gradcheck). Whenever you add a new operation to SameDiff, add tests for the forward pass and gradient checks as well.

The third set of relevant tests is stored in [imports](https://github.com/deeplearning4j/nd4j/tree/20e3d53dbcd56a14dd1b7572dd52d5e200e9a4ba/nd4j-backends/nd4j-tests/src/test/java/org/nd4j/imports) and contains test for importing TensorFlow and ONNX graphs. On a side note, the resources for these import tests are generated in our [TFOpsTests](https://github.com/deeplearning4j/TFOpTests) project.

### Creating and exposing new SameDiff ops

We've seen how ND4J operations get picked up by `DifferentialFunctionFactory` and `SameDiff` to expose them to SameDiff at various levels. As for actually implementing these ops, you need to know a few things. In libnd4j you find two classes of operations, which are described [here](https://github.com/deeplearning4j/libnd4j/blob/5dea2d228c61cdec7535d1c0c6aa093a15fef9fa/AddingNewOps.md) in detail. We'll show how to implement both op types.

All operations go [here](https://github.com/deeplearning4j/nd4j/tree/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl), and most of the time it's obvious where exactly to put the ops. Special attention goes to `layers`, which is reserved for deep learning layer implementations (like [`Conv2D`](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl/layers/convolution/Conv2D.java)). These higher-level ops are based on the concept of [Modules](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/BaseModule.java), similar to modules in pytorch or layers in TensorFlow. These layer op implementation also provide a source of more involved op implementations.

#### Implementing legacy operations

Legacy (or XYZ) operations are the old breed of ND4J operations with a characteristic "xyz" signature. Here's how to implement cosine in ND4J by wrapping the `cos` legacy op from libn4j: [Cosine implementation](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl/transforms/Cos.java#L38-L72). When it comes to SameDiff, the good thing about legacy ops is that they're already available in ND4J, but need to be augmented by SameDiff specific functionality to pass the muster. Since the cosine function does not have any properties, this implementation is straightforward. The parts that make this op SameDiff compliant are:

- You specify SameDiff constructors [here](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl/transforms/Cos.java#L38-L51)
- You implement `doDiff` [here] (https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl/transforms/Cos.java#L38-L51)
- You specify a SameDiff `opName`, a TensorFlow `tensorflowName` and an ONNX `onnxName` [here](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl/transforms/Cos.java#L74-L93).

If you look closely, this is only part of the truth, since `Cos` extends `BaseTransformOp`, which implements other SameDiff functionality. (Note that `BaseTransformOp` is a [`BaseOp`](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/BaseOp.java), which extends `DifferentialFunction` from earlier.) For instance, `calculateOutputShape` is [implemented there](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/BaseTransformOp.java#L195-L207). If you want to implement a new transform, you can simply inherit from `BaseTransformOp`, too. For other op types like reductions etc. there are op base classes available as well, meaning you only need to address the three bullet points above.

In the rare case you need to write a legacy op from scratch, you'll have to find the respective op number from libn4j, which can be found in `legacy_ops.h`.

#### Implementing Dynamic Custom Operations

[`DynamicCustomOp`](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/DynamicCustomOp.java) is the new kind of operation from libnd4j and all recent additions are implemented as such. This operation type in ND4J directly extends `DifferentialFunction`.

[Here's](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl/transforms/BatchToSpace.java) an example of the `BatchToSpace` operation, which inherits from `DynamicCustomOp`:

- BatchToSpace is [initialized](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl/transforms/BatchToSpace.java#L49-L67) with two properties, `blocks` and `crops`. Note how `blocks` and `crops`, which are both of integer type, get added to _integer arguments_ for the operation by calling `addIArgument`. For float arguments and other _types_, use `addTArgument` instead.
- The operation gets its own name and [names for import](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl/transforms/BatchToSpace.java#L69-L82),
- and `doDiff` is [implemented](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl/transforms/BatchToSpace.java#L84-L89).

The BatchToSpace operation is then integrated into `DifferentialFunctionFactory` [here](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/functions/DifferentialFunctionFactory.java#L840-L844), exposed to `SameDiff` [here](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/SameDiff.java#L2105-L2107) and tested [here](https://github.com/deeplearning4j/nd4j/blob/4c00b19ad4972399264233b6f0b0f5a22493235b/nd4j-backends/nd4j-tests/src/test/java/org/nd4j/autodiff/gradcheck/GradCheckTransforms.java#L151-L191).

The only thing BatchToSpace is currently missing is _property mapping_. We call the properties for this operation `blocks` and `crops`, but in ONNX or TensorFlow they might be called and stored quite differently. To look up the differences for mappings this correctly, see [`ops.proto`](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/resources/ops.proto) for TensorFlow and [`onnxops.json`](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/resources/onnxops.json) for ONNX.


Let's look at another operation that does property mapping right, namely [`DynamicPartition`](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl/transforms/DynamicPartition.java). This op has precisely one property, called `numPartitions` in SameDiff. To map and use this property, you do the following:

- Implement a little helper method called [`addArgs`](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl/transforms/DynamicPartition.java#L59-L61) that is used in the constructor of the op and in an import helper one-liner that we're discussing next. It's not necessary, but encouraged to do this and call it `addArgs` consistently, for clarity.
- Override [`initFromTensorFlow` method](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl/transforms/DynamicPartition.java#L63-L67) that maps properties for us using a `TFGraphMapper` instance and adding arguments with `addArgs`. Note that since ONNX does not support dynamic partitioning at the time of this writing (hence no `onnxName`) there's also no `initFromOnnx` method, which works pretty much the same way as `initFromTensorFlow`.
- For the TensorFlow import to work, we also need to [override `mappingsForFunction`](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl/transforms/DynamicPartition.java#L70-L83). This example of a mapping is very simple, all it does is map TensorFlow's property name `num_partititions` to our name `numPartitions`.

Note that while `DynamicPartition` has proper property mapping, it currently does not have a working `doDiff` implementation.

As a last example, we show one that has a little more interesting property mapping setup, namely [`Dilation2D`](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl/transforms/Dilation2D.java). Not only has this op far more properties to map, as you can see in [`mappingsForFunction`](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl/transforms/Dilation2D.java#L59-L104), the properties also come with _property values_, as defined in [`attributeAdaptersForFunction`](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl/transforms/Dilation2D.java#L106-L132). We've chosen to show this op because it is one that has property mapping, but is neither exposed to `DifferentialFunctionFactory` not `SameDiff`.

Hence, the three `DynamicCustomOp` examples shown each come with their own defects and represent examples of the work that has to be done for SameDiff. To summarize, to add a new SameDiff op you need to:

- Create a new operation in ND4J that extends `DifferentialFunction`. How exactly this implementation is set up depends on the
  - op generation (legacy vs. dynamic custom)
  - op type (transform, reduction, etc.)
- Define an own op name, as well as TensorFlow and ONNX names.
- Define necessary SameDiff constructors
- Use `addArgs` to add op arguments in a reusable way.
- Expose the operation in `DifferentialFunctionFactory` first and wrap it then in `SameDiff` (or `SDVariable` for variable methods).
- Implement `doDiff` for automatic differentiation.
- Override `mappingsForFunction` to map properties for TensorFlow and ONNX
- If necessary, also provide an attribute adapter by overriding `attributeAdaptersForFunction`.
- Add import one-liners for TensorFlow and ONNX by adding `initFromTensorFlow` and `initFromOnnx` (using `addArgs`).
- Test, test, test.
