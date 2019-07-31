---
title: Operations in SameDiff
short_title: Ops
description: What kind of operations is there in `SameDiff` and how to use them
category: SameDiff
weight: 4
---

# SameDiff operations

Operations in `SameDiff` work mostly the way you'd expect them to. You take variables - in our framework, those are 
objects of type `SDVariable` - apply operations to them, and thus produce new variables. Before we proceed to the 
overview of the available operations, let us list some of their common properties.

## Common properties of operations

- Variables of any *variable type* may be used in any operation, as long as their *data types* match those that are 
required by the operation (again, see our [variables](./samediff/variables) section for what variable types are). Most
often an operation will require its `SDVariable` to have a floating point data type.
- Variables created by operations have `ARRAY` variable type.
- For all operations, you may define a `String` name of your resulting variable, although for most operations this
is not obligatory. The name goes as the first argument in each operation, like so:
```java
SDVariable linear = weights.mmul("matrix_product", input).add(bias); 
SDVariable output = sameDiff.nn.sigmoid("output", linear);
``` 
Named variables may be accessed from outside using a `SameDiff` method `getVariable(String name)`. For the code above, 
this method will allow you to infer the value of both `output` as well as the result of `mmul` operation. Note that we 
haven't even explicitly defined this result as a separate `SDVariable`, and yet a corresponding `SDVariable` will be 
created internally and added to our instance of `SameDiff` under the `String` name `"matrix_product"`. In fact, a unique
`String` name is given to every `SDVariable` you produce by operations: if you don't give a name explicitly, it is 
assigned to the resulting `SDVariable` automatically based on the operation's name. 


## Overview of operations
The number of currently available operations, including overloads totals several hundreds, they range in complexity from s
imple additions and multiplications via producing outputs of convolutional layers to creation of dedicated recurrent 
neural network modules, and much more. The sheer number of operations would've made it cumbersome to list them all on a 
single page. So, if you are already looking for something specific, you'll be better off checking our 
[javadoc](https://deeplearning4j.org/api/latest/), which already contains a detailed information on each operation, or 
by simply browsing through autocompletion suggestions (if your IDE supports that). Here we rather try to give you an 
idea of what operations you may expect to find and where to seek for them.

All operations may be split into two major branches: those which are methods of `SDVariable` and those of `SameDiff` 
classes. Let us have a closer look at each:

### `SDVariable` operations
We have already seen `SDVariable` operations in previous examples, in expressions like 
```java
SDVariable z = x.add(y);
```
where `x` and `y` are `SDVariable`'s. 

Among `SDVariable` methods, you will find:
- `BLAS`-type operations to perform linear algebra: things like `add`, `neg`, `mul` (used for both scaling and elementwise
multiplication) and `mmul` (matrix multiplication), `dot`, `rdiv`,  etc.;
- comparison operations like `gt` or `lte`, used both to compare each element to a fixed `double` value as well as for 
elementwise comparison with another `SDVariable` of the same shape, and alike;
- basic reduction operations: things like `min`, `sum`, `prod` (product of elements in array), `mean`, `norm2`,
`argmax` (index of the maximal element), `squaredDifference` and so on, which may be taken along specified dimensions; 
- basic statistics operations for computing mean and standard deviation along given dimensions: `mean` and `std`. 
- operations for restructuring of the underlying array: `reshape` and `permute`, along with `shape` - an operation that 
delivers the shape of a variable as an array of integers - the dimension sizes; 

`SDVariable` operations may be easily chained, producing lines like:
```java
SDVariable regressionCost = weights.mmul(input).add("regression_prediction", bias).squaredDifference(labels);
```

### `SameDiff` operations
The operations that are methods of `SameDiff` are called via one of 6 auxiliary objects present in each `SameDiff`, 
which split all operations into 6 uneven branches:
- `math` - for general mathematical operations;
- `random` - creating different random number generators;
- `nn` - general neural network tools;
- `cnn` - convolutional neural network tools;
- `rnn` - recurrent neural network tools;
- `loss` - loss functions;
In order to use a particular operation, you need to call one of these 6 objects form your `SameDiff` instance, and then 
an operation itself, like that:
```java
SDVariable y = sameDiff.math.sin(x);
```
or 
```java
SDVariable y = samediff.math().sin(x);
```
The distribution of operations among the auxiliary objects has no structural bearing beyond organizing things in a more 
intuitive way. So, for instance, if you're not sure whether to seek for, say, `tanh` operation in `math` or in `nn`, 
don't worry: we have it in both. 

Let us briefly describe what kinds of operations you may expect to find in each of the branches: 

### `math` - basic mathematical operations
Math module mostly consists of general mathematical functions and statistics methods. Those include:

- power functions, e.g. `square`, `cube`, `sqrt`, `pow`, `reciprocal` etc.;
- trigonometric functions, e.g. `sin`, `atan` etc.;
- exponential/hyperbolic functions, like `exp`, `sinh`, `log`, `atanh` etc.;
- miscellaneous elementwise operations, like taking absolute value, rounding and clipping, such as `abs`, `sign`, 
`ceil`, `round`, `clipByValue`, `clipByNorm` etc.; 
- reductions along specified dimensions: `min`, `amax`, `mean`, `asum`, `logEntropy`, and similar; 
- distance (reduction) operations, such as `euclideanDistance`, `manhattanDistance`, `jaccardDistance`, `cosineDistance`, 
`hammingDistance`, `cosineSimilarity`, along specified dimensions, for two identically shaped `SDVariables`;
- specific matrix operations: `matrixInverse`, `matrixDeterminant`, `diag` (creating a diagonal matrix), `trace`, `eye` 
(creating identity matrix with variable dimensions), and several others;
- more statistics operations: `standardize`, `moment`, `normalizeMoments`, `erf` and `erfc` (Gaussian error function and
its complementary);
- counting and indexing reductions: methods like `conuntZero` (number of zero elements), `iamin` (index of the element 
with the smallest absolute value), `firstIndex` (an index of the first element satisfying a specified `Condition` function);
- reductions indicating properties of the underlying arrays. These include e.g. `isNaN` (elementwise checking), `isMax` 
(shape-preserving along specified dimensions), `isNonDecreasing` (reduction along specified dimensions);
- elementwise logical operations: `and`, `or`, `xor`, `not`.

Most operations in `math` have very simple structure, and are inferred like that:
```java
SDVariable activation = sameDiff.math.cube(input);
```
Operations may be chained, although in a more cumbersome way in comparison to the `SDVariable` operations, e.g.:
```java
SDVariable matrixNorm1 = sameDiff.math.max(sameDiff.math.sum(sameDiff.math.abs(matrix), 1));
```
Observe that the (integer) argument `1` in the `sum` operation tells us that we have to take maximum absolute value 
along the `1`'s dimension, i.e. the column of the matrix.

### `random` - creating random values Random
These operations create variables whose underlying arrays will be filled with random numbers following some distribution 
- say, Bernoulli, normal, binomial etc.. These values will be reset at each iteration. If you wish, for instance,
to create a variable that will add a Gaussian noise to entries of the MNIST database, you may do something like:
```java
double mean = 0.;
double deviation = 0.05;
long[] shape = new long[28, 28];
SDVariable noise_mnist = sameDiff.random.normal("noise_mnist", mean, deviation, shape);
```
The shape of you random variable may vary. Suppose, for instance, that you have audio signals of varying length, and you
want to add noise to them. Then, you need to specify an `SDVariable`, say, `windowShape` with an integer 
[data type](./samediff/variabeles/datatype!!!), and proceed like that
```java
SDVariabel noise_audio = sameDiff.random.normal("noise_audio", mean, deviation, windowShape);
```

### `nn` - general neural network tools
Here we store methods for neural networks that are not necessarily associated with convolutional ones. Among them are
- creation of dense linear and ReLU layers (with or without bias), and separate bias addition: `linear`, `reluLayer`, 
`biasAdd`;
- popular activation functions, e.g. `relu`, `sigmoid`, `tanh`, `softmax` as well as their less used versions like 
`leakyRelu`, `elu`, `hardTanh`, and many more;
- padding for 2d arrays with method `pad`, supporting several padding types, with both constant and variable padding width;
- explosion/overfitting prevention, such as `dropout`, `layerNorm`  and `batchNorm` for layer resp. batch normalization;

Some methods were created for internal use, but are openly available. Those include: 
- derivatives for several popular activation functions - these are mostly designed for speeding up 
backpropagation;
- attention modules - basically, building blocks for recurrent neural networks we shall discuss below.

While activations in `nn` are fairly simple, other operations become more involved. Say, to create a linear 
or a ReLU layer, up to three predefined `SDVariable` objects may be required, as in the following code:
```java
SDVariable denseReluLayer = sameDiff.nn.reluLayer(input, weights, bias);
```
where `input`, `weights` and `bias` need to have dimensions suiting each other. 

To create, say, a dense layer with softmax activation, you may proceed as follows: 
```java
SDVariable linear = sameDiff.nn.linear(input, weight, bias);
SDVariable output = sameDiff.nn.softmax(linear);
``` 

### `cnn` - convolutional neural networks tools
The `cnn` module contains layers and operations typically used in convolutional neural networks - 
different activations may be picked up from the `nn` module. Among `cnn` operations we currently have creation of:
- linear convolution layers, currently for tensors of dimension up to 3 (minibatch not included): `conv1d`, `conv2d`, 
`conv3d`, `depthWiseConv2d`, `separableConv2D`/`sconv2d`; 
- linear deconvolution layers, currently `deconv1d`, `deconv2d`, `deconv3d`; 
- pooling, e.g. `maxPoooling2D`, `avgPooling1D`;
- specialized reshaping methods: `batchToSpace`, `spaceToDepth`, `col2Im` and alike;
- upsampling, currently presented by `upsampling2d` operation;
- local response normalization: `localResponseNormalization`, currently for 2d convolutional layers only;

Convolution and deconvolution operations are specified by a number of static parameters like kernel size, 
dilation, having or not having bias etc.. To facilitate the creation process, we pack the required parameters into 
easily constructable and alterable configuration objects. Desired activations may be borrowed from the `nn` module. So, 
for example, if we want to create a 3x3 convolutional layer with `relu` activation, we may proceed as follows:
```java
Conv2DConfig config2d = new Conv2DConfig().builder().kW(3).kH(3).pW(2).pH(2).build();
SDVariable convolution2dLinear = sameDiff.cnn.conv2d(input, weights, config2d);
SDVariable convolution2dOutput = sameDiff.nn.relu(convolution2dLinear);
``` 
In the first line, we construct a convolution configuration using its default constructor. Then we specify the 
kernel size (this is mandatory) and optional padding size, keeping other settings default (unit stride, no 
dilation, no bias, `NCHW` data format). We then employ this configuration to create a linear convolution with predefined
`SDVariables` for input and weights; the shape of `weights` is to be tuned to that of `input` and to `config` 
beforehand. Thus, if in the above example `input` has shape, say, `[-1, nIn, height, width]`, then `weights` are to have
a form `[nIn, nOut, 3, 3]` (because we have 3x3 convolution kernel). The shape of the resulting variable `convoluton2d` 
will be predetermined by these parameters (in our case, it will be `[-1, nOut, height, width]`). Finally, in the last 
line we apply a `relu` activation.

### `rnn` - Recurrent neural networks

This module contains arguably the most sophisticated methods in the framework. Currently it allows you to create 
- simple recurrent units, using `sru` and `sruCell` methods;
- LSTM units, using `lstmCell`, `lstmBlockCell` and `lstmLayer`;
- Graves LSTM units, using `gru` methods.

As of now, recurrent operations require special configuration objects as input, in which you need to pack all the 
variables that will be used in a unit. This is subject to change in the later versions. For instance, to 
create a simple recurrent unit, you need to proceed like that:
```java
SRUConfiguration sruConfig = new SRUConfiguration(input, weights, bias, init);
SDVariable sruOutput = samediff.rnn().sru(sruConfig);
```
Here, the arguments in the `SRUConfiguration` constructor are variables that are to be defined beforehand. Obviously 
their shapes should be matching, and these shapes predetermine the shape of `output`.

### `loss` - Loss functions
In this branch we keep common loss functions. Most loss functions may be created quite simply, like that:
```java
SDVariable logLoss = sameDiff.loss.logLoss("logLoss", label, predictions);
```
where `labels` and `predictions` are `SDVariable`'s. A `String` name is a mandatory parameter in most `loss` methods, 
yet it may be set to `null` - in this case, the name will be generated automatically. You may also create weighted loss
functions by adding another `SDVariable` parameters containing weights, as well as specify a reduction method (see below) 
for the loss over the minibatch. Thus, a full-fledged `logLoss` operation may 
look like:
```java
SDVariable wLogLossMean = sameDiff.loss.logLoss("wLogLossMean", label, predictions, weights, LossReduce.MEAN_BY_WEIGHT);
```
Some loss operations may allow/require further arguments, depending on their type: e.g. a dimension along which the 
loss is to be computed (as in `cosineLoss`), or some real-valued parameters.  

As for reduction methods, over the minibatch, there are currently 4 of them available. Thus, initially loss values for 
each sample of the minibatch are computed, then they are multiplied by weights (if specified), and finally one of the 
following routines takes place: 
- `NONE` - leaving the resulting (weighted)loss values as-is; the result is an `INDArray` with the length of the 
minibatch: `sum_loss = sum(weights * loss_per_sample)`.
- `SUM` - summing the values, producing a scalar result.
- `MEAN_BY_WEIGHT` - first computes the sum as above, and then divides it by the sum of all weights, producing a scalar 
value: `mean_loss = sum(weights * loss_per_sample) / sum(weights)`. If weights are not
specified, they all are set to `1.0` and this reduction is equivalent to getting mean loss value over the minibatch.  
- `MEAN_BY_NONZERO_WEIGHT_COUNT` - divides the weighted sum by the number of nonzero weight, producing a scalar: 
`mean_count_loss = sum(weights * loss_per_sample) / count(weights != 0)`. Useful e.g. when you want to compute the mean
only over a subset of *valid* samples, setting weights by either `0.` or `1.`. When weights are not given, it just 
produces mean, and thus equivalent to `MEAN_BY_WEIGHT`.
 

## The *don'ts* of operations 

In order for `SameDiff` operations to work properly, several main rules are to be upheld. Failing to do so may result in
an exception or, worse even, to a working code producing undesired results. All the things we mention in the current 
section describe what **you better not** do.

- All variables in an operation have to belong to the same instance of `SamdeDiff` (see the [variables](./samediff/variables)
section on how variables are added to a `SameDiff` instance). In other words, **you better not** 
```java
SDVariable x = sameDiff0.var(DataType.FLOAT, 1);
SDVariable y = sameDiff1.placeHolder(DataType.FLOAT, 1);
SDVariable z = x.add(y);
```
- At best, a new variable is to be created for a result of an operation or a chain of operations. In other words, **you 
better not** redefine existing variables **and better not** leave operations returning no result. In other words, try to 
**avoid** the code like this:
```java
SDVariable z = x.add(y);
//DON'T!!!
z.mul(2);
x = z.mul(y);
``` 
A properly working version of the above code (if we've desired to obtain 2xy+2y<sup>2</sup> in an unusual way) will be
```java
SDVariable z = x.add(y);
SDVariable _2z = z.mul(2);
w = _2z.mul(y);
```
 To learn more why it functions like that, see our [graph section](./samediff/graph).
