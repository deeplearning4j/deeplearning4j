---
title: ND4J Syntax
short_title: Syntax
description: General syntax and structure of the ND4J API.
category: ND4J
weight: 10
---


For the complete nd4j-api index, please consult the [Javadoc](../doc).

There are three types of operations used in ND4J: scalars, transforms and accumulations. We’ll use the word op synonymously with operation. You can see the lists of those three kinds of [ND4J ops under the directories here]( https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl
). Each Java file in each list is an op.

Most of the ops just take [enums](https://docs.oracle.com/javase/tutorial/java/javaOO/enum.html), or a list of discrete values that you can autocomplete. Activation functions are the exception, because they take strings such as `"relu"` or `"tanh"`.

Scalars, transforms and accumulations each have their own patterns. Transforms are the simplest, since the take a single argument and perform an operation on it. Absolute value is a transform that takes the argument `x` like so `abs(IComplexNDArray ndarray)` and produces the result which is the absolute value of x. Similarly, you would apply to the sigmoid transform `sigmoid()` to produce the "sigmoid of x".

Scalars just take two arguments: the input and the scalar to be applied to that input. For example, `ScalarAdd()` takes two arguments: the input `INDArray x` and the scalar `Number num`; i.e. `ScalarAdd(INDArray x, Number num)`. The same format applies to every Scalar op.

Finally, we have accumulations, which are also known as reductions in GPU-land. Accumulations add arrays and vectors to one another and can *reduce* the dimensions of those arrays in the result by adding their elements in a rowwise op. For example, we might run an accumulation on the array
```java
[1 2
3 4]
```
Which would give us the vector
```
[3
7]
```
Reducing the columns (i.e. dimensions) from two to one.

Accumulations can be either pairwise or scalar. In a pairwise reduction, we might be dealing with two arrays, x and y, which have the same shape. In that case, we could calculate the cosine similarity of x and y by taking their elements two by two.

        cosineSim(x[i], y[i])

Or take `EuclideanDistance(arr, arr2)`, a reduction between one array `arr` and another `arr2`.

Many ND4J ops are overloaded, meaning methods sharing a common name have different argument lists. Below we will explain only the simplest configurations.

As you can see, there are three possible argument types with ND4J ops: inputs, optional arguments and outputs. The outputs are specified in the ops' constructor. The inputs are specified in the parentheses following the method name, always in the first position, and the optional arguments are used to transform the inputs; e.g. the scalar to add; the coefficient to multiply by, always in the second position.

|Method| What it does |
|:----|:-------------:|
|**Transforms**||
|ACos(INDArray x)|Trigonometric inverse cosine, elementwise. The inverse of cos such that, if `y = cos(x)`, then `x = ACos(y)`.|
|ASin(INDArray x)|Also known as arcsin. Inverse sine, elementwise.|
|ATan(INDArray x)|Trigonometric inverse tangent, elementwise. The inverse of tan, such that, if `y = tan(x)` then `x = ATan(y)`.|
|Transforms.tanh(myArray)|Hyperbolic tangent: a sigmoidal function. This applies elementwise tanh inplace.|
|Nd4j.getExecutioner().exec(Nd4j.getOpFactory() .createTransform("tanh", myArray))|equivalent to the above|

For other transforms, [please see this page](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/ops/transforms/Transforms.java).

Here are two examples of performing `z = tanh(x)`, in which the original array `x` is unmodified.
```java
INDArray x = Nd4j.rand(3,2);	//input
INDArray z = Nd4j.create(3,2); //output
Nd4j.getExecutioner().exec(new Tanh(x,z));
Nd4j.getExecutioner().exec(Nd4j.getOpFactory().createTransform("tanh",x,z));
```
The latter two examples above use ND4J's basic convention for all ops, in which we have 3 NDArrays, x, y and z.
```
x is input, always required
y is (optional) input, only used in some ops (like CosineSimilarity, AddOp etc)
z is output
```
Frequently, `z = x` (this is the default if you use a constructor with only one argument). But there are exceptions for situations like `x = x + y`. Another possibility is `z = x + y`, etc.

## Accumulations  

Most accumulations are accessable directly via the INDArray interface.

For example, to add up all elements of an NDArray:
```java
double sum = myArray.sumNumber().doubleValue();
```
Accum along dimension example - i.e., sum values in each row:
```java
INDArray tenBy3 = Nd4j.ones(10,3);	//10 rows, 3 columns
INDArray sumRows = tenBy3.sum(0);
System.out.println(sumRows);	//Output: [ 10.00, 10.00, 10.00]
```
Accumulations along dimensions generalize, so you can sum along two dimensions of any array with two or more dimensions.

## Subset Operations on Arrays

A simple example:
```java
INDArray random = Nd4j.rand(3, 3);
System.out.println(random);
[[0.93,0.32,0.18]
[0.20,0.57,0.60]
[0.96,0.65,0.75]]

INDArray lastTwoRows = random.get(NDArrayIndex.interval(1,3),NDArrayIndex.all());
```
Interval is fromInclusive, toExclusive; note that can equivalently use inclusive version: NDArrayIndex.interval(1,2,true);
```java
System.out.println(lastTwoRows);
[[0.20,0.57,0.60]
[0.96,0.65,0.75]]

INDArray twoValues = random.get(NDArrayIndex.point(1),NDArrayIndex.interval(0, 2));
System.out.println(twoValues);
[ 0.20, 0.57]
```
These are views of the underlying array, **not** copy operations (which provides greater flexibility and doesn't have cost of copying).
```java
twoValues.addi(5.0);
System.out.println(twoValues);
[ 5.20, 5.57]

System.out.println(random);
[[0.93,0.32,0.18]
[5.20,5.57,0.60]
[0.96,0.65,0.75]]
```
To avoid in-place behaviour, random.get(...).dup() to make a copy.

|**Scalar**||
|INDArray.add(number)|Returns the result of adding `number` to each entry of `INDArray x`; e.g. myArray.add(2.0)|
|INDArray.addi(number)|Returns the result of adding `number` to each entry of `INDArray x`.|
|ScalarAdd(INDArray x, Number num)|Returns the result of adding `num` to each entry of `INDArray x`.|
|ScalarDivision(INDArray x, Number num)|Returns the result of dividing each entry of `INDArray x` by `num`.|
|ScalarMax(INDArray x, Number num)|Compares each entry of `INDArray x` to `num` and returns the higher quantity.|
|ScalarMultiplication(INDArray x, Number num)|Returns the result of multiplying each entry of `INDArray x` by `num`.|
|ScalarReverseDivision(INDArray x, Number num)|Returns the result of dividing `num` by each element of `INDArray x`.|
|ScalarReverseSubtraction(INDArray x, Number num)|Returns the result of subtracting each entry of `INDArray x` from `num`.|
|ScalarSet(INDArray x, Number num)|This sets the value of each entry of `INDArray x` to `num`.|
|ScalarSubtraction(INDArray x, Number num)|Returns the result of subtracting `num` from each entry of `INDArray x`.|


If you do not understand the explanation of ND4J's syntax, cannot find a definition for a method, or would like to request that a function be added, please let us know on [Gitter live chat](https://gitter.im/deeplearning4j/deeplearning4j).