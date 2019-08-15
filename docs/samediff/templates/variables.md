---
title: Types of variables in SameDiff
short_title: Variables
description: What types of variables are used in SameDiff, their properties and how to switch these types.
category: SameDiff
weight: 3
---

# Variables in `SameDiff`

## What are variables

All values defining or passing through each `SameDiff` instance - be it weights, bias, inputs, activations or 
general parameters - all are handled by objects of class `SDVariable`. 

Observe that by variables we normally mean not just single values - as it is done in various online examples describing 
autodifferentiation - but rather whole multidimensional arrays of them.

## Variable types

All variables in `SameDiff` belong to one of four *variable types*, constituting an enumeration `VariableType`. 
Here they are:

- `VARIABLE`: are trainable parameters of your network, e.g. weights and bias of a layer. Naturally, we want them
to be both stored for further usage - we say, that they are *persistent* - as well as being updated during training.
- `CONSTANT`: are those parameters which, like variables, are persistent for the network, but are not being 
trained; they, however, may be changed externally by the user. 
- `PLACEHOLDER`: store temporary values that are to be supplied from the outside, like inputs and labels. 
Accordingly, since new placeholders' values are provided at each iteration, they are not stored: in other words, 
unlike `VARIABLE` and `CONSTANT`, `PLACEHOLDER` is *not* persistent.
- `ARRAY`: are temporary values as well, representing outputs of [operations](./samediff/ops) within a `SameDiff`, for 
instance sums of vectors, activations of a layer, and many more. They are being recalculated at each iteration, and 
therefor, like `PLACEHOLDER`, are not persistent.

To infer the type of a particular variable, you may use the method `getVariableType`, like so:
```java
VariableType varType = yourVariable.getVariableType();
```
The current value of a variable in a form of `INDArray` may be obtained using `getArr` or `getArr(true)` - the latter 
one if you wish the program to throw an exception if the variable's value is not initialized. 

## Data types

The data within each variable also has its *data type*, contained in `DataType` enum. Currently in `DataType` there 
are three *floating point* types: `FLOAT`, `DOUBLE` and `HALF`; four *integer* types: `LONG`, `INT`, `SHORT` and 
`UBYTE`; one *boolean* type `BOOL` - all of them will be referred as *numeric* types. In addition, there is a 
*string* type dubbed `UTF8`; and two helper data types `COMPRESSED` and `UNKNOWN`. The 16-bit floating point format `BFLOAT16` and unsigned integer types (`UINT16`, `UINT32` and `UINT64`) will be available in `1.0.0-beta5`.

To infer the data type of your variable, use
```java
DataType dataType = yourVariable.dataType();
```
You may need to trace your variable's data type since at times it does matter, which types you use in an operation. For 
example, a convolution product, like this one
```java
SDVariable prod = samediff.cnn.conv1d(input, weights, config);
```
will require its `SDVariable` arguments `input` and `weights` to be of one of the floating point data types, and will
throw an exception otherwise. Also, as we shall discuss just below, all the `SDVariables` of type `VARIABLE` are 
supposed to be of floating point type.

## Common features of variables

Before we go to the differences between variables, let us first look at the properties they all share 
- All variables are ultimately derived from an instance of `SameDiff`, serving as parts of its 
[graph](./samediff/graphs). In fact, each variable has a `SameDiff` as one of its fields.
- Results (outputs) of all operations are of `ARRAY` type. 
- All `SDVariable`'s involved in an operation are to belong to the *same* `SameDiff`. 
- All variables may or may not be given names - in the latter case, a name is actually created automatically. Either
way, the names need to be/are created unique. We shall come back to naming below.

## Differences between variable types

Let us now have a closer look at each type of variables, and what distinguish them from each other.

### Variables

Variables are the trainable parameters of your network. This predetermines their nature in `SameDiff`. As we briefly 
mentioned above, variables' values need to be 
both preserved for application, and updated during training. Training means, that we iteratively 
update the values by small fractions of their gradients, and this only makes sense if variables are of *floating 
point* types (see data types above).

Variables may be added to your `SameDiff` using different versions of `var` function from your `SameDiff` instance. 
For example, the code
```java
SDVariable weights = samediff.var("weights", DataType.FLOAT, 784, 10);
```
adds a variable constituting of a 784x10 array of `float` numbers - weights for a single layer MNIST perceptron 
in this case - to a pre-existing `SameDiff` instance `samediff`.

However, this way the values within a variable will be set as zeros. You may also create a variable with values from
a preset `INDArray`. Say
```java
SDVariable weights = samediff.var("weigths", Nd4j.nrand(784, 10).div(28));
```
will create a variable filled with normally distributed randomly generated numbers with variance `1/28`. You may put
any other array creation methods instead of `nrand`, or any preset array, of course. Also, you may use some popular 
initialization scheme, like so:

```java
SDVariable weights = samediff.var("weights", new XavierInitScheme('c', 784, 10), DataType.FLOAT, 784, 10);
```
Now, the weights will be randomly initialized using the Xavier scheme. There are other ways to create and 

fill variables: you may look them up in the 'known subclasses' section [of our javadoc](https://deeplearning4j.org/api/latest/org/nd4j/weightinit/WeightInitScheme.html").

### Constants

Constants hold values that are stored, but - unlike variables - remain unchanged during training. These, for
instance, may be some hyperparamters you wish to have in your network and be able to access from the outside. Or 
they may be pretrained weights of a neural network that you wish to keep unchanged (see more on that in 
[Changing Variable Type](https://deeplearning4j.org/api/latest/) below). Constants may be of any data type 
- so e.g. `int` and `boolean` are allowed alongside with `float` and `double`.

In general, constants are added to `SameDiff` by means of `constant` methods. A constant may be created form an 
`INDArray`, like that:
```java
SDVariable constant = samediff.constant("constants", Nd4j.create(new float[] {3.1415f, 42f}));
```
A constant consisting of a single scalar value may be created using one of the `scalar` methods:
```java
INDArray someScalar = samediff.scalar("scalar", 42);
```
Again, we refer to the [javadoc](https://deeplearning4j.org/api/latest/) for the whole reference.

### Placeholders

The most common placeholders you'll normally have in a `SameDiff` are inputs and, when applicable, labels. You may 
create placeholders of any data type, depending on the operations you use them in. To add a placeholder to a `SameDiff`, 
you may call one of `placeHolder` methods, e.g. like that:
```java
SDVariable in = samediff.placeHolder("input", DataType.FLOAT, -1, 784);
```
as in MNIST example. Here we specify name, data type and then shape of your placeholder - here, we have 
28x28 grayscale pictures rendered as 1d vectors (therefore 784) coming in batches of length we don't know beforehand 
(therefore -1). 

### Arrays

Variables of `ARRAY` type appear as outputs of [operations](./samediff/ops) within `SameDiff`. 
Accordingly, the data type of an array-type variable depends on the kind of operation it is produced by and variable 
type(s) ot its argument(s). Arrays are not persistent - they are one-time values that will be recalculated from scratch 
at the next step. However, unlike placeholders, gradients are computed for them, as those are needed to update the values
of `VARIABLE`'s. 

There are as many ways array-type variables are created as there are operations, so you're better up focusing on
our [operations section](./samediff/ops), our [javadoc](https://deeplearning4j.org/api/latest/) and [examples](./samediff/exampes).

## Recap table

Let us summarize the main properties of variable types in one table:

|                | Trainable   | Gradients | Persistent | Workspaces | Datatypes   | Instantiated from    | 
| ----------     | ----------- | --------- | ---------- | -----------| ----------  | ----------           |
| `VARIABLE`     | Yes         | Yes       | Yes        | Yes        | Float only  | Instance             |
| `CONSTANT`     | No          | No        | Yes        | No         | Any         | Instance             |
| `PLACEHOLDER`  | No          | No        | No         | No         | Any         | Instance             |
| `ARRAY`        | No          | Yes       | No         | Yes        | Any         | Operations           |

We haven't discussed what 'Workspaces' mean - if you do not know, do not worry, this is an internal technical term that 
basically describes how memory is managed internally.

## Changing variable types

You may change variable types as well. For now, there are three of such options:

### Variable to constant
At times - for instance if you perform transfer learning - you may wish to turn a variable into a constant. This is 
done like so:
```java
samediff.convertToConstant(someVariable);
```
where `someVariable` is an instance of `SDVariable` of `VARIABLE` type. The variable `someVariable` will not be trained
any more.

### Constant to variable
Conversely, constants - if they are of *floating point* data type - may be converted to variables. So, for instance, if 
you wish your frozen weights to become trainable again
```java
samediff.convertToVariable(frozenWeights); //not frozen any more
```
### Placeholder to constant
Placeholders may be converted to constants as well - for instance, if you need to freeze one of the inputs. There are no 
restrictions on the data type, yet, since placeholder values are not persistent, their value should be set before you 
turn them into constants. This can be done as follows
```java
placeHolder.setArray(someArray);
samediff.convertToConstant(placeHolder);
```
For now it is not possible to turn a constant back into a placeholder, we may consider adding this functionality if 
there is a need for that. For now, if you wish to effectively freeze your placeholder but be able to use it again, 
consider supplying it with constant values rather than turning it into a constant.

## Variables' names and values
### Getting variables from `SameDiff`
Recall that every variable in an instance of `SameDiff` has its unique `String` name. Your `SameDiff` actually tracks your 
variables by their names, and allows you to retrieve them by using `getVariable(String name)` method.

Consider the following line:
```java
SDVariable regressionCost = weights.mmul(input).sub("regression_prediction", bias).squaredDifference(labels);
```
Here, in the function `sub` we actually have implicitly introduced a variable (of type `ARRAY`) that holds the 
result of the subtraction. By adding a name into the operations's argument, we've secured ourselves the possibility
to retrieve the variable from elsewhere: say, if later you need to infer the difference between the labels and the
prediction as a vector, you may just write:
```java
SDVariable errorVector = samediff.getVariable("regressionPrediction").sub(labels);
```
This becomes especially handy if your whole `SameDiff` instance is initialized elsewhere, and you still need to get
hold of some of its variables - say, multiple outputs. 

You can get and set the name of an `SDVariable` the methods `getVarName` and `setVarName` 
respectively. When renaming, note that variable's name is to remain unique within its `SameDiff`.

### Getting variable's value
You may retrieve any variable's current value as an `INDArray` using the method `eval()`. Note that for non-persistent 
variables, the value should first be set. For variables with gradients, the gradient's value may also be inferred using
the method `getGradient`. 
 



