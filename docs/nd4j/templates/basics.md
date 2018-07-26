---
title: Elementwise Operations and Basic Usage
short_title: Basic Usage
description: How to use elementwise operations and other beginner concepts in ND4J.
category: ND4J
weight: 1
---


## Introduction

The basic operations of linear algebra are matrix creation, addition and multiplication. This guide will show you how to perform those operations with ND4J, as well as various advanced transforms.

* [Matrix Operations](../matrixwise.html)
* [Reshape/Transpose Matrices](../reshapetranspose.html)
* [Functions](../functions.html)
* [Swapping CPUs for GPUs](../gpu_native_backends.html)

The Java code below will create a simple 2 x 2 matrix, populate it with integers, and place it in the nd-array variable nd:
```java
INDArray nd = Nd4j.create(new float[]{1,2,3,4},new int[]{2,2});
```
If you print out this array
```java
System.out.println(nd);
```
you'll see this
```java
[[1.0 ,3.0]
[2.0 ,4.0]
]
```
A matrix with two rows and two columns, which orders its elements by column and which we'll call matrix nd.

A matrix that ordered its elements by row would look like this:
```java
[[1.0 ,2.0]
[3.0 ,4.0]
]
```
## Elementwise scalar operations

The simplest operations you can perform on a matrix are elementwise scalar operations; for example, adding the scalar 1 to each element of the matrix, or multiplying each element by the scalar 5. Let's try it.
```java
nd.add(1);
```
This line of code represents this operation:
```java
[[1.0 + 1 ,3.0 + 1]
[2.0 + 1,4.0 + 1]
]
```
and here is the result
```java
[[2.0 ,4.0]
[3.0 ,5.0]
]
```
There are two ways to perform any operation in ND4J, destructive and nondestructive; i.e. operations that change the underlying data, or operations that simply work with a copy of the data. Destructive operations will have an "i" at the end -- addi, subi, muli, divi.  The "i" means the operation is performed "in place," directly on the data rather than a copy, while nd.add() leaves the original untouched.

Elementwise scalar multiplication looks like this:
```java
nd.mul(5);
```
And produces this:
```java
[[10.0 ,20.0]
[15.0 ,25.0]
]
```
Subtraction and division follow a similar pattern:
```java
nd.subi(3);
nd.divi(2);
```
If you perform all these operations on your initial 2 x 2 matrix, you should end up with this matrix:
```java
[[3.5 ,8.5]
[6.0 ,11.0]
]
```
## Elementwise vector operations

When performed with simple units like scalars, the operations of arithmetic are unambiguous. But working with matrices, addition and multiplication can mean several things. With vector-on-matrix operations, you have to know what kind of addition or multiplication you're performing in each case.

First, we'll create a 2 x 2 matrix, a column vector and a row vector.
```java
INDArray nd = Nd4j.create(new float[]{1,2,3,4},new int[]{2,2});
INDArray nd2 = Nd4j.create(new float[]{5,6},new int[]{2,1}); //vector as column
INDArray nd3 = Nd4j.create(new float[]{5,6},new int[]{2}); //vector as row
```
Notice that the shape of the two vectors is specified with their final parameters. {2,1} means the vector is vertical, with elements populating two rows and one column. A simple {2} means the vector populates along a single row that spans two columns -- horizontal. You're first matrix will look like this
```java
[[1.00, 2.00],
 [3.00, 4.00]]
```
Here's how you add a column vector to a matrix:

        nd.addColumnVector(nd2);

And here's the best way to visualize what's happening. The top element of the column vector combines with the top elements of each column in the matrix, and so forth. The sum matrix represents the march of that column vector across the matrix from left to right, adding itself along the way.
```java
[1.0 ,2.0]     [5.0]    [6.0 ,7.0]
[3.0 ,4.0]  +  [6.0] =  [9.0 ,10.0]
```
But let's say you preserved the initial matrix and instead added a row vector.
```java
nd.addRowVector(nd3);
```
Then your equation is best visualized like this:
```java
[1.0 ,2.0]                   [6.0 ,8.0]
[3.0 ,4.0]  +  [5.0 ,6.0] =  [8.0 ,10.0]
```
In this case, the leftmost element of the row vector combines with the leftmost elements of each row in the matrix, and so forth. The sum matrix represents that row vector falling down the matrix from top to bottom, adding itself at each level.

So vector addition can lead to different results depending on the orientation of your vector. The same is true for multiplication, subtraction and division and every other vector operation.

In ND4J, row vectors and column vectors look the same when you print them out with
```java
System.out.println(nd);
```
They will appear like this.
```java
[5.0 ,6.0]
```
Don't be fooled. Getting the parameters right at the beginning is crucial. addRowVector and addColumnVector will not produce different results when using the same initial vector, because they do not change a vector's orientation as row or column.

## Elementwise matrix operations

To carry out scalar and vector elementwise operations, we basically pretend we have two matrices of equal shape. Elementwise scalar multiplication can be represented several ways.
```java
    [1.0 ,3.0]   [c , c]   [1.0 ,3.0]   [1c ,3c]
c * [2.0 ,4.0] = [c , c] * [2.0 ,4.0] = [2c ,4c]
```
So you see, elementwise operations match the elements of one matrix with their precise counterparts in another matrix. The element in row 1, column 1 of matrix nd will only be added to the element in row one column one of matrix c.

This is clearer when we start elementwise vector operations. We imaginee the vector, like the scalar, as populating a matrix of equal dimensions to matrix nd. Below, you can see why row and column vectors lead to different sums.

Column vector:
```java
[1.0 ,3.0]     [5.0]   [1.0 ,3.0]   [5.0 ,5.0]   [6.0 ,8.0]
[2.0 ,4.0]  +  [6.0] = [2.0 ,4.0] + [6.0 ,6.0] = [8.0 ,10.0]
```
Row vector:
```java
[1.0 ,3.0]                   [1.0 ,3.0]    [5.0 ,6.0]   [6.0 ,9.0]    
[2.0 ,4.0]  +  [5.0 ,6.0] =  [2.0 ,4.0] +  [5.0 ,6.0] = [7.0 ,10.0]
```
Now you can see why row vectors and column vectors produce different results. They are simply shorthand for different matrices.

Given that we've already been doing elementwise matrix operations implicitly with scalars and vectors, it's a short hop to do them with more varied matrices:
```java
INDArray nd4 = Nd4j.create(new float[]{5,6,7,8},new int[]{2,2});

nd.add(nd4);
```
Here's how you can visualize that command:
```java
[1.0 ,3.0]   [5.0 ,7.0]   [6.0 ,10.0]
[2.0 ,4.0] + [6.0 ,8.0] = [8.0 ,12.0]
```
Muliplying the initial matrix nd with matrix nd4 works the same way:
```java
nd.muli(nd4);

[1.0 ,3.0]   [5.0 ,7.0]   [5.0 ,21.0]
[2.0 ,4.0] * [6.0 ,8.0] = [12.0 ,32.0]
```
The term of art for this particular matrix manipulation is a [Hadamard product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices)).

These toy matrices are a useful heuristic to introduce the ND4J interface as well as basic ideas in linear algebra. This framework, however, is built to handle billions of parameters in n dimensions (and beyond...).

Next, we'll look at more complicated [matrix operations](../matrixwise.html).