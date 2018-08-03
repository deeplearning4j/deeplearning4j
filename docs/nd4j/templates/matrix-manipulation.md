---
title: Matrix Manipulation
short_title: Matrix Manipulation
description: Operations for matrix manipulation, including transpose, reshape, in ND4J.
category: ND4J
weight: 10
---


There are several other basic matrix manipulations to highlight as you learn ND4J's workings. ([Example code](https://github.com/SkymindIO/nd4j-examples/blob/master/src/main/java/org/nd4j/examples/ReshapeOperationExample.java).)

### Transpose

The transpose of a matrix is its mirror image. An element located in row 1, column 2, in matrix A will be located in row 2, column 1, in the the transpose of matrix A, whose mathematical notation is A to the T, or A^T. Notice that the elements along the diagonal of a square matrix do not move -- they are at the hinge of the reflection. In ND4J, transpose matrices like this:
```java
INDArray nd = Nd4j.create(new float[]{1, 2, 3, 4}, new int[]{2, 2});

[1.0 ,3.0]
[2.0 ,4.0]                                                                                                                      
nd.transpose();

[1.0 ,2.0]
[3.0 ,4.0]
```
And a long matrix like this
```java
[1.0 ,3.0 ,5.0 ,7.0 ,9.0 ,11.0]
[2.0 ,4.0 ,6.0 ,8.0 ,10.0 ,12.0]
```
Looks like this when it is transposed
```java
[1.0 ,2.0]
[3.0 ,4.0]
[5.0 ,6.0]
[7.0 ,8.0]
[9.0 ,10.0]
[11.0 ,12.0]
```
In fact, transpose is just an important subset of a more general operation: reshape.

### Reshape

Yes, matrices can be reshaped. You can change the number of rows and columns they have. The reshaped matrix has to fulfill one condition: the product of its rows and columns must equal the product of the row and columns of the original matrix. For example, proceeding columnwise, you can reshape a 3 by 4 matrix into a 2 by 6 matrix:
```java
    INDArray nd2 = Nd4j.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, new int[]{2, 6});
```
The array nd2 looks like this
```java
[1.0 ,3.0 ,5.0 ,7.0 ,9.0 ,11.0]
[2.0 ,4.0 ,6.0 ,8.0 ,10.0 ,12.0]
```
Reshaping it is easy, and follows the same convention by which we gave it shape to begin with
```java
nd2.reshape(3,4);

[1.0 ,4.0 ,7.0 ,10.0]
[2.0 ,5.0 ,8.0 ,11.0]
[3.0 ,6.0 ,9.0 ,12.0]
```
### Linear view

This is straight view of an arbitrary nd-array. You can go through the nd-array like a vector, linearly, squashing it into one long line. Linear view allows you to do nondestructive operations (reshape and other operations can be destructive because elements are changed within the nd-array). Linear views are only good for elementwise operations (rather than matrix operations), since the views do not preserve the order of the buffer.
```java
nd2.linearView();

[1.0 ,2.0 ,3.0 ,4.0 ,5.0 ,6.0 ,7.0 ,8.0 ,9.0 ,10.0 ,11.0 ,12.0]
```
### Broadcast

Broadcast is advanced. It usually happens in the background without having to be called. The simplest way to understand it is by working with one long row vector, like the one above.
```java
nd2 = Nd4j.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
```
Broadcasting will actually take multiple copies of that row vector and put them together into a larger matrix. The first parameter is the number of copies you want "broadcast," as well as the number of rows involved. In order not to throw a compiler error, make the second parameter of broadcast equal to the number of elements in your row vector.
```java
nd2.broadcast(new int[]{3,12});

[1.0 ,4.0 ,7.0 ,10.0 ,1.0 ,4.0 ,7.0 ,10.0 ,1.0 ,4.0 ,7.0 ,10.0]
[2.0 ,5.0 ,8.0 ,11.0 ,2.0 ,5.0 ,8.0 ,11.0 ,2.0 ,5.0 ,8.0 ,11.0]
[3.0 ,6.0 ,9.0 ,12.0 ,3.0 ,6.0 ,9.0 ,12.0 ,3.0 ,6.0 ,9.0 ,12.0]

nd2.broadcast(new int[]{6,12});

[1.0 ,7.0 ,1.0 ,7.0 ,1.0 ,7.0 ,1.0 ,7.0 ,1.0 ,7.0 ,1.0 ,7.0]
[2.0 ,8.0 ,2.0 ,8.0 ,2.0 ,8.0 ,2.0 ,8.0 ,2.0 ,8.0 ,2.0 ,8.0]
[3.0 ,9.0 ,3.0 ,9.0 ,3.0 ,9.0 ,3.0 ,9.0 ,3.0 ,9.0 ,3.0 ,9.0]
[4.0 ,10.0 ,4.0 ,10.0 ,4.0 ,10.0 ,4.0 ,10.0 ,4.0 ,10.0 ,4.0 ,10.0]
[5.0 ,11.0 ,5.0 ,11.0 ,5.0 ,11.0 ,5.0 ,11.0 ,5.0 ,11.0 ,5.0 ,11.0]
[6.0 ,12.0 ,6.0 ,12.0 ,6.0 ,12.0 ,6.0 ,12.0 ,6.0 ,12.0 ,6.0 ,12.0]
```
For various other matrix transforms, see our page on [Functions](../functions.html).