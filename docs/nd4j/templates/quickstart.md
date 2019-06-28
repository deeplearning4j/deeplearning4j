---
title: Quickstart
short_title: Quick start tutorial
description: ND4J Key features and brief samples.
category: ND4J
weight: 1
---
<!--- Comments are standard html. Tripple dash based on stackoverflow: https://stackoverflow.com/questions/4823468/comments-in-markdown -->

<!--- Borrowing the layout of the Numpy quickstart to get started. -->

## Introduction
<!--- What is ND4J and why is it important. From the nd4j repo readme.  -->
ND4J is a scientific computing library for the JVM. It is meant to be used in production environments rather than as a research tool, which means routines are designed to run fast with minimum RAM requirements. The main features are:
* A versatile n-dimensional array object.
* Linear algebra and signal processing functions.
* Multiplatform functionality including GPUs.

This quickstart follows the same layout and approach of the [Numpy quickstart](https://docs.scipy.org/doc/numpy/user/quickstart.html). This should help people familiar with Python and Numpy get started quickly with Nd4J.

## Prerequisites
<!--- // Java, Maven, git. Coding skills and hello world example. -->

To follow the examples in this quick start you will need to know some Java. You will also need to install the following software on your computer:
<!--- from the dl4j quickstart, pointing to the dl4j quiclstart for details. -->
* [Java (developer version)](./deeplearning4j-quickstart#Java) 1.7 or later (Only 64-Bit versions supported)
* [Apache Maven](./deeplearning4j-quickstart#Maven) (automated build and dependency manager)
<!--- git allows us to start with a cleaner project than mvn create. -->
* [Git](./deeplearning4j-quickstart#Git)(distributed version control system)

If you are confident you know how to use maven and git, please feel free to skip to the [Basics](#Basics). In the remainder of this section we will build a small 'hello ND4J' application to verify the prequisites are set up correctly.

Execute the following commands to get the project from github. 

<!--- TODO: Create HelloNd4J or Quickstart-nd4j repo in Deeplearning4J. -->
```shell
git clone https://github.com/RobAltena/HelloNd4J.git

cd HelloNd4J

mvn install

mvn exec:java -Dexec.mainClass="HelloNd4j"
```

When everything is set up correctly you should see the following output:

```shell
SLF4J: Failed to load class "org.slf4j.impl.StaticLoggerBinder".
SLF4J: Defaulting to no-operation (NOP) logger implementation
SLF4J: See http://www.slf4j.org/codes.html#StaticLoggerBinder for further details.
[         0,         0]
```

## Basics
<!--- TODO: We will put some into this page. Start with refering to existing doc. -->
While this quickstart is being build, please refer to our existing 
[basics usage](./nd4j-basics) document.

The main feature of Nd4j is the versatile n-dimensional array interface called INDArray. To improve performance Nd4j uses off-heap memory to store data. The INDArray is different from standard Java arrays.

Some of the key properties and methods for an INDArray x are as follows:

```java
// The number of axes (dimensions) of the array.
int dimensions = x.shape().length;

// The dimensions of the array. The size in each dimension.
long[] shape = x.shape();

// The total number of elements.
length = x.length();

// The type of the array elements. 
DataType dt = x.dataType();
```
<!--- staying away from itemsize and data buffer. The numpy quickstart has these. -->

### Array Creation
To create INDArrays you use the static factory methods of the [Nd4j](https://deeplearning4j.org/api/latest/org/nd4j/linalg/factory/Nd4j.html ) class.

<!--- We have good docs on creating INDArrays already.  -->
<!--- https://deeplearning4j.org/docs/latest/nd4j-overview#creating -->
The `Nd4j.create` function is overloaded to make it easy to create INDArrays from regular Java arrays. The example below uses Java `double` arrays. Similar create methods are overloaded for `float`, `int` and `long`.

```java
double arr_2d[][]={{1.0,2.0,3.0},{4.0,5.0,6.0},{7.0,8.0,9.0}};
INDArray x_2d = Nd4j.create(arr_2d);

double arr_1d[]={1.0,2.0,3.0};
INDArray  x_1d = Nd4j.create(arr_1d);
```

There are overloaded `create` functions for INDArrays up to 4 dimensions. To create INDArrays with any shape use the `create` functions that take a shape as one of their arguments.

```java
double[] flat = ArrayUtil.flattenDoubleArray(myDoubleArray);
int[] shape = ...;	//Array shape here
INDArray myArr = Nd4j.create(flat,shape,'c');
```

Nd4j can create arrays initialized with zeros and ones using the functions `zeros` and `ones`. The `rand` function allows you to create an array initialized with random values.
The default datatype of the INDarray created is `float`. Some overloads allow you to set the datatype.

```java
INDArray  x = Nd4j.zeros(5);
//[         0,         0,         0,         0,         0], FLOAT

int [] shape = {5};
x = Nd4j.zeros(shape, DataType.DOUBLE);
//[         0,         0,         0,         0,         0], DOUBLE

// For higher dimensions you can provide a shape array. 2D random matrix example:
int rows = 4;
int cols = 5;
int[] shape = {rows, cols};
INDArray x = Nd4j.rand(shape);
```

Use the `arange` functions to create an array of evenly spaces values:

```java
INDArray  x = Nd4j.arange(5);
// [         0,    1.0000,    2.0000,    3.0000,    4.0000]

INDArray  x = Nd4j.arange(2, 7);
// [    2.0000,    3.0000,    4.0000,    5.0000,    6.0000]
```

The `linspace` function allows you to specify the number of points generated:
```java
INDArray  x = Nd4j.linspace(1, 10, 5);
// [    1.0000,    3.2500,    5.5000,    7.7500,   10.0000]

// Evaluate a function over many points.
import static org.nd4j.linalg.ops.transforms.Transforms.sin;
INDArray  x = Nd4j.linspace(0.0, Math.PI, 100, DataType.DOUBLE);
INDArray  y = sin(x);  
```

### Printing Arrays
The INDArray supports Java's `toString()` method. The output is similar to printing NumPy arrays:
```java
INDArray  x = Nd4j.arange(6);  //1d array
System.out.println(x);
// [         0,    1.0000,    2.0000,    3.0000,    4.0000,    5.0000]

int [] shape = {4,3};
x = Nd4j.arange(12).reshape(shape);   //2d array
System.out.println(x);
/*
[[         0,    1.0000,    2.0000], 
 [    3.0000,    4.0000,    5.0000], 
 [    6.0000,    7.0000,    8.0000], 
 [    9.0000,   10.0000,   11.0000]]
*/

int [] shape2 = {2,3,4};
x = Nd4j.arange(24).reshape(shape2);  //3d array
System.out.println(x);
/*
[[[         0,    1.0000,    2.0000,    3.0000], 
  [    4.0000,    5.0000,    6.0000,    7.0000], 
  [    8.0000,    9.0000,   10.0000,   11.0000]], 

 [[   12.0000,   13.0000,   14.0000,   15.0000], 
  [   16.0000,   17.0000,   18.0000,   19.0000], 
  [   20.0000,   21.0000,   22.0000,   23.0000]]]
*/
```

### Basic Operations
You will have to use INDArray methods to perform operations on your arrays. There are  in-place and copy overloads and scalar and element wise overloaded versions. The in-place operators return a reference to the array so you can conveniently chain operations together.

```java
//Copy
arr_new = arr.add(scalar);    // return a new array with scalar added to each element of arr.
arr_new = arr.add(other_arr); // return a new array with element wise addition of arr and other_arr.

//in place.
arr_new = arr.addi(scalar); //Heads up: arr_new points to the same array as arr.
arr_new = arr.addi(other_arr);
```

addition: arr.add(...), arr.addi(...)
substraction: arr.sub(...), arr.subi(...)
multiplication: arr.mul(...), arr.muli(...)
division: arr.div(...), arr.divi(...)

When you perform the basic operations you must make sure the underlying data types are the same.
```java
int [] shape = {5};
INDArray  x = Nd4j.zeros(shape, DataType.DOUBLE);
INDArray  x2 = Nd4j.zeros(shape, DataType.INT);
INDArray  x3 = x.add(x2);
// java.lang.IllegalArgumentException: Op.X and Op.Y must have the same data type, but got INT vs DOUBLE
```
<!--- Moving matrix operations after Transforms. In Nd4j the dot product is a transform. -->

The IndArray has methods implementing operations such as `sum`, `min`, `max`.
```java
int [] shape = {2,3};
INDArray  x = Nd4j.rand(shape);
System.out.println(x);

System.out.println(x.sum());
System.out.println(x.min());
System.out.println(x.max());
/*
[[    0.8621,    0.9224,    0.8407], 
 [    0.1504,    0.5489,    0.9584]]
4.2830
0.1504
0.9584
*/
```

Provide a dimension argument to apply the operation across the specified dimension:

```java
int [] shape = {3,4};
INDArray x = Nd4j.arange(12).reshape(shape);
System.out.println(x);
/*
[[         0,    1.0000,    2.0000,    3.0000], 
 [    4.0000,    5.0000,    6.0000,    7.0000], 
 [    8.0000,    9.0000,   10.0000,   11.0000]]
*/        

System.out.println(x.sum(0)); // Sum of each column.
//[   12.0000,   15.0000,   18.0000,   21.0000]

System.out.println(x.min(1)); // Min of each row
//[         0,    4.0000,    8.0000]

System.out.println(x.cumsum(1)); // cumulative sum across each row,
/*
[[         0,    1.0000,    3.0000,    6.0000], 
 [    4.0000,    9.0000,   15.0000,   22.0000], 
 [    8.0000,   17.0000,   27.0000,   38.0000]]
*/

```

<!--- The numpy quickstart calls them Universal Functions. -->
### Transform operation
Nd4j provides familiar mathematical functions such as sin, cos, and exp. These are called transform operations. The result is returned as an INDArray.

```java
INDArray x = Nd4j.arange(3);
System.out.println(x);
// [         0,    1.0000,    2.0000]
System.out.println(exp(x));
// [    1.0000,    2.7183,    7.3891]
System.out.println(sqrt(x));
// [         0,    1.0000,    1.4142]
```

You can check out a complete list of transform operations in the [Javadoc](https://deeplearning4j.org/api/latest/org/nd4j/linalg/api/ops/impl/transforms/package-summary.html )


### Matrix multiplication
We have already seen the element wise multiplcation in the basic operations. The other Matrix operations have their own methods:

```java
int[] shape = {3, 4};
INDArray x = Nd4j.arange(12).reshape(shape);
int[] shape2 = {4, 3};
INDArray y = Nd4j.arange(12).reshape(shape2);

System.out.println(x);
/*
[[         0,    1.0000,    2.0000,    3.0000], 
 [    4.0000,    5.0000,    6.0000,    7.0000], 
 [    8.0000,    9.0000,   10.0000,   11.0000]]
*/

System.out.println(y);
/*
[[         0,    1.0000,    2.0000], 
 [    3.0000,    4.0000,    5.0000], 
 [    6.0000,    7.0000,    8.0000], 
 [    9.0000,   10.0000,   11.0000]]
*/

System.out.println(x.mmul(y));  // matrix product.
/*
[[   42.0000,   48.0000,   54.0000], 
 [  114.0000,  136.0000,  158.0000], 
 [  186.0000,  224.0000,  262.0000]]
*/

// dot product.
INDArray x = Nd4j.arange(12);
INDArray y = Nd4j.arange(12);
System.out.println(dot(x, y));  
//506.0000
```

### Indexing, Slicing and Iterating
Indexing, Slicing and Iterating is harder in Java than in Python. 
To retreive individual values from an INDArray you can use the `getDouble`, `getFloat` or `getInt` methods. INDArrays cannot be indexed like Java arrays. You can get a Java array from an INDArray using `.data().asDouble()`. 

```java

INDArray x = Nd4j.arange(12);
// [         0,    1.0000,    2.0000,    3.0000,    4.0000,    5.0000,    6.0000,    7.0000,    8.0000,    9.0000,   10.0000,   11.0000]

float f = x.getFloat(3);  // Single element access. Other methods: getDouble, getInt, ...
// 3.0

float []  fArr = x.data().asFloat(); //Convert to Java array.
// [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]

INDArray x2 = x.get(NDArrayIndex.interval(2, 6));
// [    2.0000,    3.0000,    4.0000,    5.0000]

// On a copy of x: From start to position 6, exclusive, set every 2nd element to -1.0
INDArray y = x.dup();
y.get(NDArrayIndex.interval(0, 2, 6)).assign(-1.0);
//[   -1.0000,    1.0000,   -1.0000,    3.0000,   -1.0000,    5.0000,    6.0000,    7.0000,    8.0000,    9.0000,   10.0000,   11.0000]

// reversed copy of y.
INDArray y2 = Nd4j.reverse(y.dup());
//[   11.0000,   10.0000,    9.0000,    8.0000,    7.0000,    6.0000,    5.0000,   -1.0000,    3.0000,   -1.0000,    1.0000,   -1.0000]

```

For multidimensional arrays you should use `INDArray.get(NDArrayIndex...)`. The example below shows how to iterate over the rows and columns of a 2D array. Note that for 2D arrays we could have used the `getColumn` and `getRow` convenience methods. 

```java
// Iterate over the rows and columns of a 2d arrray.
int rows = 4;
int cols = 5;
int[] shape = {rows, cols};

INDArray x = Nd4j.rand(shape);
/*
[[    0.2228,    0.2871,    0.3880,    0.7167,    0.9951], 
 [    0.7181,    0.8106,    0.9062,    0.9291,    0.5115], 
 [    0.5483,    0.7515,    0.3623,    0.7797,    0.5887], 
 [    0.6822,    0.7785,    0.4456,    0.4231,    0.9157]]
*/
        
for (int row=0; row<rows; row++) {
	INDArray y = x.get(NDArrayIndex.point(row), NDArrayIndex.all());
	}
/*
[    0.2228,    0.2871,    0.3880,    0.7167,    0.9951]
[    0.7181,    0.8106,    0.9062,    0.9291,    0.5115]
[    0.5483,    0.7515,    0.3623,    0.7797,    0.5887]
[    0.6822,    0.7785,    0.4456,    0.4231,    0.9157]
*/
	
for (int col=0; col<cols; col++) {
	INDArray y = x.get(NDArrayIndex.all(), NDArrayIndex.point(col));
	}
/*
[    0.2228,    0.7181,    0.5483,    0.6822]
[    0.2871,    0.8106,    0.7515,    0.7785]
[    0.3880,    0.9062,    0.3623,    0.4456]
[    0.7167,    0.9291,    0.7797,    0.4231]
[    0.9951,    0.5115,    0.5887,    0.9157]
*/
	
```

## Shape Manipulation
### Changing the shape of an array
### Stacking together different arrays
### Splitting one array into several smaller ones

## Copies and View
### No Copy at All
### View or Shallow Copy
### Deep Copy

## Functions and Methods Overview
// List of links. Start with similar methods as the numpy quickstart.

// From here the Numpy quickstart goes deeper. For now we stop here.