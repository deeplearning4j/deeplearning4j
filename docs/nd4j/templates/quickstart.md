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
    * all major operating systems: win/linux/osx/android.
    * architectures: x86, arm, ppc.

This quickstart follows the same layout and approach of the [Numpy quickstart](https://docs.scipy.org/doc/numpy/user/quickstart.html). This should help people familiar with Python and Numpy get started quickly with Nd4J.

## Prerequisites
<!--- // Java, Maven, git. Coding skills and hello world example. -->

To follow the examples in this quick start you will need to know some Java. You can use Nd4J from any [JVM Language](https://en.wikipedia.org/wiki/List_of_JVM_languages). (For example: Scala, Kotlin). You will also need to install the following software on your computer:
<!--- from the dl4j quickstart, pointing to the dl4j quiclstart for details. -->
* [Java (developer version)](./deeplearning4j-quickstart#Java) 1.7 or later (Only 64-Bit versions supported)
* [Apache Maven](./deeplearning4j-quickstart#Maven) (automated build and dependency manager)
<!--- git allows us to start with a cleaner project than mvn create. -->
* [Git](./deeplearning4j-quickstart#Git)(distributed version control system)

To improve readability we show you the output of `System.out.println(...)`.  But we have not show the print statement in the sample code. If you are confident you know how to use maven and git, please feel free to skip to the [Basics](#Basics). In the remainder of this section we will build a small 'hello ND4J' application to verify the prequisites are set up correctly.

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
[         0,         0]
```

## Basics
<!--- TODO: We will put some into this page. Start with refering to existing doc. -->
While this quickstart is being build, please refer to our existing 
[basics usage](./nd4j-basics) document.

The main feature of Nd4j is the versatile n-dimensional array interface called INDArray. To improve performance Nd4j uses off-heap memory to store data. The INDArray is different from standard Java arrays.

Some of the key properties and methods for an INDArray x are as follows:

```java
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.buffer.DataType;
        
INDArray x = Nd4j.zeros(3,4);

// The number of axes (dimensions) of the array.
int dimensions = x.rank();

// The dimensions of the array. The size in each dimension.
long[] shape = x.shape();

// The total number of elements.
long length = x.length();

// The type of the array elements. 
DataType dt = x.dataType();
```
<!--- staying away from itemsize and data buffer. The numpy quickstart has these. -->

### Array Creation
To create INDArrays you use the static factory methods of the [Nd4j](https://deeplearning4j.org/api/latest/org/nd4j/linalg/factory/Nd4j.html ) class.

<!--- We have good docs on creating INDArrays already.  -->
<!--- https://deeplearning4j.org/docs/latest/nd4j-overview#creating -->
The `Nd4j.createFromArray` function is overloaded to make it easy to create INDArrays from regular Java arrays. The example below uses Java `double` arrays. Similar create methods are overloaded for `float`, `int` and `long`. The `Nd4j.createFromArray` function has overloads up to 4d for all types.

```java
double arr_2d[][]={{1.0,2.0,3.0},{4.0,5.0,6.0},{7.0,8.0,9.0}};
INDArray x_2d = Nd4j.createFromArray(arr_2d);

double arr_1d[]={1.0,2.0,3.0};
INDArray  x_1d = Nd4j.createFromArray(arr_1d);
```

Nd4j can create arrays initialized with zeros and ones using the functions `zeros` and `ones`. The `rand` function allows you to create an array initialized with random values.
The default datatype of the INDArray created is `float`. Some overloads allow you to set the datatype.

```java
INDArray  x = Nd4j.zeros(5);
//[         0,         0,         0,         0,         0], FLOAT

int [] shape = {5};
x = Nd4j.zeros(DataType.DOUBLE, 5);
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
INDArray  x = Nd4j.linspace(1, 10, 5); //start, stop, count.
// [    1.0000,    3.2500,    5.5000,    7.7500,   10.0000]

// Evaluate a function over many points.
import static org.nd4j.linalg.ops.transforms.Transforms.sin;
INDArray  x = Nd4j.linspace(0.0, Math.PI, 100, DataType.DOUBLE);
INDArray  y = sin(x);  
```

### Printing Arrays
The INDArray supports Java's `toString()` method. The current implementation has limited precision and a limited number of elements.  The output is similar to printing NumPy arrays:
```java
INDArray  x = Nd4j.arange(6);  //1d array
System.out.println(x);  //We just give the output of the print command from here on.
// [         0,    1.0000,    2.0000,    3.0000,    4.0000,    5.0000]

int [] shape = {4,3};
x = Nd4j.arange(12).reshape(shape);   //2d array
/*
[[         0,    1.0000,    2.0000], 
 [    3.0000,    4.0000,    5.0000], 
 [    6.0000,    7.0000,    8.0000], 
 [    9.0000,   10.0000,   11.0000]]
*/

int [] shape2 = {2,3,4};
x = Nd4j.arange(24).reshape(shape2);  //3d array
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
You will have to use INDArray methods to perform operations on your arrays. There are  in-place and copy overloads and scalar and element wise overloaded versions. The in-place operators return a reference to the array so you can conveniently chain operations together. Use in-place operators where possible to improve performance. Copy operators have new array creation overhead.

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

// casting x2 to DOUBLE solves the problem:
INDArray x3 = x.add(x2.castTo(DataType.DOUBLE));
```
<!--- Moving matrix operations after Transforms. In Nd4j the dot product is a transform. -->

The INDArray has methods implementing reduction/accumulation operations such as `sum`, `min`, `max`.
```java
int [] shape = {2,3};
INDArray  x = Nd4j.rand(shape);
x;
x.sum();
x.min();
x.max();
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
INDArray x = Nd4j.arange(12).reshape(3, 4);
/*
[[         0,    1.0000,    2.0000,    3.0000], 
 [    4.0000,    5.0000,    6.0000,    7.0000], 
 [    8.0000,    9.0000,   10.0000,   11.0000]]
*/        

x.sum(0); // Sum of each column.
//[   12.0000,   15.0000,   18.0000,   21.0000]

x.min(1); // Min of each row
//[         0,    4.0000,    8.0000]

x.cumsum(1); // cumulative sum across each row,
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
import static org.nd4j.linalg.ops.transforms.Transforms.exp;
import static org.nd4j.linalg.ops.transforms.Transforms.sqrt;

INDArray x = Nd4j.arange(3);
// [         0,    1.0000,    2.0000]
exp(x);
// [    1.0000,    2.7183,    7.3891]
sqrt(x);
// [         0,    1.0000,    1.4142]
```

You can check out a complete list of transform operations in the [Javadoc](https://deeplearning4j.org/api/latest/org/nd4j/linalg/api/ops/impl/transforms/package-summary.html )


### Matrix multiplication
We have already seen the element wise multiplcation in the basic operations. The other Matrix operations have their own methods:

```java
INDArray x = Nd4j.arange(12).reshape(3, 4);
/*
[[         0,    1.0000,    2.0000,    3.0000], 
 [    4.0000,    5.0000,    6.0000,    7.0000], 
 [    8.0000,    9.0000,   10.0000,   11.0000]]
*/

INDArray y = Nd4j.arange(12).reshape(4, 3);
/*
[[         0,    1.0000,    2.0000], 
 [    3.0000,    4.0000,    5.0000], 
 [    6.0000,    7.0000,    8.0000], 
 [    9.0000,   10.0000,   11.0000]]
*/

x.mmul(y);  // matrix product.
/*
[[   42.0000,   48.0000,   54.0000], 
 [  114.0000,  136.0000,  158.0000], 
 [  186.0000,  224.0000,  262.0000]]
*/

// dot product.
INDArray x = Nd4j.arange(12);
INDArray y = Nd4j.arange(12);
dot(x, y);  
//506.0000
```

### Indexing, Slicing and Iterating
Indexing, Slicing and Iterating is harder in Java than in Python. 
To retreive individual values from an INDArray you can use the `getDouble`, `getFloat` or `getInt` methods. INDArrays cannot be indexed like Java arrays. You can get a Java array from an INDArray using `toDoubleVector()`,  `toDoubleMatrix()`, `toFloatVector()` and `toFloatMatrix()` 

```java

INDArray x = Nd4j.arange(12);
// [         0,    1.0000,    2.0000,    3.0000,    4.0000,    5.0000,    6.0000,    7.0000,    8.0000,    9.0000,   10.0000,   11.0000]

float f = x.getFloat(3);  // Single element access. Other methods: getDouble, getInt, ...
// 3.0

float []  fArr = x.toFloatVector(); //Convert to Java array.
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
The number of elements along each axis is in the shape. The shape can be changed with various methods,.
```java
INDArray x = Nd4j.rand(3,4);
x.shape();
// [3, 4]

INDArray x2 = x.ravel();
x2.shape();
// [12]

INDArray x3 = x.reshape(6,2).shape();
x3.shape();
//[6, 2]

// Be aware that x, x2, and x3 share the same data. 
x2.putScalar(5, -1.0);

System.out.println( x);
/*
[[    0.0270,    0.3799,    0.5576,    0.3086], 
 [    0.2266,   -1.0000,    0.1107,    0.4895], 
 [    0.8431,    0.6011,    0.2996,    0.7500]]
*/

System.out.println( x2);
// [    0.0270,    0.3799,    0.5576,    0.3086,    0.2266,   -1.0000,    0.1107,    0.4895,    0.8431,    0.6011,    0.2996,    0.7500]

System.out.println( x3);
/*        
[[    0.0270,    0.3799], 
 [    0.5576,    0.3086], 
 [    0.2266,   -1.0000], 
 [    0.1107,    0.4895], 
 [    0.8431,    0.6011], 
 [    0.2996,    0.7500]]
*/
```

### Stacking together different arrays
Arrays can be stacked together using the `vstack` and `hstack` methods.

```java
INDArray x = Nd4j.rand(2,2);
INDArray y = Nd4j.rand(2,2);

x
/*
[[    0.1462,    0.5037], 
 [    0.1418,    0.8645]]
*/

y;
/*
[[    0.2305,    0.4798], 
 [    0.9407,    0.9735]]
*/
 
Nd4j.vstack(x, y);
/*
[[    0.1462,    0.5037], 
 [    0.1418,    0.8645], 
 [    0.2305,    0.4798], 
 [    0.9407,    0.9735]]
*/

Nd4j.hstack(x, y);
/*
[[    0.1462,    0.5037,    0.2305,    0.4798], 
 [    0.1418,    0.8645,    0.9407,    0.9735]]
*/
```

<!--- No hsplit and vsplit functions in Nd4J. -->
<!--- ### Splitting one array into several smaller ones -->


## Copies and View
When working with INDArrays the data is not always copied. Here are three cases you should be aware of.

### No Copy at All
Simple assignments make no copy of the data. Java passes objects by reference. No copies are made on a method call. 

```java
INDArray x = Nd4j.rand(2,2);
INDArray y = x; // y and x point to the same INData object.

public static void f(INDArray x){
    // No copy is made. Any changes to x are visible after the function call.
    }

```

### View or Shallow Copy
Some functions will return a view of an array. 

```java
INDArray x = Nd4j.rand(3,4);
INDArray  x2 = x.ravel();
INDArray  x3 = x.reshape(6,2);

x2.putScalar(5, -1.0); // Changes x, x2 and x3

x
/*
[[    0.8546,    0.1509,    0.0331,    0.1308], 
 [    0.1753,   -1.0000,    0.2277,    0.1998], 
 [    0.2741,    0.8257,    0.6946,    0.6851]]
*/

x2
// [    0.8546,    0.1509,    0.0331,    0.1308,    0.1753,   -1.0000,    0.2277,    0.1998,    0.2741,    0.8257,    0.6946,    0.6851]

x3
/*
[[    0.8546,    0.1509], 
 [    0.0331,    0.1308], 
 [    0.1753,   -1.0000], 
 [    0.2277,    0.1998], 
 [    0.2741,    0.8257], 
 [    0.6946,    0.6851]]
*/

```

### Deep Copy
To make a copy of the array use the `dup` method. This will give you a new array with new data.

```java
INDArray x = Nd4j.rand(3,4);
INDArray  x2 = x.ravel().dup();

x2.putScalar(5, -1.0); // Now only changes x2.
        
x
/*
[[    0.1604,    0.0322,    0.8910,    0.4604], 
 [    0.7724,    0.1267,    0.1617,    0.7586], 
 [    0.6117,    0.5385,    0.1251,    0.6886]]
*/

x2
// [    0.1604,    0.0322,    0.8910,    0.4604,    0.7724,   -1.0000,    0.1617,    0.7586,    0.6117,    0.5385,    0.1251,    0.6886]
```

## Functions and Methods Overview

### Array Creation
 [arange](https://deeplearning4j.org/api/latest/org/nd4j/linalg/factory/Nd4j.html#arange-double-double- ),
 [create](https://deeplearning4j.org/api/latest/org/nd4j/linalg/factory/Nd4j.html#create-org.nd4j.linalg.api.buffer.DataBuffer- ),
 [copy](https://deeplearning4j.org/api/latest/org/nd4j/linalg/factory/Nd4j.html#copy-org.nd4j.linalg.api.ndarray.INDArray-org.nd4j.linalg.api.ndarray.INDArray- ),
 [empty](https://deeplearning4j.org/api/latest/org/nd4j/linalg/factory/Nd4j.html#empty-org.nd4j.linalg.api.buffer.DataBuffer.Type- ), 
 [empty_like](https://deeplearning4j.org/api/latest/org/nd4j/linalg/factory/Nd4j.html#emptyLike-org.nd4j.linalg.api.ndarray.INDArray- ), 
 [eye]( https://deeplearning4j.org/api/latest/org/nd4j/linalg/factory/Nd4j.html#eye-long- ), 
 [linspace](https://deeplearning4j.org/api/latest/org/nd4j/linalg/factory/Nd4j.html#linspace-double-double-long- ),
 [meshgrid](https://deeplearning4j.org/api/latest/org/nd4j/linalg/factory/Nd4j.html#meshgrid-org.nd4j.linalg.api.ndarray.INDArray-org.nd4j.linalg.api.ndarray.INDArray- ), 
 [ones](https://deeplearning4j.org/api/latest/org/nd4j/linalg/factory/Nd4j.html#ones-int...- ),
 [ones_like](https://deeplearning4j.org/api/latest/org/nd4j/linalg/factory/Nd4j.html#onesLike-org.nd4j.linalg.api.ndarray.INDArray- ), 
 [rand](https://deeplearning4j.org/api/latest/org/nd4j/linalg/factory/Nd4j.html#rand-int-int- ),
 [readTxt](https://deeplearning4j.org/api/latest/org/nd4j/linalg/factory/Nd4j.html#readTxt-java.lang.String- ),
 [zeros](https://deeplearning4j.org/api/latest/org/nd4j/linalg/factory/Nd4j.html#zeros-int...- ), 
 [zeros_like](https://deeplearning4j.org/api/latest/org/nd4j/linalg/factory/Nd4j.html#zerosLike-org.nd4j.linalg.api.ndarray.INDArray- )

### Conversions
[convertToDoubles](https://deeplearning4j.org/api/latest/org/nd4j/linalg/api/ndarray/INDArray.html#convertToDoubles-- ), 
[convertToFloats](https://deeplearning4j.org/api/latest/org/nd4j/linalg/api/ndarray/INDArray.html#convertToFloats-- ), 
[convertToHalfs](https://deeplearning4j.org/api/latest/org/nd4j/linalg/api/ndarray/INDArray.html#convertToHalfs-- )

### Manipulations 
[concatenate](https://deeplearning4j.org/api/latest/org/nd4j/linalg/factory/Nd4j.html#concat-int-org.nd4j.linalg.api.ndarray.INDArray...- ),
[hstack](https://deeplearning4j.org/api/latest/org/nd4j/linalg/factory/Nd4j.html#hstack-org.nd4j.linalg.api.ndarray.INDArray...- ), 
 [ravel](https://deeplearning4j.org/api/latest/org/nd4j/linalg/api/ndarray/INDArray.html#ravel-- ),
 [repeat](https://deeplearning4j.org/api/latest/org/nd4j/linalg/api/ndarray/INDArray.html#repeat-int-long...- ), 
 [reshape](https://deeplearning4j.org/api/latest/org/nd4j/linalg/api/ndarray/INDArray.html#reshape-long...- ),
[squeeze](https://deeplearning4j.org/api/latest/org/nd4j/linalg/factory/Nd4j.html#squeeze-org.nd4j.linalg.api.ndarray.INDArray-int- ), 
[swapaxes](https://deeplearning4j.org/api/latest/org/nd4j/linalg/api/ndarray/INDArray.html#swapAxes-int-int- ),
[tear](https://deeplearning4j.org/api/latest/org/nd4j/linalg/factory/Nd4j.html#tear-org.nd4j.linalg.api.ndarray.INDArray-int...- ),
[transpose](https://deeplearning4j.org/api/latest/org/nd4j/linalg/api/ndarray/INDArray.html#transpose-- ),
[vstack](https://deeplearning4j.org/api/latest/org/nd4j/linalg/factory/Nd4j.html#vstack-org.nd4j.linalg.api.ndarray.INDArray...- )

### Ordering
[argmax](https://deeplearning4j.org/api/latest/org/nd4j/linalg/api/ndarray/INDArray.html#argMax-int...- ),
[max](https://deeplearning4j.org/api/latest/org/nd4j/linalg/api/ndarray/INDArray.html#max-int...- ),
[min](https://deeplearning4j.org/api/latest/org/nd4j/linalg/api/ndarray/INDArray.html#min-int...- ),
[sort](https://deeplearning4j.org/api/latest/org/nd4j/linalg/factory/Nd4j.html#sort-org.nd4j.linalg.api.ndarray.INDArray-int-boolean- )

### Operations
[choice](https://deeplearning4j.org/api/latest/org/nd4j/linalg/factory/Nd4j.html#choice-org.nd4j.linalg.api.ndarray.INDArray-org.nd4j.linalg.api.ndarray.INDArray-org.nd4j.linalg.api.ndarray.INDArray- ),
[cumsum](https://deeplearning4j.org/api/latest/org/nd4j/linalg/api/ndarray/INDArray.html#cumsum-int- ), 
[mmul](https://deeplearning4j.org/api/latest/org/nd4j/linalg/api/ndarray/INDArray.html#mmul-org.nd4j.linalg.api.ndarray.INDArray- ), 
[prod](https://deeplearning4j.org/api/latest/org/nd4j/linalg/api/ndarray/INDArray.html#prod-int...- ), 
[put](https://deeplearning4j.org/api/latest/org/nd4j/linalg/api/ndarray/INDArray.html#put-org.nd4j.linalg.api.ndarray.INDArray-org.nd4j.linalg.api.ndarray.INDArray- ),
[putWhere](https://deeplearning4j.org/api/latest/org/nd4j/linalg/api/ndarray/INDArray.html#putWhere-org.nd4j.linalg.api.ndarray.INDArray-org.nd4j.linalg.api.ndarray.INDArray-org.nd4j.linalg.indexing.conditions.Condition- ),
 [sum](https://deeplearning4j.org/api/latest/org/nd4j/linalg/api/ndarray/INDArray.html#sum-int...- )

### Basic Statistics
[covarianceMatrix](https://deeplearning4j.org/api/latest/org/nd4j/linalg/dimensionalityreduction/PCA.html#getCovarianceMatrix--),
[mean](https://deeplearning4j.org/api/latest/org/nd4j/linalg/api/ndarray/INDArray.html#mean-int...- ),
[std](https://deeplearning4j.org/api/latest/org/nd4j/linalg/api/ndarray/INDArray.html#std-int...- ),
[var](https://deeplearning4j.org/api/latest/org/nd4j/linalg/api/ndarray/INDArray.html#var-int...- )

### Basic Linear Algebra
<!--- Not too happy with these links and (lack of) javadoc in them. -->
[cross](https://deeplearning4j.org/api/latest/org/nd4j/linalg/api/ops/impl/shape/Cross.html ),
 [dot](https://deeplearning4j.org/api/latest/org/nd4j/linalg/api/ops/impl/accum/Dot.html ), [gesvd](https://deeplearning4j.org/api/latest/org/nd4j/linalg/api/blas/Lapack.html#gesvd-org.nd4j.linalg.api.ndarray.INDArray-org.nd4j.linalg.api.ndarray.INDArray-org.nd4j.linalg.api.ndarray.INDArray-org.nd4j.linalg.api.ndarray.INDArray- ),
[mmul](https://deeplearning4j.org/api/latest/org/nd4j/linalg/api/ndarray/INDArray.html#mmul-org.nd4j.linalg.api.ndarray.INDArray-)
