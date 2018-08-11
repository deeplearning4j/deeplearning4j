---
title: ND4J Overview
short_title: Overview
description: Comprehensive programming guide for ND4J.
category: ND4J
weight: 0
---

This user guide is designed to explain (and provide examples for) the main functionality in ND4J.

* <a href="#intro">Introduction</a>
  * <a href="#inmemory">NDArrays: How Are They Stored in Memory?</a>
  * <a href="#views">Views: When an two or more INDArrays refer to the same data</a>
* <a href="#creating">Creating NDArrays</a>
  * <a href="#createzero">Zero, One and Scalar-Value Initialized Arrays</a>
  * <a href="#createrandom">Random arrays</a>
  * <a href="#createfromjava">Creating NDArrays from Java arrays</a>
  * <a href="#createfromndarray">Creating NDArrays from other NDArrays</a>
  * <a href="#createother">Miscellaneous NDArray Creation Methods</a>
* <a href="#individual">Getting and Setting Individual Values</a>
* <a href="#getset">Getting and Setting Parts of NDArrays</a>
  * <a href="#getsetrows">getRow() and putRow()</a>
  * <a href="#getsetsub">Sub-arrays: get(), put() and NDArrayIndex</a>
  * <a href="#getsettensor">Tensor Along Dimension</a>
  * <a href="#getsetslice">Slice</a>
* <a href="#ops">Performing Operations on NDArrays</a>
  * <a href="#opsscalar">Scalar Ops</a>
  * <a href="#opstransform">Transform Ops</a>
  * <a href="#accumops">Accumulation (Reduction) Ops</a>
  * <a href="#indexaccumops">Index Accumulation Ops,</a>
  * <a href="#opsbroadcast">Broadcast and Vector Operations</a>
* <a href="#boolean">Boolean Indexing: Selectively Apply Operations Based on a Condition</a>
* <a href="#workspaces">Workspaces</a>
  * <a href="#workspaces-panic">Workspaces: Scope Panic</a>
* <a href="#misc">Advanced and Miscellaneous Topics</a>
  * <a href="#miscdatatype">Setting the data type</a>
  * Reshaping
  * <a href="#miscflattening">Flattening<a>
  * Permute
  * sortRows/sortColumns
  * Directly accessing BLAS operations
* <a href="#serialization">Serialization</a>
* <a href="#quickref">Quick Reference: A Summary Overview of ND4J Methods</a>
* <a href="#faq">FAQ: Frequently Asked Questions</a>


## <a name="intro">Introduction</a>

An NDArray is in essence n-dimensional array: i.e., a rectangular array of numbers, with some number of dimensions.

Some concepts you should be familiar with:

* The *rank* of a NDArray is the number of dimensions. 2d NDArrays have a rank of 2, 3d arrays have a rank of 3, and so on. You can create NDArrays with any arbitrary rank.
* The *shape* of an NDArray defines the size of each of the dimensions. Suppose we have a 2d array with 3 rows and 5 columns. This NDArray would have shape `[3,5]`
* The *length* of an NDArray defines the total number of elements in the array. The length is always equal to the product of the values that make up the shape.
* The *stride* of an NDArray is defined as the separation (in the underlying data buffer) of contiguous elements in each dimension. Stride is defined per dimension, so a rank N NDArray has N stride values, one for each dimension. Note that most of the time, you don't need to know (or concern yourself with) the stride - just be aware that this is how ND4J operates internally. The next section has an example of strides.
* The *data type* of an NDArray refers to the type of data of an NDArray (for example, *float* or *double* precision). Note that this is set globally in ND4J, so all NDArrays should have the same data type. Setting the data type is discussed later in this document.

In terms of indexing there are a few things to know. First, rows are dimension 0, and columns are dimension 1: thus `INDArray.size(0)` is the number of rows, and `INDArray.size(1)` is the number of columns. Like normal arrays in most programming languages, indexing is zero-based: thus rows have indexes `0` to `INDArray.size(0)-1`, and so on for the other dimensions.

Throughout this document, we'll use the term `NDArray` to refer to the general concept of an n-dimensional array; the term `INDArray` refers specifically to the [Java interface](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ndarray/INDArray.java) that ND4J defines. In practice, these two terms can be used interchangeably.

### <a name="inmemory">NDArrays: How Are They Stored in Memory?</a>

The next few paragraphs describe some of architecture behind ND4J. Understanding this is not strictly necessary in order to use ND4J, but it may help you to understand what is going on behind the scenes.
NDArrays are stored in memory as a single flat array of numbers (or more generally, as a single contiguous block of memory), and hence differs a lot from typical Java multidimensional arrays such as a `float[][]` or `double[][][]`.

Physically, the data that backs an INDArray is stored off-heap: that is, it is stored outside of the Java Virtual Machine (JVM). This has numerous benefits, including performance, interoperability with high-performance BLAS libraries, and the ability to avoid some shortcomings of the JVM in high-performance computing (such as issues with Java arrays being limited to 2^31 -1 (2.14 billion) elements due to integer indexing).

In terms of encoding, an NDArray can be encoded in either C (row-major) or Fortran (column-major) order. For more details on row vs. column major order, see [Wikipedia](https://en.wikipedia.org/wiki/Row-major_order). Nd4J may use a combination of C and F order arrays together, at the same time. Most users can just use the default array ordering, but note that is is possible to use a specific ordering for a given array, should the need arise.


The following image shows how a simple 3x3 (2d) NDArray is stored in memory,

![C vs. F order](./images/guide/c_vs_f_order.png)

In the above array, we have:

* `Shape = [3,3]` (3 rows, 3 columns)
* `Rank = 2` (2 dimensions)
* `Length = 9` (3x3=9)
* Stride
  * C order stride: `[3,1]`: the values in consecutive rows are separated in the buffer by 3, and the values consecutive columns are separated in the buffer by 1
  * F order stride: `[1,3]`: the values in consecutive rows are separated in the buffer by 1, and the values in consecutive columns are separated in the buffer by 3




### <a name="views">Views: When Two or More NDArrays Refer to the Same Data</a>

A key concept in ND4J is the fact that two NDArrays can actually point to the same underlying data in memory. Usually, we have one NDArray referring to some subset of another array, and this only occurs for certain operations (such as `INDArray.get()`, `INDArray.transpose()`, `INDArray.getRow()` etc. This is a powerful concept, and one that is worth understanding.

There are two primary motivations for  this:

1. There are considerable performance benefits, most notably in avoiding copying arrays
2. We gain a lot of power in terms of how we can perform operations on our NDArrays

Consider a simple operation like a matrix transpose on a large (10,000 x 10,000) matrix. Using views, we can perform this matrix transpose in constant time without performing any copies (i.e., O(1) in [big O notation](https://en.wikipedia.org/wiki/Big_O_notation)), avoiding the considerable cost copying all of the array elements. Of course, sometimes we *do* want to make a copy - at which point we can use the `INDArray.dup()` to get a copy. For example, to get a *copy* of a transposed matrix, use `INDArray out = myMatrix.transpose().dup()`. After this `dup()` call, there will be no link between the original array `myMatrix` and the array `out` (thus, changes to one will not impact the other).


So see how views can be powerful, consider a simple task: adding 1.0 to the first row of a larger array, `myArray`. We can do this easily, in one line:

`myArray.getRow(0).addi(1.0)`

Let's break down what is happening here. First, the `getRow(0)` operation returns an INDArray that is a view of the original. Note that both `myArrays` and `myArray.getRow(0)` point to the same area in memory:

![getRow(0)](./images/guide/row_addi.png)

then, after the addi(1.0) is performed, we have the following situation:

![getRow(0).addi(1.0)](./images/guide/row_addi_2.png)

As we can see, changes to the NDArray returned by `myArray.getRow(0)` will be reflected in the original array `myArray`; similarly, changes to `myArray` will be reflected in the row vector.

## <a name="creating"> Creating NDArrays </a>

### <a name="createzero">Zero, One and Scalar-Value Initialized Arrays</a>

Two of the most commonly used methods of creating arrays are:

* `Nd4j.zeros(int...)`
* `Nd4j.ones(int...)`

The shape of the arrays are specified as integers. For example, to create a zero-filled array with 3 rows and 5 columns, use `Nd4j.zeros(3,5)`.

These can often be combined with other operations to create arrays with other values. For example, to create an array filled with 10s:

`INDArray tens = Nd4j.zeros(3,5).addi(10)`

The above initialization works in two steps: first by allocating a 3x5 array filled with zeros, and then by adding 10 to each value.

### <a name="createrandom">Random Arrays</a>

Nd4j provides a few methods to generate INDArrays, where the contents are pseudo-random numbers.

To generate uniform random numbers in the range 0 to 1, use `Nd4j.rand(int nRows, int nCols)` (for 2d arrays), or `Nd4j.rand(int[])` (for 3 or more dimensions).

Similarly, to generate Gaussian random numbers with mean zero and standard deviation 1, use `Nd4j.randn(int nRows, int nCols)` or `Nd4j.randn(int[])`.

For repeatability (i.e., to set Nd4j's random number generator seed) you can use `Nd4j.getRandom().setSeed(long)`

### <a name="createfromjava">Creating NDArrays from Java arrays</a>

Nd4j provides convenience methods for the creation of arrays from Java float and double arrays.

To create a 1d NDArray from a 1d Java array, use:

* Row vector: `Nd4j.create(float[])` or `Nd4j.create(double[])`
* Column vector: `Nd4j.create(float[],new int[]{length,1})` or `Nd4j.create(double[],new int[]{length,1})`

For 2d arrays, use `Nd4j.create(float[][])` or `Nd4j.create(double[][])`.

For creating NDArrays from Java primitive arrays with 3 or more dimensions (`double[][][]` etc), one approach is to use the following:

```java
double[] flat = ArrayUtil.flattenDoubleArray(myDoubleArray);
int[] shape = ...;	//Array shape here
INDArray myArr = Nd4j.create(flat,shape,'c');
```

### <a name="createfromndarray"> Creating NDArrays from Other NDArrays </a>

There are three primary ways of creating arrays from other arrays:

* Creating an exact copy of an existing NDArray using `INDArray.dup()`
* Create the array as a subset of an existing NDArray
* Combine a number of existing NDArrays to create a new NDArray

For the second case, you can use getRow(), get(), etc. See <a href="#getset">Getting and Setting Parts of NDArrays</a> for details on this.

Two methods for combining NDArrays are `Nd4j.hstack(INDArray...)` and `Nd4j.vstack(INDArray...)`.

`hstack` (horizontal stack) takes as argument a number of matrices that have the same number of rows, and stacks them horizontally to produce a new array. The input NDArrays can have a different number of columns, however.

Example:

```
int nRows = 2;
int nColumns = 2;
// Create INDArray of zeros
INDArray zeros = Nd4j.zeros(nRows, nColumns);
// Create one of all ones
INDArray ones = Nd4j.ones(nRows, nColumns);
//hstack
INDArray hstack = Nd4j.hstack(ones,zeros);
System.out.println("### HSTACK ####");
System.out.println(hstack);

```

Output:

```
### HSTACK ####
[[1.00, 1.00, 0.00, 0.00],
[1.00, 1.00, 0.00, 0.00]]
```

`vstack` (vertical stack) is the vertical equivalent of hstack. The input arrays must have the same number of columns.

Example:

```
int nRows = 2;
int nColumns = 2;
// Create INDArray of zeros
INDArray zeros = Nd4j.zeros(nRows, nColumns);
// Create one of all ones
INDArray ones = Nd4j.ones(nRows, nColumns);
//vstack
INDArray vstack = Nd4j.vstack(ones,zeros);
System.out.println("### VSTACK ####");
System.out.println(vstack);
```

Output:

```
### VSTACK ####
[[1.00, 1.00],
 [1.00, 1.00],
 [0.00, 0.00],
 [0.00, 0.00]]
```

`ND4J.concat` combines arrays along a dimension. 

Example:

```
int nRows = 2;
int nColumns = 2;
//INDArray of zeros
INDArray zeros = Nd4j.zeros(nRows, nColumns);
// Create one of all ones
INDArray ones = Nd4j.ones(nRows, nColumns);
// Concat on dimension 0
INDArray combined = Nd4j.concat(0,zeros,ones);
System.out.println("### COMBINED dimension 0####");
System.out.println(combined);
//Concat on dimension 1
INDArray combined2 = Nd4j.concat(1,zeros,ones);
System.out.println("### COMBINED dimension 1 ####");
System.out.println(combined2);
```

Output:
```
### COMBINED dimension 0####
[[0.00, 0.00],
 [0.00, 0.00],
 [1.00, 1.00],
 [1.00, 1.00]]
### COMBINED dimension 1 ####
[[0.00, 0.00, 1.00, 1.00],
 [0.00, 0.00, 1.00, 1.00]]
```

`ND4J.pad` is used to pad an array.

Example:
```
int nRows = 2;
int nColumns = 2;
// Create INDArray of all ones
INDArray ones = Nd4j.ones(nRows, nColumns);
// pad the INDArray
INDArray padded = Nd4j.pad(ones, new int[]{1,1}, Nd4j.PadMode.CONSTANT );
System.out.println("### Padded ####");
System.out.println(padded);
```

Output:


```
### Padded ####
[[0.00, 0.00, 0.00, 0.00],
 [0.00, 1.00, 1.00, 0.00],
 [0.00, 1.00, 1.00, 0.00],
 [0.00, 0.00, 0.00, 0.00]]
```


One other method that can occasionally be useful is `Nd4j.diag(INDArray in)`. This method has two uses, depending on the argument `in`:

* If `in` in a vector, diag outputs a NxN matrix with the diagonal equal to the array `in` (where N is the length of `in`)
* If `in` is a NxN matrix, diag outputs a vector taken from the diagonal of `in`



### <a name="createother"> Miscellaneous NDArray Creation Methods </a>

To create an [identity matrix](https://en.wikipedia.org/wiki/Identity_matrix) of size N, you can use `Nd4j.eye(N)`.

To create a row vector with elements `[a, a+1, a+2, ..., b]` you can use the linspace command:

`Nd4j.linspace(a, b, b-a+1)`

Linspace can be combined with a reshape operation to get other shapes. For example, if you want a 2d NDArray with 5 rows and 5 columns, with values 1 to 25 inclusive, you can use the following:

`Nd4j.linspace(1,25,25).reshape(5,5) `


## <a name="individual">Getting and Setting Individual Values</a>

For an INDArray, you can get or set values using the indexes of the element you want to get or set. For a rank N array (i.e., an array with N dimensions) you need N indices.

Note: getting or setting values individually (for example, one at a time in a for loop) is generally a bad idea in terms of performance. When possible, try to use other INDArray methods that operate on a large number of elements at a time.

To get values from a 2d array, you can use: `INDArray.getDouble(int row, int column)`

For arrays of any dimensionality, you can use `INDArray.getDouble(int...)`. For example, to get the value at index `i,j,k` use `INDArray.getDouble(i,j,k)`


To set values, use one of the putScalar methods:

* `INDArray.putScalar(int[],double)`
* `INDArray.putScalar(int[],float)`
* `INDArray.putScalar(int[],int)`

Here, the `int[]` is the index, and the `double/float/int` is the value to be placed at that index.


Some additional functionality that might be useful in certain circumstances is the `NDIndexIterator` class. The NDIndexIterator allows you to get the indexes in a defined order (specifially, the C-order traversal order: [0,0,0], [0,0,1], [0,0,2], ..., [0,1,0], ... etc for a rank 3 array).

To iterate over the values in a 2d array, you can use:

```java
NdIndexIterator iter = new NdIndexIterator(nRows, nCols);
while (iter.hasNext()) {
    int[] nextIndex = iter.next();
    double nextVal = myArray.getDouble(nextIndex);
    //do something with the value
}
```


## <a name="getset">Getting and Setting Parts of NDArrays</a>



### <a name="getsetrows">getRow() and putRow()</a>

In order to get a single row from an INDArray, you can use `INDArray.getRow(int)`. This will obviously return a row vector.
Of note here is that this row is a view: changes to the returned row will impact the original array. This can be quite useful at times (for example: `myArr.getRow(3).addi(1.0)` to add 1.0 to the third row of a larger array); if you want a copy of a row, use `getRow(int).dup()`.

Simiarly, to get multiple rows, use `INDArray.getRows(int...)`. This returns an array with the rows stacked; note however that this will be a copy (not a view) of the original rows, a view is not possible here due to the way NDArrays are stored in memory.

For setting a single row, you can use `myArray.putRow(int rowIdx,INDArray row)`. This will set the `rowIdx`th row of `myArray` to the values contained in the INDArray `row`.


### <a name="getsetsub">Sub-Arrays: get(), put() and NDArrayIndex</a>

**Get:**

A more powerful and general method is to use `INDArray.get(NDArrayIndex...)`. This functionality allows you to get an arbitrary sub-arrays based on certain indexes.
This is perhaps best explained by some examples:

To get a single row (and all columns), you can use:

`myArray.get(NDArrayIndex.point(rowIdx), NDArrayIndex.all()) `


To get a range of rows (row `a` (inclusive) to row `b` (exclusive)) and all columns, you can use:

`myArray.get(NDArrayIndex.interval(a,b), NDArrayIndex.all())`

To get all rows and every second column, you can use:

`myArray.get(NDArrayIndex.all(),NDArrayIndex.interval(0,2,nCols)) `

Though the above examples are for 2d arrays only, the NDArrayIndex approach extends to 3 or more dimensions. For 3 dimension, you would provide 3 INDArrayIndex objects instead of just two, as above.


Note that the `NDArrayIndex.interval(...)`, `.all()` and `.point(int)` methods always return views of the underlying arrays. Thus, changes to the arrays returned by `.get()` will be reflected in the original array.


**Put:**

The same NDArrayIndex approach is also used to put elements to another array: in this case you use the `INDArray.put(INDArrayIndex[], INDArray toPut)` method. Clearly, the size of the NDArray `toPut` must match the size implied by the provided indexes.


Also note that `myArray.put(NDArrayIndex[],INDArray other)` is functionally equivalent to doing `myArray.get(INDArrayIndex...).assign(INDArray other)`. Again, this is because `.get(INDArrayIndex...)` returns a view of the underlying array, not a copy.


### <a name="getsettensor">Tensor Along Dimension</a>

(Note: ND4J versions 0.4-rc3.8 and earlier returned slightly different results for tensor along dimension, as compared to current versions).

Tensor along dimension is a powerful technique, but can be a little hard to understand at first. The idea behind tensor along dimension (hereafter refered to as TAD) is to get a lower rank sub-array that is a <a href="#views">view</a> of the original array.

The tensor along dimension method takes two arguments:

- The *index* of the tensor to return (in the range of 0 to numTensors-1)
- The *dimensions* (1 or more values) along which to execute the TAD operation

The simplest case is a tensor along a single row or column of a 2d array. Consider the following diagram (where dimension 0 (rows) are indexed going down the page, and dimension 1 (columns) are indexed going across the page):

![Tensor Along Dimension](./images/guide/tad_2d.png)

Note that the output of the tensorAlongDimension call with one dimension is a row vector in all cases.

To understand why we get this output: consider the first case in the above diagram. There, we are taking the 0th (first) tensor *along* dimension 0 (dimension 0 being rows); the values (1,5,2) are in a line as we move along dimension 0, hence the output. Similarly, the `tensorAlongDimension(1,1)` is the second (*index=1*) tensor along dimension 1; values (5,3,5) are in a line as we move along dimension 1.


The TAD operation can also be executed along multiple dimensions. For example, by specifying two dimensions to execute the TAD operation along, we can use it to get a 2d sub-array from a 3d (or 4d, or 5d...) array. Similarly, by specifying 3 dimensions, we can use it to get a 3d from 4d or higher.

There are two things we need to know about the output, for the TAD operation to be useful.

First, we need to the number of tensors that we can get, for a given set of dimensions. To determine this, we can use the "number of tensors along dimensions" method, `INDArray.tensorssAlongDimension(int... dimensions)`. This method simply returns the number of tensors along the specified dimensions. In the examples above, we have:

* `myArray.tensorssAlongDimension(0) = 3`
* `myArray.tensorssAlongDimension(1) = 3`
* `myArray.tensorssAlongDimension(0,1) = 1`
* `myArray.tensorssAlongDimension(1,0) = 1`

(In the latter 2 cases, note that tensor along dimension would give us the same array out as the original array in - i.e., we get a 2d output from a 2d array).

More generally, the *number* of tensors is given by the product of the remaining dimensions, and the *shape* of the tensors is given by the size of the specified dimensions in the original shape.


Here's some examples:

- For input shape [a,b,c], tensorssAlongDimension(0) gives b*c tensors, and tensorAlongDimension(i,0) returns tensors of shape [1,a].
- For input shape [a,b,c], tensorssAlongDimension(1) gives a*c tensors, and tensorAlongDimension(i,1) returns tensors of shape [1,b].
- For input shape [a,b,c], tensorssAlongDimension(0,1) gives c tensors, and tensorAlongDimension(i,0,1) returns tensors of shape [a,b].
- For input shape [a,b,c], tensorssAlongDimension(1,2) gives a tensors, and tensorAlongDimension(i,1,2) returns tensors of shape [b,c].
- For input shape [a,b,c,d], tensorssAlongDimension(1,2) gives a*d tensors, and tensorAlongDimension(i,1,2) returns tensors of shape [b,c].
- For input shape [a,b,c,d], tensorssAlongDimension(0,2,3) gives b tensors, and tensorAlongDimension(i,0,2,3) returns tensors of shape [a,c,d].


### <a name="getsetslice">Slice</a>

[This section: Forthcoming.]

## <a name="ops">Performing Operations on NDArrays</a>

Nd4J has the concept of ops (operations) for many things you might want to do with (or to) an INDArray.
For example, ops are used to apply things like tanh operations, or add a scalar, or do element-wise operations.

ND4J defines five types of operations:

* Scalar
* Transform
* Accumulation
* Index Accumulation
* Broadcast

And two methods of executing each:

* Directly on the entire INDArray, or
* Along a dimension

Before getting into the specifics of these operations, let's take a moment to consider the difference between *in-place* and *copy* operations.

Many ops have both in-place and copy operations. Suppose we want to add two arrays. Nd4j defines two methods for this: `INDArray.add(INDArray)` and `INDArray.addi(INDArray)`. The former (add) is a copy operation; the latter is an in-place operation - the *i* in *addi* means in-place. This convention (*...i* means in-place, no *i* means copy) holds for other ops that are accessible via the INDArray interface.

Suppose we have two INDArrays `x` and `y` and we do `INDArray z = x.add(y)` or `INDArray z = x.addi(y)`. The results of these operations are shown below.

![Add](./images/guide/add_v_addi_1.png)

![Addi](./images/guide/add_v_addi_2.png)


Note that with the `x.add(y)` operation, the original array `x` is not modified. Comparatively, with the in-place version `x.addi(y)`, the array `x` is modified. In both versions of the add operation, an INDArray is returned that contains the result. Note however that in the case of the `addi` operation, the result array us actually just the original array `x`.




### <a name="opsscalar">Scalar Ops</a>

[Scalar ops](https://github.com/deeplearning4j/nd4j/tree/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl/scalar) are element-wise operations that also take a scalar (i.e., a number). Examples of scalar ops are  add, max, multiply, set and divide operations (see the previous link for a full list).

A number of the methods such as `INDArray.addi(Number)` and `INDArray.divi(Number)` actually execute scalar ops behind the scenes, so when available, it is more convenient to use these methods.

To execute a scalar op more directly, you can use for example:

`Nd4j.getExecutioner().execAndReturn(new ScalarAdd(myArray,1.0))`

Note that `myArray` is modified by this operation. If this is not what you want, use `myArray.dup()`.

Unlike the remaining ops, scalar ops don't have a sensible interpretation of executing them along a dimension.

### <a name="opstransform">Transform Ops</a>

Transform ops are operations such as element-wise logarithm, cosine, tanh, rectified linear, etc. Other examples include add, subtract and copy operations. Transform ops are commonly used in an element-wise manner (such as tanh on each element), but this is not always the case - for example, softmax is typically executed along a dimension.

To execute an element-wise tanh operation directly (on the full NDArray) you can use:

`INDArray tanh = Nd4j.getExecutioner().execAndReturn(new Tanh(myArr))`
As with scalar ops mentioned above, transform operations using the above method are *in-place* operations: that is, the NDArray myArr is modified, and the returned array `tanh` is actually the same object as the input `myArr`. Again, you can use `myArr.dup()` if you want a copy.

The [Transforms class](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/ops/transforms/Transforms.java) also defines some convenience methods, such as: `INDArray tanh = Transforms.tanh(INDArray in,boolean copy);` This is equivalent to the method using `Nd4j.getExecutioner()` above.

### <a name="opsaccum">Accumulation (Reduction) Ops</a>

When it comes to executing accumulations, there is a key difference between executing the accumulation on the entire NDArray, versus executing along a particular dimension (or dimensions). In the first case (executing on the entire array), only a single value is returned. In the second case (accumulating along a dimension) a new INDArray is returned.

To get the sum of all values in the array:

`double sum = Nd4j.getExecutioner().execAndReturn(new Sum(myArray)).getFinalResult().doubleValue();`

or equivalently (and more conveniently)

`double sum = myArray.sumNumber().doubleValue();`


Accumulation ops can also be executed along a dimension. For example, to get the sum of all values in each column (in each column = along dimension 0, or "for values in each row"), you can use:

`INDArray sumOfColumns = Nd4j.getExecutioner().exec(new Sum(myArray),0);`

or equivalently,

`INDArray sumOfColumns = myArray.sum(0)`

Suppose this was executed on a 3x3 input array. Visually, this sum operation along dimension 0 operation looks like:

![Sum along dimension 0](./images/guide/sum_dim0.png)

Note that here, the input has shape `[3,3]` (3 rows, 3 columns) and the output has shape `[1,3]` (i.e., our output is a row vector). Had we instead done the operation along dimension 1, we would get a column vector with shape `[3,1]`, with values `(12,13,11)`.

Accumulations along dimensions also generalize to NDArrays with 3 or more dimensions.

### <a name="opsindexaccum">Index Accumulation Ops</a>

[Index accumulation ops](https://github.com/deeplearning4j/nd4j/tree/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl/indexaccum) are very similar to accumulation ops. The difference is that they return an integer index, instead of a double values.

Examples of index accumulation ops are IMax (argmax), IMin (argmin) and IAMax (argmax of absolute values).

To get the index of the maximum value in the array:

`int idx = Nd4j.getExecutioner().execAndReturn(new IAMax(myArray)).getFinalResult();`

Index accumulation ops are often most useful when executed along a dimension. For example, to get the index of the maximum value in each column (in each column = along dimension 0), you can use:

`INDArray idxOfMaxInEachColumn = Nd4j.getExecutioner().exec(new IAMax(myArray),0);`

Suppose this was executed on a 3x3 input array. Visually, this argmax/IAMax operation along dimension 0 operation looks like:

![Argmax / IAMax](./images/guide/argmax_dim0.png)

As with the accumulation op described above, the output has shape `[1,3]`. Again, had we instead done the operation along dimension 1, we would get a column vector with shape `[3,1]`, with values `(1,0,2)`.


### <a name="opsbroadcast">Broadcast and Vector Ops</a>

ND4J also defines broadcast and vector operations.

Some of the more useful operations are vector operations, such as addRowVector and muliColumnVector.

Consider for example the operation `x.addRowVector(y)` where `x` is a matrix and `y` is a row vector. In this case, the `addRowVector` operation adds the row vector `y` to each row of the matrix `x`, as shown below.

![addRowVector](./images/guide/addrowvector.png)

As with other ops, there are inplace and copy versions. There are also column column versions of these operations, such as `addColumnVector`, which adds a column vector to each column of the original INDArray.



## <a name="boolean"> Boolean Indexing: Selectively Apply Operations Based on a Condition</a>

[This section: Forthcoming.]

[Link: Boolean Indexing Unit Tests](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-tests/src/test/java/org/nd4j/linalg/indexing/BooleanIndexingTest.java)


## <a name="workspaces">Workspaces</a>

Workspaces are a feature of ND4J used to improve performance, by means of more efficient memory allocation and management. Specifically, workspaces are designed for cyclical workloads - such as training neural networks - as they allow for off-heap memory reuse (instead of continually allocating and deallocating memory on each iteration of the loop). The net effect is improved performance and reduced memory use.

For more details on workspaces, see the following links:

* <a href="https://deeplearning4j.org/workspaces">Deeplearning4j Guide to Workspaces</a>
* <a href="https://github.com/deeplearning4j/dl4j-examples/blob/master/nd4j-examples/src/main/java/org/nd4j/examples/Nd4jEx15_Workspaces.java">Workspaces Examples</a>

### <a name="workspaces-panic">Workspaces: Scope Panic</a>

Sometimes with workspaces, you may encounter an exception such as:
```
org.nd4j.linalg.exception.ND4JIllegalStateException: Op [set] Y argument uses leaked workspace pointer from workspace [LOOP_EXTERNAL]
For more details, see the ND4J User Guide: nd4j.org/userguide#workspaces-panic
```
or
```
org.nd4j.linalg.exception.ND4JIllegalStateException: Op [set] Y argument uses outdated workspace pointer from workspace [LOOP_EXTERNAL]
For more details, see the ND4J User Guide: nd4j.org/userguide#workspaces-panic
```


**Understanding Scope Panic Exceptions**

In short: these exceptions mean that an INDArray that has been allocated in a workspace is being used incorrectly (for example, a bug or incorrect implementation of some method). This can occur for two reasons:

1. The INDArray has 'leaked out' of the workspace in which is was defined
2. The INDArray is used within the correct workspace, but from a previous iteration

In both cases, the underlying off-heap memory that the INDArray points to has been invalidated, and can no longer be used.

An example sequence of events leading to a workspace leak:
1. Workspace W is opened
2. INDArray X is allocated in workspace W
3. Workspace W is closed, and hence the memory for X is no longer valid.
4. INDArray X is used in some operation, resulting in an exception

An example sequence of events, leading to an outdated workspace pointer:
1. Workspace W is opened (iteration 1)
2. INDArray X is allocated in workspace W (iteration 1)
3. Workspace W is closed (iteration 1)
4. Workspace W is opened (iteration 2)
5. INDArray X (from iteration 1) is used in some operation, resulting in an exception

**Workarounds and Fixes for Scope Panic Exceptions**

There are two basic solutions, depending on the cause.

First. if you have implemented some custom code (or are using workspaces manually), this usually indicates a bug in your code.
Generally, you have two options:
1. Detach the INDArray from all workspace, using the ```INDArray.detach()``` method. The consequence is that the returned array is no longer associated with a workspace, and can be used freely within or outside of any workspace.
2. Don't allocate the array in the workspace in the first place. You can temporarily 'turn off' a workspace using: ```try(MemoryWorkspace scopedOut = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()){ <your code here> }```. The consequence is that any new arrays (created via Nd4j.create, for example) within the try block will not be associated with a workspace, and can be used outside of a workspace 
3. Move/copy the array to a parent workspace, using one of the ```INDArray.leverage()``` or ```leverageTo(String)``` or ```migrate()``` methods. See the Javadoc of these methods for more details.


Second, if you are using workspaces as part of Deeplearning4j and have not implemented any custom functionality (i.e., you have not written your own layer, data pipeline, etc), then (on the off-chance you run into this), this most likely indicates a bug in the underlying library, which usually should be reported via a Github issue. One possible workaround in the mean time is to disable workspaces using the following code:
```
.trainingWorkspaceMode(WorkspaceMode.NONE)
.inferenceWorkspaceMode(WorkspaceMode.NONE)
```

If the exception is due to an issue in the data pipeline, you can try wrapping your ```DataSetIterator``` or ```MultiDataSetIterator``` in an ```AsyncShieldDataSetIterator``` or ```AsyncShieldMultiDataSetIterator```.


For either cause, a final solution - if you are sure your code is correct - is to try disabling scope panic. *Note that this is NOT recommended and can crash the JVM if a legitimate issue is present*. To do this, use ```Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.DISABLED);``` before executing your code.


## <a name="misc">Advanced and Miscellaneous Topics</a>


### <a name="miscdatatype">Setting the data type</a>

ND4J currently allows INDArrays to be backed by either float or double-precision values. The default is single-precision (float). To set the order that ND4J uses for arrays globally to double precision, you can use:

```java
Nd4j.setDataType(DataBuffer.Type.DOUBLE);
```

Note that this should be done before using ND4J operations or creating arrays.

Alternatively, you can set the property when launching the JVM:
```
-Ddtype=double
```


### Reshaping

[This section: Forthcoming.]


### <a name="miscflattening">Flattening</a>

Flattening is the process of taking a or more INDArrays and converting them into a single flat array (a row vector), given some traversal order of the arrays.

Nd4j provides the following methods for this:

```java
Nd4j.toFlattened(char order, INDArray... arrays)
Nd4j.toFlattened(char order, Collection<INDArray>)
```
Nd4j also provides overloaded toFlattened methods with the default ordering. The order argument must be 'c' or 'f', and defines the order in which values are taken from the arrays: c order results in the arrays being flattened using array indexes in an order like [0,0,0], [0,0,1], etc (for 3d arrays) whereas f order results in values being taken in order [0,0,0], [1,0,0], etc.



### Permute

[This section: Forthcoming.]


### sortRows/sortColumns

[This section: Forthcoming.]


### Directly accessing BLAS operations

[This section: Forthcoming.]


### Serialization

Nd4j provides serialization of INDArrays many formats. Here are some examples for binary and text serialization:
```java
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.serde.binary.BinarySerde;

import java.io.*;
import java.nio.ByteBuffer;

INDArray arrWrite = Nd4j.linspace(1,10,10);
INDArray arrRead;

//1. Binary format
//   Close the streams manually or use try with resources.
try (DataOutputStream sWrite = new DataOutputStream(new FileOutputStream(new File("tmp.bin")))) {
	Nd4j.write(arrWrite, sWrite);
    }

try (DataInputStream sRead = new DataInputStream(new FileInputStream(new File("tmp.bin")))) {
	arrRead = Nd4j.read(sRead);
    }

//2. Binary format using java.nio.ByteBuffer;
ByteBuffer buffer = BinarySerde.toByteBuffer(arrWrite);
arrRead = BinarySerde.toArray(buffer);

//3. Text format
Nd4j.writeTxt(arrWrite, "tmp.txt");
arrRead = Nd4j.readTxt("tmp.txt");

// To read csv format:
// The writeNumpy method has been deprecated.
arrRead =Nd4j.readNumpy("tmp.csv", ", ");
```

The [nd4j-serde](https://github.com/deeplearning4j/nd4j/tree/master/nd4j-serde) directory provides packages for Aeron, base64, camel-routes, gsom, jackson and kryo.  


## <a name="quickref">Quick Reference: A Summary Overview of ND4J Methods</a>

This section lists the most commonly used operations in ND4J, in a summary form. More details on most of these can be found later in this page.

In this section, assume that `arr`, `arr1` etc are INDArrays.

**Creating NDArrays**:

* Create a zero-initialized array: `Nd4j.zeros(nRows, nCols)` or `Nd4j.zeros(int...)`
* Create a one-initialized array: `Nd4j.ones(nRows, nCols)`
* Create a copy (duplicate) of an NDArray: `arr.dup()`
* Create a row/column vector from a `double[]`: `myRow = Nd4j.create(myDoubleArr)`, `myCol = Nd4j.create(myDoubleArr,new int[]{10,1})`
* Create a 2d NDArray from a `double[][]`: `Nd4j.create(double[][])`
* Stacking a set of arrays to make a larger array: `Nd4j.hstack(INDArray...)`, `Nd4j.vstack(INDArray...)` for horizontal and vertical respectively
* Uniform random NDArrays: `Nd4j.rand(int,int)`, `Nd4j.rand(int[])` etc
* Normal(0,1) random NDArrays: `Nd4j.randn(int,int)`, `Nd4j.randn(int[])`

**Determining the Size/Dimensions of an INDArray**:

The following methods are defined by the INDArray interface:

* Get the number of dimensions: `rank()`
* For 2d NDArrays only: `rows()`, `columns()`
* Size of the ith dimension: `size(i)`
* Get the size of all dimensions, as an int[]: `shape()`
* Determine the total number of elements in array: `arr.length()`
* See also: `isMatrix()`, `isVector()`, `isRowVector()`, `isColumnVector()`

**Getting and Setting Single Values**:

* Get the value at row i, column j: `arr.getDouble(i,j)`
* Getting a values from a 3+ dimenional array: `arr.getDouble(int[])`
* Set a single value in an array: `arr.putScalar(int[],double)`

**Scalar operations**:
Scalar operations take a double/float/int value and do an operation for each As with element-wise operations, there are in-place and copy operations.

* Add a scalar: arr1.add(myDouble)
* Substract a scalar: arr1.sub(myDouble)
* Multiply by a scalar: arr.mul(myDouble)
* Divide by a scalar: arr.div(myDouble)
* Reverse subtract (scalar - arr1): arr1.rsub(myDouble)
* Reverse divide (scalar / arr1): arr1.rdiv(myDouble)


**Element-Wise Operations**:
Note: there are copy (add, mul, etc) and in-place (addi, muli) operations. The former: arr1 is not modified. In the latter: arr1 is modified

* Adding: `arr1.add(arr2)`
* Subtract: `arr.sub(arr2)`
* Multiply: `add1.mul(arr2)`
* Divide: `arr1.div(arr2)`
* Assignment (set each value in arr1 to those in arr2): `arr1.assign(arr2)`

**Reduction Operations (sum, etc)**;
Note that these operations operate on the entire array. Call `.doubleValue()` to get a double out of the returned Number.

* Sum of all elements: `arr.sumNumber()`
* Product of all elements: `arr.prod()`
* L1 and L2 norms: `arr.norm1()` and `arr.norm2()`
* Standard deviation of all elements: `arr.stdNumber()`

**Linear Algebra Operations**:

* Matrix multiplication: `arr1.mmul(arr2)`
* Transpose a matrix: `transpose()`
* Get the diagonal of a matrix: `Nd4j.diag(INDArray)`
* Matrix inverse: `InvertMatrix.invert(INDArray,boolean)`

**Getting Parts of a Larger NDArray**:
Note: all of these methods return

* Getting a row (2d NDArrays only): `getRow(int)`
* Getting multiple rows as a matrix (2d only): `getRows(int...)`
* Setting a row (2d NDArrays only): `putRow(int,INDArray)`
* Getting the first 3 rows, all columns: `Nd4j.create(0).get(NDArrayIndex.interval(0,3),NDArrayIndex.all());`

**Element-Wise Transforms (Tanh, Sigmoid, Sin, Log etc)**:

* Using [Transforms](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/ops/transforms/Transforms.java): `Transforms.sin(INDArray)`, `Transforms.log(INDArray)`, `Transforms.sigmoid(INDArray)` etc
* Directly (method 1): `Nd4j.getExecutioner().execAndReturn(new Tanh(INDArray))`
* Directly (method 2) `Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("tanh",INDArray))`



## <a name="faq">FAQ: Frequently Asked Questions</a>

**Q: Does ND4J support sparse arrays?**

At present: no. Support for spase arrays is planned for the future.

**Q: Is it possible to dynamically grow or shrink the size on an INDArray?**
In the current version of ND4J, this is not possible. We may add this functionality in the future, however.

There are two possible work-arounds:

1. Allocate a new array and do a copy (for example, a .put() operation)
2. Initially, pre-allocate a larger than required NDArray, and then operate on a view of that array. Then, as you need a larger array, get a larger view on the original pre-allocated array.