package org.deeplearning4j.linalg.api.ndarray;

import org.deeplearning4j.linalg.indexing.NDArrayIndex;
import org.deeplearning4j.linalg.ops.reduceops.Ops;

/**
 * Interface for an ndarray
 *
 * @author Adam Gibson
 */
public interface INDArray {

    /**
     * Returns the number of possible vectors for a given dimension
     * @param dimension the dimension to calculate the number of vectors for
     * @return the number of possible vectors along a dimension
     */
    public int vectorsAlongDimension(int dimension);

    /**
     * Get the vector along a particular dimension
     * @param index the index of the vector to get
     * @param dimension the dimension to get the vector from
     * @return the vector along a particular dimension
     */
    public INDArray vectorAlongDimension(int index,int dimension);

    /**
     * Cumulative sum along a dimension
     * @param dimension the dimension to perform cumulative sum along
     * @return the cumulative sum along the specified dimension
     */
    public INDArray cumsumi(int dimension);
    /**
     * Cumulative sum along a dimension (in place)
     * @param dimension the dimension to perform cumulative sum along
     * @return the cumulative sum along the specified dimension
     */
    public INDArray cumsum(int dimension);

    /**
     * Assign all of the elements in the given
     * ndarray to this ndarray
     * @param arr the elements to assign
     * @return this
     */
    public INDArray assign(INDArray arr);

    public INDArray putScalar(int i,Number value);

    public INDArray putScalar(int[] i,Number value);

    public INDArray lt(Number other);

    public INDArray lti(Number other);

    public INDArray eq(Number other);

    public INDArray eqi(Number other);

    public INDArray gt(Number other);

    public INDArray gti(Number other);



    public INDArray lt(INDArray other);

    public INDArray lti(INDArray other);

    public INDArray eq(INDArray other);

    public INDArray eqi(INDArray other);

    public INDArray gt(INDArray other);

    public INDArray gti(INDArray other);

    public INDArray neg();

    public INDArray negi();

    public INDArray rdiv(Number n);

    public INDArray rdivi(Number n);

    public INDArray rsub(Number n);

    public INDArray rsubi(Number n);


    public INDArray div(Number n);

    public INDArray divi(Number n);


    public INDArray mul(Number n);

    public INDArray muli(Number n);


    public INDArray sub(Number n);

    public INDArray subi(Number n);

    public INDArray add(Number n);

    public INDArray addi(Number n);

    /**
     * Returns a subset of this array based on the specified
     * indexes
     * @param indexes the indexes in to the array
     * @return a view of the array with the specified indices
     */
    public INDArray get(NDArrayIndex...indexes);


    /**
     * Get a list of specified columns
     * @param columns
     * @return
     */
    INDArray getColumns(int[] columns);

    /**
     * Get a list of rows
     * @param rows
     * @return
     */
    INDArray getRows(int[] rows);

    /**
     * Reverse division
     * @param other the matrix to divide from
     * @return
     */
    INDArray rdiv(INDArray other);

    /**
     * Reverse divsion (in place)
     * @param other
     * @return
     */
    INDArray rdivi(INDArray other);


    /**
     * Reverse division
     * @param other the matrix to subtract from
     * @param result the result ndarray
     * @return
     */
    INDArray rdiv(INDArray other,INDArray result);

    /**
     * Reverse division (in-place)
     * @param other the other ndarray to subtract
     * @param result the result ndarray
     * @return the ndarray with the operation applied
     */
    INDArray rdivi(INDArray other,INDArray result);

    /**
     * Reverse subtraction
     * @param other the matrix to subtract from
     * @param result the result ndarray
     * @return
     */
    INDArray rsub(INDArray other,INDArray result);


    /**
     *
     * @param other
     * @return
     */
    INDArray rsub(INDArray other);

    /**
     *
     * @param other
     * @return
     */
    INDArray rsubi(INDArray other);

    /**
     * Reverse subtraction (in-place)
     * @param other the other ndarray to subtract
     * @param result the result ndarray
     * @return the ndarray with the operation applied
     */
    INDArray rsubi(INDArray other,INDArray result);

    /**
     * Set the value of the ndarray to the specified value
     * @param value the value to assign
     * @return the ndarray with the values
     */
    INDArray assign(Number value);


    /**
     * Get the linear index of the data in to
     * the array
     * @param i the index to get
     * @return the linear index in to the data
     */
    public int linearIndex(int i);

    /**
     * Iterate over every row of every slice
     * @param op the operation to apply
     */
    public void iterateOverAllRows(SliceOp op);


    /**
     * Fetch a particular number on a multi dimensional scale.
     * @param indexes the indexes to getFromOrigin a number from
     * @return the number at the specified indices
     */
    public INDArray getScalar(int... indexes);

    /**
     * Validate dimensions are equal
     * @param other the other ndarray to compare
     *
     */

    public void checkDimensions(INDArray other);
    /**
     * Gives the indices for the ending of each slice
     * @return the off sets for the beginning of each slice
     */
    public int[] endsForSlices();


    /**
     *
     * http://docs.scipy.org/doc/numpy/reference/generated/numpy.ufunc.reduce.html
     * @param op the operation to do
     * @param dimension the dimension to return from
     * @return the results of the reduce (applying the operation along the specified
     * dimension)t
     */
    public INDArray reduce(Ops.DimensionOp op,int dimension);

    /**
     * Assigns the given matrix (put) to the specified slice
     * @param slice the slice to assign
     * @param put the slice to applyTransformToDestination
     * @return this for chainability
     */
    public INDArray putSlice(int slice,INDArray put);


    /**
     * Iterate along a dimension.
     * This encapsulates the process of sum, mean, and other processes
     * take when iterating over a dimension.
     * @param dimension the dimension to iterate over
     * @param op the operation to apply
     * @param modify whether to modify this array while iterating
     */
    public void iterateOverDimension(int dimension,SliceOp op,boolean modify);


    /**
     * Replicate and tile array to fill out to the given shape
     * @param shape the new shape of this ndarray
     * @return the shape to fill out to
     */
    public INDArray repmat(int[] shape);

    /**
     * Insert a row in to this array
     * Will throw an exception if this
     * ndarray is not a matrix
     * @param row the row insert into
     * @param toPut the row to insert
     * @return this
     */
    public INDArray putRow(int row,INDArray toPut);

    /**
     * Insert a column in to this array
     * Will throw an exception if this
     * ndarray is not a matrix
     * @param column the column to insert
     * @param toPut the array to put
     * @return this
     */
    public INDArray putColumn(int column,INDArray toPut);

    /**
     * Returns the element at the specified row/column
     * This will throw an exception if the
     * @param row the row of the element to return
     * @param column the row of the element to return

     * @return a scalar indarray of the element at this index
     */
    public INDArray getScalar(int row,int column);

    /**
     * Returns the element at the specified index
     * @param i the index of the element to return
     * @return a scalar ndarray of the element at this index
     */
    public INDArray getScalar(int i);



    /**
     * Returns the squared (Euclidean) distance.
     */
    public double squaredDistance(INDArray other);

    /**
     * Returns the (euclidean) distance.
     */
    public double distance2(INDArray other);

    /**
     * Returns the (1-norm) distance.
     */
    public double distance1(INDArray other);



    public INDArray put(NDArrayIndex[] indices,INDArray element);


    public INDArray put(NDArrayIndex[] indices,Number element);

    /**
     * Inserts the element at the specified index
     * @param indices the indices to insert into
     * @param element a scalar ndarray
     * @return a scalar ndarray of the element at this index
     */
    public INDArray put(int [] indices,INDArray element);



    /**
     * Inserts the element at the specified index
     * @param i the row insert into
     * @param j the column to insert into
     * @param element a scalar ndarray
     * @return a scalar ndarray of the element at this index
     */
    public INDArray put(int i,int j,INDArray element);



    /**
     * Inserts the element at the specified index
     * @param i the row insert into
     * @param j the column to insert into
     * @param element a scalar ndarray
     * @return a scalar ndarray of the element at this index
     */
    public INDArray put(int i,int j,Number element);


    /**
     * Inserts the element at the specified index
     * @param i the index insert into
     * @param element a scalar ndarray
     * @return a scalar ndarray of the element at this index
     */
    public INDArray put(int i,INDArray element);


    /**
     * In place addition of a column vector
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    public INDArray diviColumnVector(INDArray columnVector);
    /**
     * In place addition of a column vector
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    public INDArray divColumnVector(INDArray columnVector);

    /**
     * In place addition of a column vector
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    public INDArray diviRowVector(INDArray rowVector);
    /**
     * In place addition of a column vector
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    public INDArray divRowVector(INDArray rowVector);


    /**
     * In place addition of a column vector
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    public INDArray muliColumnVector(INDArray columnVector);
    /**
     * In place addition of a column vector
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    public INDArray mulColumnVector(INDArray columnVector);

    /**
     * In place addition of a column vector
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    public INDArray muliRowVector(INDArray rowVector);
    /**
     * In place addition of a column vector
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    public INDArray mulRowVector(INDArray rowVector);




    /**
     * In place addition of a column vector
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    public INDArray subiColumnVector(INDArray columnVector);
    /**
     * In place addition of a column vector
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    public INDArray subColumnVector(INDArray columnVector);

    /**
     * In place addition of a column vector
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    public INDArray subiRowVector(INDArray rowVector);
    /**
     * In place addition of a column vector
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    public INDArray subRowVector(INDArray rowVector);

    /**
     * In place addition of a column vector
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    public INDArray addiColumnVector(INDArray columnVector);
    /**
     * In place addition of a column vector
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    public INDArray addColumnVector(INDArray columnVector);

    /**
     * In place addition of a column vector
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    public INDArray addiRowVector(INDArray rowVector);
    /**
     * In place addition of a column vector
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    public INDArray addRowVector(INDArray rowVector);

    /**
     * Perform a copy matrix multiplication
     * @param other the other matrix to perform matrix multiply with
     * @return the result of the matrix multiplication
     */
    public INDArray mmul(INDArray other);


    /**
     * Perform an copy matrix multiplication
     * @param other the other matrix to perform matrix multiply with
     * @param result the result ndarray
     * @return the result of the matrix multiplication
     */
    public INDArray mmul(INDArray other,INDArray result);


    /**
     * in place (element wise) division of two matrices
     * @param other the second ndarray to divide
     * @return the result of the divide
     */
    public INDArray div(INDArray other);

    /**
     * copy (element wise) division of two matrices
     * @param other the second ndarray to divide
     * @param result the result ndarray
     * @return the result of the divide
     */
    public INDArray div(INDArray other,INDArray result);


    /**
     * copy (element wise) multiplication of two matrices
     * @param other the second ndarray to multiply
     * @return the result of the addition
     */
    public INDArray mul(INDArray other);

    /**
     * copy (element wise) multiplication of two matrices
     * @param other the second ndarray to multiply
     * @param result the result ndarray
     * @return the result of the multiplication
     */
    public INDArray mul(INDArray other,INDArray result);

    /**
     * copy subtraction of two matrices
     * @param other the second ndarray to subtract
     * @return the result of the addition
     */
    public INDArray sub(INDArray other);

    /**
     * copy subtraction of two matrices
     * @param other the second ndarray to subtract
     * @param result the result ndarray
     * @return the result of the subtraction
     */
    public INDArray sub(INDArray other,INDArray result);

    /**
     * copy addition of two matrices
     * @param other the second ndarray to add
     * @return the result of the addition
     */
    public INDArray add(INDArray other);

    /**
     * copy addition of two matrices
     * @param other the second ndarray to add
     * @param result the result ndarray
     * @return the result of the addition
     */
    public INDArray add(INDArray other,INDArray result);










    /**
     * Perform an copy matrix multiplication
     * @param other the other matrix to perform matrix multiply with
     * @return the result of the matrix multiplication
     */
    public INDArray mmuli(INDArray other);


    /**
     * Perform an copy matrix multiplication
     * @param other the other matrix to perform matrix multiply with
     * @param result the result ndarray
     * @return the result of the matrix multiplication
     */
    public INDArray mmuli(INDArray other,INDArray result);


    /**
     * in place (element wise) division of two matrices
     * @param other the second ndarray to divide
     * @return the result of the divide
     */
    public INDArray divi(INDArray other);

    /**
     * in place (element wise) division of two matrices
     * @param other the second ndarray to divide
     * @param result the result ndarray
     * @return the result of the divide
     */
    public INDArray divi(INDArray other,INDArray result);


    /**
     * in place (element wise) multiplication of two matrices
     * @param other the second ndarray to multiply
     * @return the result of the addition
     */
    public INDArray muli(INDArray other);

    /**
     * in place (element wise) multiplication of two matrices
     * @param other the second ndarray to multiply
     * @param result the result ndarray
     * @return the result of the multiplication
     */
    public INDArray muli(INDArray other,INDArray result);

    /**
     * in place subtraction of two matrices
     * @param other the second ndarray to subtract
     * @return the result of the addition
     */
    public INDArray subi(INDArray other);

    /**
     * in place subtraction of two matrices
     * @param other the second ndarray to subtract
     * @param result the result ndarray
     * @return the result of the subtraction
     */
    public INDArray subi(INDArray other,INDArray result);

    /**
     * in place addition of two matrices
     * @param other the second ndarray to add
     * @return the result of the addition
     */
    public INDArray addi(INDArray other);

    /**
     * in place addition of two matrices
     * @param other the second ndarray to add
     * @param result the result ndarray
     * @return the result of the addition
     */
    public INDArray addi(INDArray other,INDArray result);


    /**
     * Returns the normmax along the specified dimension
     * @param dimension  the dimension to getScalar the norm1 along
     * @return the norm1 along the specified dimension
     */
    public INDArray normmax(int dimension);




    /**
     * Returns the norm2 along the specified dimension
     * @param dimension  the dimension to getScalar the norm2 along
     * @return the norm2 along the specified dimension
     */
    public INDArray norm2(int dimension);


    /**
     * Returns the norm1 along the specified dimension
     * @param dimension  the dimension to getScalar the norm1 along
     * @return the norm1 along the specified dimension
     */
    public INDArray norm1(int dimension);


    /**
     * Standard deviation of an ndarray along a dimension
     * @param dimension the dimension to get the std along
     * @return the standard deviation along a particular dimension
     */
    public INDArray std(int dimension);

    /**
     * Returns the product along a given dimension
     * @param dimension the dimension to getScalar the product along
     * @return the product along the specified dimension
     */
    public INDArray prod(int dimension);


    /**
     * Returns the overall mean of this ndarray
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    public INDArray mean(int dimension);


    /**
     * Returns the overall variance of this ndarray
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    public INDArray var(int dimension);


    /**
     * Returns the overall max of this ndarray
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    public INDArray max(int dimension);

    /**
     * Returns the overall min of this ndarray
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    public INDArray min(int dimension);

    /**
     * Returns the sum along the last dimension of this ndarray
     * @param dimension  the dimension to getScalar the sum along
     * @return the sum along the specified dimension of this ndarray
     */
    public INDArray sum(int dimension);





    /**
     * Returns the elements at the the specified indices
     * @param indices the indices to getScalar
     * @return the array with the specified elements
     */
    public INDArray get(int[] indices);


    /**
     * Return a copy of this ndarray
     * @return a copy of this ndarray
     */
    public INDArray dup();


    /**
     * Returns a flattened version (row vector) of this ndarray
     * @return a flattened version (row vector) of this ndarray
     */
    public INDArray ravel();


    /**
     * Returns the number of slices in this ndarray
     * @return the number of slices in this ndarray
     */
    public int slices();


    /**
     * Returns the specified slice of this ndarray
     * @param i the index of the slice to return
     * @param dimension the dimension to return the slice for
     * @return the specified slice of this ndarray
     */
    public INDArray slice(int i,int dimension);


    /**
     * Returns the specified slice of this ndarray
     * @param i the index of the slice to return
     * @return the specified slice of this ndarray
     */
    public INDArray slice(int i);


    /**
     * Returns the start of where the ndarray is
     * for the underlying data
     * @return the starting offset
     */
    public int offset();

    /**
     * Reshapes the ndarray (can't change the length of the ndarray)
     * @param newShape the new shape of the ndarray
     * @return the reshaped ndarray
     */
    public INDArray reshape(int[] newShape);


    /**
     * Reshapes the ndarray (can't change the length of the ndarray)
     * @param rows the rows of the matrix
     * @param columns the columns of the matrix
     * @return the reshaped ndarray
     */
    public INDArray reshape(int rows,int columns);

    /**
     * Flip the rows and columns of a matrix
     * @return the flipped rows and columns of a matrix
     */
    public INDArray transpose();

    /**
     * Mainly here for people coming from numpy.
     * This is equivalent to a call to permute
     * @param dimension the dimension to swap
     * @param with the one to swap it with
     * @return the swapped axes view
     */
    public INDArray swapAxes(int dimension,int with);

    /**
     * See: http://www.mathworks.com/help/matlab/ref/permute.html
     * @param rearrange the dimensions to swap to
     * @return the newly permuted array
     */
    public INDArray permute(int[] rearrange);


    /**
     * Returns the specified column.
     * Throws an exception if its not a matrix
     * @param i the column to getScalar
     * @return the specified column
     */
    INDArray getColumn(int i);

    /**
     * Returns the specified row.
     * Throws an exception if its not a matrix
     * @param i the row to getScalar
     * @return the specified row
     */
    INDArray getRow(int i);

    /**
     * Returns the number of columns in this matrix (throws exception if not 2d)
     * @return the number of columns in this matrix
     */
    int columns();

    /**
     * Returns the number of rows in this matrix (throws exception if not 2d)
     * @return the number of rows in this matrix
     */
    int rows();

    /**
     * Returns true if the number of columns is 1
     * @return true if the number of columns is 1
     */
    boolean isColumnVector();
    /**
     * Returns true if the number of rows is 1
     * @return true if the number of rows is 1
     */
    boolean isRowVector();

    /**
     * Returns true if this ndarray is a vector
     * @return whether this ndarray is a vector
     */
    boolean isVector();

    /**
     * Returns true if this ndarray is a matrix
     * @return whether this ndarray is a matrix
     */
    boolean isMatrix();

    /**
     * Returns true if this ndarray is a scalar
     * @return whether this ndarray is a scalar
     */
    boolean isScalar();


    /**
     * Returns the shape of this ndarray
     * @return the shape of this ndarray
     */
    int[] shape();


    /**
     * Returns the stride of this ndarray
     * @return the stride of this ndarray
     */
    int[] stride();

    /**
     * Returns the size along a specified dimension
     * @param dimension the dimension to return the size for
     * @return the size of the array along the specified dimension
     */
    int size(int dimension);

    /**
     * Returns the total number of elements in the ndarray
     * @return the number of elements in the ndarray
     */
    int length();



    /**
     * Broadcasts this ndarray to be the specified shape
     * @param shape the new shape of this ndarray
     * @return the broadcasted ndarray
     */
    INDArray broadcast(int[] shape);

    /**
     * Broadcasts this ndarray to be the specified shape
     * @param shape the new shape of this ndarray
     * @return the broadcasted ndarray
     */
    INDArray broadcasti(int[] shape);


    /**
     * Returns a scalar (individual element)
     * of a scalar ndarray
     * @return the individual item in this ndarray
     */
    Object element();

    /**
     * Returns a linear double array representation of this ndarray
     * @return the linear double array representation of this ndarray
     */
    public double[] data();


    void setData(double[] data);


    /**
     * Returns a linear float array representation of this ndarray
     * @return the linear float array representation of this ndarray
     */
    public float[] floatData();


    void setData(float[] data);
}
