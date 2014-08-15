package org.deeplearning4j.linalg.api.ndarray;


import org.deeplearning4j.linalg.ops.reduceops.Ops;

/**
 * Created by mjk on 8/12/14.
 */
public class JCUBlasNDArray  implements INDArray  {


    /**
     * Reverse division
     *
     * @param other the matrix to divide from
     * @return
     */
    @Override
    public INDArray rdiv(INDArray other) {
        return null;
    }

    /**
     * Reverse divsion (in place)
     *
     * @param other
     * @return
     */
    @Override
    public INDArray rdivi(INDArray other) {
        return null;
    }

    /**
     * Reverse division
     *
     * @param other  the matrix to subtract from
     * @param result the result ndarray
     * @return
     */
    @Override
    public INDArray rdiv(INDArray other, INDArray result) {
        return null;
    }

    /**
     * Reverse division (in-place)
     *
     * @param other  the other ndarray to subtract
     * @param result the result ndarray
     * @return the ndarray with the operation applied
     */
    @Override
    public INDArray rdivi(INDArray other, INDArray result) {
        return null;
    }

    /**
     * Reverse subtraction
     *
     * @param other  the matrix to subtract from
     * @param result the result ndarray
     * @return
     */
    @Override
    public INDArray rsub(INDArray other, INDArray result) {
        return null;
    }

    /**
     * @param other
     * @return
     */
    @Override
    public INDArray rsub(INDArray other) {
        return null;
    }

    /**
     * @param other
     * @return
     */
    @Override
    public INDArray rsubi(INDArray other) {
        return null;
    }

    /**
     * Reverse subtraction (in-place)
     *
     * @param other  the other ndarray to subtract
     * @param result the result ndarray
     * @return the ndarray with the operation applied
     */
    @Override
    public INDArray rsubi(INDArray other, INDArray result) {
        return null;
    }

    /**
     * Set the value of the ndarray to the specified value
     *
     * @param value the value to assign
     * @return the ndarray with the values
     */
    @Override
    public INDArray assign(Number value) {
        return null;
    }

    /**
     * Get the linear index of the data in to
     * the array
     *
     * @param i the index to get
     * @return the linear index in to the data
     */
    @Override
    public int linearIndex(int i) {
        return 0;
    }

    /**
     * Iterate over every row of every slice
     *
     * @param op the operation to apply
     */
    @Override
    public void iterateOverAllRows(SliceOp op) {

    }

    /**
     * Fetch a particular number on a multi dimensional scale.
     *
     * @param indexes the indexes to getFromOrigin a number from
     * @return the number at the specified indices
     */
    @Override
    public INDArray getScalar(int... indexes) {
        return null;
    }

    /**
     * Validate dimensions are equal
     *
     * @param other the other ndarray to compare
     */
    @Override
    public void checkDimensions(INDArray other) {

    }

    /**
     * Gives the indices for the ending of each slice
     *
     * @return the off sets for the beginning of each slice
     */
    @Override
    public int[] endsForSlices() {
        return new int[0];
    }

    /**
     * http://docs.scipy.org/doc/numpy/reference/generated/numpy.ufunc.reduce.html
     *
     * @param op        the operation to do
     * @param dimension the dimension to return from
     * @return the results of the reduce (applying the operation along the specified
     * dimension)t
     */
    @Override
    public INDArray reduce(Ops.DimensionOp op, int dimension) {
        return null;
    }

    /**
     * Assigns the given matrix (put) to the specified slice
     *
     * @param slice the slice to assign
     * @param put   the slice to applyTransformToDestination
     * @return this for chainability
     */
    @Override
    public INDArray putSlice(int slice, INDArray put) {
        return null;
    }

    /**
     * Iterate along a dimension.
     * This encapsulates the process of sum, mean, and other processes
     * take when iterating over a dimension.
     *
     * @param dimension the dimension to iterate over
     * @param op        the operation to apply
     * @param modify    whether to modify this array while iterating
     */
    @Override
    public void iterateOverDimension(int dimension, SliceOp op, boolean modify) {

    }

    /**
     * Replicate and tile array to fill out to the given shape
     *
     * @param shape the new shape of this ndarray
     * @return the shape to fill out to
     */
    @Override
    public INDArray repmat(int[] shape) {
        return null;
    }

    /**
     * Insert a row in to this array
     * Will throw an exception if this
     * ndarray is not a matrix
     *
     * @param row   the row insert into
     * @param toPut the row to insert
     * @return this
     */
    @Override
    public INDArray putRow(int row, INDArray toPut) {
        return null;
    }

    /**
     * Insert a column in to this array
     * Will throw an exception if this
     * ndarray is not a matrix
     *
     * @param column the column to insert
     * @param toPut  the array to put
     * @return this
     */
    @Override
    public INDArray putColumn(int column, INDArray toPut) {
        return null;
    }

    /**
     * Returns the element at the specified row/column
     * This will throw an exception if the
     *
     * @param row    the row of the element to return
     * @param column the row of the element to return
     * @return a scalar indarray of the element at this index
     */
    @Override
    public INDArray getScalar(int row, int column) {
        return null;
    }

    /**
     * Returns the element at the specified index
     *
     * @param i the index of the element to return
     * @return a scalar ndarray of the element at this index
     */
    @Override
    public INDArray getScalar(int i) {
        return null;
    }

    /**
     * Inserts the element at the specified index
     *
     * @param indices the indices to insert into
     * @param element a scalar ndarray
     * @return a scalar ndarray of the element at this index
     */
    @Override
    public INDArray put(int[] indices, INDArray element) {
        return null;
    }

    /**
     * Inserts the element at the specified index
     *
     * @param i       the row insert into
     * @param j       the column to insert into
     * @param element a scalar ndarray
     * @return a scalar ndarray of the element at this index
     */
    @Override
    public INDArray put(int i, int j, INDArray element) {
        return null;
    }

    /**
     * Inserts the element at the specified index
     *
     * @param i       the index insert into
     * @param element a scalar ndarray
     * @return a scalar ndarray of the element at this index
     */
    @Override
    public INDArray put(int i, INDArray element) {
        return null;
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray diviColumnVector(INDArray columnVector) {
        return null;
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray divColumnVector(INDArray columnVector) {
        return null;
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray diviRowVector(INDArray rowVector) {
        return null;
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray divRowVector(INDArray rowVector) {
        return null;
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray muliColumnVector(INDArray columnVector) {
        return null;
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray mulColumnVector(INDArray columnVector) {
        return null;
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray muliRowVector(INDArray rowVector) {
        return null;
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray mulRowVector(INDArray rowVector) {
        return null;
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray subiColumnVector(INDArray columnVector) {
        return null;
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray subColumnVector(INDArray columnVector) {
        return null;
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray subiRowVector(INDArray rowVector) {
        return null;
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray subRowVector(INDArray rowVector) {
        return null;
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray addiColumnVector(INDArray columnVector) {
        return null;
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray addColumnVector(INDArray columnVector) {
        return null;
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray addiRowVector(INDArray rowVector) {
        return null;
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray addRowVector(INDArray rowVector) {
        return null;
    }

    /**
     * Perform a copy matrix multiplication
     *
     * @param other the other matrix to perform matrix multiply with
     * @return the result of the matrix multiplication
     */
    @Override
    public INDArray mmul(INDArray other) {
        return null;
    }

    /**
     * Perform an copy matrix multiplication
     *
     * @param other  the other matrix to perform matrix multiply with
     * @param result the result ndarray
     * @return the result of the matrix multiplication
     */
    @Override
    public INDArray mmul(INDArray other, INDArray result) {
        return null;
    }

    /**
     * in place (element wise) division of two matrices
     *
     * @param other the second ndarray to divide
     * @return the result of the divide
     */
    @Override
    public INDArray div(INDArray other) {
        return null;
    }

    /**
     * copy (element wise) division of two matrices
     *
     * @param other  the second ndarray to divide
     * @param result the result ndarray
     * @return the result of the divide
     */
    @Override
    public INDArray div(INDArray other, INDArray result) {
        return null;
    }

    /**
     * copy (element wise) multiplication of two matrices
     *
     * @param other the second ndarray to multiply
     * @return the result of the addition
     */
    @Override
    public INDArray mul(INDArray other) {
        return null;
    }

    /**
     * copy (element wise) multiplication of two matrices
     *
     * @param other  the second ndarray to multiply
     * @param result the result ndarray
     * @return the result of the multiplication
     */
    @Override
    public INDArray mul(INDArray other, INDArray result) {
        return null;
    }

    /**
     * copy subtraction of two matrices
     *
     * @param other the second ndarray to subtract
     * @return the result of the addition
     */
    @Override
    public INDArray sub(INDArray other) {
        return null;
    }

    /**
     * copy subtraction of two matrices
     *
     * @param other  the second ndarray to subtract
     * @param result the result ndarray
     * @return the result of the subtraction
     */
    @Override
    public INDArray sub(INDArray other, INDArray result) {
        return null;
    }

    /**
     * copy addition of two matrices
     *
     * @param other the second ndarray to add
     * @return the result of the addition
     */
    @Override
    public INDArray add(INDArray other) {
        return null;
    }

    /**
     * copy addition of two matrices
     *
     * @param other  the second ndarray to add
     * @param result the result ndarray
     * @return the result of the addition
     */
    @Override
    public INDArray add(INDArray other, INDArray result) {
        return null;
    }

    /**
     * Perform an copy matrix multiplication
     *
     * @param other the other matrix to perform matrix multiply with
     * @return the result of the matrix multiplication
     */
    @Override
    public INDArray mmuli(INDArray other) {
        return null;
    }

    /**
     * Perform an copy matrix multiplication
     *
     * @param other  the other matrix to perform matrix multiply with
     * @param result the result ndarray
     * @return the result of the matrix multiplication
     */
    @Override
    public INDArray mmuli(INDArray other, INDArray result) {
        return null;
    }

    /**
     * in place (element wise) division of two matrices
     *
     * @param other the second ndarray to divide
     * @return the result of the divide
     */
    @Override
    public INDArray divi(INDArray other) {
        return null;
    }

    /**
     * in place (element wise) division of two matrices
     *
     * @param other  the second ndarray to divide
     * @param result the result ndarray
     * @return the result of the divide
     */
    @Override
    public INDArray divi(INDArray other, INDArray result) {
        return null;
    }

    /**
     * in place (element wise) multiplication of two matrices
     *
     * @param other the second ndarray to multiply
     * @return the result of the addition
     */
    @Override
    public INDArray muli(INDArray other) {
        return null;
    }

    /**
     * in place (element wise) multiplication of two matrices
     *
     * @param other  the second ndarray to multiply
     * @param result the result ndarray
     * @return the result of the multiplication
     */
    @Override
    public INDArray muli(INDArray other, INDArray result) {
        return null;
    }

    /**
     * in place subtraction of two matrices
     *
     * @param other the second ndarray to subtract
     * @return the result of the addition
     */
    @Override
    public INDArray subi(INDArray other) {
        return null;
    }

    /**
     * in place subtraction of two matrices
     *
     * @param other  the second ndarray to subtract
     * @param result the result ndarray
     * @return the result of the subtraction
     */
    @Override
    public INDArray subi(INDArray other, INDArray result) {
        return null;
    }

    /**
     * in place addition of two matrices
     *
     * @param other the second ndarray to add
     * @return the result of the addition
     */
    @Override
    public INDArray addi(INDArray other) {
        return null;
    }

    /**
     * in place addition of two matrices
     *
     * @param other  the second ndarray to add
     * @param result the result ndarray
     * @return the result of the addition
     */
    @Override
    public INDArray addi(INDArray other, INDArray result) {
        return null;
    }

    /**
     * Returns the normmax along the specified dimension
     *
     * @param dimension the dimension to getScalar the norm1 along
     * @return the norm1 along the specified dimension
     */
    @Override
    public INDArray normmax(int dimension) {
        return null;
    }

    /**
     * Returns the norm2 along the specified dimension
     *
     * @param dimension the dimension to getScalar the norm2 along
     * @return the norm2 along the specified dimension
     */
    @Override
    public INDArray norm2(int dimension) {
        return null;
    }

    /**
     * Returns the norm1 along the specified dimension
     *
     * @param dimension the dimension to getScalar the norm1 along
     * @return the norm1 along the specified dimension
     */
    @Override
    public INDArray norm1(int dimension) {
        return null;
    }

    /**
     * Returns the product along a given dimension
     *
     * @param dimension the dimension to getScalar the product along
     * @return the product along the specified dimension
     */
    @Override
    public INDArray prod(int dimension) {
        return null;
    }

    /**
     * Returns the overall mean of this ndarray
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    @Override
    public INDArray mean(int dimension) {
        return null;
    }

    /**
     * Returns the overall max of this ndarray
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    @Override
    public INDArray max(int dimension) {
        return null;
    }

    /**
     * Returns the overall min of this ndarray
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    @Override
    public INDArray min(int dimension) {
        return null;
    }

    /**
     * Returns the sum along the last dimension of this ndarray
     *
     * @param dimension the dimension to getScalar the sum along
     * @return the sum along the specified dimension of this ndarray
     */
    @Override
    public INDArray sum(int dimension) {
        return null;
    }

    /**
     * Returns the elements at the the specified indices
     *
     * @param indices the indices to getScalar
     * @return the array with the specified elements
     */
    @Override
    public INDArray get(int[] indices) {
        return null;
    }

    /**
     * Return a copy of this ndarray
     *
     * @return a copy of this ndarray
     */
    @Override
    public INDArray dup() {
        return null;
    }

    /**
     * Returns a flattened version (row vector) of this ndarray
     *
     * @return a flattened version (row vector) of this ndarray
     */
    @Override
    public INDArray ravel() {
        return null;
    }

    /**
     * Returns the number of slices in this ndarray
     *
     * @return the number of slices in this ndarray
     */
    @Override
    public int slices() {
        return 0;
    }

    /**
     * Returns the specified slice of this ndarray
     *
     * @param i         the index of the slice to return
     * @param dimension the dimension to return the slice for
     * @return the specified slice of this ndarray
     */
    @Override
    public INDArray slice(int i, int dimension) {
        return null;
    }

    /**
     * Returns the specified slice of this ndarray
     *
     * @param i the index of the slice to return
     * @return the specified slice of this ndarray
     */
    @Override
    public INDArray slice(int i) {
        return null;
    }

    /**
     * Returns the start of where the ndarray is
     * for the underlying data
     *
     * @return the starting offset
     */
    @Override
    public int offset() {
        return 0;
    }

    /**
     * Reshapes the ndarray (can't change the length of the ndarray)
     *
     * @param newShape the new shape of the ndarray
     * @return the reshaped ndarray
     */
    @Override
    public INDArray reshape(int[] newShape) {
        return null;
    }

    /**
     * Flip the rows and columns of a matrix
     *
     * @return the flipped rows and columns of a matrix
     */
    @Override
    public INDArray transpose() {
        return null;
    }

    /**
     * Mainly here for people coming from numpy.
     * This is equivalent to a call to permute
     *
     * @param dimension the dimension to swap
     * @param with      the one to swap it with
     * @return the swapped axes view
     */
    @Override
    public INDArray swapAxes(int dimension, int with) {
        return null;
    }

    /**
     * See: http://www.mathworks.com/help/matlab/ref/permute.html
     *
     * @param rearrange the dimensions to swap to
     * @return the newly permuted array
     */
    @Override
    public INDArray permute(int[] rearrange) {
        return null;
    }

    /**
     * Returns the specified column.
     * Throws an exception if its not a matrix
     *
     * @param i the column to getScalar
     * @return the specified column
     */
    @Override
    public INDArray getColumn(int i) {
        return null;
    }

    /**
     * Returns the specified row.
     * Throws an exception if its not a matrix
     *
     * @param i the row to getScalar
     * @return the specified row
     */
    @Override
    public INDArray getRow(int i) {
        return null;
    }

    /**
     * Returns the number of columns in this matrix (throws exception if not 2d)
     *
     * @return the number of columns in this matrix
     */
    @Override
    public int columns() {
        return 0;
    }

    /**
     * Returns the number of rows in this matrix (throws exception if not 2d)
     *
     * @return the number of rows in this matrix
     */
    @Override
    public int rows() {
        return 0;
    }

    /**
     * Returns true if the number of columns is 1
     *
     * @return true if the number of columns is 1
     */
    @Override
    public boolean isColumnVector() {
        return false;
    }

    /**
     * Returns true if the number of rows is 1
     *
     * @return true if the number of rows is 1
     */
    @Override
    public boolean isRowVector() {
        return false;
    }

    /**
     * Returns true if this ndarray is a vector
     *
     * @return whether this ndarray is a vector
     */
    @Override
    public boolean isVector() {
        return false;
    }

    /**
     * Returns true if this ndarray is a matrix
     *
     * @return whether this ndarray is a matrix
     */
    @Override
    public boolean isMatrix() {
        return false;
    }

    /**
     * Returns true if this ndarray is a scalar
     *
     * @return whether this ndarray is a scalar
     */
    @Override
    public boolean isScalar() {
        return false;
    }

    /**
     * Returns the shape of this ndarray
     *
     * @return the shape of this ndarray
     */
    @Override
    public int[] shape() {
        return new int[0];
    }

    /**
     * Returns the stride of this ndarray
     *
     * @return the stride of this ndarray
     */
    @Override
    public int[] stride() {
        return new int[0];
    }

    /**
     * Returns the size along a specified dimension
     *
     * @param dimension the dimension to return the size for
     * @return the size of the array along the specified dimension
     */
    @Override
    public int size(int dimension) {
        return 0;
    }

    /**
     * Returns the total number of elements in the ndarray
     *
     * @return the number of elements in the ndarray
     */
    @Override
    public int length() {
        return 0;
    }

    /**
     * Broadcasts this ndarray to be the specified shape
     *
     * @param shape the new shape of this ndarray
     * @return the broadcasted ndarray
     */
    @Override
    public INDArray broadcast(int[] shape) {
        return null;
    }

    /**
     * Broadcasts this ndarray to be the specified shape
     *
     * @param shape the new shape of this ndarray
     * @return the broadcasted ndarray
     */
    @Override
    public INDArray broadcasti(int[] shape) {
        return null;
    }

    /**
     * Returns a scalar (individual element)
     * of a scalar ndarray
     *
     * @return the individual item in this ndarray
     */
    @Override
    public Object element() {
        return null;
    }

    /**
     * Returns a linear double array representation of this ndarray
     *
     * @return the linear double array representation of this ndarray
     */
    @Override
    public double[] data() {
        return new double[0];
    }

    /**
     * Returns a linear float array representation of this ndarray
     *
     * @return the linear float array representation of this ndarray
     */
    @Override
    public float[] floatData() {
        return new float[0];
    }
}
