/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.api.ndarray;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Condition;

import java.io.Serializable;
import java.util.List;

/**
 * Interface for an ndarray
 *
 * @author Adam Gibson
 */
public interface INDArray extends Serializable {


    /**
     * Returns true if the ndarray has already been freed
     * @return
     */
    boolean isCleanedUp();

    /**
     * Cleanup resources
     */
    void cleanup();

    /**
     * Reference to the ndarray
     *
     * @return the id for this ndarray
     */
    String id();

    /**
     * Resets the linear view
     */
    void resetLinearView();

    /**
     * Return the second stride for an ndarray.
     * Think of this as the stride for the next element in a column.
     *
     * @return the secondary stride for an ndarray
     */
    int secondaryStride();

    /**
     * Return the major stride for an ndarray
     *
     * @return the major stride for an ndarray
     */
    int majorStride();

    /**
     * Returns a linear view reference of shape
     * 1,length(ndarray)
     *
     * @return the linear view of this ndarray
     */
    public INDArray linearView();

    /**
     * Returns a linear view reference of shape
     * 1,length(ndarray)
     *
     * @return the linear view of this ndarray
     */
    public INDArray linearViewColumnOrder();

    /**
     * Returns the number of possible vectors for a given dimension
     *
     * @param dimension the dimension to calculate the number of vectors for
     * @return the number of possible vectors along a dimension
     */
    public int vectorsAlongDimension(int dimension);

    /**
     * Get the vector along a particular dimension
     *
     * @param index     the index of the vector to getScalar
     * @param dimension the dimension to getScalar the vector from
     * @return the vector along a particular dimension
     */
    public INDArray vectorAlongDimension(int index, int dimension);

    /**
     * Cumulative sum along a dimension
     *
     * @param dimension the dimension to perform cumulative sum along
     * @return the cumulative sum along the specified dimension
     */
    public INDArray cumsumi(int dimension);

    /**
     * Cumulative sum along a dimension (in place)
     *
     * @param dimension the dimension to perform cumulative sum along
     * @return the cumulative sum along the specified dimension
     */
    public INDArray cumsum(int dimension);

    /**
     * Assign all of the elements in the given
     * ndarray to this ndarray
     *
     * @param arr the elements to assign
     * @return this
     */
    public INDArray assign(INDArray arr);

    /**
     * Insert the number linearly in to the ndarray
     *
     * @param i     the index to insert into
     * @param value the value to insert
     * @return this
     */
    public INDArray putScalar(int i, double value);

    /**
     * Insert a scalar float at the specified index
     *
     * @param i
     * @param value
     * @return
     */
    INDArray putScalar(int i, float value);

    /**
     * Insert a scalar int at the specified index
     *
     * @param i
     * @param value
     * @return
     */
    INDArray putScalar(int i, int value);

    /**
     * Insert the item at the specified indices
     *
     * @param i     the indices to insert at
     * @param value the number to insert
     * @return this
     */
    public INDArray putScalar(int[] i, double value);

    /**
     * Returns an ndarray with 1 if the element is less than
     * the given element 0 other wise
     *
     * @param other the number to compare
     * @return a copied ndarray with the given
     * binary conditions
     */
    public INDArray lt(Number other);

    /**
     * In place less than comparison:
     * If the given number is less than the
     * comparison number the item is 0 otherwise 1
     *
     * @param other the number to compare
     * @return
     */
    public INDArray lti(Number other);

    INDArray putScalar(int[] indexes, float value);

    INDArray putScalar(int[] indexes, int value);

    /**
     * Returns an ndarray with 1 if the element is epsilon equals
     *
     * @param other the number to compare
     * @return a copied ndarray with the given
     * binary conditions
     */
    public INDArray eps(Number other);


    /**
     * Returns an ndarray with 1 if the element is epsilon equals
     *
     * @param other the number to compare
     * @return a copied ndarray with the given
     * binary conditions
     */
    public INDArray epsi(Number other);


    /**
     * Returns an ndarray with 1 if the element is less than
     * the given element 0 other wise
     *
     * @param other the number to compare
     * @return a copied ndarray with the given
     * binary conditions
     */
    public INDArray eq(Number other);

    /**
     * In place less than comparison:
     * If the given number is less than the
     * comparison number the item is 0 otherwise 1
     *
     * @param other the number to compare
     * @return
     */
    public INDArray eqi(Number other);

    /**
     * Greater than boolean (copying)(
     *
     * @param other
     * @return
     */
    public INDArray gt(Number other);

    /**
     * In place greater than comparison:
     * If the given number is less than the
     * comparison number the item is 0 otherwise 1
     *
     * @param other the number to compare
     * @return
     */
    public INDArray gti(Number other);

    /**
     * less than comparison:
     * If the given number is less than the
     * comparison number the item is 0 otherwise 1
     *
     * @param other the number to compare
     * @return the result ndarray
     */

    public INDArray lt(INDArray other);

    /**
     * In place less than comparison:
     * If the given number is less than the
     * comparison number the item is 0 otherwise 1
     *
     * @param other the number to compare
     * @return
     */
    public INDArray lti(INDArray other);


    /**
     * epsilon equals than comparison:
     * If the given number is less than the
     * comparison number the item is 0 otherwise 1
     *
     * @param other the number to compare
     * @return
     */
    public INDArray eps(INDArray other);

    /**
     * In place epsilon equals than comparison:
     * If the given number is less than the
     * comparison number the item is 0 otherwise 1
     *
     * @param other the number to compare
     * @return
     */
    public INDArray epsi(INDArray other);


    INDArray neq(Number other);

    INDArray neqi(Number other);

    INDArray neq(INDArray other);

    INDArray neqi(INDArray other);

    /**
     * equal than comparison:
     * If the given number is less than the
     * comparison number the item is 0 otherwise 1
     *
     * @param other the number to compare
     * @return
     */
    public INDArray eq(INDArray other);

    /**
     * In place equal than comparison:
     * If the given number is less than the
     * comparison number the item is 0 otherwise 1
     *
     * @param other the number to compare
     * @return
     */
    public INDArray eqi(INDArray other);

    /**
     * greater than comparison:
     * If the given number is less than the
     * comparison number the item is 0 otherwise 1
     *
     * @param other the number to compare
     * @return
     */
    public INDArray gt(INDArray other);

    /**
     * In place greater than comparison:
     * If the given number is less than the
     * comparison number the item is 0 otherwise 1
     *
     * @param other the number to compare
     * @return
     */
    public INDArray gti(INDArray other);


    /**
     * Returns the ndarray negative (cloned)
     *
     * @return
     */
    public INDArray neg();

    /**
     * In place setting of the negative version of this ndarray
     *
     * @return
     */
    public INDArray negi();

    /**
     * Reverse division
     *
     * @param n
     * @return
     */
    public INDArray rdiv(Number n);

    /**
     * In place reverse division
     *
     * @param n
     * @return
     */
    public INDArray rdivi(Number n);

    /**
     * Reverse subtraction with duplicates
     *
     * @param n
     * @return
     */
    public INDArray rsub(Number n);

    public INDArray rsubi(Number n);


    /**
     * Division by a number
     *
     * @param n
     * @return
     */
    public INDArray div(Number n);

    /**
     * In place scalar division
     *
     * @param n
     * @return
     */
    public INDArray divi(Number n);


    /**
     * Scalar multiplication (copy)
     *
     * @param n the number to multiply by
     * @return a copy of this ndarray multiplied by the given number
     */
    public INDArray mul(Number n);

    /**
     * In place scalar multiplication
     *
     * @param n
     * @return
     */
    public INDArray muli(Number n);


    /**
     * Scalar subtraction (copied)
     *
     * @param n the number to subtract by
     * @return this ndarray - the given number
     */
    public INDArray sub(Number n);


    /**
     * In place scalar subtraction
     *
     * @param n
     * @return
     */
    public INDArray subi(Number n);

    /**
     * Scalar addition (cloning)
     *
     * @param n the number to add
     * @return a clone with this matrix + the given number
     */
    public INDArray add(Number n);

    /**
     * In place scalar addition
     *
     * @param n
     * @return
     */
    public INDArray addi(Number n);


    /**
     * Reverse division (number / ndarray)
     *
     * @param n      the number to divide by
     * @param result
     * @return
     */
    public INDArray rdiv(Number n, INDArray result);


    /**
     * Reverse in place division
     *
     * @param n      the number to divide by  by
     * @param result the result ndarray
     * @return the result ndarray
     */
    public INDArray rdivi(Number n, INDArray result);

    /**
     * Reverse subtraction
     *
     * @param n      the number to subtract by
     * @param result the result ndarray
     * @return
     */
    public INDArray rsub(Number n, INDArray result);

    /**
     * Reverse in place subtraction
     *
     * @param n      the number to subtract by
     * @param result the result ndarray
     * @return the result ndarray
     */
    public INDArray rsubi(Number n, INDArray result);


    /**
     * @param n
     * @param result
     * @return
     */
    public INDArray div(Number n, INDArray result);

    /**
     * In place division of this ndarray
     *
     * @param n      the number to divide by
     * @param result the result ndarray
     * @return
     */
    public INDArray divi(Number n, INDArray result);


    public INDArray mul(Number n, INDArray result);


    /**
     * In place multiplication of this ndarray
     *
     * @param n      the number to divide by
     * @param result the result ndarray
     * @return
     */
    public INDArray muli(Number n, INDArray result);


    public INDArray sub(Number n, INDArray result);

    /**
     * In place subtraction of this ndarray
     *
     * @param n      the number to subtract by
     * @param result the result ndarray
     * @return the result ndarray
     */
    public INDArray subi(Number n, INDArray result);

    public INDArray add(Number n, INDArray result);

    /**
     * In place addition
     *
     * @param n      the number to add
     * @param result the result ndarray
     * @return the result ndarray
     */
    public INDArray addi(Number n, INDArray result);


    /**
     * Returns a subset of this array based on the specified
     * indexes
     *
     * @param indexes the indexes in to the array
     * @return a view of the array with the specified indices
     */
    public INDArray get(NDArrayIndex... indexes);


    /**
     * Get a list of specified columns
     *
     * @param columns
     * @return
     */
    INDArray getColumns(int[] columns);

    /**
     * Get a list of rows
     *
     * @param rows
     * @return
     */
    INDArray getRows(int[] rows);

    /**
     * Reverse division
     *
     * @param other the matrix to divide from
     * @return
     */
    INDArray rdiv(INDArray other);

    /**
     * Reverse divsion (in place)
     *
     * @param other
     * @return
     */
    INDArray rdivi(INDArray other);


    /**
     * Reverse division
     *
     * @param other  the matrix to subtract from
     * @param result the result ndarray
     * @return
     */
    INDArray rdiv(INDArray other, INDArray result);

    /**
     * Reverse division (in-place)
     *
     * @param other  the other ndarray to subtract
     * @param result the result ndarray
     * @return the ndarray with the operation applied
     */
    INDArray rdivi(INDArray other, INDArray result);

    /**
     * Reverse subtraction
     *
     * @param other  the matrix to subtract from
     * @param result the result ndarray
     * @return
     */
    INDArray rsub(INDArray other, INDArray result);


    /**
     * @param other
     * @return
     */
    INDArray rsub(INDArray other);

    /**
     * @param other
     * @return
     */
    INDArray rsubi(INDArray other);

    /**
     * Reverse subtraction (in-place)
     *
     * @param other  the other ndarray to subtract
     * @param result the result ndarray
     * @return the ndarray with the operation applied
     */
    INDArray rsubi(INDArray other, INDArray result);

    /**
     * Set the value of the ndarray to the specified value
     *
     * @param value the value to assign
     * @return the ndarray with the values
     */
    INDArray assign(Number value);


    /**
     * Get the linear index of the data in to
     * the array
     *
     * @param i the index to getScalar
     * @return the linear index in to the data
     */
    public int linearIndex(int i);

    /**
     * Iterate over every row of every slice
     *
     * @param op the operation to apply
     */
    public void iterateOverAllRows(SliceOp op);

    /**
     * Iterate over every column of every slice
     *
     * @param op the operation to apply
     */
    public void iterateOverAllColumns(SliceOp op);


    /**
     * Validate dimensions are equal
     *
     * @param other the other ndarray to compare
     */

    public void checkDimensions(INDArray other);

    /**
     * Gives the indices for the ending of each slice
     *
     * @return the off sets for the beginning of each slice
     */
    public int[] endsForSlices();

    void sliceVectors(List<INDArray> list);


    /**
     * Assigns the given matrix (put) to the specified slice
     *
     * @param slice the slice to assign
     * @param put   the slice to applyTransformToDestination
     * @return this for chainability
     */
    public INDArray putSlice(int slice, INDArray put);

    /**
     * 1 in the ndarray if the element matches
     * the condition 0 otherwise
     *
     * @param condition
     * @return
     */
    INDArray cond(Condition condition);

    /**
     * 1 in the ndarray if the element matches
     * the condition 0 otherwise
     *
     * @param condition
     * @return
     */
    INDArray condi(Condition condition);

    /**
     * Iterate along a dimension.
     * This encapsulates the process of sum, mean, and other processes
     * take when iterating over a dimension.
     *
     * @param dimension the dimension to iterate over
     * @param op        the operation to apply
     * @param modify    whether to modify this array while iterating
     */
    public void iterateOverDimension(int dimension, SliceOp op, boolean modify);


    /**
     * Replicate and tile array to fill out to the given shape
     *
     * @param shape the new shape of this ndarray
     * @return the shape to fill out to
     */
    public INDArray repmat(int[] shape);

    /**
     * Insert a row in to this array
     * Will throw an exception if this
     * ndarray is not a matrix
     *
     * @param row   the row insert into
     * @param toPut the row to insert
     * @return this
     */
    public INDArray putRow(int row, INDArray toPut);

    /**
     * Insert a column in to this array
     * Will throw an exception if this
     * ndarray is not a matrix
     *
     * @param column the column to insert
     * @param toPut  the array to put
     * @return this
     */
    public INDArray putColumn(int column, INDArray toPut);

    /**
     * Returns the element at the specified row/column
     * This will throw an exception if the
     *
     * @param row    the row of the element to return
     * @param column the row of the element to return
     * @return a scalar indarray of the element at this index
     */
    public INDArray getScalar(int row, int column);

    /**
     * Returns the element at the specified index
     *
     * @param i the index of the element to return
     * @return a scalar ndarray of the element at this index
     */
    public INDArray getScalar(int i);


    /**
     * Return the linear index of the specified row and column
     *
     * @param row    the row to getScalar the linear index for
     * @param column the column to getScalar the linear index for
     * @return the linear index of the given row and column
     */
    int index(int row, int column);

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


    /**
     * Put the elements of the ndarray
     * in to the specified indices
     *
     * @param indices the indices to put the ndarray in to
     * @param element the ndarray to put
     * @return this ndarray
     */
    public INDArray put(NDArrayIndex[] indices, INDArray element);

    /**
     * Put the elements of the ndarray
     * in to the specified indices
     *
     * @param indices the indices to put the ndarray in to
     * @param element the ndarray to put
     * @return this ndarray
     */
    public INDArray put(NDArrayIndex[] indices, Number element);

    /**
     * Inserts the element at the specified index
     *
     * @param indices the indices to insert into
     * @param element a scalar ndarray
     * @return a scalar ndarray of the element at this index
     */
    public INDArray put(int[] indices, INDArray element);


    /**
     * Inserts the element at the specified index
     *
     * @param i       the row insert into
     * @param j       the column to insert into
     * @param element a scalar ndarray
     * @return a scalar ndarray of the element at this index
     */
    public INDArray put(int i, int j, INDArray element);


    /**
     * Inserts the element at the specified index
     *
     * @param i       the row insert into
     * @param j       the column to insert into
     * @param element a scalar ndarray
     * @return a scalar ndarray of the element at this index
     */
    public INDArray put(int i, int j, Number element);


    /**
     * Inserts the element at the specified index
     *
     * @param i       the index insert into
     * @param element a scalar ndarray
     * @return a scalar ndarray of the element at this index
     */
    public INDArray put(int i, INDArray element);


    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    public INDArray diviColumnVector(INDArray columnVector);

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    public INDArray divColumnVector(INDArray columnVector);

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    public INDArray diviRowVector(INDArray rowVector);

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    public INDArray divRowVector(INDArray rowVector);


    /**
     * In place reverse divison of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    public INDArray rdiviColumnVector(INDArray columnVector);

    /**
     * In place reverse division of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    public INDArray rdivColumnVector(INDArray columnVector);

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    public INDArray rdiviRowVector(INDArray rowVector);

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    public INDArray rdivRowVector(INDArray rowVector);


    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    public INDArray muliColumnVector(INDArray columnVector);

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    public INDArray mulColumnVector(INDArray columnVector);

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    public INDArray muliRowVector(INDArray rowVector);

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    public INDArray mulRowVector(INDArray rowVector);


    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    public INDArray rsubiColumnVector(INDArray columnVector);

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    public INDArray rsubColumnVector(INDArray columnVector);

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    public INDArray rsubiRowVector(INDArray rowVector);

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    public INDArray rsubRowVector(INDArray rowVector);

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    public INDArray subiColumnVector(INDArray columnVector);

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    public INDArray subColumnVector(INDArray columnVector);

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    public INDArray subiRowVector(INDArray rowVector);

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    public INDArray subRowVector(INDArray rowVector);

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    public INDArray addiColumnVector(INDArray columnVector);

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    public INDArray addColumnVector(INDArray columnVector);

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    public INDArray addiRowVector(INDArray rowVector);

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    public INDArray addRowVector(INDArray rowVector);

    /**
     * Perform a copy matrix multiplication
     *
     * @param other the other matrix to perform matrix multiply with
     * @return the result of the matrix multiplication
     */
    public INDArray mmul(INDArray other);


    /**
     * Perform an copy matrix multiplication
     *
     * @param other  the other matrix to perform matrix multiply with
     * @param result the result ndarray
     * @return the result of the matrix multiplication
     */
    public INDArray mmul(INDArray other, INDArray result);


    /**
     * in place (element wise) division of two matrices
     *
     * @param other the second ndarray to divide
     * @return the result of the divide
     */
    public INDArray div(INDArray other);

    /**
     * copy (element wise) division of two matrices
     *
     * @param other  the second ndarray to divide
     * @param result the result ndarray
     * @return the result of the divide
     */
    public INDArray div(INDArray other, INDArray result);


    /**
     * copy (element wise) multiplication of two matrices
     *
     * @param other the second ndarray to multiply
     * @return the result of the addition
     */
    public INDArray mul(INDArray other);

    /**
     * copy (element wise) multiplication of two matrices
     *
     * @param other  the second ndarray to multiply
     * @param result the result ndarray
     * @return the result of the multiplication
     */
    public INDArray mul(INDArray other, INDArray result);

    /**
     * copy subtraction of two matrices
     *
     * @param other the second ndarray to subtract
     * @return the result of the addition
     */
    public INDArray sub(INDArray other);

    /**
     * copy subtraction of two matrices
     *
     * @param other  the second ndarray to subtract
     * @param result the result ndarray
     * @return the result of the subtraction
     */
    public INDArray sub(INDArray other, INDArray result);

    /**
     * copy addition of two matrices
     *
     * @param other the second ndarray to add
     * @return the result of the addition
     */
    public INDArray add(INDArray other);

    /**
     * copy addition of two matrices
     *
     * @param other  the second ndarray to add
     * @param result the result ndarray
     * @return the result of the addition
     */
    public INDArray add(INDArray other, INDArray result);


    /**
     * Perform an copy matrix multiplication
     *
     * @param other the other matrix to perform matrix multiply with
     * @return the result of the matrix multiplication
     */
    public INDArray mmuli(INDArray other);


    /**
     * Perform an copy matrix multiplication
     *
     * @param other  the other matrix to perform matrix multiply with
     * @param result the result ndarray
     * @return the result of the matrix multiplication
     */
    public INDArray mmuli(INDArray other, INDArray result);


    /**
     * in place (element wise) division of two matrices
     *
     * @param other the second ndarray to divide
     * @return the result of the divide
     */
    public INDArray divi(INDArray other);

    /**
     * in place (element wise) division of two matrices
     *
     * @param other  the second ndarray to divide
     * @param result the result ndarray
     * @return the result of the divide
     */
    public INDArray divi(INDArray other, INDArray result);


    /**
     * in place (element wise) multiplication of two matrices
     *
     * @param other the second ndarray to multiply
     * @return the result of the addition
     */
    public INDArray muli(INDArray other);

    /**
     * in place (element wise) multiplication of two matrices
     *
     * @param other  the second ndarray to multiply
     * @param result the result ndarray
     * @return the result of the multiplication
     */
    public INDArray muli(INDArray other, INDArray result);

    /**
     * in place subtraction of two matrices
     *
     * @param other the second ndarray to subtract
     * @return the result of the addition
     */
    public INDArray subi(INDArray other);

    /**
     * in place subtraction of two matrices
     *
     * @param other  the second ndarray to subtract
     * @param result the result ndarray
     * @return the result of the subtraction
     */
    public INDArray subi(INDArray other, INDArray result);

    /**
     * in place addition of two matrices
     *
     * @param other the second ndarray to add
     * @return the result of the addition
     */
    public INDArray addi(INDArray other);

    /**
     * in place addition of two matrices
     *
     * @param other  the second ndarray to add
     * @param result the result ndarray
     * @return the result of the addition
     */
    public INDArray addi(INDArray other, INDArray result);


    /**
     * Returns the normmax along the specified dimension
     *
     * @param dimension the dimension to getScalar the norm1 along
     * @return the norm1 along the specified dimension
     */
    public INDArray normmax(int dimension);


    /**
     * Returns the norm2 along the specified dimension
     *
     * @param dimension the dimension to getScalar the norm2 along
     * @return the norm2 along the specified dimension
     */
    public INDArray norm2(int dimension);


    /**
     * Returns the norm1 along the specified dimension
     *
     * @param dimension the dimension to getScalar the norm1 along
     * @return the norm1 along the specified dimension
     */
    public INDArray norm1(int dimension);


    /**
     * Standard deviation of an ndarray along a dimension
     *
     * @param dimension the dimension to getScalar the std along
     * @return the standard deviation along a particular dimension
     */
    public INDArray std(int dimension);

    /**
     * Returns the product along a given dimension
     *
     * @param dimension the dimension to getScalar the product along
     * @return the product along the specified dimension
     */
    public INDArray prod(int dimension);


    /**
     * Returns the overall mean of this ndarray
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    public INDArray mean(int dimension);


    /**
     * Returns the overall variance of this ndarray
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    public INDArray var(int dimension);


    /**
     * Returns the overall max of this ndarray
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    public INDArray max(int dimension);

    /**
     * Returns the overall min of this ndarray
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    public INDArray min(int dimension);

    /**
     * Returns the sum along the last dimension of this ndarray
     *
     * @param dimension the dimension to getScalar the sum along
     * @return the sum along the specified dimension of this ndarray
     */
    public INDArray sum(int dimension);


    public void setStride(int[] stride);

    /**
     * @param offsets
     * @param shape
     * @param stride
     * @return
     */
    public INDArray subArray(int[] offsets, int[] shape, int[] stride);

    /**
     * Returns the elements at the the specified indices
     *
     * @param indices the indices to getScalar
     * @return the array with the specified elements
     */
    public INDArray getScalar(int[] indices);


    int getInt(int... indices);

    double getDouble(int... indices);

    /**
     * Returns the elements at the the specified indices
     *
     * @param indices the indices to getScalar
     * @return the array with the specified elements
     */
    public float getFloat(int[] indices);


    double getDouble(int i);

    double getDouble(int i, int j);

    /**
     * Return the item at the linear index i
     *
     * @param i the index of the item to getScalar
     * @return the item at index j
     */
    public float getFloat(int i);

    /**
     * Return the item at row i column j
     * Note that this is the same as calling getScalar(new int[]{i,j}
     *
     * @param i the row to getScalar
     * @param j the column to getScalar
     * @return the item at row i column j
     */
    public float getFloat(int i, int j);


    /**
     * Return a copy of this ndarray
     *
     * @return a copy of this ndarray
     */
    public INDArray dup();


    /**
     * Returns a flattened version (row vector) of this ndarray
     *
     * @return a flattened version (row vector) of this ndarray
     */
    public INDArray ravel();


    void setData(DataBuffer data);

    /**
     * Returns the number of slices in this ndarray
     *
     * @return the number of slices in this ndarray
     */
    public int slices();


    /**
     * Returns the specified slice of this ndarray
     *
     * @param i         the index of the slice to return
     * @param dimension the dimension to return the slice for
     * @return the specified slice of this ndarray
     */
    public INDArray slice(int i, int dimension);


    /**
     * Returns the specified slice of this ndarray
     *
     * @param i the index of the slice to return
     * @return the specified slice of this ndarray
     */
    public INDArray slice(int i);


    /**
     * Returns the start of where the ndarray is
     * for the underlying data
     *
     * @return the starting offset
     */
    public int offset();

    /**
     * Reshapes the ndarray (can't change the length of the ndarray)
     *
     * @param newShape the new shape of the ndarray
     * @return the reshaped ndarray
     */
    public INDArray reshape(int... newShape);


    /**
     * Reshapes the ndarray (can't change the length of the ndarray)
     *
     * @param rows    the rows of the matrix
     * @param columns the columns of the matrix
     * @return the reshaped ndarray
     */
    public INDArray reshape(int rows, int columns);

    /**
     * Flip the rows and columns of a matrix
     *
     * @return the flipped rows and columns of a matrix
     */
    public INDArray transpose();


    /**
     * Flip the rows and columns of a matrix
     *
     * @return the flipped rows and columns of a matrix
     */
    public INDArray transposei();

    /**
     * Mainly here for people coming from numpy.
     * This is equivalent to a call to permute
     *
     * @param dimension the dimension to swap
     * @param with      the one to swap it with
     * @return the swapped axes view
     */
    public INDArray swapAxes(int dimension, int with);

    /**
     * See: http://www.mathworks.com/help/matlab/ref/permute.html
     *
     * @param rearrange the dimensions to swap to
     * @return the newly permuted array
     */
    public INDArray permute(int... rearrange);

    /**
     * Dimshuffle: an extension of permute that adds the ability
     * to broadcast various dimensions.
     * This will only accept integers and xs.
     * <p/>
     * An x indicates a dimension should be broadcasted rather than permuted.
     *
     * @param rearrange     the dimensions to swap to
     * @param newOrder      the new order (think permute)
     * @param broadCastable (whether the dimension is broadcastable) (must be same length as new order)
     * @return the newly permuted array
     */
    public INDArray dimShuffle(Object[] rearrange, int[] newOrder, boolean[] broadCastable);

    /**
     * Returns the specified column.
     * Throws an exception if its not a matrix
     *
     * @param i the column to getScalar
     * @return the specified column
     */
    INDArray getColumn(int i);

    /**
     * Returns the specified row.
     * Throws an exception if its not a matrix
     *
     * @param i the row to getScalar
     * @return the specified row
     */
    INDArray getRow(int i);

    /**
     * Returns the number of columns in this matrix (throws exception if not 2d)
     *
     * @return the number of columns in this matrix
     */
    int columns();

    /**
     * Returns the number of rows in this matrix (throws exception if not 2d)
     *
     * @return the number of rows in this matrix
     */
    int rows();

    /**
     * Returns true if the number of columns is 1
     *
     * @return true if the number of columns is 1
     */
    boolean isColumnVector();

    /**
     * Returns true if the number of rows is 1
     *
     * @return true if the number of rows is 1
     */
    boolean isRowVector();

    /**
     * Returns true if this ndarray is a vector
     *
     * @return whether this ndarray is a vector
     */
    boolean isVector();


    /**
     * Returns whether the matrix
     * has the same rows and columns
     *
     * @return true if the matrix has the same rows and columns
     * false otherwise
     */
    boolean isSquare();

    /**
     * Returns true if this ndarray is a matrix
     *
     * @return whether this ndarray is a matrix
     */
    boolean isMatrix();

    /**
     * Returns true if this ndarray is a scalar
     *
     * @return whether this ndarray is a scalar
     */
    boolean isScalar();


    /**
     * Returns the shape of this ndarray
     *
     * @return the shape of this ndarray
     */
    int[] shape();


    /**
     * Returns the stride of this ndarray
     *
     * @return the stride of this ndarray
     */
    int[] stride();

    char ordering();

    /**
     * Returns the size along a specified dimension
     *
     * @param dimension the dimension to return the size for
     * @return the size of the array along the specified dimension
     */
    int size(int dimension);

    /**
     * Returns the total number of elements in the ndarray
     *
     * @return the number of elements in the ndarray
     */
    int length();


    /**
     * Broadcasts this ndarray to be the specified shape
     *
     * @param shape the new shape of this ndarray
     * @return the broadcasted ndarray
     */
    INDArray broadcast(int... shape);


    /**
     * Returns a scalar (individual element)
     * of a scalar ndarray
     *
     * @return the individual item in this ndarray
     */
    Object element();

    /**
     * Returns a linear double array representation of this ndarray
     *
     * @return the linear double array representation of this ndarray
     */
    public DataBuffer data();


    void setData(float[] data);

    public IComplexNDArray rdiv(IComplexNumber n);

    public IComplexNDArray rdivi(IComplexNumber n);

    public IComplexNDArray rsub(IComplexNumber n);

    public IComplexNDArray rsubi(IComplexNumber n);


    public IComplexNDArray div(IComplexNumber n);

    public IComplexNDArray divi(IComplexNumber n);


    public IComplexNDArray mul(IComplexNumber n);

    public IComplexNDArray muli(IComplexNumber n);


    public IComplexNDArray sub(IComplexNumber n);

    public IComplexNDArray subi(IComplexNumber n);

    public IComplexNDArray add(IComplexNumber n);

    public IComplexNDArray addi(IComplexNumber n);


    public IComplexNDArray rdiv(IComplexNumber n, IComplexNDArray result);

    public IComplexNDArray rdivi(IComplexNumber n, IComplexNDArray result);

    public IComplexNDArray rsub(IComplexNumber n, IComplexNDArray result);

    public IComplexNDArray rsubi(IComplexNumber n, IComplexNDArray result);


    public IComplexNDArray div(IComplexNumber n, IComplexNDArray result);

    public IComplexNDArray divi(IComplexNumber n, IComplexNDArray result);


    public IComplexNDArray mul(IComplexNumber n, IComplexNDArray result);

    public IComplexNDArray muli(IComplexNumber n, IComplexNDArray result);


    public IComplexNDArray sub(IComplexNumber n, IComplexNDArray result);

    public IComplexNDArray subi(IComplexNumber n, IComplexNDArray result);

    public IComplexNDArray add(IComplexNumber n, IComplexNDArray result);

    public IComplexNDArray addi(IComplexNumber n, IComplexNDArray result);


}
