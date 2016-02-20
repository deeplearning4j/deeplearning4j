/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.api.ndarray;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.ShapeOffsetResolution;
import org.nd4j.linalg.indexing.conditions.Condition;

import java.io.Serializable;
import java.nio.IntBuffer;
import java.util.List;

/**
 * Interface for an ndarray
 *
 * @author Adam Gibson
 */
public interface INDArray extends Serializable  {

    /**
     * Shape info
     * @return
     */
    IntBuffer shapeInfo();
    /**
     * Returns true if this array is a view or not
     * @return
     */
    boolean isView();

    /**
     * Set the ndarray to wrap around
     * @param wrapAround thewrap around
     */
    void setWrapAround(boolean wrapAround);

    /**
     * Returns true if the ndarray
     * on linear indexing wraps around
     * based on the stride(1) of the ndarray
     * This is a useful optimization in linear view
     * where strides that might otherwise
     * go out of bounds but wrap around instead.
     *
     * @return true if this ndarray wraps around on linear
     * indexing, false otherwise
     */
    boolean isWrapAround();

    /**
     * The rank of the ndarray (the number of dimensions
     * @return the rank for the ndarray
     */
    int rank();

    /**
     * Calculate the stride along a particular dimension
     * @param dimension the dimension to get the stride for
     * @return the stride for a particular dimension
     */
    int stride(int dimension);

    /**
     * Element stride (one element to the next,
     * also called the defualt stride: 1 for normal
     * 2 for complex)
     * @return
     */
    int elementStride();


    /**
     * Element wise stride
     */
    int elementWiseStride();

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
     * Get a scalar
     * at the given linear offset
     * @param offset the offset to get at
     * @return this
     */
    double getDoubleUnsafe(int offset);

    /**
     * Insert a scalar
     * at the given linear offset
     * @param offset the offset to insert at
     * @param value the value to insert
     * @return this
     */
    INDArray putScalarUnsafe(int offset,double value);

    /**
     * Return the major stride for an ndarray
     *
     * @return the major stride for an ndarray
     */
    int majorStride();

    /**
     * Get the inner most stride
     * wrt the ordering of the array
     * @return
     */
    int innerMostStride();

    /**
     * Returns a linear view reference of shape
     * 1,length(ndarray)
     *
     * @return the linear view of this ndarray
     */
    INDArray linearView();



    /**
     * Returns a linear view reference of shape
     * 1,length(ndarray)
     *
     * @return the linear view of this ndarray
     */
    INDArray linearViewColumnOrder();

    /**
     * Returns the number of possible vectors for a given dimension
     *
     * @param dimension the dimension to calculate the number of vectors for
     * @return the number of possible vectors along a dimension
     */
    int vectorsAlongDimension(int dimension);

    /**
     * Get the vector along a particular dimension
     *
     * @param index     the index of the vector to getScalar
     * @param dimension the dimension to getScalar the vector from
     * @return the vector along a particular dimension
     */
    INDArray vectorAlongDimension(int index, int dimension);
    /**
     * Returns the number of possible vectors for a given dimension
     *
     * @param dimension the dimension to calculate the number of vectors for
     * @return the number of possible vectors along a dimension
     */
    int tensorssAlongDimension(int...dimension);

    /**
     * Get the vector along a particular dimension
     *
     * @param index     the index of the vector to getScalar
     * @param dimension the dimension to getScalar the vector from
     * @return the vector along a particular dimension
     */
    INDArray tensorAlongDimension(int index, int...dimension);


    /**
     * Cumulative sum along a dimension
     *
     * @param dimension the dimension to perform cumulative sum along
     * @return the cumulative sum along the specified dimension
     */
    INDArray cumsumi(int dimension);

    /**
     * Cumulative sum along a dimension (in place)
     *
     * @param dimension the dimension to perform cumulative sum along
     * @return the cumulative sum along the specified dimension
     */
    INDArray cumsum(int dimension);

    /**
     * Assign all of the elements in the given
     * ndarray to this ndarray
     *
     * @param arr the elements to assign
     * @return this
     */
    INDArray assign(INDArray arr);

    /**
     * Insert the number linearly in to the ndarray
     *
     * @param i     the index to insert into
     * @param value the value to insert
     * @return this
     */
    INDArray putScalar(int i, double value);

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
    INDArray putScalar(int[] i, double value);

    /**
     * Returns an ndarray with 1 if the element is less than
     * the given element 0 other wise
     *
     * @param other the number to compare
     * @return a copied ndarray with the given
     * binary conditions
     */
    INDArray lt(Number other);

    /**
     * In place less than comparison:
     * If the given number is less than the
     * comparison number the item is 0 otherwise 1
     *
     * @param other the number to compare
     * @return
     */
    INDArray lti(Number other);

    /**
     *
     * @param indexes
     * @param value
     * @return
     */
    INDArray putScalar(int[] indexes, float value);

    /**
     *
     * @param indexes
     * @param value
     * @return
     */
    INDArray putScalar(int[] indexes, int value);

    /**
     * Returns an ndarray with 1 if the element is epsilon equals
     *
     * @param other the number to compare
     * @return a copied ndarray with the given
     * binary conditions
     */
    INDArray eps(Number other);


    /**
     * Returns an ndarray with 1 if the element is epsilon equals
     *
     * @param other the number to compare
     * @return a copied ndarray with the given
     * binary conditions
     */
    INDArray epsi(Number other);


    /**
     * Returns an ndarray with 1 if the element is less than
     * the given element 0 other wise
     *
     * @param other the number to compare
     * @return a copied ndarray with the given
     * binary conditions
     */
    INDArray eq(Number other);

    /**
     * In place less than comparison:
     * If the given number is less than the
     * comparison number the item is 0 otherwise 1
     *
     * @param other the number to compare
     * @return
     */
    INDArray eqi(Number other);

    /**
     * Greater than boolean (copying)(
     *
     * @param other
     * @return
     */
    INDArray gt(Number other);

    /**
     * In place greater than comparison:
     * If the given number is less than the
     * comparison number the item is 0 otherwise 1
     *
     * @param other the number to compare
     * @return
     */
    INDArray gti(Number other);

    /**
     * less than comparison:
     * If the given number is less than the
     * comparison number the item is 0 otherwise 1
     *
     * @param other the number to compare
     * @return the result ndarray
     */

    INDArray lt(INDArray other);

    /**
     * In place less than comparison:
     * If the given number is less than the
     * comparison number the item is 0 otherwise 1
     *
     * @param other the number to compare
     * @return
     */
    INDArray lti(INDArray other);


    /**
     * epsilon equals than comparison:
     * If the given number is less than the
     * comparison number the item is 0 otherwise 1
     *
     * @param other the number to compare
     * @return
     */
    INDArray eps(INDArray other);

    /**
     * In place epsilon equals than comparison:
     * If the given number is less than the
     * comparison number the item is 0 otherwise 1
     *
     * @param other the number to compare
     * @return
     */
    INDArray epsi(INDArray other);

    /**
     * Not equal
     * @param other
     * @return
     */
    INDArray neq(Number other);

    /**
     * In place not equal
     * @param other
     * @return
     */
    INDArray neqi(Number other);

    /**
     *
     * @param other
     * @return
     */
    INDArray neq(INDArray other);

    /**
     *
     * @param other
     * @return
     */
    INDArray neqi(INDArray other);

    /**
     * equal than comparison:
     * If the given number is less than the
     * comparison number the item is 0 otherwise 1
     *
     * @param other the number to compare
     * @return
     */
    INDArray eq(INDArray other);

    /**
     * In place equal than comparison:
     * If the given number is less than the
     * comparison number the item is 0 otherwise 1
     *
     * @param other the number to compare
     * @return
     */
    INDArray eqi(INDArray other);

    /**
     * greater than comparison:
     * If the given number is less than the
     * comparison number the item is 0 otherwise 1
     *
     * @param other the number to compare
     * @return
     */
    INDArray gt(INDArray other);

    /**
     * In place greater than comparison:
     * If the given number is less than the
     * comparison number the item is 0 otherwise 1
     *
     * @param other the number to compare
     * @return
     */
    INDArray gti(INDArray other);


    /**
     * Returns the ndarray negative (cloned)
     *
     * @return
     */
    INDArray neg();

    /**
     * In place setting of the negative version of this ndarray
     *
     * @return
     */
    INDArray negi();

    /**
     * Reverse division
     *
     * @param n
     * @return
     */
    INDArray rdiv(Number n);

    /**
     * In place reverse division
     *
     * @param n
     * @return
     */
    INDArray rdivi(Number n);

    /**
     * Reverse subtraction with duplicates
     *
     * @param n
     * @return
     */
    INDArray rsub(Number n);


    /**
     *
     * @param n
     * @return
     */
    INDArray rsubi(Number n);


    /**
     * Division by a number
     *
     * @param n
     * @return
     */
    INDArray div(Number n);

    /**
     * In place scalar division
     *
     * @param n
     * @return
     */
    INDArray divi(Number n);


    /**
     * Scalar multiplication (copy)
     *
     * @param n the number to multiply by
     * @return a copy of this ndarray multiplied by the given number
     */
    INDArray mul(Number n);

    /**
     * In place scalar multiplication
     *
     * @param n
     * @return
     */
    INDArray muli(Number n);


    /**
     * Scalar subtraction (copied)
     *
     * @param n the number to subtract by
     * @return this ndarray - the given number
     */
    INDArray sub(Number n);


    /**
     * In place scalar subtraction
     *
     * @param n
     * @return
     */
    INDArray subi(Number n);

    /**
     * Scalar addition (cloning)
     *
     * @param n the number to add
     * @return a clone with this matrix + the given number
     */
    INDArray add(Number n);

    /**
     * In place scalar addition
     *
     * @param n
     * @return
     */
    INDArray addi(Number n);


    /**
     * Reverse division (number / ndarray)
     *
     * @param n      the number to divide by
     * @param result
     * @return
     */
    INDArray rdiv(Number n, INDArray result);


    /**
     * Reverse in place division
     *
     * @param n      the number to divide by  by
     * @param result the result ndarray
     * @return the result ndarray
     */
    INDArray rdivi(Number n, INDArray result);

    /**
     * Reverse subtraction
     *
     * @param n      the number to subtract by
     * @param result the result ndarray
     * @return
     */
    INDArray rsub(Number n, INDArray result);

    /**
     * Reverse in place subtraction
     *
     * @param n      the number to subtract by
     * @param result the result ndarray
     * @return the result ndarray
     */
    INDArray rsubi(Number n, INDArray result);


    /**
     * @param n
     * @param result
     * @return
     */
    INDArray div(Number n, INDArray result);

    /**
     * In place division of this ndarray
     *
     * @param n      the number to divide by
     * @param result the result ndarray
     * @return
     */
    INDArray divi(Number n, INDArray result);


    INDArray mul(Number n, INDArray result);


    /**
     * In place multiplication of this ndarray
     *
     * @param n      the number to divide by
     * @param result the result ndarray
     * @return
     */
    INDArray muli(Number n, INDArray result);


    INDArray sub(Number n, INDArray result);

    /**
     * In place subtraction of this ndarray
     *
     * @param n      the number to subtract by
     * @param result the result ndarray
     * @return the result ndarray
     */
    INDArray subi(Number n, INDArray result);

    INDArray add(Number n, INDArray result);

    /**
     * In place addition
     *
     * @param n      the number to add
     * @param result the result ndarray
     * @return the result ndarray
     */
    INDArray addi(Number n, INDArray result);


    /**
     * Returns a subset of this array based on the specified
     * indexes
     *
     * @param indexes the indexes in to the array
     * @return a view of the array with the specified indices
     */
    INDArray get(INDArrayIndex... indexes);




    /**
     * Get a list of specified columns
     *
     * @param columns
     * @return
     */
    INDArray getColumns(int...columns);

    /**
     * Get a list of rows
     *
     * @param rows
     * @return
     */
    INDArray getRows(int...rows);

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
    int linearIndex(int i);




    /**
     * Validate dimensions are equal
     *
     * @param other the other ndarray to compare
     */

    void checkDimensions(INDArray other);



    /**
     *
     * @param list
     */
    void sliceVectors(List<INDArray> list);


    /**
     * Assigns the given matrix (put) to the specified slice
     *
     * @param slice the slice to assign
     * @param put   the slice to applyTransformToDestination
     * @return this for chainability
     */
    INDArray putSlice(int slice, INDArray put);

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
     * Replicate and tile array to fill out to the given shape
     *
     * @param shape the new shape of this ndarray
     * @return the shape to fill out to
     */
    INDArray repmat(int...shape);


    /**
     * Repeat elements along a specified dimension.
     *
     * @param dimension the dimension to repeat
     * @param repeats the number of elements to repeat on each element
     * @return
     */
    INDArray repeat(int dimension,int...repeats);

    /**
     * Returns a flat array
     * with the elements repeated k times along each given dimension
     * @param repeats
     * @return
     */
    INDArray repeat(int...repeats);

    /**
     * Insert a row in to this array
     * Will throw an exception if this
     * ndarray is not a matrix
     *
     * @param row   the row insert into
     * @param toPut the row to insert
     * @return this
     */
    INDArray putRow(int row, INDArray toPut);

    /**
     * Insert a column in to this array
     * Will throw an exception if this
     * ndarray is not a matrix
     *
     * @param column the column to insert
     * @param toPut  the array to put
     * @return this
     */
    INDArray putColumn(int column, INDArray toPut);

    /**
     * Returns the element at the specified row/column
     * This will throw an exception if the
     *
     * @param row    the row of the element to return
     * @param column the row of the element to return
     * @return a scalar indarray of the element at this index
     */
    INDArray getScalar(int row, int column);

    /**
     * Returns the element at the specified index
     *
     * @param i the index of the element to return
     * @return a scalar ndarray of the element at this index
     */
    INDArray getScalar(int i);


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
    double squaredDistance(INDArray other);

    /**
     * Returns the (euclidean) distance.
     */
    double distance2(INDArray other);

    /**
     * Returns the (1-norm) distance.
     */
    double distance1(INDArray other);


    /**
     * Put the elements of the ndarray
     * in to the specified indices
     *
     * @param indices the indices to put the ndarray in to
     * @param element the ndarray to put
     * @return this ndarray
     */
    INDArray put(INDArrayIndex[] indices, INDArray element);

    /**
     * Put the elements of the ndarray
     * in to the specified indices
     *
     * @param indices the indices to put the ndarray in to
     * @param element the ndarray to put
     * @return this ndarray
     */
    INDArray put(INDArrayIndex[] indices, Number element);

    /**
     * Inserts the element at the specified index
     *
     * @param indices the indices to insert into
     * @param element a scalar ndarray
     * @return a scalar ndarray of the element at this index
     */
    INDArray put(int[] indices, INDArray element);


    /**
     * Inserts the element at the specified index
     *
     * @param i       the row insert into
     * @param j       the column to insert into
     * @param element a scalar ndarray
     * @return a scalar ndarray of the element at this index
     */
    INDArray put(int i, int j, INDArray element);


    /**
     * Inserts the element at the specified index
     *
     * @param i       the row insert into
     * @param j       the column to insert into
     * @param element a scalar ndarray
     * @return a scalar ndarray of the element at this index
     */
    INDArray put(int i, int j, Number element);


    /**
     * Inserts the element at the specified index
     *
     * @param i       the index insert into
     * @param element a scalar ndarray
     * @return a scalar ndarray of the element at this index
     */
    INDArray put(int i, INDArray element);


    /**
     * In place division of a column vector
     *
     * @param columnVector the column vector used for division
     * @return the result of the division 
     */
    INDArray diviColumnVector(INDArray columnVector);

    /**
     * Division of a column vector (copy)
     *
     * @param columnVector the column vector used for division
     * @return the result of the division 
     */
    INDArray divColumnVector(INDArray columnVector);

    /**
     * In place division of a row vector
     *
     * @param rowVector the row vector used for division
     * @return the result of the division 
     */
    INDArray diviRowVector(INDArray rowVector);

    /**
     * Division of a row vector (copy)
     *
     * @param rowVector the row vector used for division
     * @return the result of the division 
     */
    INDArray divRowVector(INDArray rowVector);


    /**
     * In place reverse divison of a column vector
     *
     * @param columnVector the column vector used for division
     * @return the result of the division 
     */
    INDArray rdiviColumnVector(INDArray columnVector);

    /**
     * Reverse division of a column vector (copy)
     *
     * @param columnVector the column vector used for division
     * @return the result of the division 
     */
    INDArray rdivColumnVector(INDArray columnVector);

    /**
     * In place reverse division of a column vector
     *
     * @param rowVector the row vector used for division
     * @return the result of the division 
     */
    INDArray rdiviRowVector(INDArray rowVector);

    /**
     * Reverse division of a column vector (copy)
     *
     * @param rowVector the row vector used for division
     * @return the result of the division 
     */
    INDArray rdivRowVector(INDArray rowVector);


    /**
     * In place multiplication of a column vector
     *
     * @param columnVector the column vector used for multiplication
     * @return the result of the multiplication
     */
    INDArray muliColumnVector(INDArray columnVector);

    /**
     * Multiplication of a column vector (copy)
     *
     * @param columnVector the column vector used for multiplication
     * @return the result of the multiplication
     */
    INDArray mulColumnVector(INDArray columnVector);

    /**
     * In place multiplication of a row vector
     *
     * @param rowVector the row vector used for multiplication
     * @return the result of the multiplication
     */
    INDArray muliRowVector(INDArray rowVector);

    /**
     * Multiplication of a row vector (copy)
     *
     * @param rowVector the row vector used for multiplication
     * @return the result of the multiplication
     */
    INDArray mulRowVector(INDArray rowVector);


    /**
     * In place reverse subtraction of a column vector
     *
     * @param columnVector the column vector to subtract
     * @return the result of the subtraction
     */
    INDArray rsubiColumnVector(INDArray columnVector);

    /**
     * Reverse subtraction of a column vector (copy)
     *
     * @param columnVector the column vector to subtract
     * @return the result of the subtraction
     */
    INDArray rsubColumnVector(INDArray columnVector);

    /**
     * In place reverse subtraction of a row vector
     *
     * @param rowVector the row vector to subtract
     * @return the result of the subtraction
     */
    INDArray rsubiRowVector(INDArray rowVector);

    /**
     * Reverse subtraction of a row vector (copy)
     *
     * @param rowVector the row vector to subtract
     * @return the result of the subtraction
     */
    INDArray rsubRowVector(INDArray rowVector);

    /**
     * In place subtraction of a column vector
     *
     * @param columnVector the column vector to subtract
     * @return the result of the subtraction
     */
    INDArray subiColumnVector(INDArray columnVector);

    /**
     * Subtraction of a column vector (copy)
     *
     * @param columnVector the column vector to subtract
     * @return the result of the subtraction
     */
    INDArray subColumnVector(INDArray columnVector);

    /**
     * In place subtraction of a row vector
     *
     * @param rowVector the row vector to subtract
     * @return the result of the subtraction
     */
    INDArray subiRowVector(INDArray rowVector);

    /**
     * Subtraction of a row vector (copy)
     *
     * @param rowVector the row vector to subtract
     * @return the result of the subtraction
     */
    INDArray subRowVector(INDArray rowVector);

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    INDArray addiColumnVector(INDArray columnVector);

    /**
     * Addition of a column vector (copy)
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    INDArray addColumnVector(INDArray columnVector);

    /**
     * In place addition of a row vector
     *
     * @param rowVector the row vector to add
     * @return the result of the addition
     */
    INDArray addiRowVector(INDArray rowVector);

    /**
     * Addition of a row vector (copy)
     *
     * @param rowVector the row vector to add
     * @return the result of the addition
     */
    INDArray addRowVector(INDArray rowVector);

    /**
     * Perform a copy matrix multiplication
     *
     * @param other the other matrix to perform matrix multiply with
     * @return the result of the matrix multiplication
     */
    INDArray mmul(INDArray other);


    /**
     * Perform an copy matrix multiplication
     *
     * @param other  the other matrix to perform matrix multiply with
     * @param result the result ndarray
     * @return the result of the matrix multiplication
     */
    INDArray mmul(INDArray other, INDArray result);


    /**
     * Copy (element wise) division of two NDArrays
     *
     * @param other the second ndarray to divide
     * @return the result of the divide
     */
    INDArray div(INDArray other);

    /**
     * copy (element wise) division of two NDArrays
     *
     * @param other  the second ndarray to divide
     * @param result the result ndarray
     * @return the result of the divide
     */
    INDArray div(INDArray other, INDArray result);


    /**
     * copy (element wise) multiplication of two NDArrays
     *
     * @param other the second ndarray to multiply
     * @return the result of the addition
     */
    INDArray mul(INDArray other);

    /**
     * copy (element wise) multiplication of two NDArrays
     *
     * @param other  the second ndarray to multiply
     * @param result the result ndarray
     * @return the result of the multiplication
     */
    INDArray mul(INDArray other, INDArray result);

    /**
     * copy subtraction of two NDArrays
     *
     * @param other the second ndarray to subtract
     * @return the result of the addition
     */
    INDArray sub(INDArray other);

    /**
     * copy subtraction of two NDArrays
     *
     * @param other  the second ndarray to subtract
     * @param result the result ndarray
     * @return the result of the subtraction
     */
    INDArray sub(INDArray other, INDArray result);

    /**
     * copy addition of two NDArrays
     *
     * @param other the second ndarray to add
     * @return the result of the addition
     */
    INDArray add(INDArray other);

    /**
     * copy addition of two NDArrays
     *
     * @param other  the second ndarray to add
     * @param result the result ndarray
     * @return the result of the addition
     */
    INDArray add(INDArray other, INDArray result);


    /**
     * Perform an inplace matrix multiplication
     *
     * @param other the other matrix to perform matrix multiply with
     * @return the result of the matrix multiplication
     */
    INDArray mmuli(INDArray other);


    /**
     * Perform an inplace matrix multiplication
     *
     * @param other  the other matrix to perform matrix multiply with
     * @param result the result ndarray
     * @return the result of the matrix multiplication
     */
    INDArray mmuli(INDArray other, INDArray result);


    /**
     * in place (element wise) division of two NDArrays
     *
     * @param other the second ndarray to divide
     * @return the result of the divide
     */
    INDArray divi(INDArray other);

    /**
     * in place (element wise) division of two NDArrays
     *
     * @param other  the second ndarray to divide
     * @param result the result ndarray
     * @return the result of the divide
     */
    INDArray divi(INDArray other, INDArray result);


    /**
     * in place (element wise) multiplication of two NDArrays
     *
     * @param other the second ndarray to multiply
     * @return the result of the addition
     */
    INDArray muli(INDArray other);

    /**
     * in place (element wise) multiplication of two NDArrays
     *
     * @param other  the second ndarray to multiply
     * @param result the result ndarray
     * @return the result of the multiplication
     */
    INDArray muli(INDArray other, INDArray result);

    /**
     * in place subtraction of two NDArrays
     *
     * @param other the second ndarray to subtract
     * @return the result of the addition
     */
    INDArray subi(INDArray other);

    /**
     * in place subtraction of two NDArrays
     *
     * @param other  the second ndarray to subtract
     * @param result the result ndarray
     * @return the result of the subtraction
     */
    INDArray subi(INDArray other, INDArray result);

    /**
     * in place addition of two NDArrays
     *
     * @param other the second ndarray to add
     * @return the result of the addition
     */
    INDArray addi(INDArray other);

    /**
     * in place addition of two NDArrays
     *
     * @param other  the second ndarray to add
     * @param result the result ndarray
     * @return the result of the addition
     */
    INDArray addi(INDArray other, INDArray result);


    /**
     * Returns the normmax along the specified dimension
     *
     * @param dimension the dimension to getScalar the norm1 along
     * @return the norm1 along the specified dimension
     */
    INDArray normmax(int...dimension);

    Number normmaxNumber();

    IComplexNumber normmaxComplex();

    /**
     * Returns the norm2 along the specified dimension
     *
     * @param dimension the dimension to getScalar the norm2 along
     * @return the norm2 along the specified dimension
     */
    INDArray norm2(int...dimension);

    Number norm2Number();

    IComplexNumber norm2Complex();

    /**
     * Returns the norm1 along the specified dimension
     *
     * @param dimension the dimension to getScalar the norm1 along
     * @return the norm1 along the specified dimension
     */
    INDArray norm1(int...dimension);

    Number norm1Number();

    IComplexNumber norm1Complex();

    /**
     * Standard deviation of an ndarray along a dimension
     *
     * @param dimension the dimension to getScalar the std along
     * @return the standard deviation along a particular dimension
     */
    INDArray std(int...dimension);

    Number stdNumber();

    IComplexNumber stdComplex();

    /**
     * Returns the product along a given dimension
     *
     * @param dimension the dimension to getScalar the product along
     * @return the product along the specified dimension
     */
    INDArray prod(int...dimension);

    Number prodNumber();

    IComplexNumber prodComplex();

    /**
     * Returns the overall mean of this ndarray
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    INDArray mean(int...dimension);

    Number meanNumber();

    IComplexNumber meanComplex();

    /**
     * Returns the overall variance of this ndarray
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    INDArray var(int...dimension);

    /**
     * Returns the overall variance of this ndarray
     *
     * @param biasCorrected boolean on whether to apply corrected bias
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    INDArray var(boolean biasCorrected, int...dimension);

    Number varNumber();

    IComplexNumber varComplex();

    /**
     * Returns the overall max of this ndarray
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    INDArray max(int...dimension);

    Number maxNumber();

    IComplexNumber maxComplex();

    /**
     * Returns the overall min of this ndarray
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    INDArray min(int...dimension);

    Number minNumber();

    IComplexNumber minComplex();

    /**
     * Returns the sum along the last dimension of this ndarray
     *
     * @param dimension the dimension to getScalar the sum along
     * @return the sum along the specified dimension of this ndarray
     */
    INDArray sum(int...dimension);

    /**
     * Sum the entire array
     * @return
     */
    Number sumNumber();

    /**
     * Sum the entire array
     * @return
     */
    IComplexNumber sumComplex();

    /**
     * stride setter
     * @param stride
     */
    void setStride(int...stride);

    /**
     * Shape setter
     * @param shape
     */
    void setShape(int...shape);

    /**
     * Set the ordering
     * @param order the ordering to set
     */
    void setOrder(char order);

    /**
     * Sub array based on the
     * pre calculated shape,strides, offsets
     * @param resolution the resolution to use
     * @return the sub array based on the calculations from the resolution
     */
    INDArray subArray(ShapeOffsetResolution resolution);

    /**
     * @param offsets
     * @param shape
     * @param stride
     * @return
     */
    INDArray subArray(int[] offsets, int[] shape, int[] stride);

    /**
     * Returns the elements at the the specified indices
     *
     * @param indices the indices to getScalar
     * @return the array with the specified elements
     */
    INDArray getScalar(int[] indices);

    /**
     *
     * @param indices
     * @return
     */
    int getInt(int... indices);

    /**
     *
     * @param indices
     * @return
     */
    double getDouble(int... indices);

    /**
     * Returns the elements at the the specified indices
     *
     * @param indices the indices to getScalar
     * @return the array with the specified elements
     */
    float getFloat(int[] indices);


    /**
     *
     * @param i
     * @return
     */
    double getDouble(int i);

    /**
     *
     * @param i
     * @param j
     * @return
     */
    double getDouble(int i, int j);

    /**
     * Return the item at the linear index i
     *
     * @param i the index of the item to getScalar
     * @return the item at index j
     */
    float getFloat(int i);

    /**
     * Return the item at row i column j
     * Note that this is the same as calling getScalar(new int[]{i,j}
     *
     * @param i the row to getScalar
     * @param j the column to getScalar
     * @return the item at row i column j
     */
    float getFloat(int i, int j);


    /**
     * Return a copy of this ndarray
     *
     * @return a copy of this ndarray
     */
    INDArray dup();

    /**Return a copy of this ndarray, where the returned ndarray has the specified order
     * @param order Order of the NDArray. 'f' or 'c'
     * @return copy of ndarray with specified order
     */
    INDArray dup(char order);

    /**
     * Returns a flattened version (row vector) of this ndarray
     *
     * @return a flattened version (row vector) of this ndarray
     */
    INDArray ravel();


    /**
     * Returns a flattened version (row vector) of this ndarray
     *
     * @return a flattened version (row vector) of this ndarray
     */
    INDArray ravel(char order);


    /**
     *
     * @param data
     */
    void setData(DataBuffer data);

    /**
     * Returns the number of slices in this ndarray
     *
     * @return the number of slices in this ndarray
     */
    int slices();

    /**
     *
     * @return
     */
    int getTrailingOnes();

    /**
     *
     * @return
     */
    int getLeadingOnes();

    /**
     * Returns the specified slice of this ndarray
     *
     * @param i         the index of the slice to return
     * @param dimension the dimension to return the slice for
     * @return the specified slice of this ndarray
     */
    INDArray slice(int i, int dimension);


    /**
     * Returns the specified slice of this ndarray
     *
     * @param i the index of the slice to return
     * @return the specified slice of this ndarray
     */
    INDArray slice(int i);


    /**
     * Returns the start of where the ndarray is
     * for the underlying data
     *
     * @return the starting offset
     */
    int offset();


    /**
     * Returns the start of where the ndarray is for the original data buffer
     * @return
     */
    int originalOffset();


    /**
     * Reshapes the ndarray (can't change the length of the ndarray)
     *
     * @param newShape the new shape of the ndarray
     * @return the reshaped ndarray
     */
    INDArray reshape(char order,int... newShape);


    /**
     * Reshapes the ndarray (can't change the length of the ndarray)
     *
     * @param rows    the rows of the matrix
     * @param columns the columns of the matrix
     * @return the reshaped ndarray
     */
    INDArray reshape(char order,int rows, int columns);


    /**
     * Reshapes the ndarray (can't change the length of the ndarray)
     *
     * @param newShape the new shape of the ndarray
     * @return the reshaped ndarray
     */
    INDArray reshape(int... newShape);


    /**
     * Reshapes the ndarray (can't change the length of the ndarray)
     *
     * @param rows    the rows of the matrix
     * @param columns the columns of the matrix
     * @return the reshaped ndarray
     */
    INDArray reshape(int rows, int columns);

    /**
     * Flip the rows and columns of a matrix
     *
     * @return the flipped rows and columns of a matrix
     */
    INDArray transpose();


    /**
     * Flip the rows and columns of a matrix
     *
     * @return the flipped rows and columns of a matrix
     */
    INDArray transposei();

    /**
     * Mainly here for people coming from numpy.
     * This is equivalent to a call to permute
     *
     * @param dimension the dimension to swap
     * @param with      the one to swap it with
     * @return the swapped axes view
     */
    INDArray swapAxes(int dimension, int with);

    /**
     * See: http://www.mathworks.com/help/matlab/ref/permute.html
     *
     * @param rearrange the dimensions to swap to
     * @return the newly permuted array
     */
    INDArray permute(int... rearrange);

    /**
     * Dimshuffle: an extension of permute that adds the ability
     * to broadcast various dimensions.
     * This will only accept integers and xs.
     * <p/>
     * An x indicates a dimension should be broadcasted rather than permuted.
     *
     * Examples originally from the theano docs:
     * http://deeplearning.net/software/theano/library/tensor/basic.html
     *
     *  Returns a view of this tensor with permuted dimensions. Typically the pattern will include the integers 0, 1, ... ndim-1, and any number of ‘x’ characters in dimensions where this tensor should be broadcasted.

     A few examples of patterns and their effect:

     (‘x’) -> make a 0d (scalar) into a 1d vector
     (0, 1) -> identity for 2d vectors
     (1, 0) -> inverts the first and second dimensions
     (‘x’, 0) -> make a row out of a 1d vector (N to 1xN)
     (0, ‘x’) -> make a column out of a 1d vector (N to Nx1)
     (2, 0, 1) -> AxBxC to CxAxB
     (0, ‘x’, 1) -> AxB to Ax1xB
     (1, ‘x’, 0) -> AxB to Bx1xA
     (1,) -> This remove dimensions 0. It must be a broadcastable dimension (1xA to A)

     * @param rearrange     the dimensions to swap to
     * @param newOrder      the new order (think permute)
     * @param broadCastable (whether the dimension is broadcastable) (must be same length as new order)
     * @return the newly permuted array
     */
    INDArray dimShuffle(Object[] rearrange, int[] newOrder, boolean[] broadCastable);



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

    /**
     * Return the ordering (fortran or c)
     * of this ndarray
     * @return the ordering of this ndarray
     */
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
    DataBuffer data();


    /**
     *
     * @param n
     * @return
     */
    IComplexNDArray rdiv(IComplexNumber n);

    /**
     *
     * @param n
     * @return
     */
    IComplexNDArray rdivi(IComplexNumber n);

    /**
     *
     * @param n
     * @return
     */
    IComplexNDArray rsub(IComplexNumber n);

    /**
     *
     * @param n
     * @return
     */
    IComplexNDArray rsubi(IComplexNumber n);

    /**
     *
     * @param n
     * @return
     */
    IComplexNDArray div(IComplexNumber n);

    /**
     *
     * @param n
     * @return
     */
    IComplexNDArray divi(IComplexNumber n);

    /**
     *
     * @param n
     * @return
     */
    IComplexNDArray mul(IComplexNumber n);

    /**
     *
     * @param n
     * @return
     */
    IComplexNDArray muli(IComplexNumber n);

    /**
     *
     * @param n
     * @return
     */
    IComplexNDArray sub(IComplexNumber n);

    /**
     *
     * @param n
     * @return
     */
    IComplexNDArray subi(IComplexNumber n);

    /**
     *
     * @param n
     * @return
     */
    IComplexNDArray add(IComplexNumber n);

    /**
     *
     * @param n
     * @return
     */
    IComplexNDArray addi(IComplexNumber n);

    /**
     *
     * @param n
     * @param result
     * @return
     */
    IComplexNDArray rdiv(IComplexNumber n, IComplexNDArray result);

    /**
     *
     * @param n
     * @param result
     * @return
     */
    IComplexNDArray rdivi(IComplexNumber n, IComplexNDArray result);

    /**
     *
     * @param n
     * @param result
     * @return
     */
    IComplexNDArray rsub(IComplexNumber n, IComplexNDArray result);

    /**
     *
     * @param n
     * @param result
     * @return
     */
    IComplexNDArray rsubi(IComplexNumber n, IComplexNDArray result);

    /**
     *
     * @param n
     * @param result
     * @return
     */
    IComplexNDArray div(IComplexNumber n, IComplexNDArray result);

    /**
     *
     * @param n
     * @param result
     * @return
     */
    IComplexNDArray divi(IComplexNumber n, IComplexNDArray result);

    /**
     *
     * @param n
     * @param result
     * @return
     */
    IComplexNDArray mul(IComplexNumber n, IComplexNDArray result);

    /**
     *
     * @param n
     * @param result
     * @return
     */
    IComplexNDArray muli(IComplexNumber n, IComplexNDArray result);

    /**
     *
     * @param n
     * @param result
     * @return
     */
    IComplexNDArray sub(IComplexNumber n, IComplexNDArray result);

    /**
     *
     * @param n
     * @param result
     * @return
     */
    IComplexNDArray subi(IComplexNumber n, IComplexNDArray result);

    /**
     *
     * @param n
     * @param result
     * @return
     */
    IComplexNDArray add(IComplexNumber n, IComplexNDArray result);

    /**
     *
     * @param n
     * @param result
     * @return
     */
    IComplexNDArray addi(IComplexNumber n, IComplexNDArray result);



}
