/*-
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

package org.nd4j.linalg.api.complex;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Condition;

/**
 * Complex numbers
 *
 * @author Adam Gibson
 */
public interface IComplexNDArray extends INDArray {
    /**
     * For blas operations, this is the offset / 2
     * when offset is > 0
     *
     * @return the blas offset
     */
    int blasOffset();

    /**
     * Returns a linear view reference of shape
     * length(ndarray),1
     *
     * @return the linear view of this ndarray
     */
    @Override
    IComplexNDArray linearViewColumnOrder();

    /**
     * Returns a linear view reference of shape
     * 1,length(ndarray)
     *
     * @return the linear view of this ndarray
     */
    @Override
    IComplexNDArray linearView();

    /**
     * Reshapes the ndarray (can't change the length of the ndarray)
     *
     * @param rows    the rows of the matrix
     * @param columns the columns of the matrix
     * @return the reshaped ndarray
     */
    @Override
    IComplexNDArray reshape(long rows, long columns);


    /**
     * Cumulative sum along a dimension
     *
     * @param dimension the dimension to perform cumulative sum along
     * @return the cumulative sum along the specified dimension
     */
    @Override
    IComplexNDArray cumsumi(int dimension);

    /**
     * Cumulative sum along a dimension (in place)
     *
     * @param dimension the dimension to perform cumulative sum along
     * @return the cumulative sum along the specified dimension
     */
    @Override
    IComplexNDArray cumsum(int dimension);

    /**
     * Get the vector along a particular dimension
     *
     * @param index     the index of the vector to getScalar
     * @param dimension the dimension to getScalar the vector from
     * @return the vector along a particular dimension
     */
    @Override
    IComplexNDArray vectorAlongDimension(int index, int dimension);


    /**
     * 1 in an ndarray when condition is fulfilled,
     * 0 otherwise(copying)(
     *
     * @param condition the condition to be fulfilled
     * @return
     */
    IComplexNDArray cond(Condition condition);

    /**
     * 1 in an ndarray when condition is fulfilled,
     * 0 otherwise(copying)(
     *
     * @param condition the condition to be fulfilled
     * @return
     */
    IComplexNDArray condi(Condition condition);

    @Override
    IComplexNDArray neq(INDArray other);

    @Override
    IComplexNDArray neqi(INDArray other);

    @Override
    IComplexNDArray neqi(Number other);

    @Override
    IComplexNDArray neq(Number other);

    /**
     * Assign all of the elements in the given
     * ndarray to this nedarray
     *
     * @param arr the elements to assign
     * @return this
     */
    IComplexNDArray assign(IComplexNDArray arr);


    /**
     *
     * @param indices
     * @param element
     * @return
     */
    IComplexNDArray put(INDArrayIndex[] indices, IComplexNumber element);

    /**
     *
     * @param indices
     * @param element
     * @return
     */
    IComplexNDArray put(INDArrayIndex[] indices, IComplexNDArray element);

    /**
     *
     * @param indices the indices to put the ndarray in to
     * @param element the ndarray to put
     * @return
     */
    IComplexNDArray put(INDArrayIndex[] indices, Number element);


    /**
     *
     * @param i
     * @param value
     * @return
     */
    IComplexNDArray putScalar(int i, IComplexNumber value);


    /**
     *
     * @param i     the index to insert into
     * @param value the value to insert
     * @return
     */
    IComplexNDArray putScalar(int i, double value);


    /**
     * Returns an ndarray with 1 if the element is epsilon equals
     *
     * @param other the number to compare
     * @return a copied ndarray with the given
     * binary conditions
     */

    IComplexNDArray eps(IComplexNumber other);

    /**
     * Returns an ndarray with 1 if the element is epsilon equals
     *
     * @param other the number to compare
     * @return a copied ndarray with the given
     * binary conditions
     */

    IComplexNDArray epsi(IComplexNumber other);


    /**
     * Returns an ndarray with 1 if the element is epsilon equals
     *
     * @param other the number to compare
     * @return a copied ndarray with the given
     * binary conditions
     */
    @Override
    IComplexNDArray eps(Number other);

    /**
     * Returns an ndarray with 1 if the element is epsilon equals
     *
     * @param other the number to compare
     * @return a copied ndarray with the given
     * binary conditions
     */
    @Override
    IComplexNDArray epsi(Number other);

    /**
     * epsilon equals than comparison:
     * If the given number is less than the
     * comparison number the item is 0 otherwise 1
     *
     * @param other the number to compare
     * @return
     */
    @Override
    IComplexNDArray eps(INDArray other);

    /**
     * In place epsilon equals than comparison:
     * If the given number is less than the
     * comparison number the item is 0 otherwise 1
     *
     * @param other the number to compare
     * @return
     */
    @Override
    IComplexNDArray epsi(INDArray other);

    @Override
    IComplexNDArray lt(Number other);

    @Override
    IComplexNDArray lti(Number other);

    @Override
    IComplexNDArray eq(Number other);

    @Override
    IComplexNDArray eqi(Number other);

    @Override
    IComplexNDArray gt(Number other);

    @Override
    IComplexNDArray gti(Number other);

    @Override
    IComplexNDArray lt(INDArray other);

    @Override
    IComplexNDArray lti(INDArray other);

    @Override
    IComplexNDArray eq(INDArray other);

    @Override
    IComplexNDArray eqi(INDArray other);

    @Override
    IComplexNDArray gt(INDArray other);

    @Override
    IComplexNDArray gti(INDArray other);

    INDArray putScalar(int[] i, IComplexNumber complexNumber);

    @Override
    IComplexNDArray neg();

    @Override
    IComplexNDArray negi();


    @Override
    IComplexNDArray addi(IComplexNumber n, IComplexNDArray result);

    @Override
    IComplexNDArray add(IComplexNumber n, IComplexNDArray result);

    @Override
    IComplexNDArray subi(IComplexNumber n, IComplexNDArray result);

    @Override
    IComplexNDArray sub(IComplexNumber n, IComplexNDArray result);

    @Override
    IComplexNDArray muli(IComplexNumber n, IComplexNDArray result);

    @Override
    IComplexNDArray mul(IComplexNumber n, IComplexNDArray result);

    @Override
    IComplexNDArray divi(IComplexNumber n, IComplexNDArray result);

    @Override
    IComplexNDArray div(IComplexNumber n, IComplexNDArray result);

    @Override
    IComplexNDArray rsubi(IComplexNumber n, IComplexNDArray result);

    @Override
    IComplexNDArray rsub(IComplexNumber n, IComplexNDArray result);

    @Override
    IComplexNDArray rdivi(IComplexNumber n, IComplexNDArray result);

    @Override
    IComplexNDArray rdiv(IComplexNumber n, IComplexNDArray result);

    /**
     *
     * @param n
     * @param result
     * @return
     */
    IComplexNDArray rdiv(IComplexNumber n, INDArray result);


    IComplexNDArray rdivi(IComplexNumber n, INDArray result);


    IComplexNDArray rsub(IComplexNumber n, INDArray result);


    IComplexNDArray rsubi(IComplexNumber n, INDArray result);


    @Override
    IComplexNDArray rdiviColumnVector(INDArray columnVector);

    @Override
    IComplexNDArray rdivColumnVector(INDArray columnVector);

    @Override
    IComplexNDArray rdiviRowVector(INDArray rowVector);

    @Override
    IComplexNDArray rdivRowVector(INDArray rowVector);

    @Override
    IComplexNDArray rsubiColumnVector(INDArray columnVector);

    @Override
    IComplexNDArray rsubColumnVector(INDArray columnVector);

    @Override
    IComplexNDArray rsubiRowVector(INDArray rowVector);

    @Override
    IComplexNDArray rsubRowVector(INDArray rowVector);

    IComplexNDArray div(IComplexNumber n, INDArray result);


    IComplexNDArray divi(IComplexNumber n, INDArray result);


    IComplexNDArray mul(IComplexNumber n, INDArray result);


    IComplexNDArray muli(IComplexNumber n, INDArray result);


    IComplexNDArray sub(IComplexNumber n, INDArray result);


    IComplexNDArray subi(IComplexNumber n, INDArray result);


    IComplexNDArray add(IComplexNumber n, INDArray result);


    IComplexNDArray addi(IComplexNumber n, INDArray result);


    IComplexNDArray rdiv(IComplexNumber n);


    IComplexNDArray rdivi(IComplexNumber n);


    IComplexNDArray rsub(IComplexNumber n);


    IComplexNDArray rsubi(IComplexNumber n);


    IComplexNDArray div(IComplexNumber n);


    IComplexNDArray divi(IComplexNumber n);


    IComplexNDArray mul(IComplexNumber n);


    IComplexNDArray muli(IComplexNumber n);


    IComplexNDArray sub(IComplexNumber n);


    IComplexNDArray subi(IComplexNumber n);


    IComplexNDArray add(IComplexNumber n);


    IComplexNDArray addi(IComplexNumber n);


    @Override
    IComplexNDArray rdiv(Number n, INDArray result);

    @Override
    IComplexNDArray rdivi(Number n, INDArray result);

    @Override
    IComplexNDArray rsub(Number n, INDArray result);

    @Override
    IComplexNDArray rsubi(Number n, INDArray result);

    @Override
    IComplexNDArray div(Number n, INDArray result);

    @Override
    IComplexNDArray divi(Number n, INDArray result);

    @Override
    IComplexNDArray mul(Number n, INDArray result);

    @Override
    IComplexNDArray muli(Number n, INDArray result);

    @Override
    IComplexNDArray sub(Number n, INDArray result);

    @Override
    IComplexNDArray subi(Number n, INDArray result);

    @Override
    IComplexNDArray add(Number n, INDArray result);

    @Override
    IComplexNDArray addi(Number n, INDArray result);

    @Override
    IComplexNDArray rdiv(Number n);

    @Override
    IComplexNDArray rdivi(Number n);

    @Override
    IComplexNDArray rsub(Number n);

    @Override
    IComplexNDArray rsubi(Number n);

    @Override
    IComplexNDArray div(Number n);

    @Override
    IComplexNDArray divi(Number n);

    @Override
    IComplexNDArray mul(Number n);

    @Override
    IComplexNDArray muli(Number n);

    @Override
    IComplexNDArray sub(Number n);

    @Override
    IComplexNDArray subi(Number n);

    @Override
    IComplexNDArray add(Number n);

    @Override
    IComplexNDArray addi(Number n);

    /**
     * Returns a subset of this array based on the specified
     * indexes
     *
     * @param indexes the indexes in to the array
     * @return a view of the array with the specified indices
     */
    @Override
    IComplexNDArray get(INDArrayIndex... indexes);

    @Override
    IComplexNDArray getColumns(int[] columns);

    @Override
    IComplexNDArray getRows(int[] rows);

    /**
     * Returns the overall min of this ndarray
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    @Override
    IComplexNDArray min(int... dimension);

    /**
     * Returns the overall max of this ndarray
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    @Override
    IComplexNDArray max(int... dimension);

    /**
     * Inserts the element at the specified index
     *
     * @param i       the row insert into
     * @param j       the column to insert into
     * @param element a scalar ndarray
     * @return a scalar ndarray of the element at this index
     */
    @Override
    IComplexNDArray put(int i, int j, INDArray element);

    /**
     * Inserts the element at the specified index
     *
     * @param indices the indices to insert into
     * @param element a scalar ndarray
     * @return a scalar ndarray of the element at this index
     */
    @Override
    IComplexNDArray put(int[] indices, INDArray element);

    /**
     * Assigns the given matrix (put) to the specified slice
     *
     * @param slice the slice to assign
     * @param put   the slice to applyTransformToDestination
     * @return this for chainability
     */
    @Override
    IComplexNDArray putSlice(int slice, INDArray put);

    /**
     * Get the imaginary component at the specified index
     *
     * @param i
     * @return
     */
    double getImag(int i);

    /**
     * Get the real component at the specified index
     *
     * @param i
     * @return
     */
    double getReal(int i);


    /**
     *
     * @param indices
     * @param value
     * @return
     */
    IComplexNDArray putReal(int[] indices, float value);

    /**
     *
     * @param indices
     * @param value
     * @return
     */
    IComplexNDArray putImag(int[] indices, float value);

    /**
     *
     * @param indices
     * @param value
     * @return
     */
    IComplexNDArray putReal(int[] indices, double value);

    /**
     *
     * @param indices
     * @param value
     * @return
     */
    IComplexNDArray putImag(int[] indices, double value);


    /**
     *
     * @param rowIndex
     * @param columnIndex
     * @param value
     * @return
     */
    IComplexNDArray putReal(int rowIndex, int columnIndex, float value);

    /**
     *
     * @param rowIndex
     * @param columnIndex
     * @param value
     * @return
     */
    IComplexNDArray putImag(int rowIndex, int columnIndex, float value);

    /**
     *
     * @param rowIndex
     * @param columnIndex
     * @param value
     * @return
     */
    IComplexNDArray putReal(int rowIndex, int columnIndex, double value);

    /**
     *
     * @param rowIndex
     * @param columnIndex
     * @param value
     * @return
     */
    IComplexNDArray putImag(int rowIndex, int columnIndex, double value);

    /**
     *
     * @param i
     * @param v
     * @return
     */
    IComplexNDArray putReal(int i, float v);

    /**
     *
     * @param i
     * @param v
     * @return
     */
    IComplexNDArray putImag(int i, float v);

    /**
     * Return all the real components in this ndarray
     * @return
     */
    INDArray real();

    /**
     * Return all of the imaginary components in this ndarray
     * @return
     */
    INDArray imag();



    /**
     * Put a scalar ndarray at the specified index
     *
     * @param i
     * @param element
     * @return
     */
    IComplexNDArray put(int i, IComplexNDArray element);

    /**
     * Fetch a particular number on a multi dimensional scale.
     *
     * @param indexes the indexes to getFromOrigin a number from
     * @return the number at the specified indices
     */
    @Override
    IComplexNDArray getScalar(int... indexes);

    /**
     * Validate dimensions are equal
     *
     * @param other the other ndarray to compare
     */
    @Override
    void checkDimensions(INDArray other);

    @Override
    IComplexNDArray reshape(char order, long... newShape);

    @Override
    IComplexNDArray reshape(char order, int rows, int columns);

    /**
     * Set the value of the ndarray to the specified value
     *
     * @param value the value to assign
     * @return the ndarray with the values
     */
    @Override
    IComplexNDArray assign(Number value);

    /**
     * Reverse division
     *
     * @param other the matrix to divide from
     * @return
     */
    @Override
    IComplexNDArray rdiv(INDArray other);

    /**
     * Reverse divsion (in place)
     *
     * @param other
     * @return
     */
    @Override
    IComplexNDArray rdivi(INDArray other);

    /**
     * Reverse division
     *
     * @param other  the matrix to subtract from
     * @param result the result ndarray
     * @return
     */
    @Override
    IComplexNDArray rdiv(INDArray other, INDArray result);

    /**
     * Reverse division (in-place)
     *
     * @param other  the other ndarray to subtract
     * @param result the result ndarray
     * @return the ndarray with the operation applied
     */
    @Override
    IComplexNDArray rdivi(INDArray other, INDArray result);

    /**
     * Reverse subtraction
     *
     * @param other  the matrix to subtract from
     * @param result the result ndarray
     * @return
     */
    @Override
    IComplexNDArray rsub(INDArray other, INDArray result);

    /**
     * @param other
     * @return
     */
    @Override
    IComplexNDArray rsub(INDArray other);

    /**
     * @param other
     * @return
     */
    @Override
    IComplexNDArray rsubi(INDArray other);

    /**
     * Reverse subtraction (in-place)
     *
     * @param other  the other ndarray to subtract
     * @param result the result ndarray
     * @return the ndarray with the operation applied
     */
    @Override
    IComplexNDArray rsubi(INDArray other, INDArray result);

    /**
     * a Hermitian matrix is a square matrix with complex entries that is equal to its own conjugate transpose
     *
     * @return the hermitian of this ndarray
     */
    IComplexNDArray hermitian();

    /**
     * Compute complex conj.
     */

    IComplexNDArray conj();


    /**
     * Compute complex conj (in-place).
     */

    IComplexNDArray conji();

    /**
     * Gets the real portion of this complex ndarray
     *
     * @return the real portion of this complex ndarray
     */
    INDArray getReal();

    /**
     * Replicate and tile array to fill out to the given shape
     *
     * @param shape the new shape of this ndarray
     * @return the shape to fill out to
     */
    @Override
    IComplexNDArray repmat(int... shape);

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
    IComplexNDArray putRow(long row, INDArray toPut);

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
    IComplexNDArray putColumn(int column, INDArray toPut);

    /**
     * Returns the element at the specified row/column
     * This will throw an exception if the
     *
     * @param row    the row of the element to return
     * @param column the row of the element to return
     * @return a scalar indarray of the element at this index
     */
    @Override
    IComplexNDArray getScalar(long row, long column);

    /**
     * Returns the element at the specified index
     *
     * @param i the index of the element to return
     * @return a scalar ndarray of the element at this index
     */
    @Override
    IComplexNDArray getScalar(long i);


    /**
     * Inserts the element at the specified index
     *
     * @param i       the index insert into
     * @param element a scalar ndarray
     * @return a scalar ndarray of the element at this index
     */
    @Override
    IComplexNDArray put(int i, INDArray element);


    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    IComplexNDArray diviColumnVector(INDArray columnVector);

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    IComplexNDArray divColumnVector(INDArray columnVector);

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    IComplexNDArray diviRowVector(INDArray rowVector);

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    IComplexNDArray divRowVector(INDArray rowVector);


    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    IComplexNDArray muliColumnVector(INDArray columnVector);

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    IComplexNDArray mulColumnVector(INDArray columnVector);

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    IComplexNDArray muliRowVector(INDArray rowVector);

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    IComplexNDArray mulRowVector(INDArray rowVector);


    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    IComplexNDArray subiColumnVector(INDArray columnVector);

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    IComplexNDArray subColumnVector(INDArray columnVector);

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    IComplexNDArray subiRowVector(INDArray rowVector);

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    IComplexNDArray subRowVector(INDArray rowVector);

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    IComplexNDArray addiColumnVector(INDArray columnVector);

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    IComplexNDArray addColumnVector(INDArray columnVector);

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    IComplexNDArray addiRowVector(INDArray rowVector);

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    IComplexNDArray addRowVector(INDArray rowVector);

    /**
     * Perform a copy matrix multiplication
     *
     * @param other the other matrix to perform matrix multiply with
     * @return the result of the matrix multiplication
     */
    @Override
    IComplexNDArray mmul(INDArray other);


    /**
     * Perform an copy matrix multiplication
     *
     * @param other  the other matrix to perform matrix multiply with
     * @param result the result ndarray
     * @return the result of the matrix multiplication
     */
    @Override
    IComplexNDArray mmul(INDArray other, INDArray result);


    /**
     * in place (element wise) division of two matrices
     *
     * @param other the second ndarray to divide
     * @return the result of the divide
     */
    @Override
    IComplexNDArray div(INDArray other);

    /**
     * copy (element wise) division of two matrices
     *
     * @param other  the second ndarray to divide
     * @param result the result ndarray
     * @return the result of the divide
     */
    @Override
    IComplexNDArray div(INDArray other, INDArray result);


    /**
     * copy (element wise) multiplication of two matrices
     *
     * @param other the second ndarray to multiply
     * @return the result of the addition
     */
    @Override
    IComplexNDArray mul(INDArray other);

    /**
     * copy (element wise) multiplication of two matrices
     *
     * @param other  the second ndarray to multiply
     * @param result the result ndarray
     * @return the result of the multiplication
     */
    @Override
    IComplexNDArray mul(INDArray other, INDArray result);

    /**
     * copy subtraction of two matrices
     *
     * @param other the second ndarray to subtract
     * @return the result of the addition
     */
    @Override
    IComplexNDArray sub(INDArray other);

    /**
     * copy subtraction of two matrices
     *
     * @param other  the second ndarray to subtract
     * @param result the result ndarray
     * @return the result of the subtraction
     */
    @Override
    IComplexNDArray sub(INDArray other, INDArray result);

    /**
     * copy addition of two matrices
     *
     * @param other the second ndarray to add
     * @return the result of the addition
     */
    @Override
    IComplexNDArray add(INDArray other);

    /**
     * copy addition of two matrices
     *
     * @param other  the second ndarray to add
     * @param result the result ndarray
     * @return the result of the addition
     */
    @Override
    IComplexNDArray add(INDArray other, INDArray result);


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
    @Override
    IComplexNDArray dimShuffle(Object[] rearrange, int[] newOrder, boolean[] broadCastable);

    /**
     * Perform an copy matrix multiplication
     *
     * @param other the other matrix to perform matrix multiply with
     * @return the result of the matrix multiplication
     */
    @Override
    IComplexNDArray mmuli(INDArray other);


    /**
     * Perform an copy matrix multiplication
     *
     * @param other  the other matrix to perform matrix multiply with
     * @param result the result ndarray
     * @return the result of the matrix multiplication
     */
    @Override
    IComplexNDArray mmuli(INDArray other, INDArray result);


    /**
     * in place (element wise) division of two matrices
     *
     * @param other the second ndarray to divide
     * @return the result of the divide
     */
    @Override
    IComplexNDArray divi(INDArray other);

    /**
     * in place (element wise) division of two matrices
     *
     * @param other  the second ndarray to divide
     * @param result the result ndarray
     * @return the result of the divide
     */
    @Override
    IComplexNDArray divi(INDArray other, INDArray result);


    /**
     * in place (element wise) multiplication of two matrices
     *
     * @param other the second ndarray to multiply
     * @return the result of the addition
     */
    @Override
    IComplexNDArray muli(INDArray other);

    /**
     * in place (element wise) multiplication of two matrices
     *
     * @param other  the second ndarray to multiply
     * @param result the result ndarray
     * @return the result of the multiplication
     */
    @Override
    IComplexNDArray muli(INDArray other, INDArray result);

    /**
     * in place subtraction of two matrices
     *
     * @param other the second ndarray to subtract
     * @return the result of the addition
     */
    IComplexNDArray subi(INDArray other);

    /**
     * in place subtraction of two matrices
     *
     * @param other  the second ndarray to subtract
     * @param result the result ndarray
     * @return the result of the subtraction
     */
    @Override
    IComplexNDArray subi(INDArray other, INDArray result);

    /**
     * in place addition of two matrices
     *
     * @param other the second ndarray to add
     * @return the result of the addition
     */
    @Override
    IComplexNDArray addi(INDArray other);

    /**
     * in place addition of two matrices
     *
     * @param other  the second ndarray to add
     * @param result the result ndarray
     * @return the result of the addition
     */
    @Override
    IComplexNDArray addi(INDArray other, INDArray result);


    /**
     * Returns the normmax along the specified dimension
     *
     * @param dimension the dimension to getScalar the norm1 along
     * @return the norm1 along the specified dimension
     */
    @Override
    IComplexNDArray normmax(int... dimension);


    /**
     * Returns the norm2 along the specified dimension
     *
     * @param dimension the dimension to getScalar the norm2 along
     * @return the norm2 along the specified dimension
     */
    @Override
    IComplexNDArray norm2(int... dimension);


    /**
     * Returns the norm1 along the specified dimension
     *
     * @param dimension the dimension to getScalar the norm1 along
     * @return the norm1 along the specified dimension
     */
    @Override
    IComplexNDArray norm1(int... dimension);


    /**
     * Returns the product along a given dimension
     *
     * @param dimension the dimension to getScalar the product along
     * @return the product along the specified dimension
     */
    @Override
    IComplexNDArray prod(int... dimension);


    /**
     * Returns the overall mean of this ndarray
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    @Override
    IComplexNDArray mean(int... dimension);

    /**
     * Returns the sum along the last dimension of this ndarray
     *
     * @param dimension the dimension to getScalar the sum along
     * @return the sum along the specified dimension of this ndarray
     */
    @Override
    IComplexNDArray sum(int... dimension);


    @Override
    IComplexNDArray var(int... dimension);

    @Override
    IComplexNDArray std(int... dimension);

    /**
     *
     * @param i
     * @return
     */
    IComplexNumber getComplex(int i);

    /**
     *
     * @param i
     * @param j
     * @return
     */
    IComplexNumber getComplex(int i, int j);

    /**
     *
     * @param i
     * @param result
     * @return
     */
    IComplexNumber getComplex(int i, IComplexNumber result);

    /**
     *
     * @param i
     * @param j
     * @param result
     * @return
     */
    IComplexNumber getComplex(int i, int j, IComplexNumber result);

    /**
     * Return a copy of this ndarray
     *
     * @return a copy of this ndarray
     */
    @Override
    IComplexNDArray dup();


    /**
     * Returns a flattened version (row vector) of this ndarray
     *
     * @return a flattened version (row vector) of this ndarray
     */
    @Override
    IComplexNDArray ravel();


    /**
     * Returns the specified slice of this ndarray
     *
     * @param i         the index of the slice to return
     * @param dimension the dimension to return the slice for
     * @return the specified slice of this ndarray
     */
    IComplexNDArray slice(int i, int dimension);


    /**
     * Returns the specified slice of this ndarray
     *
     * @param i the index of the slice to return
     * @return the specified slice of this ndarray
     */
    IComplexNDArray slice(int i);


    /**
     * Reshapes the ndarray (can't change the length of the ndarray)
     *
     * @param newShape the new shape of the ndarray
     * @return the reshaped ndarray
     */
    @Override
    IComplexNDArray reshape(long... newShape);

    /**
     * Flip the rows and columns of a matrix
     *
     * @return the flipped rows and columns of a matrix
     */
    @Override
    IComplexNDArray transpose();

    @Override
    IComplexNDArray transposei();

    IComplexNDArray put(int[] indexes, float value);

    IComplexNDArray put(int[] indexes, double value);

    IComplexNDArray putSlice(int slice, IComplexNDArray put);

    /**
     * Get the complex number at the given indices
     * @param indices the indices to
     *                get the complex number at
     * @return the complex number at the given indices
     */
    IComplexNumber getComplex(int... indices);



    /**
     * Mainly here for people coming from numpy.
     * This is equivalent to a call to permute
     *
     * @param dimension the dimension to swap
     * @param with      the one to swap it with
     * @return the swapped axes view
     */
    @Override
    IComplexNDArray swapAxes(int dimension, int with);

    /**
     * See: http://www.mathworks.com/help/matlab/ref/permute.html
     *
     * @param rearrange the dimensions to swap to
     * @return the newly permuted array
     */
    @Override
    IComplexNDArray permute(int... rearrange);


    /**
     * Returns the specified column.
     * Throws an exception if its not a matrix
     *
     * @param i the column to getScalar
     * @return the specified column
     */
    @Override
    IComplexNDArray getColumn(long i);

    /**
     * Returns the specified row.
     * Throws an exception if its not a matrix
     *
     * @param i the row to getScalar
     * @return the specified row
     */
    @Override
    IComplexNDArray getRow(long i);


    /**
     * Broadcasts this ndarray to be the specified shape
     *
     * @param shape the new shape of this ndarray
     * @return the broadcasted ndarray
     */
    @Override
    IComplexNDArray broadcast(long[] shape);

    IComplexNDArray putScalar(int j, int i, IComplexNumber conji);


    IComplexNDArray neqi(IComplexNumber other);


    IComplexNDArray neq(IComplexNumber other);


    IComplexNDArray lt(IComplexNumber other);


    IComplexNDArray lti(IComplexNumber other);


    IComplexNDArray eq(IComplexNumber other);


    IComplexNDArray eqi(IComplexNumber other);


    IComplexNDArray gt(IComplexNumber other);


    IComplexNDArray gti(IComplexNumber other);


    void assign(IComplexNumber aDouble);

    IComplexNDArray put(int i, int j, IComplexNumber complex);
}
