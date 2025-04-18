/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.ndarray;

import com.google.flatbuffers.FlatBufferBuilder;
import lombok.NonNull;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.exception.Nd4jNoSuchWorkspaceException;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Condition;

import java.io.Serializable;
import java.nio.LongBuffer;
import java.util.List;

import org.nd4j.linalg.profiler.data.array.eventlog.Nd4jEventLog;
import org.nd4j.linalg.profiler.data.array.event.NDArrayEvent;
import org.nd4j.linalg.string.NDArrayStrings;
import org.nd4j.nativeblas.OpaqueNDArray;

public interface INDArray extends Serializable, AutoCloseable {


    /**
     * Create an {@link OpaqueNDArray}
     * and cache the result.
     * This created array will
     * be destroyed when {@link INDArray#close()}
     * is called.
     * @return
     */
    OpaqueNDArray getOrCreateOpaqueNDArray();


    /**
     * The underlying event log for all ndarrays.
     * @return
     */
    Nd4jEventLog log();

    /**
     * Adds an ndarray event to the log.
     * @param event
     */
    void addEvent(NDArrayEvent event);

    List<NDArrayEvent> writeEvents();
    /**
     * When an INDArray is created and {@link Environment#isFuncTracePrintAllocate()}
     * or {@link Environment#isFuncTracePrintJavaOnly()} is true, the stack trace will be recorded.
     * is true, the stack trace will be recorded and saved as a string on the array object.
     *
     * @return
     */
    StackTraceElement[] allocationTrace();

    /**
     * Returns the shape information debugging information
     * @return the shape information.
     */
    String shapeInfoToString();

    /**
     * Shape info
     * @return Shape info
     */
    DataBuffer shapeInfoDataBuffer();

    /**
     * Shape info
     * @return Shape info
     */
    LongBuffer shapeInfo();

    /**
     * Check if this array is a view or not.
     * @return true if array is a view.
     */
    boolean isView();

    /**
     * Check if this array is sparse
     * @return true if this array is sparse.
     */
    boolean isSparse();

    /**
     * Check if this array is compressed.
     * @return true if this array is compressed.
     */
    boolean isCompressed();

    /**
     * This method marks INDArray instance as compressed
     * PLEASE NOTE: Do not use this method unless you 100% have to
     *
     * @param reallyCompressed new value for compressed.
     */
    void markAsCompressed(boolean reallyCompressed);

    /**
     * Returns the rank of the ndarray (the number of dimensions).
     *
     * @return the rank for the ndarray.
     */
    int rank();

    /**
     * Calculate the stride along a particular dimension
     * @param dimension the dimension to get the stride for
     * @return the stride for a particular dimension
     */
    int stride(int dimension);

    /**
     * Element wise stride
     * @return the element wise stride
     */
    int elementWiseStride();

    // TODO: Unused untested method.
    /**
     * Get a double at the given linear offset unsafe, without checks.
     * @param offset the offset to get at
     * @return double value at offset
     */
    double getDoubleUnsafe(long offset);

    /**
     * Get string value at given index.
     * @param index index to retreive
     * @return string value at index.
     */
    String getString(long index);

    // TODO: Unused untested method.
    /**
     * Insert a scalar at the given linear offset
     * @param offset the offset to insert at
     * @param value the value to insert
     * @return this
     */
    INDArray putScalarUnsafe(long offset, double value);

    /**
     * Returns the number of possible vectors for a given dimension
     *
     * @param dimension the dimension to calculate the number of vectors for
     * @return the number of possible vectors along a dimension
     */
    long vectorsAlongDimension(int dimension);

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
    long tensorsAlongDimension(long... dimension);

    /**
     * Get the vector along a particular dimension
     *
     * @param index     the index of the vector to getScalar
     * @param dimension the dimension to getScalar the vector from
     * @return the vector along a particular dimension
     */
    INDArray tensorAlongDimension(long index, long... dimension);




    /**
     * Returns the cumulative sum along a dimension. In-place method.
     *
     * @param dimension the dimension to perform cumulative sum along.
     * @return this object.
     */
    INDArray cumsumi(int dimension);

    /**
     * Returns the cumulative sum along a dimension.
     *
     * @param dimension the dimension to perform cumulative sum along.
     * @return the cumulative sum along the specified dimension
     */
    INDArray cumsum(int dimension);

    /**
     * Assign all of the elements in the given ndarray to this ndarray
     *
     * @param arr the elements to assign
     * @return this
     */
    INDArray assign(INDArray arr);

    // TODO: Unused untested method.
    /**
     * Assign all elements from given ndarray that are matching given condition,
     * ndarray to this ndarray
     *
     * @param arr the elements to assign
     * @return this
     */
    INDArray assignIf(INDArray arr, Condition condition);


    /**
     * Replaces all elements in this ndarray that are matching give condition, with corresponding elements from given array
     *
     * @param arr       Source array
     * @param condition Condition to apply
     * @return New array with values conditionally replaced
     */
    INDArray replaceWhere(INDArray arr, Condition condition);


    /**
     * Insert the number linearly in to the ndarray
     *
     * @param i     the index to insert into
     * @param value the value to insert
     * @return this
     */
    INDArray putScalar(long i, double value);

    /**
     * Insert a scalar float at the specified index
     *
     * @param i     The index to insert into
     * @param value Value to insert
     * @return This array
     */
    INDArray putScalar(long i, float value);

    INDArray putScalar(long i, boolean b);


    /**
     * Insert a scalar int at the specified index
     *
     * @param i     The index to insert into
     * @param value Value to insert
     * @return This array
     */
    INDArray putScalar(long i, int value);

    /**
     * Insert the item at the specified indices
     *
     * @param i     the indices to insert at
     * @param value the number to insert
     * @return this
     */
    INDArray putScalar(int[] i, double value);

    /**
     * See {@link #putScalar(int[], double)}
     */
    INDArray putScalar(long[] i, double value);

    /**
     * See {@link #putScalar(int[], double)}
     */
    INDArray putScalar(long[] i, float value);

    /**
     * See {@link #putScalar(int[], double)}
     */
    INDArray putScalar(long[] i, int value);

    /**
     * Insert the value at the specified indices, in a 2d (rank 2) NDArray<br>
     * Equivalent to {@link #putScalar(int[], double)} but avoids int[] creation
     * @param row      Row (dimension 0) index
     * @param col      Column (dimension 1) index
     * @param value    Value to put
     * @return         This INDArray
     */
    INDArray putScalar(long row, long col, double value);

    /**
     * Insert the value at the specified indices, in a 3d (rank 3) NDArray<br>
     * Equivalent to {@link #putScalar(int[], double)} but avoids int[] creation
     * @param dim0     Dimension 0 index
     * @param dim1     Dimension 1 index
     * @param dim2     Dimension 2 index
     * @param value    Value to put
     * @return         This INDArray
     */
    INDArray putScalar(long dim0, long dim1, long dim2, double value);

    /**
     * Insert the value at the specified indices, in a 4d (rank 4) NDArray<br>
     * Equivalent to {@link #putScalar(int[], double)} but avoids int[] creation
     * @param dim0     Dimension 0 index
     * @param dim1     Dimension 1 index
     * @param dim2     Dimension 2 index
     * @param dim3     Dimension 3 index
     * @param value    Value to put
     * @return         This INDArray
     */
    INDArray putScalar(long dim0, long dim1, long dim2, long dim3, double value);

    /**
     * Returns the binary ndarray for "Less" comparison.
     *
     * @param other the number to compare.
     * @return the binary ndarray for "Less" comparison.
     */
    INDArray lt(Number other);

    /**
     * Put the specified float value at the specified indices in this array
     *
     * @param indexes Indices to place the value
     * @param value   Value to insert
     * @return This array
     */
    INDArray putScalar(int[] indexes, float value);

    /**
     * Put the specified integer value at the specified indices in this array
     *
     * @param indexes Indices to place the value
     * @param value   Value to insert
     * @return This array
     */
    INDArray putScalar(int[] indexes, int value);

    /**
     * Returns the binary ndarray for "Epsilon equals" comparison.
     *
     * @param other the number to compare.
     * @return the binary ndarray for "Epsilon equals" comparison.
     */
    INDArray eps(Number other);

    /**
     * Returns the binary ndarray for "Equals" comparison.
     *
     * @param other the number to compare.
     * @return the binary ndarray for "Equals" comparison.
     */
    INDArray eq(Number other);

    /**
     * Returns the binary ndarray for "Greater" comparison.
     *
     * @param other the number to compare.
     * @return the binary ndarray for "Greater" comparison.
     */
    INDArray gt(Number other);

    /**
     * Returns binary ndarray for "Greter or equals" comparison.
     *
     * @param other the number to compare.
     * @return binary ndarray for "Greter or equals" comparison.
     */
    INDArray gte(Number other);

    /**
     * Returns the binary ndarray for "Less or equals" comparison.
     *
     * @param other the number to compare.
     * @return the binary ndarray for "Less or equals" comparison.
     */
    INDArray lte(Number other);

    /**
     * Returns the binary ndarray for "Less" comparison.
     *
     * @param other the ndarray to compare.
     * @return the binary ndarray for "Less" comparison.
     */
    INDArray lt(INDArray other);

    /**
     * Returns the binary ndarray for "Epsilon equals" comparison.
     *
     * @param other the ndarray to compare.
     * @return the binary ndarray for "Epsilon equals" comparison.
     */
    INDArray eps(INDArray other);

    /**
     * Returns the binary ndarray for "Not equals" comparison.
     *
     * @param other the number to compare.
     * @return the binary ndarray for "Not equals" comparison.
     */
    INDArray neq(Number other);

    /**
     * Returns the binary ndarray for "Not equals" comparison.
     *
     * @param other the ndarray to compare.
     * @return the binary ndarray for "Not equals" comparison.
     */
    INDArray neq(INDArray other);

    /**
     * Returns the binary ndarray for "Equals" comparison.
     *
     * @param other the ndarray to compare.
     * @return the binary ndarray for "Equals" comparison.
     */
    INDArray eq(INDArray other);

    /**
     * Returns the binary ndarray for "Greater Than" comparison.
     *
     * @param other the ndarray to compare.
     * @return the binary ndarray for "Greater Than" comparison.
     */
    INDArray gt(INDArray other);

    /**
     * Returns the binary NDArray with value true where this array's entries are infinite, or false where they
     * are not infinite
     */
    INDArray isInfinite();

    /**
     * Returns the binary NDArray with value true where this array's entries are NaN, or false where they
     * are not infinite
     */
    INDArray isNaN();

    /**
     * Returns the ndarray negative (cloned)
     *
     * @return Array copy with all values negated
     */
    INDArray neg();

    /**
     * In place setting of the negative version of this ndarray
     *
     * @return This array with all values negated
     */
    INDArray negi();

    /**
     * Reverse division with a scalar - i.e., (n / thisArrayValues)
     *
     * @param n Value to use for reverse division
     * @return  Copy of array after applying reverse division
     */
    INDArray rdiv(Number n);

    /**
     * In place reverse division - i.e., (n / thisArrayValues)
     *
     * @param n Value to use for reverse division
     * @return This array after applying reverse division
     */
    INDArray rdivi(Number n);

    /**
     * Reverse subtraction with duplicates - i.e., (n - thisArrayValues)
     *
     * @param n Value to use for reverse subtraction
     * @return Copy of array after reverse subtraction
     */
    INDArray rsub(Number n);

    /**
     * Reverse subtraction in place - i.e., (n - thisArrayValues)
     *
     * @param n Value to use for reverse subtraction
     * @return This array after reverse subtraction
     */
    INDArray rsubi(Number n);

    /**
     * Division by a number
     *
     * @param n Number to divide values by
     * @return Copy of array after division
     */
    INDArray div(Number n);

    /**
     * In place scalar division
     *
     * @param n Number to divide values by
     * @return This array, after applying division operation
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
     * @param n The number to multiply by
     * @return This array, after applying scaler multiplication
     */
    INDArray muli(Number n);

    /**
     * Scalar subtraction (copied)
     *
     * @param n the number to subtract by
     * @return Copy of this array after applying subtraction operation
     */
    INDArray sub(Number n);

    /**
     * In place scalar subtraction
     *
     * @param n Number to subtract
     * @return This array, after applying subtraction operation
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
     * @param n Number to add
     * @return This array, after adding value
     */
    INDArray addi(Number n);

    /**
     * Reverse division (number / ndarray)
     *
     * @param n      the number to divide by
     * @param result Array to place the result in. Must match shape of this array
     * @return Result array
     */
    INDArray rdiv(Number n, INDArray result);

    /**
     * Reverse in place division
     *
     * @param n      the number to divide by
     * @param result the result ndarray
     * @return the result ndarray
     */
    INDArray rdivi(Number n, INDArray result);

    /**
     * Reverse subtraction
     *
     * @param n      the number to subtract by
     * @param result the result ndarray
     * @return the result ndarray
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
     * Division if ndarray by number
     *
     * @param n      the number to divide by
     * @param result the result ndarray
     * @return the result ndarray
     */
    INDArray div(Number n, INDArray result);

    /**
     * In place division of this ndarray
     *
     * @param n      the number to divide by
     * @param result the result ndarray
     * @return the result ndarray
     */
    INDArray divi(Number n, INDArray result);

    /**
     * Multiplication of ndarray.
     *
     * @param n the number to multiply by
     * @param result the result ndarray
     * @return the result ndarray
     */
    INDArray mul(Number n, INDArray result);

    /**
     * In place multiplication of this ndarray
     *
     * @param n      the number to divide by
     * @param result the result ndarray
     * @return the result ndarray
     */
    INDArray muli(Number n, INDArray result);

    /**
     * Subtraction of this ndarray
     *
     * @param n      the number to subtract by
     * @param result the result ndarray
     * @return the result ndarray
     */
    INDArray sub(Number n, INDArray result);

    /**
     * In place subtraction of this ndarray
     *
     * @param n      the number to subtract by
     * @param result the result ndarray
     * @return the result ndarray
     */
    INDArray subi(Number n, INDArray result);

    /**
     * Addition of this ndarray.
     * @param n      the number to add
     * @param result the result ndarray
     * @return the result ndarray
     */
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
     * Returns a subset of this array based on the specified indexes
     *
     * @param indexes the indexes in to the array
     * @return a view of the array with the specified indices
     */
    INDArray get(INDArrayIndex... indexes);




    /**
     * Return a mask on whether each element matches the given condition
     * @param comp
     * @param condition
     * @return
     */
    INDArray match(INDArray comp,Condition condition);



    /**
     * Returns a mask
     * @param comp
     * @param condition
     * @return
     */
    INDArray match(Number comp,Condition condition);

    /**
     * Boolean indexing:
     * Return the element if it fulfills the condition in
     * result array
     * @param comp the comparison array
     * @param condition the condition to apply
     * @return the array fulfilling the criteria
     */
    INDArray getWhere(INDArray comp,Condition condition);

    /**
     * Boolean indexing:
     * Return the element if it fulfills the condition in
     * result array
     * @param comp the comparison array
     * @param condition the condition to apply
     * @return the array fulfilling the criteria
     */
    INDArray getWhere(Number comp,Condition condition);

    //TODO: unused / untested method. (only used to forward calls from putWhere(Number,INDArray ,Condition).
    /**
     * Assign the element according to the comparison array
     * @param comp the comparison array
     * @param put the elements to put
     * @param condition the condition for masking on
     * @return a copy of this array with the conditional assignments.
     */
    INDArray putWhere(INDArray comp,INDArray put,Condition condition);

    //TODO: unused / untested method.
    /**
     * Assign the element according to the comparison array
     * @param comp the comparison array
     * @param put the elements to put
     * @param condition the condition for masking on
     * @return a copy of this array with the conditional assignments.
     */
    INDArray putWhere(Number comp,INDArray put,Condition condition);

    //TODO: unused / untested method. (only used to forward calls from  other putWhereWithMask implementations.
    /**
     * Use a pre computed mask for assigning arrays
     * @param mask the mask to use
     * @param put the array to put
     * @return a copy of this array with the conditional assignments.
     */
    INDArray putWhereWithMask(INDArray mask,INDArray put);

    //TODO: unused / untested method.
    /**
     * Use a pre computed mask for assigning arrays
     * @param mask the mask to use
     * @param put the array to put
     * @return a copy of this array with the conditional assignments.
     */
    INDArray putWhereWithMask(INDArray mask,Number put);

    //TODO: unused / untested method.
    /**
     * Assign the element according to the comparison array
     * @param comp the comparison array
     * @param put the elements to put
     * @param condition the condition for masking on
     * @return a copy of this array with the conditional assignments.
     */
    INDArray putWhere(Number comp,Number put,Condition condition);

    /**
     * Get the elements from this ndarray based on the specified indices
     * @param indices an ndaray of the indices to get the elements for
     * @return the elements to get the array for
     */
    INDArray get(INDArray indices);

    /**
     * Get an INDArray comprised of the specified columns only. Copy operation.
     *
     * @param columns Columns to extract out of the current array
     * @return Array with only the specified columns
     */
    INDArray getColumns(int... columns);

    /**
     * Get an INDArray comprised of the specified rows only. Copy operation
     *
     * @param rows Rose to extract from this array
     * @return Array with only the specified rows
     */
    INDArray getRows(int... rows);

    /**
     * Reverse division, elements wise. i.e., other / this
     *
     * @param other the matrix to divide from
     * @return Copy of this array after performing element wise reverse division
     */
    INDArray rdiv(INDArray other);

    /**
     * Reverse divsion (in place). i.e., other / this
     *
     * @param other The matrix to divide from
     * @return This array after performing element wise reverse division
     */
    INDArray rdivi(INDArray other);

    //TODO: unused / untested method.
    /**
     * Reverse division
     *
     * @param other  the matrix to divide from
     * @param result the result ndarray
     * @return the result ndarray
     */
    INDArray rdiv(INDArray other, INDArray result);

    /**
     * Reverse division (in-place)
     *
     * @param other  the matrix to divide from
     * @param result the result ndarray
     * @return the ndarray with the operation applied
     */
    INDArray rdivi(INDArray other, INDArray result);

    /**
     * Reverse subtraction
     *
     * @param other  the matrix to subtract from
     * @param result the result ndarray
     * @return the result ndarray
     */
    INDArray rsub(INDArray other, INDArray result);

    /**
     * Element-wise reverse subtraction (copy op). i.e., other - this
     *
     * @param other Other array to use in reverse subtraction
     * @return Copy of this array, after applying reverse subtraction
     */
    INDArray rsub(INDArray other);

    /**
     * Element-wise reverse subtraction (in the place op) - i.e., other - this
     *
     * @param other Other way to use in reverse subtraction operation
     * @return This array, after applying reverse subtraction
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
     * Set all entries of the ndarray to the specified value
     *
     * @param value the value to assign
     * @return the ndarray with the values
     */
    INDArray assign(Number value);

    /**
     * Set all entries of the ndarray to the specified value
     *
     * @param value the value to assign
     * @return the ndarray with the values
     */
    INDArray assign(boolean value);

    //TODO: unused / untested method. only used recursively.
    /**
     * Flattens the array for linear indexing in list.
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
     * Returns a binary INDArray with value 'true' if the element matches the specified condition and 'false' otherwise
     *
     * @param condition Condition to apply
     * @return Copy of this array with values 0 (condition does not apply), or one (condition applies)
     */
    INDArray cond(Condition condition);

    /**
     * Replicate and tile array to fill out to the given shape
     * See:
     * https://github.com/numpy/numpy/blob/master/numpy/matlib.py#L310-L358
     * @param shape the new shape of this ndarray
     * @return the shape to fill out to
     */
    INDArray repmat(long... shape);

    @Deprecated
    INDArray repmat(int... shape);

    /**
     * Repeat elements along a specified dimension.
     *
     * @param dimension the dimension to repeat
     * @param repeats the number of elements to repeat on each element
     * @return Repeated array
     */
    INDArray repeat(int dimension, long... repeats);

    /**
     * Insert a row in to this array
     * Will throw an exception if this ndarray is not a matrix
     *
     * @param row   the row insert into
     * @param toPut the row to insert
     * @return this
     */
    INDArray putRow(long row, INDArray toPut);

    /**
     * Insert a column in to this array
     * Will throw an exception if this ndarray is not a matrix
     *
     * @param column the column to insert
     * @param toPut  the array to put
     * @return this
     */
    INDArray putColumn(int column, INDArray toPut);

    /**
     * Returns the element at the specified row/column
     *
     * @param row    the row of the element to return
     * @param column the row of the element to return
     * @return a scalar indarray of the element at this index
     */
    INDArray getScalar(long row, long column);

    /**
     * Returns the element at the specified index
     *
     * @param i the index of the element to return
     * @return a scalar ndarray of the element at this index
     */
    INDArray getScalar(long i);

    /**
     * Returns the square of the Euclidean distance.
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
     * Put element in to the indices denoted by
     * the indices ndarray.
     * In numpy this is equivalent to:
     * a[indices] = element
     *
     * @param indices the indices to put
     * @param element the element array to put
     * @return this array
     */
    INDArray put(INDArray indices,INDArray element);

    /**
     * Put the elements of the ndarray in to the specified indices
     *
     * @param indices the indices to put the ndarray in to
     * @param element the ndarray to put
     * @return this ndarray
     */
    INDArray put(INDArrayIndex[] indices, INDArray element);

    /**
     * Put the elements of the ndarray in to the specified indices
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

    //TODO: unused / untested method.
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

    //TODO: unused / untested method.
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
     * In place assignment of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    INDArray putiColumnVector(INDArray columnVector);

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
     * in place assignment of row vector, to each row of this array
     *
     * @param rowVector Row vector to put
     * @return This array, after assigning every road to the specified value
     */
    INDArray putiRowVector(INDArray rowVector);

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
    INDArray mmul(INDArray other, MMulTranspose mMulTranspose);

    /**
     * Perform a copy matrix multiplication
     *
     * @param other the other matrix to perform matrix multiply with
     * @return the result of the matrix multiplication
     */
    INDArray mmul(INDArray other);

    /**
     * Perform a copy matrix multiplication
     * @param other other the other matrix to perform matrix multiply with
     * @param resultOrder either C or F order for result array
     * @return the result of the matrix multiplication
     */
    INDArray mmul(INDArray other, char resultOrder);

    /**
     * Convert this ndarray to a 2d double matrix.
     * Note that THIS SHOULD NOT BE USED FOR SPEED.
     * This is mainly used for integrations with other libraries.
     * Due to nd4j's off  heap nature, moving data on heap is very expensive
     * and should not be used if possible.
     * @return a copy of this array as a 2d double array
     */
    double[][] toDoubleMatrix();

    /**
     * Convert this ndarray to a 1d double matrix.
     * Note that THIS SHOULD NOT BE USED FOR SPEED.
     * This is mainly used for integrations with other libraries.
     * Due to nd4j's off  heap nature, moving data on heap is very expensive
     * and should not be used if possible.
     * @return a copy of this array as a 1d double array
     */
    double[] toDoubleVector();

    /**
     * Convert this ndarray to a 1d float vector.
     * Note that THIS SHOULD NOT BE USED FOR SPEED.
     * This is mainly used for integrations with other libraries.
     * Due to nd4j's off  heap nature, moving data on heap is very expensive
     * and should not be used if possible.
     * @return a copy of this array as a 1d float array
     */
    float[] toFloatVector();

    /**
     * Convert this ndarray to a 2d float matrix.
     * Note that THIS SHOULD NOT BE USED FOR SPEED.
     * This is mainly used for integrations with other libraries.
     * Due to nd4j's off  heap nature, moving data on heap is very expensive
     * and should not be used if possible.
     * @return a copy of this array as a 2d float array
     */
    float[][] toFloatMatrix();

    /**
     * Convert this ndarray to a 1d int matrix.
     * Note that THIS SHOULD NOT BE USED FOR SPEED.
     * This is mainly used for integrations with other libraries.
     * Due to nd4j's off  heap nature, moving data on heap is very expensive
     * and should not be used if possible.
     * @return a copy of this array as a 1d int array
     */
    int[] toIntVector();

    /**
     * Convert this ndarray to a 1d long matrix.
     * Note that THIS SHOULD NOT BE USED FOR SPEED.
     * This is mainly used for integrations with other libraries.
     * Due to nd4j's off  heap nature, moving data on heap is very expensive
     * and should not be used if possible.
     * @return a copy of this array as a 1d long array
     */
    long[] toLongVector();

    /**
     * Convert this ndarray to a 2d int matrix.
     * Note that THIS SHOULD NOT BE USED FOR SPEED.
     * This is mainly used for integrations with other libraries.
     * Due to nd4j's off  heap nature, moving data on heap is very expensive
     * and should not be used if possible.
     * @return a copy of this array as a 2d int array
     */
    long[][] toLongMatrix();

    /**
     * Convert this ndarray to a 2d int matrix.
     * Note that THIS SHOULD NOT BE USED FOR SPEED.
     * This is mainly used for integrations with other libraries.
     * Due to nd4j's off  heap nature, moving data on heap is very expensive
     * and should not be used if possible.
     * @return a copy of this array as a 2d int array
     */
    int[][] toIntMatrix();

    /**
     * Perform an copy matrix multiplication
     *
     * @param other  the other matrix to perform matrix multiply with
     * @param result the result ndarray
     * @return the result of the matrix multiplication
     */
    INDArray mmul(INDArray other, INDArray result);

    /**
     * Perform an copy matrix multiplication
     *
     * @param other  the other matrix to perform matrix multiply with
     * @param result the result ndarray
     * @param mMulTranspose the transpose status of each array
     * @return the result of the matrix multiplication
     */
    INDArray mmul(INDArray other, INDArray result,MMulTranspose mMulTranspose);

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
     * Element-wise copy addition of two NDArrays
     *
     * @param other the second ndarray to add
     * @return the result of the addition
     */
    INDArray add(INDArray other);

    /**
     * Element-wise copy addition of two NDArrays
     *
     * @param other  the second ndarray to add
     * @param result the result ndarray
     * @return the result of the addition
     */
    INDArray add(INDArray other, INDArray result);

    /**
     * Perform an copy matrix multiplication
     *
     * @param other the other matrix to perform matrix multiply with
     * @param transpose the transpose status of each ndarray
     * @return the result of the matrix multiplication
     */
    INDArray mmuli(INDArray other, MMulTranspose transpose);

    /**
     * Perform an inplace matrix multiplication
     *
     * @param other the other matrix to perform matrix multiply with
     * @return the result of the matrix multiplication
     */
    INDArray mmuli(INDArray other);

    /**
     * Perform an in place matrix multiplication
     *
     * @param other  the other matrix to perform matrix multiply with
     * @param result the result ndarray
     * @return the result of the matrix multiplication
     */
    INDArray mmuli(INDArray other, INDArray result, MMulTranspose transpose);

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
     * @return the result of the multiplication
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
     * in place (element wise) subtraction of two NDArrays
     *
     * @param other the second ndarray to subtract
     * @return the result of the subtraction
     */
    INDArray subi(INDArray other);

    /**
     * in place (element wise) subtraction of two NDArrays
     *
     * @param other  the second ndarray to subtract
     * @param result the result ndarray
     * @return the result of the subtraction
     */
    INDArray subi(INDArray other, INDArray result);

    /**
     * in place (element wise) addition of two NDArrays
     *
     * @param other the second ndarray to add
     * @return the result of the addition
     */
    INDArray addi(INDArray other);

    /**
     * in place (element wise) addition of two NDArrays
     *
     * @param other  the second ndarray to add
     * @param result the result ndarray
     * @return the result of the addition
     */
    INDArray addi(INDArray other, INDArray result);

    /**
     * Returns the max norm (aka infinity norm, equal to the maximum absolute value) along the specified dimension(s)
     *
     * @param dimension the dimension to the max norm along
     * @return Max norm along the specified dimension
     */
    INDArray normmax(long... dimension);

    /**
     * Returns the max norm (aka infinity norm, equal to the maximum absolute value) along the specified dimension(s)
     *
     * @param dimension the dimension to the max norm along
     * @param keepDims whether to keep reduced dimensions as dimensions of size 1
     * @return Max norm along the specified dimension
     */
    INDArray normmax(boolean keepDims, long... dimension);

    /**
     * Return the max norm (aka infinity norm, equal to the maximum absolute value) for the entire array
     *
     * @return Max norm for the entire array
     */
    Number normmaxNumber();

    /**
     * Returns the norm2 (L2 norm, sqrt(sum(x_i^2), also known as Euclidean norm) along the specified dimension(s)
     *
     * @param dimension the dimension to getScalar the norm2 along
     * @return the norm2 along the specified dimension
     */
    INDArray norm2(long... dimension);

    /**
     * Returns the norm2 (L2 norm, sqrt(sum(x_i^2), also known as Euclidean norm) along the specified dimension(s)
     *
     * @param dimension the dimension to getScalar the norm2 along
     * @param keepDims whether to keep reduced dimensions as dimensions of size 1
     * @return the norm2 along the specified dimension
     */
    INDArray norm2(boolean keepDims, long... dimension);

    /**
     * Return the norm2 (L2 norm, sqrt(sum(x_i^2), also known as Euclidean norm) for the entire array
     *
     * @return L2 norm for the array
     */
    Number norm2Number();

    /**
     * Returns the norm1 (L1 norm, i.e., sum of absolute values; also known as Taxicab or Manhattan norm) along the
     * specified dimension
     *
     * @param dimension the dimension to getScalar the norm1 along
     * @return the norm1 along the specified dimension
     */
    INDArray norm1(long... dimension);

    /**
     * Returns the norm1 (L1 norm, i.e., sum of absolute values; also known as Taxicab or Manhattan norm) along the
     * specified dimension
     *
     * @param dimension the dimension to getScalar the norm1 along
     * @param keepDims whether to keep reduced dimensions as dimensions of size 1
     * @return the norm1 along the specified dimension
     */
    INDArray norm1(boolean keepDims, long... dimension);

    /**
     * Calculate and return norm1 (L1 norm, i.e., sum of absolute values; also known as Taxicab or Manhattan norm) for
     * the entire array
     *
     * @return Norm 1 for the array
     */
    Number norm1Number();

    /**
     * Standard deviation of an INDArray along one or more dimensions
     *
     * @param dimension the dimension to getScalar the std along
     * @return the standard deviation along a particular dimension
     */
    INDArray std(long... dimension);

    /**
     * Calculate the standard deviation for the entire array
     *
     * @return standard deviation
     */
    Number stdNumber();

    /**
     * Standard deviation of an ndarray along a dimension
     *
     * @param dimension the dimension to getScalar the std along
     * @return the standard deviation along a particular dimension
     */
    INDArray std(boolean biasCorrected, long... dimension);

    /**
     * Standard deviation of an ndarray along a dimension
     *
     * @param dimension the dimension to getScalar the std along
     * @param keepDims whether to keep reduced dimensions as dimensions of size 1
     * @return the standard deviation along a particular dimension
     */
    INDArray std(boolean biasCorrected, boolean keepDims, long... dimension);

    /**
     * Calculate the standard deviation for the entire array, specifying whether it is bias corrected or not
     *
     * @param biasCorrected If true: bias corrected standard deviation. False: not bias corrected
     * @return Standard dev
     */
    Number stdNumber(boolean biasCorrected);

    /**
     * Returns the product along a given dimension
     *
     * @param dimension the dimension to getScalar the product along
     * @return the product along the specified dimension
     */
    INDArray prod(long... dimension);

    /**
     * Returns the product along a given dimension
     *
     * @param dimension the dimension to getScalar the product along
     * @param keepDims whether to keep reduced dimensions as dimensions of size 1
     * @return the product along the specified dimension
     */
    INDArray prod(boolean keepDims, long... dimension);

    /**
     * Calculate the product of all values in the array
     *
     * @return Product of all values in the array
     */
    Number prodNumber();

    /**
     * Returns the overall mean of this ndarray
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    INDArray mean(long... dimension);

    /**
     * Returns the overall mean of this ndarray
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    INDArray mean(INDArray result, long... dimension);

    /**
     * Returns the overall mean of this ndarray
     *
     * @param dimension the dimension to getScalar the mean along
     * @param keepDims whether to keep reduced dimensions as dimensions of size 1
     * @return the mean along the specified dimension of this ndarray
     */
    INDArray mean(boolean keepDims, long... dimension);

    /**
     * Returns the overall mean of this ndarray
     *
     * @param dimension the dimension to getScalar the mean along
     * @param keepDims whether to keep reduced dimensions as dimensions of size 1
     * @return the mean along the specified dimension of this ndarray
     */
    INDArray mean(INDArray result, boolean keepDims, long... dimension);

    /**
     * Returns the absolute overall mean of this ndarray
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the absolute mean along the specified dimension of this ndarray
     */
    INDArray amean(long... dimension);

    /**
     * Returns the overall mean of this ndarray
     *
     * @return the mean along the specified dimension of this ndarray
     */
    Number meanNumber();

    /**
     * Returns the absolute overall mean of this ndarray
     *
     * @return the mean along the specified dimension of this ndarray
     */
    Number ameanNumber();

    /**
     * Returns the overall variance of this ndarray
     *
     * @param dimension the dimension to getScalar the variance along
     * @return the variance along the specified dimension of this ndarray
     */
    INDArray var(long... dimension);

    /**
     * Returns the overall variance of this ndarray
     *
     * @param biasCorrected boolean on whether to apply corrected bias
     * @param dimension the dimension to getScalar the variance along
     * @return the variance along the specified dimension of this ndarray
     */
    INDArray var(boolean biasCorrected, long... dimension);

    /**
     * Returns the overall variance of all values in this INDArray
     *
     * @return variance
     */
    Number varNumber();

    /**
     * Returns the overall max of this ndarray along given dimensions
     *
     * @param dimension the dimension to getScalar the max along
     * @return the max along the specified dimension of this ndarray
     */
    INDArray max(long... dimension);

    /**
     * Returns the overall max of this ndarray along given dimensions
     *
     * @param dimension the dimension to getScalar the max along
     * @param keepDims whether to keep reduced dimensions as dimensions of size 1
     * @return the max along the specified dimension of this ndarray
     */
    INDArray max(boolean keepDims, long... dimension);

    /**
     * Returns the absolute overall max of this ndarray along given dimensions
     *
     * @param dimension the dimension to getScalar the amax along
     * @return the amax along the specified dimension of this ndarray
     */
    INDArray amax(long... dimension);

    /**
     * Returns maximum value in this INDArray
     * @return maximum value
     */
    Number maxNumber();

    /**
     * Returns maximum (absolute) value in this INDArray
     * @return Max absolute value
     */
    Number amaxNumber();

    /**
     * Returns the overall min of this ndarray
     *
     * @param dimension the dimension to getScalar the min along
     * @return the min along the specified dimension of this ndarray
     */
    INDArray min(long... dimension);

    /**
     * Returns the overall min of this ndarray
     *
     * @param dimension the dimension to getScalar the min along
     * @param keepDims whether to keep reduced dimensions as dimensions of size 1
     * @return the min along the specified dimension of this ndarray
     */
    INDArray min(boolean keepDims, long... dimension);

    /**
     * Returns minimum (absolute) value in this INDArray, along the specified dimensions
     *
     * @return Minimum absolute value
     */
    INDArray amin(long... dimension);

    /**
     * Returns min value in this INDArray
     * @return Minimum value in the array
     */
    Number minNumber();

    /**
     * Returns absolute min value in this INDArray
     *
     * @return Absolute min value
     */
    Number aminNumber();

    /**
     * Returns the sum along the last dimension of this ndarray
     *
     * @param dimension the dimension to getScalar the sum along
     * @return the sum along the specified dimension of this ndarray
     */
    INDArray sum(long... dimension);

    /**
     * Returns the sum along the last dimension of this ndarray
     *
     * @param dimension the dimension to getScalar the sum along
     * @param keepDims whether to keep reduced dimensions as dimensions of size 1
     * @return the sum along the specified dimension of this ndarray
     */
    INDArray sum(boolean keepDims, long... dimension);

    /**
     * This method takes boolean condition, and returns number of elements matching this condition
     *
     * @param condition Condition to calculate matches for
     * @return Number of elements matching condition
     */
    Number scan(Condition condition);

    /**
     * Returns the sum along the last dimension of this ndarray
     *
     * @param result result of this operation will be stored here
     * @param dimension the dimension to getScalar the sum along
     * @return the sum along the specified dimension of this ndarray
     */
    INDArray sum(INDArray result, long... dimension);

    /**
     * Returns the sum along the last dimension of this ndarray
     *
     * @param result result of this operation will be stored here
     * @param keepDims whether to keep reduced dimensions as dimensions of size 1
     * @param dimension the dimension to getScalar the sum along
     * @return the sum along the specified dimension of this ndarray
     */
    INDArray sum(INDArray result, boolean keepDims, long... dimension);

    /**
     * Sum the entire array
     * @return Sum of array
     */
    Number sumNumber();

    /**
     * Returns entropy value for this INDArray
     * @return entropy value
     */
    Number entropyNumber();

    /**
     * Returns non-normalized Shannon entropy value for this INDArray
     * @return non-normalized Shannon entropy
     */
    Number shannonEntropyNumber();

    /**
     * Returns log entropy value for this INDArray
     * @return log entropy value
     */
    Number logEntropyNumber();

    /**
     * Returns entropy value for this INDArray along specified dimension(s)
     * @param dimension specified dimension(s)
     * @return entropy value
     */
    INDArray entropy(long... dimension);

    /**
     * Returns Shannon entropy value for this INDArray along specified dimension(s)
     * @param dimension specified dimension(s)
     * @return Shannon entropy
     */
    INDArray shannonEntropy(long... dimension);

    /**
     * Returns log entropy value for this INDArray along specified dimension(s)
     * @param dimension specified dimension(s)
     * @return log entropy value
     */
    INDArray logEntropy(long... dimension);

    /**
     * Shape and stride setter
     * @param shape new value for shape
     * @param stride new value for stride
     */
    void setShapeAndStride(int[] shape, int[] stride);

    /**
     * Set the ordering
     * @param order the ordering to set
     */
    void setOrder(char order);

    /**
     * Returns the elements at the specified indices
     *
     * @param indices the indices to getScalar
     * @return the array with the specified elements
     */
    INDArray getScalar(int... indices);

    /**
     * See {@link #getScalar(int[])}
     */
    INDArray getScalar(long... indices);

    /**
     * Get an integer value at the specified indices. Result will be cast to an integer, precision loss is possible.
     * @param indices Indices to get the integer at. Number of indices must match the array rank.
     * @return Integer value at the specified index
     */
    int getInt(int... indices);

    /**
     * Get a long value at the specified index.
     * @param index Index to get the integer at.
     * @return long value at the specified index
     */
    long getLong(long index);

    /**
     * Get a long value at the specified indices.
     * @param indices Indices to get the double at. Number of indices must match the array rank.
     * @return long value at the specified index
     */
    long getLong(long... indices);

    /**
     * Get the numeric value at the specified index.
     * @param index index to retreive.
     * @return numeric value at the specified index.
     */
    Number getNumber(long index);

    /**
     * Get a numeric value at the specified indices.
     * @param indices Indices to get the value from. Number of indices must match the array rank.
     * @return Numeric value at the specified index
     */
    Number getNumber(long... indices);

    /**
     * Get a double value at the specified indices.
     * @param indices Indices to get the double at. Number of indices must match the array rank.
     * @return Double value at the specified index
     */
    double getDouble(int... indices);

    /**
     * See {@link #getDouble(int[])}
     */
    double getDouble(long... indices);

    /**
     * Returns the elements at the specified indices
     *
     * @param indices the indices to getScalar
     * @return the array with the specified elements
     */
    float getFloat(int... indices);

    /**
     * See {@link #getFloat(int...)}
     */
    float getFloat(long... indices);

    /**
     * Get the double value at the specified linear index in the array
     *
     * @param i Index
     * @return Double value at the specified index
     */
    double getDouble(long i);

    /**
     * Get the double value at the specified indices. Can only be used for 2D (rank 2) arrays.
     *
     * @param i Dimension 0 (row) index
     * @param j Dimension 1 (column) index
     * @return double value at the specified indices
     */
    double getDouble(long i, long j);

    /**
     * Return the item at the linear index i
     *
     * @param i the index of the item to getScalar
     * @return the item at index j
     */
    float getFloat(long i);

    /**
     * Return the item at row i column j
     * Note that this is the same as calling getScalar(new int[]{i,j}
     *
     * @param i the row to getScalar
     * @param j the column to getScalar
     * @return the item at row i column j
     */
    float getFloat(long i, long j);

    /**
     * Returns a copy of this ndarray
     *
     * @return a copy of this ndarray
     */
    INDArray dup();

    /**
     * Returns a copy of this ndarray, where the returned ndarray has the specified order
     *
     * @param order order of the NDArray. 'f' or 'c'
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
     * Set the data for this ndarray.
     * @param data new value for the ndarray data.
     */
    void setData(DataBuffer data);

    /**
     * Returns the number of slices in this ndarray
     *
     * @return the number of slices in this ndarray
     */
    long slices();

    /**
     * Get the number of trailing ones in the array shape. For example, a rank 3 array with shape [10, 1, 1] would
     * return 2 for this method
     *
     * @return Number of trailing ones in shape
     */
    int getTrailingOnes();

    /**
     * Get the number of leading ones in the array shape. For example, a rank 3 array with shape [1, 10, 1] would
     * return value 1 for this method
     *
     * @return Number of leading ones in shape
     */
    int getLeadingOnes();

    /**
     * Returns the slice of this from the specified dimension
     *
     * @param i the index of the slice to return
     * @param dimension the dimension of the slice to return
     * @return the slice of this matrix from the specified dimension
     * and dimension
     */
    INDArray slice(long i, int dimension);

    /**
     * Returns the specified slice of this ndarray
     *
     * @param i the index of the slice to return
     * @return the specified slice of this ndarray
     */
    INDArray slice(long i);

    /**
     * Returns the start of where the ndarray is for the underlying data
     *
     * @return the starting offset
     */
    long offset();

    // TODO: Unused untested method.
    /**
     * Returns the start of where the ndarray is for the original data buffer
     *
     * @return original offset.
     */
    long originalOffset();

    /**
     * Reshapes the ndarray (can't change the length of the ndarray). Typically this will be a view, unless reshaping
     * without copying is impossible.
     *
     * @param newShape the new shape of the ndarray
     * @return the reshaped ndarray
     */
    INDArray reshape(char order, long... newShape);

    /**
     * Reshapes the ndarray (can't change the length of the ndarray). Typically this will be a view, unless reshaping
     * without copying is impossible.
     *
     * @param newShape the new shape of the ndarray
     * @return the reshaped ndarray
     */
    INDArray reshape(char order, int... newShape);

    /**
     * Reshapes the ndarray (note: it's not possible to change the length of the ndarray).
     * Typically this will be a view, unless reshaping without copying (i.e., returning a view) is impossible.<br>
     * In that case, the behaviour will depend on the enforceView argument:
     * enforceView == true: throw an exception<br>
     * enforceView == false: return a copy<br>
     *
     * @param newShape the new shape of the ndarray
     * @return the reshaped ndarray
     */
    INDArray reshape(char order, boolean enforceView, long... newShape);

    /**
     * Reshapes the ndarray (can't change the length of the ndarray). Typically this will be a view, unless reshaping
     * without copying is impossible.
     *
     * @param rows    the rows of the matrix
     * @param columns the columns of the matrix
     * @return the reshaped ndarray
     */
    INDArray reshape(char order, int rows, int columns);

    /**
     * Reshapes the ndarray (can't change the length of the ndarray). Typically this will be a view, unless reshaping
     * without copying is impossible.
     *
     * @param newShape the new shape of the ndarray
     * @return the reshaped ndarray
     */
    INDArray reshape(long... newShape);

    /**
     * See {@link #reshape(long[])}
     */
    INDArray reshape(int[] shape);

    /**
     * Reshapes the ndarray (can't change the length of the ndarray). Typically this will be a view, unless reshaping
     * without copying is impossible.
     *
     * @param rows    the rows of the matrix
     * @param columns the columns of the matrix
     * @return the reshaped ndarray
     */
    INDArray reshape(long rows, long columns);

    /**
     * Flip the rows and columns of a matrix
     *
     * @return the flipped rows and columns of a matrix
     */
    INDArray transpose();

    /**
     * Flip the rows and columns of a matrix, in-place
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
    INDArray permute(long... rearrange);

    /**
     * An <b>in-place</b> version of permute. The array  shape information (shape, strides)
     * is modified by this operation (but not the data itself)
     * See: http://www.mathworks.com/help/matlab/ref/permute.html
     *
     * @param rearrange the dimensions to swap to
     * @return the current array
     */
    INDArray permutei(long... rearrange);

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
     *  Returns a view of this tensor with permuted dimensions. Typically the pattern will include the integers 0, 1, ... ndim-1, and any number of 'x' characters in dimensions where this tensor should be broadcasted.

     A few examples of patterns and their effect:

     ('x') -> make a 0d (scalar) into a 1d vector
     (0, 1) -> identity for 2d vectors
     (1, 0) -> inverts the first and second dimensions
     ('x', 0) -> make a row out of a 1d vector (N to 1xN)
     (0, 'x') -> make a column out of a 1d vector (N to Nx1)
     (2, 0, 1) -> AxBxC to CxAxB
     (0, 'x', 1) -> AxB to Ax1xB
     (1, 'x', 0) -> AxB to Bx1xA
     (1,) -> This remove dimensions 0. It must be a broadcastable dimension (1xA to A)

     * @param rearrange     the dimensions to swap to
     * @param newOrder      the new order (think permute)
     * @param broadCastable (whether the dimension is broadcastable) (must be same length as new order)
     * @return the newly permuted array
     */
    INDArray dimShuffle(Object[] rearrange, int[] newOrder, boolean[] broadCastable);

    /**
     * See {@link #dimShuffle(Object[], int[], boolean[])
     */
    INDArray dimShuffle(Object[] rearrange, long[] newOrder, boolean[] broadCastable);

    /**
     * Returns the specified column.
     * Throws an exception if its not a matrix
     *
     * @param i the column to getScalar
     * @return the specified column
     */
    INDArray getColumn(long i);

    /**
     * Returns the specified column. Throws an exception if its not a matrix (rank 2).
     * Returned array will either be 1D (keepDim = false) or 2D (keepDim = true) with shape [length, 1]
     *
     * @param i the row to get
     * @param keepDim If true: return [length, 1] array. Otherwise: return [length] array
     * @return the specified row
     */
    INDArray getColumn(long i, boolean keepDim);

    /**
     * Returns the specified row as a 1D vector.
     * Throws an exception if its not a matrix
     *
     * @param i the row to getScalar
     * @return the specified row
     */
    INDArray getRow(long i);

    /**
     * Returns the specified row. Throws an exception if its not a matrix.
     * Returned array will either be 1D (keepDim = false) or 2D (keepDim = true) with shape [1, length]
     *
     * @param i the row to get
     * @param keepDim If true: return [1,length] array. Otherwise: return [length] array
     * @return the specified row
     */
    INDArray getRow(long i, boolean keepDim);

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
     * Returns true if the number of columns is 1
     *
     * @return true if the number of columns is 1
     */
    boolean isColumnVectorOrScalar();

    /**
     * Returns true if the number of rows is 1
     *
     * @return true if the number of rows is 1
     */
    boolean isRowVectorOrScalar();

    /**
     * Returns true if this ndarray is a vector
     *
     * @return whether this ndarray is a vector
     */
    boolean isVector();

    /**
     * Returns true if this ndarray is a vector or scalar
     *
     * @return whether this ndarray is a vector or scalar
     */
    boolean isVectorOrScalar();

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
    long[] shape();

    /**
     * Returns shape descriptor of this ndarray
     * @return shape descriptor
     */
    LongShapeDescriptor shapeDescriptor();

    /**
     * Returns the stride of this ndarray
     *
     * @return the stride of this ndarray
     */
    long[] stride();

    /**
     * Return the ordering (fortran or c  'f' and 'c' respectively) of this ndarray
     * @return the ordering of this ndarray
     */
    char ordering();

    /**
     * Returns the size along a specified dimension
     *
     * @param dimension the dimension to return the size for
     * @return the size of the array along the specified dimension
     */
    long size(int dimension);


    default long size(long dimension) {
        return size((int) dimension);
    }

    /**
     * Returns the total number of elements in the ndarray
     *
     * @return the number of elements in the ndarray
     */
    long length();

    /**
     * Broadcasts this ndarray to be the specified shape
     *
     * @param shape the new shape of this ndarray
     * @return the broadcasted ndarray
     */
    INDArray broadcast(long... shape);

    /**
     * Broadcasts this ndarray to be the specified shape
     *
     * @return the broadcasted ndarray
     */
    INDArray broadcast(INDArray result);

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
     * This method checks 2 INDArrays equality with given eps
     *
     * @param o   INDArray to compare against.
     * @param eps Epsilon value to use for the quality operation
     * @return True if ndarrays are equal within eps.
     */
    boolean equalsWithEps(Object o, double eps);

    /**
     * This method checks 2 INDArrays for equal shapes.<br>
     * Shapes are considered equal if:<br>
     * (a) Both arrays have equal rank, and<br>
     * (b) size(0)...size(rank()-1) are equal for both arrays
     * @param other Other
     * @return True if shap
     */
    boolean equalShapes(INDArray other);

    /**
     * Perform efficient (but unsafe) duplication. Don't use this method unless you know exactly what you are doing.
     * Instead, use {@link #dup()}
     *
     * @return Unsafe duplicate of array
     */
    INDArray unsafeDuplication();

    /**
     * Perform efficient (but unsafe) duplication. Don't use this method unless you know exactly what you are doing.
     * Instead, use {@link #dup()}
     *
     * @return Unsafe duplicate of array
     */
    INDArray unsafeDuplication(boolean blocking);

    /**
     * Remainder operator
     * @param denominator the denominator
     * @return remainder
     */
    INDArray remainder(INDArray denominator);

    /**
     * Remainder operator
     * @param denominator the denominator
     * @param result the result array to put this in
     * @return Remainder
     */
    INDArray remainder(INDArray denominator, INDArray result);

    /**
     * The scalar remainder
     * @param denominator the denominator as a scalar
     * @return Remainder
     */
    INDArray remainder(Number denominator);

    /**
     * The scalar remainder
     * @param denominator the denominator as a scalar
     * @param result the result array to put this in
     * @return Remainder
     */
    INDArray remainder(Number denominator, INDArray result);

    // TODO: Unused untested method.
    /**
     * In place remainder
     * @param denominator the denominator
     * @return Remainder
     */
    INDArray remainderi(INDArray denominator);

    // TODO: Unused untested method.
    /**
     * In place remainder
     * @param denominator the denominator
     * @return Remainder
     */
    INDArray remainderi(Number denominator);

    /**
     * remainder of division
     * @param denominator the array of denominators for each element in this array
     * @return array of remainders
     */
    INDArray fmod(INDArray denominator);

    /**
     *  remainder of division
     * @param denominator the array of denominators for each element in this array
     * @param result the result array
     * @return array of remainders
     */
    INDArray fmod(INDArray denominator, INDArray result);

    /**
     * remainder of division by scalar.
     *
     * @param denominator the denominator
     * @return array of remainders
     */
    INDArray fmod(Number denominator);

    /**
     * remainder of division by scalar.
     *
     * @param denominator the denominator
     * @param result the result array
     * @return array of remainders
     */
    INDArray fmod(Number denominator, INDArray result);

    // TODO: Unused untested method.
    /**
     * In place fmod
     * @param denominator the array of denominators for each element in this array
     * @return array of remainders
     */
    INDArray fmodi(INDArray denominator);

    // TODO: Unused untested method.
    /**
     * In place fmod
     * @param denominator the denominator as a scalar
     * @return array of remainders
     */
    INDArray fmodi(Number denominator);

    /**
     * This method returns index of highest value along specified dimension(s)
     *
     * @param dimension Dimension along which to perform the argMax operation
     * @return Array containing indices
     */
    INDArray argMax(long... dimension);

    /**
     * This method returns True, if this INDArray instance is attached to some Workspace. False otherwise.
     * @return True if attached to workspace, false otherwise
     */
    boolean isAttached();

    /**
     * This method checks, if given attached INDArray is still in scope of its parent Workspace
     *
     * PLEASE NOTE: if this INDArray isn't attached to any Workspace, this method will return true
     * @return true if attached to workspace.
     */
    boolean isInScope();

    /**
     * This method detaches INDArray from Workspace, returning copy.
     * Basically it's dup() into new memory chunk.
     *
     * PLEASE NOTE: If this INDArray instance is NOT attached - it will be returned unmodified.
     *
     * @return The attached copy of array, or original if not in workspace
     */
    INDArray detach();

    /**
     * This method detaches INDArray from current Workspace, and attaches it to Workspace above, if any.
     *
     * PLEASE NOTE: If this INDArray instance is NOT attached - it will be returned unmodified.
     * PLEASE NOTE: If current Workspace is the top-tier one,
     * effect will be equal to detach() call - detached copy will be returned
     *
     * @return this ndarray or a detached copy.
     */
    INDArray leverage();

    /**
     * This method detaches INDArray from current Workspace, and attaches it to Workspace with a given Id - if a workspace
     * with that ID exists. If no workspace with the specified ID exists, the current INDArray is returned unmodified.
     *
     * @see #leverageTo(String, boolean)
     */
    INDArray leverageTo(String id);

    /**
     * This method detaches INDArray from current Workspace, and attaches it to Workspace with a given Id.
     * If enforceExistence == true, and no workspace with the specified ID exists, then an {@link Nd4jNoSuchWorkspaceException}
     * is thrown. Otherwise, if enforceExistance == false and no workspace with the specified ID exists, then the current
     * INDArray is returned unmodified (same as {@link #leverage()}
     *
     * @param id ID of the workspace to leverage to
     * @param enforceExistence If true, and the specified workspace does not exist: an {@link Nd4jNoSuchWorkspaceException}
     *                         will be thrown.
     * @return The INDArray, leveraged to the specified workspace
     * @see #leverageTo(String)
     */
    INDArray leverageTo(String id, boolean enforceExistence) throws Nd4jNoSuchWorkspaceException;

    /**
     * This method detaches INDArray from current Workspace, and attaches it to Workspace with a given Id, if a workspace
     * with the given ID is open and active.
     *
     * If the workspace does not exist, or is not active, the array is detached from any workspaces.
     *
     * @param id ID of the workspace to leverage to
     * @return The INDArray, leveraged to the specified workspace (if it exists and is active) otherwise the detached array
     * @see #leverageTo(String)
     */
    INDArray leverageOrDetach(String id);

    /**
     * This method pulls this INDArray into current Workspace.
     *
     * PLEASE NOTE: If there's no current Workspace - INDArray returned as is
     *
     * @return Migrated INDArray or <i>this</i> if no current workspace
     * @see #migrate(boolean)
     */
    INDArray migrate();

    /**
     * This method pulls this INDArray into current Workspace, or optionally detaches if no workspace is present.<br>
     * That is:<br>
     * If current workspace is present/active, INDArray is migrated to it.<br>
     * If no current workspace is present/active, one of two things occur:
     * 1. If detachOnNoWs arg is true: if there is no current workspace, INDArray is detached
     * 2. If detachOnNoWs arg is false: this INDArray is returned as-is (no-op) - equivalent to {@link #migrate()}
     *
     * @param detachOnNoWs If true: detach on no WS. If false and no workspace: return this.
     * @return Migrated INDArray
     */
    INDArray migrate(boolean detachOnNoWs);

    /**
     * This method returns percentile value for this INDArray
     *
     * @param percentile target percentile in range of 0..100
     * @return percentile value
     */
    Number percentileNumber(Number percentile);

    /**
     * This method returns median value for this INDArray
     *
     * @return Median value for array
     */
    Number medianNumber();

    /**
     * This method returns median along given dimension(s)
     * @param dimension Dimension to calculate median
     * @return Median along specified dimensions
     */
    INDArray median(long... dimension);

    /**
     * This method returns percentile along given dimension(s)
     * @param percentile target percentile in range of 0..100
     * @param dimension  Dimension to calculate percentile for
     * @return array with percentiles
     */
    INDArray percentile(Number percentile, long... dimension);

    /**
     * Add an {@link INDArray}
     * to flatbuffers builder
     * @param builder the builder to use
     * @return the offset to add
     */
    int toFlatArray(FlatBufferBuilder builder);

    /**
     * This method returns true if this INDArray is special case: no-value INDArray
     * @return True if empty.
     */
    boolean isEmpty();

    /**
     * This method returns shapeInformation as jvm long array
     * @return shapeInformation
     */
    long[] shapeInfoJava();

    /**
     * This method returns dtype for this INDArray
     * @return Datattype
     */
    DataType dataType();

    /**
     * This method checks if this INDArray instance is one of Real types
     * @return true if data type is floating point, false otherwise
     */
    boolean isR();

    /**
     * This method checks if this INDArray instance is one of integer types
     * @return true if integer type
     */
    boolean isZ();

    /**
     * This method checks if this INDArray instance has boolean type
     * @return true if boolean type.
     */
    boolean isB();

    /**
     * This method checks if this INDArray instance has String type
     * @return true if string type.
     */
    boolean isS();

    /**
     * This method cast elements of this INDArray to new data type
     *
     * @param dataType new datatype.
     * @return this if datatype matches, otherwise a new array of specified datatype.
     */
    INDArray castTo(DataType dataType);

    /**
     * This method checks if all elements within this array are non-zero (or true, in case of boolean)
     * @return true if all non-zero.
     */
    boolean all();

    /**
     * This method checks if any of the elements within this array are non-zero (or true, in case of boolean)
     * @return true if any non-zero.
     */
    boolean any();

    /**
     * This method checks if any of the elements within this array are non-zero (or true, in case of boolean)
     * @return true if any non-zero
     */
    boolean none();

    /**
     * This method checks, if this INDArray instalce can use close() method
     * @return true if array can be released, false otherwise
     */
    boolean closeable();

    /**
     * Mainly for overriding closeable in specific situations
     * where a user does not want an ndarray closed.
     * @param closeable
     */
    void setCloseable(boolean closeable);

    /**
     * This method releases exclusive off-heap resources uses by this INDArray instance.
     * If INDArray relies on shared resources, exception will be thrown instead
     *
     * PLEASE NOTE: This method is NOT safe by any means
     */
    void close();

    /**
     * This method checks if array or its buffer was closed before
     * @return true if was closed, false otherwise
     */
    boolean wasClosed();

    /**
     * This method returns empty array with the same dtype/order/shape as this one
     * @return empty array with the same dtype/order/shape
     */
    INDArray like();

    /**
     * This method returns uninitialized array with the same dtype/order/shape as this one
     * @return uninitialized array with the same dtype/order/shape
     */
    INDArray ulike();

    /**
     * Get a string representation of the array with configurable formatting
     * @param options format options
     */
    String toString(@NonNull NDArrayStrings options);

    /**
     * Get a string representation of the array
     *
     * @param maxElements Summarize if more than maxElements in the array
     * @param forceSummarize Force a summary instead of a full print
     * @param precision The number of decimals to print.  Doesn't print trailing 0s if negative
     * @return string representation of the array
     */
    String toString(long maxElements, boolean forceSummarize, int precision);

    /**
     * ToString with unlimited elements and precision
     * @see BaseNDArray#toString(long, boolean, int)
     */
    String toStringFull();

    /**
     * A unique ID for the INDArray object instance. Does not account for views.
     * @return INDArray unique ID
     */
    long getId();

    default MemoryWorkspace getWorkspace() {
        if(isEmpty())
            return null;
        return data().getParentWorkspace();
    }

}
