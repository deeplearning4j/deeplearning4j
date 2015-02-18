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

package org.nd4j.linalg.api.buffer;

import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.ops.ElementWiseOp;

import java.io.Serializable;

/**
 * A data buffer is an interface
 * for handling storage and retrieval of data
 *
 * @author Adam Gibson
 */
public interface DataBuffer extends Serializable {


    public final static int DOUBLE = 0;
    public final static int FLOAT = 1;
    public final static int INT = 2;


    /**
     * Assign the given elements to the given indices
     *
     * @param indices    the indices to assign
     * @param data       the data to assign
     * @param contiguous whether the indices are contiguous or not
     * @param inc        the number to increment by when assigning
     */
    void assign(int[] indices, float[] data, boolean contiguous, int inc);

    /**
     * Assign the given elements to the given indices
     *
     * @param indices    the indices to assign
     * @param data       the data to assign
     * @param contiguous whether the data is contiguous or not
     * @param inc        the number to increment by when assigning
     */
    void assign(int[] indices, double[] data, boolean contiguous, int inc);


    /**
     * Assign the given elements to the given indices
     *
     * @param indices    the indices to assign
     * @param data       the data to assign
     * @param contiguous whether the indices are contiguous or not
     */
    void assign(int[] indices, float[] data, boolean contiguous);

    /**
     * Assign the given elements to the given indices
     *
     * @param indices    the indices to assign
     * @param data       the data to assign
     * @param contiguous whether the data is contiguous or not
     */
    void assign(int[] indices, double[] data, boolean contiguous);

    /**
     * Get the doubles at a particular offset
     *
     * @param offset the offset to start
     * @param length the length of the array
     * @return the doubles at the specified offset and length
     */
    double[] getDoublesAt(int offset, int length);


    /**
     * Get the doubles at a particular offset
     *
     * @param offset the offset to start
     * @param length the length of the array
     * @return the doubles at the specified offset and length
     */
    float[] getFloatsAt(int offset, int length);


    /**
     * Get the doubles at a particular offset
     *
     * @param offset the offset to start
     * @param inc    the increment to use
     * @param length the length of the array
     * @return the doubles at the specified offset and length
     */
    double[] getDoublesAt(int offset, int inc, int length);


    /**
     * Get the doubles at a particular offset
     *
     * @param offset the offset to start
     * @param inc    the increment to use
     * @param length the length of the array
     * @return the doubles at the specified offset and length
     */
    float[] getFloatsAt(int offset, int inc, int length);


    /**
     * Assign the given value to the buffer
     *
     * @param value the value to assign
     */
    void assign(Number value);

    /**
     * Assign the given value to the buffer
     * starting at offset
     *
     * @param value  assign the value to set
     * @param offset the offset to start at
     */
    void assign(Number value, int offset);

    /**
     * Set the data for this buffer
     *
     * @param data the data for this buffer
     */
    void setData(int[] data);

    /**
     * Set the data for this buffer
     *
     * @param data the data for this buffer
     */
    void setData(float[] data);

    /**
     * Set the data for this buffer
     *
     * @param data the data for this buffer
     */
    void setData(double[] data);

    /**
     * Raw byte array storage
     *
     * @return the data represented as a raw byte array
     */
    byte[] asBytes();

    /**
     * The data type of the buffer
     *
     * @return the data type of the buffer
     */
    public int dataType();

    /**
     * Return the buffer as a float array
     * Relative to the datatype, this will either be a copy
     * or a reference. The reference is preferred for
     * faster access of data and no copying
     *
     * @return the buffer as a float
     */
    public float[] asFloat();

    /**
     * Return the buffer as a double array
     * Relative to the datatype, this will either be a copy
     * or a reference. The reference is preferred for
     * faster access of data and no copying
     *
     * @return the buffer as a float
     */
    public double[] asDouble();

    /**
     * Return the buffer as an int  array
     * Relative to the datatype, this will either be a copy
     * or a reference. The reference is preferred for
     * faster access of data and no copying
     *
     * @return the buffer as a float
     */
    public int[] asInt();

    /**
     * Returns the element buffer of the specified type.
     *
     * @param <E>
     * @return the element buffer of the specified type
     */
    public <E> E[] asType();

    /**
     * Get element i in the buffer as a double
     *
     * @param i the element to getFloat
     * @return the element at this index
     */
    public double getDouble(int i);

    /**
     * Get element i in the buffer as a double
     *
     * @param i the element to getFloat
     * @return the element at this index
     */
    public float getFloat(int i);

    /**
     * Get element i in the buffer as a double
     *
     * @param i the element to getFloat
     * @return the element at this index
     */
    public Number getNumber(int i);

    /**
     * Get element i in the buffer as a double
     *
     * @param i the element to getFloat
     * @return the element at this index
     */
    public <E> E getElement(int i);

    /**
     * Assign an element in the buffer to the specified index
     *
     * @param i       the index
     * @param element the element to assign
     */
    void put(int i, float element);

    /**
     * Assign an element in the buffer to the specified index
     *
     * @param i       the index
     * @param element the element to assign
     */
    void put(int i, double element);

    /**
     * Assign an element in the buffer to the specified index
     *
     * @param i       the index
     * @param element the element to assign
     */
    void put(int i, int element);

    /**
     * Assign an element in the buffer to the specified index
     *
     * @param i       the index
     * @param element the element to assign
     */
    <E> void put(int i, E element);


    /**
     * Get the complex float
     *
     * @param i the i togete
     * @return the complex float at the specified index
     */
    IComplexFloat getComplexFloat(int i);

    /**
     * Get the complex double at the specified index
     *
     * @param i the index
     * @return the complex double
     */
    IComplexDouble getComplexDouble(int i);

    /**
     * Returns a complex number
     *
     * @param i the complex number cto get
     * @return the complex number to get
     */
    IComplexNumber getComplex(int i);

    /**
     * Returns the length of the buffer
     *
     * @return the length of the buffer
     */
    int length();

    /**
     * Get the int at the specified index
     *
     * @param ix the int at the specified index
     * @return the int at the specified index
     */
    int getInt(int ix);

    /**
     * Return a copy of this buffer
     *
     * @return a copy of this buffer
     */
    DataBuffer dup();

    /**
     * Flush the data buffer
     */
    void flush();

    /**
     * Clears this buffer
     */
    void destroy();

    /**
     * Apply an element wise op to the data buffer
     *
     * @param op the operation to apply
     */
    void apply(ElementWiseOp op);

    /**
     * Apply an element wise op to the data buffer
     *
     * @param op     the operation to apply
     * @param offset the offset to start applying the function from
     */
    void apply(ElementWiseOp op, int offset);

    /**
     * Assign the contents of this buffer
     * to this buffer
     *
     * @param data the data to assign
     */
    void assign(DataBuffer data);


    /**
     * Element wise addition
     *
     * @param n the number to add
     */
    void addi(Number n);

    /**
     * Element wise addition
     *
     * @param n the number to add
     */
    void subi(Number n);

    /**
     * Element wise multiplication
     *
     * @param n the number to multiply
     */
    void muli(Number n);

    /**
     * Element wise division
     *
     * @param n the number to divide by
     */
    void divi(Number n);


    /**
     * Element wise addition
     *
     * @param n the number to add
     */
    void addi(Number n, int inc, int offset);

    /**
     * Element wise addition
     *
     * @param n the number to add
     */
    void subi(Number n, int inc, int offset);

    /**
     * Element wise multiplication
     *
     * @param n the number to multiply
     */
    void muli(Number n, int inc, int offset);

    /**
     * Element wise division
     *
     * @param n the number to divide by
     */
    void divi(Number n, int inc, int offset);

    /**
     * Add the items in this buffer by
     * the elements in the other buffer
     *
     * @param buffer the buffer to add
     */
    void addi(DataBuffer buffer);

    /**
     * Subtract the items in this buffer
     * from the other buffer
     *
     * @param buffer the buffer to subtract
     */
    void subi(DataBuffer buffer);

    /**
     * Multiply the items in this buffer
     * from the other buffer
     *
     * @param buffer the buffer to multiply
     */
    void muli(DataBuffer buffer);

    /**
     * Divide the items in this buffer
     * from the other buffer
     *
     * @param buffer the buffer to divide
     */
    void divi(DataBuffer buffer);

    /**
     * @param n
     * @param buffer
     * @param offset
     * @param incx
     * @param incy
     */
    void addi(DataBuffer buffer, int n, int offset, int yOffset, int incx, int incy);


    /**
     * @param n
     * @param buffer
     * @param offset
     * @param incx
     * @param incy
     */
    void subi(DataBuffer buffer, int n, int offset, int yOffset, int incx, int incy);

    /**
     * @param n
     * @param buffer
     * @param offset
     * @param incx
     * @param incy
     */
    void muli(DataBuffer buffer, int n, int offset, int yOffset, int incx, int incy);

    /**
     * @param n
     * @param buffer
     * @param offset
     * @param incx
     * @param incy
     */
    void divi(DataBuffer buffer, int n, int offset, int yOffset, int incx, int incy);

}
