package org.nd4j.linalg.api.buffer;

import java.io.Serializable;

/**
 * A data buffer is an interface
 * for handling storage and retrieval of data
 * @author Adam Gibson
 */
public interface DataBuffer extends Serializable {


    public final static int DOUBLE = 0;
    public final static int FLOAT = 1;
    public final static int INT = 2;

    /**
     * Set the data for this buffer
     * @param data the data for this buffer
     */
    void setData(int[] data);
    /**
     * Set the data for this buffer
     * @param data the data for this buffer
     */
    void setData(float[] data);
    /**
     * Set the data for this buffer
     * @param data the data for this buffer
     */
    void setData(double[] data);

    /**
     * Raw byte array storage
     * @return the data represented as a raw byte array
     */
    byte[] asBytes();

    /**
     * The data type of the buffer
     * @return the data type of the buffer
     */
    public int dataType();

    /**
     * Return the buffer as a float array
     * Relative to the datatype, this will either be a copy
     * or a reference. The reference is preferred for
     * faster access of data and no copying
     * @return the buffer as a float
     */
    public float[] asFloat();

    /**
     * Return the buffer as a double array
     * Relative to the datatype, this will either be a copy
     * or a reference. The reference is preferred for
     * faster access of data and no copying
     * @return the buffer as a float
     */
    public double[] asDouble();
    /**
     * Return the buffer as an int  array
     * Relative to the datatype, this will either be a copy
     * or a reference. The reference is preferred for
     * faster access of data and no copying
     * @return the buffer as a float
     */
    public int[] asInt();

    /**
     * Returns the element buffer of the specified type.
     * @param <E>
     * @return the element buffer of the specified type
     */
    public <E> E[] asType();

    /**
     * Get element i in the buffer as a double
     * @param i the element to getFloat
     * @return the element at this index
     */
    public double getDouble(int i);
    /**
     * Get element i in the buffer as a double
     * @param i the element to getFloat
     * @return the element at this index
     */
    public float getFloat(int i);
    /**
     * Get element i in the buffer as a double
     * @param i the element to getFloat
     * @return the element at this index
     */
    public Number getNumber(int i);
    /**
     * Get element i in the buffer as a double
     * @param i the element to getFloat
     * @return the element at this index
     */
    public <E> E getElement(int i);

    /**
     * Assign an element in the buffer to the specified index
     * @param i the index
     * @param element the element to assign
     */
    void put(int i,float element);
    /**
     * Assign an element in the buffer to the specified index
     * @param i the index
     * @param element the element to assign
     */
    void put(int i,double element);
    /**
     * Assign an element in the buffer to the specified index
     * @param i the index
     * @param element the element to assign
     */
    void put(int i,int element);
    /**
     * Assign an element in the buffer to the specified index
     * @param i the index
     * @param element the element to assign
     */
    <E> void put(int i,E element);


    /**
     * Returns the length of the buffer
     * @return the length of the buffer
     */
    int length();

    /**
     * Get the int at the specified index
     * @param ix the int at the specified index
     * @return the int at the specified index
     */
    int getInt(int ix);

    /**
     * Return a copy of this buffer
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

}
