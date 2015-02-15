package org.nd4j.linalg.api.buffer.factory;

import org.nd4j.linalg.api.buffer.DataBuffer;

/**
 * DataBufferFactory: Creates the data buffer wrt
 * a specified data type
 * @author Adam Gibson
 */
public interface DataBufferFactory {

    /**
     * Create a double data buffer
     * @return the new data buffer
     */
    DataBuffer createDouble(int length);

    /**
     * Create a float data buffer
     * @param length the length of the buffer
     * @return the new data buffer
     */
    DataBuffer createFloat(int length);

    /**
     * Create an int data buffer
     * @param length the length of the data buffer
     * @return the create data buffer
     */
    DataBuffer createInt(int length);


    /**
     * Creates a double data buffer
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createDouble(int[] data);
    /**
     * Creates a double data buffer
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createFloat(int[] data);
    /**
     * Creates a double data buffer
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createInt(int[] data);

    /**
     * Creates a double data buffer
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createDouble(double[] data);
    /**
     * Creates a float data buffer
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createFloat(double[] data);
    /**
     * Creates an int data buffer
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createInt(double[] data);

    /**
     * Creates a double data buffer
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createDouble(float[] data);
    /**
     * Creates a float data buffer
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createFloat(float[] data);

    /**
     * Creates an int data buffer
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createInt(float[] data);









    /**
     * Creates a double data buffer
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createDouble(int[] data,boolean copy);
    /**
     * Creates a double data buffer
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createFloat(int[] data,boolean copy);
    /**
     * Creates a double data buffer
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createInt(int[] data,boolean copy);

    /**
     * Creates a double data buffer
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createDouble(double[] data,boolean copy);
    /**
     * Creates a float data buffer
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createFloat(double[] data,boolean copy);
    /**
     * Creates an int data buffer
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createInt(double[] data,boolean copy);

    /**
     * Creates a double data buffer
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createDouble(float[] data,boolean copy);
    /**
     * Creates a float data buffer
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createFloat(float[] data,boolean copy);

    /**
     * Creates an int data buffer
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createInt(float[] data,boolean copy);


}
