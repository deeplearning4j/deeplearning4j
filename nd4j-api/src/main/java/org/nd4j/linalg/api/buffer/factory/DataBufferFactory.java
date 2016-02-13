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

package org.nd4j.linalg.api.buffer.factory;

import org.nd4j.linalg.api.buffer.DataBuffer;

import java.nio.ByteBuffer;

/**
 * DataBufferFactory: Creates the data buffer wrt
 * a specified data type
 *
 * @author Adam Gibson
 */
public interface DataBufferFactory {




    /**
     * Create int buffer
     * @param buffer
     * @param length
     * @return
     */
    DataBuffer createInt(int offset,ByteBuffer buffer, int length);

    /**
     * Create a float data buffer
     * @param buffer
     * @param length
     * @return
     */
    DataBuffer createFloat(int offset,ByteBuffer buffer, int length);

    /**
     * Creates a double data buffer
     * @param buffer
     * @param length
     * @return
     */
    DataBuffer createDouble(int offset,ByteBuffer buffer, int length);

    /**
     * Create a double data buffer
     *
     * @return the new data buffer
     */
    DataBuffer createDouble(int offset,int length);

    /**
     * Create a float data buffer
     *
     * @param length the length of the buffer
     * @return the new data buffer
     */
    DataBuffer createFloat(int offset,int length);

    /**
     * Create an int data buffer
     *
     * @param length the length of the data buffer
     * @return the create data buffer
     */
    DataBuffer createInt(int offset,int length);


    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createDouble(int offset,int[] data);

    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createFloat(int offset,int[] data);

    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createInt(int offset,int[] data);

    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createDouble(int offset,double[] data);


    /**
     * Create a double buffer buffer
     * @param data
     * @param length
     * @return
     */
    DataBuffer createDouble(int offset,byte[] data, int length);

    /**
     * Create a double buffer
     * @param data
     * @param length
     * @return
     */
    DataBuffer createFloat(int offset,byte[] data, int length);

    /**
     * Creates a float data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createFloat(int offset,double[] data);

    /**
     * Creates an int data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createInt(int offset,double[] data);

    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createDouble(int offset,float[] data);

    /**
     * Creates a float data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createFloat(int offset,float[] data);

    /**
     * Creates an int data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createInt(int offset,float[] data);


    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createDouble(int offset,int[] data, boolean copy);

    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createFloat(int offset,int[] data, boolean copy);

    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createInt(int offset,int[] data, boolean copy);

    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createDouble(int offset,double[] data, boolean copy);

    /**
     * Creates a float data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createFloat(int offset,double[] data, boolean copy);

    /**
     * Creates an int data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createInt(int offset,double[] data, boolean copy);

    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createDouble(int offset,float[] data, boolean copy);

    /**
     * Creates a float data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createFloat(int offset,float[] data, boolean copy);

    /**
     * Creates an int data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createInt(int offset,float[] data, boolean copy);


    /**
     * Create int buffer
     * @param buffer
     * @param length
     * @return
     */
    DataBuffer createInt(ByteBuffer buffer, int length);

    /**
     * Create a float data buffer
     * @param buffer
     * @param length
     * @return
     */
    DataBuffer createFloat(ByteBuffer buffer, int length);

    /**
     * Creates a double data buffer
     * @param buffer
     * @param length
     * @return
     */
    DataBuffer createDouble(ByteBuffer buffer, int length);

    /**
     * Create a double data buffer
     *
     * @return the new data buffer
     */
    DataBuffer createDouble(int length);

    /**
     * Create a float data buffer
     *
     * @param length the length of the buffer
     * @return the new data buffer
     */
    DataBuffer createFloat(int length);

    /**
     * Create an int data buffer
     *
     * @param length the length of the data buffer
     * @return the create data buffer
     */
    DataBuffer createInt(int length);


    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createDouble(int[] data);

    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createFloat(int[] data);

    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createInt(int[] data);

    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createDouble(double[] data);


    /**
     * Create a double buffer buffer
     * @param data
     * @param length
     * @return
     */
    DataBuffer createDouble(byte[] data, int length);

    /**
     * Create a double buffer
     * @param data
     * @param length
     * @return
     */
    DataBuffer createFloat(byte[] data, int length);

    /**
     * Creates a float data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createFloat(double[] data);

    /**
     * Creates an int data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createInt(double[] data);

    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createDouble(float[] data);

    /**
     * Creates a float data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createFloat(float[] data);

    /**
     * Creates an int data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createInt(float[] data);


    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createDouble(int[] data, boolean copy);

    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createFloat(int[] data, boolean copy);

    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createInt(int[] data, boolean copy);

    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createDouble(double[] data, boolean copy);

    /**
     * Creates a float data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createFloat(double[] data, boolean copy);

    /**
     * Creates an int data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createInt(double[] data, boolean copy);

    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createDouble(float[] data, boolean copy);

    /**
     * Creates a float data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createFloat(float[] data, boolean copy);

    /**
     * Creates an int data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createInt(float[] data, boolean copy);


}