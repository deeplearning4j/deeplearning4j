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

package jcuda;

import java.nio.*;

/**
 * Utility methods for handling Buffers
 *
 * (currently not used)
 */
class BufferUtils
{

    /**
     * Creates a buffer for the given number of elements
     *
     * @param elements The number of elements in the buffer
     * @return The buffer
     */
    public static ByteBuffer createByteBuffer(int elements)
    {
        return ByteBuffer.allocateDirect(elements);
    }

    /**
     * Creates a buffer for the given number of elements and
     * native byte ordering
     *
     * @param elements The number of elements in the buffer
     * @return The buffer
     */
    public static CharBuffer createCharBuffer(int elements)
    {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(elements * 2);
        byteBuffer.order(ByteOrder.nativeOrder());
        return byteBuffer.asCharBuffer();
    }

    /**
     * Creates a buffer for the given number of elements and
     * native byte ordering
     *
     * @param elements The number of elements in the buffer
     * @return The buffer
     */
    public static ShortBuffer createShortBuffer(int elements)
    {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(elements * 2);
        byteBuffer.order(ByteOrder.nativeOrder());
        return byteBuffer.asShortBuffer();
    }

    /**
     * Creates a buffer for the given number of elements and
     * native byte ordering
     *
     * @param elements The number of elements in the buffer
     * @return The buffer
     */
    public static IntBuffer createIntBuffer(int elements)
    {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(elements * 4);
        byteBuffer.order(ByteOrder.nativeOrder());
        return byteBuffer.asIntBuffer();
    }

    /**
     * Creates a buffer for the given number of elements and
     * native byte ordering
     *
     * @param elements The number of elements in the buffer
     * @return The buffer
     */
    public static FloatBuffer createFloatBuffer(int elements)
    {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(elements * 4);
        byteBuffer.order(ByteOrder.nativeOrder());
        return byteBuffer.asFloatBuffer();
    }

    /**
     * Creates a buffer for the given number of elements and
     * native byte ordering
     *
     * @param elements The number of elements in the buffer
     * @return The buffer
     */
    public static LongBuffer createLongBuffer(int elements)
    {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(elements * 8);
        byteBuffer.order(ByteOrder.nativeOrder());
        return byteBuffer.asLongBuffer();
    }

    /**
     * Creates a buffer for the given number of elements and
     * native byte ordering
     *
     * @param elements The number of elements in the buffer
     * @return The buffer
     */
    public static DoubleBuffer createDoubleBuffer(int elements)
    {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(elements * 8);
        byteBuffer.order(ByteOrder.nativeOrder());
        return byteBuffer.asDoubleBuffer();
    }


    /**
     * Private constructor to prevent instantiation.
     */
    private BufferUtils()
    {
    }
}
