/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 *
 * Copyright (c) 2009-2015 Marco Hutter - http://www.jcuda.org
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
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
