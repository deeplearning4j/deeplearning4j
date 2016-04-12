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

/**
 * Constants for sizes of primitives
 */
public class Sizeof
{
    /**
     * Size of a byte, in bytes
     */
    public static final int BYTE = Byte.SIZE / 8;

    /**
     * Size of a char, in bytes
     */
    public static final int CHAR = Character.SIZE / 8;

    /**
     * Size of a short, in bytes
     */
    public static final int SHORT = Short.SIZE / 8;

    /**
     * Size of an int, in bytes
     */
    public static final int INT = Integer.SIZE / 8;

    /**
     * Size of a float, in bytes
     */
    public static final int FLOAT = Float.SIZE / 8;

    /**
     * Size of a long, in bytes
     */
    public static final int LONG = Long.SIZE / 8;

    /**
     * Size of a double, in bytes
     */
    public static final int DOUBLE = Double.SIZE / 8;

    /**
     * Size of a (native) pointer, in bytes.
     */
    public static final int POINTER = computePointerSize();

    /**
     * Computes the size of a pointer, in bytes
     *
     * @return The size of a pointer, in bytes
     */
    private static int computePointerSize()
    {
        String bits = System.getProperty("sun.arch.data.model");
        if (bits.equals("32"))
        {
            return 4;
        }
        else if (bits.equals("64"))
        {
            return 8;
        }
        else
        {
            System.err.println(
                "Unknown value for sun.arch.data.model - assuming 32 bits");
            return 4;
        }
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private Sizeof()
    {
    }
}
