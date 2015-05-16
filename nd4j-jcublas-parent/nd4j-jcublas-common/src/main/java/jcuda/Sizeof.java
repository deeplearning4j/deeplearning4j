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
