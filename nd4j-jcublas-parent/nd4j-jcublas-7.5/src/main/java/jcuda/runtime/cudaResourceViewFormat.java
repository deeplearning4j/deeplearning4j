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
package jcuda.runtime;

/**
 * CUDA texture resource view formats
 */
public class cudaResourceViewFormat
{
    /**
     * No resource view format (use underlying resource format)
     */
    public static final int cudaResViewFormatNone                      = 0x00;

    /**
     * 1 channel unsigned 8-bit integers
     */
    public static final int cudaResViewFormatUnsignedChar1             = 0x01;

    /**
     * 2 channel unsigned 8-bit integers
     */
    public static final int cudaResViewFormatUnsignedChar2             = 0x02;

    /**
     * 4 channel unsigned 8-bit integers
     */
    public static final int cudaResViewFormatUnsignedChar4             = 0x03;

    /**
     * 1 channel signed 8-bit integers
     */
    public static final int cudaResViewFormatSignedChar1               = 0x04;

    /**
     * 2 channel signed 8-bit integers
     */
    public static final int cudaResViewFormatSignedChar2               = 0x05;

    /**
     * 4 channel signed 8-bit integers
     */
    public static final int cudaResViewFormatSignedChar4               = 0x06;

    /**
     * 1 channel unsigned 16-bit integers
     */
    public static final int cudaResViewFormatUnsignedShort1            = 0x07;

    /**
     * 2 channel unsigned 16-bit integers
     */
    public static final int cudaResViewFormatUnsignedShort2            = 0x08;

    /**
     * 4 channel unsigned 16-bit integers
     */
    public static final int cudaResViewFormatUnsignedShort4            = 0x09;

    /**
     * 1 channel signed 16-bit integers
     */
    public static final int cudaResViewFormatSignedShort1              = 0x0a;

    /**
     * 2 channel signed 16-bit integers
     */
    public static final int cudaResViewFormatSignedShort2              = 0x0b;

    /**
     * 4 channel signed 16-bit integers
     */
    public static final int cudaResViewFormatSignedShort4              = 0x0c;

    /**
     * 1 channel unsigned 32-bit integers
     */
    public static final int cudaResViewFormatUnsignedInt1              = 0x0d;

    /**
     * 2 channel unsigned 32-bit integers
     */
    public static final int cudaResViewFormatUnsignedInt2              = 0x0e;

    /**
     * 4 channel unsigned 32-bit integers
     */
    public static final int cudaResViewFormatUnsignedInt4              = 0x0f;

    /**
     * 1 channel signed 32-bit integers
     */
    public static final int cudaResViewFormatSignedInt1                = 0x10;

    /**
     * 2 channel signed 32-bit integers
     */
    public static final int cudaResViewFormatSignedInt2                = 0x11;

    /**
     * 4 channel signed 32-bit integers
     */
    public static final int cudaResViewFormatSignedInt4                = 0x12;

    /**
     * 1 channel 16-bit floating point
     */
    public static final int cudaResViewFormatHalf1                     = 0x13;

    /**
     * 2 channel 16-bit floating point
     */
    public static final int cudaResViewFormatHalf2                     = 0x14;

    /**
     * 4 channel 16-bit floating point
     */
    public static final int cudaResViewFormatHalf4                     = 0x15;

    /**
     * 1 channel 32-bit floating point
     */
    public static final int cudaResViewFormatFloat1                    = 0x16;

    /**
     * 2 channel 32-bit floating point
     */
    public static final int cudaResViewFormatFloat2                    = 0x17;

    /**
     * 4 channel 32-bit floating point
     */
    public static final int cudaResViewFormatFloat4                    = 0x18;

    /**
     * Block compressed 1
     */
    public static final int cudaResViewFormatUnsignedBlockCompressed1  = 0x19;

    /**
     * Block compressed 2
     */
    public static final int cudaResViewFormatUnsignedBlockCompressed2  = 0x1a;

    /**
     * Block compressed 3
     */
    public static final int cudaResViewFormatUnsignedBlockCompressed3  = 0x1b;

    /**
     * Block compressed 4 unsigned
     */
    public static final int cudaResViewFormatUnsignedBlockCompressed4  = 0x1c;

    /**
     * Block compressed 4 signed
     */
    public static final int cudaResViewFormatSignedBlockCompressed4    = 0x1d;

    /**
     * Block compressed 5 unsigned
     */
    public static final int cudaResViewFormatUnsignedBlockCompressed5  = 0x1e;

    /**
     * Block compressed 5 signed
     */
    public static final int cudaResViewFormatSignedBlockCompressed5    = 0x1f;

    /**
     * Block compressed 6 unsigned half-float
     */
    public static final int cudaResViewFormatUnsignedBlockCompressed6H = 0x20;

    /**
     * Block compressed 6 signed half-float
     */
    public static final int cudaResViewFormatSignedBlockCompressed6H   = 0x21;

    /**
     * Block compressed 7
     */
    public static final int cudaResViewFormatUnsignedBlockCompressed7  = 0x22;

    /**
     * Returns the String identifying the given cudaResourceViewFormat
     *
     * @param m The cudaResourceViewFormat
     * @return The String identifying the given cudaResourceViewFormat
     */
    public static String stringFor(int m)
    {
        switch (m)
        {
            case cudaResViewFormatNone                      :return"cudaResViewFormatNone";
            case cudaResViewFormatUnsignedChar1             :return"cudaResViewFormatUnsignedChar1";
            case cudaResViewFormatUnsignedChar2             :return"cudaResViewFormatUnsignedChar2";
            case cudaResViewFormatUnsignedChar4             :return"cudaResViewFormatUnsignedChar4";
            case cudaResViewFormatSignedChar1               :return"cudaResViewFormatSignedChar1";
            case cudaResViewFormatSignedChar2               :return"cudaResViewFormatSignedChar2";
            case cudaResViewFormatSignedChar4               :return"cudaResViewFormatSignedChar4";
            case cudaResViewFormatUnsignedShort1            :return"cudaResViewFormatUnsignedShort1";
            case cudaResViewFormatUnsignedShort2            :return"cudaResViewFormatUnsignedShort2";
            case cudaResViewFormatUnsignedShort4            :return"cudaResViewFormatUnsignedShort4";
            case cudaResViewFormatSignedShort1              :return"cudaResViewFormatSignedShort1";
            case cudaResViewFormatSignedShort2              :return"cudaResViewFormatSignedShort2";
            case cudaResViewFormatSignedShort4              :return"cudaResViewFormatSignedShort4";
            case cudaResViewFormatUnsignedInt1              :return"cudaResViewFormatUnsignedInt1";
            case cudaResViewFormatUnsignedInt2              :return"cudaResViewFormatUnsignedInt2";
            case cudaResViewFormatUnsignedInt4              :return"cudaResViewFormatUnsignedInt4";
            case cudaResViewFormatSignedInt1                :return"cudaResViewFormatSignedInt1";
            case cudaResViewFormatSignedInt2                :return"cudaResViewFormatSignedInt2";
            case cudaResViewFormatSignedInt4                :return"cudaResViewFormatSignedInt4";
            case cudaResViewFormatHalf1                     :return"cudaResViewFormatHalf1";
            case cudaResViewFormatHalf2                     :return"cudaResViewFormatHalf2";
            case cudaResViewFormatHalf4                     :return"cudaResViewFormatHalf4";
            case cudaResViewFormatFloat1                    :return"cudaResViewFormatFloat1";
            case cudaResViewFormatFloat2                    :return"cudaResViewFormatFloat2";
            case cudaResViewFormatFloat4                    :return"cudaResViewFormatFloat4";
            case cudaResViewFormatUnsignedBlockCompressed1  :return"cudaResViewFormatUnsignedBlockCompressed1";
            case cudaResViewFormatUnsignedBlockCompressed2  :return"cudaResViewFormatUnsignedBlockCompressed2";
            case cudaResViewFormatUnsignedBlockCompressed3  :return"cudaResViewFormatUnsignedBlockCompressed3";
            case cudaResViewFormatUnsignedBlockCompressed4  :return"cudaResViewFormatUnsignedBlockCompressed4";
            case cudaResViewFormatSignedBlockCompressed4    :return"cudaResViewFormatSignedBlockCompressed4";
            case cudaResViewFormatUnsignedBlockCompressed5  :return"cudaResViewFormatUnsignedBlockCompressed5";
            case cudaResViewFormatSignedBlockCompressed5    :return"cudaResViewFormatSignedBlockCompressed5";
            case cudaResViewFormatUnsignedBlockCompressed6H :return"cudaResViewFormatUnsignedBlockCompressed6H";
            case cudaResViewFormatSignedBlockCompressed6H   :return"cudaResViewFormatSignedBlockCompressed6H";
            case cudaResViewFormatUnsignedBlockCompressed7  :return"cudaResViewFormatUnsignedBlockCompressed7";
        }
        return "INVALID cudaResourceViewFormat: " + m;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private cudaResourceViewFormat()
    {
    }
}
