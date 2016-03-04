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

package jcuda.driver;

/**
 * Texture reference addressing modes.<br />
 * <br />
 * Most comments are taken from the CUDA reference manual.
 *
 * @see JCudaDriver#cuTexRefSetAddressMode
 * @see JCudaDriver#cuTexRefGetAddressMode
 */
public class CUaddress_mode
{
    /**
     * Wrapping address mode
     */
    public static final int CU_TR_ADDRESS_MODE_WRAP = 0;

    /**
     * Clamp to edge address mode
     */
    public static final int CU_TR_ADDRESS_MODE_CLAMP = 1;

    /**
     * Mirror address mode
     */
    public static final int CU_TR_ADDRESS_MODE_MIRROR = 2;


    /**
     * Returns the String identifying the given CUaddress_mode
     *
     * @param n The CUaddress_mode
     * @return The String identifying the given CUaddress_mode
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CU_TR_ADDRESS_MODE_WRAP: return "CU_TR_ADDRESS_MODE_WRAP";
            case CU_TR_ADDRESS_MODE_CLAMP: return "CU_TR_ADDRESS_MODE_CLAMP";
            case CU_TR_ADDRESS_MODE_MIRROR: return "CU_TR_ADDRESS_MODE_MIRROR";
        }
        return "INVALID CUaddress_mode: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private CUaddress_mode()
    {
    }

}
