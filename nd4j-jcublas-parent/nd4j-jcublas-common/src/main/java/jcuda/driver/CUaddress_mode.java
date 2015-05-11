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
