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

package jcuda.jcufft;


/**
 * Compatibility flags for CUFFT. Original documentation: <br />
 * <br />
 * Certain R2C and C2R transforms go much more slowly when FFTW memory
 * layout and behaviour is required. The default is "best performance",
 * which means not-compatible-with-fftw. Use the cufftSetCompatibilityMode
 * API to enable exact FFTW-like behaviour.<br />
 * <br />
 * These flags can be ORed together to select precise FFTW compatibility
 * behaviour. 
 */
public class cufftCompatibility
{
    /**
     * Disable any FFTW compatibility mode.
     * 
     * @deprecated as of CUDA 6.0RC
     */
    public static final int CUFFT_COMPATIBILITY_NATIVE          = 0x00;
    
    /**
     * Inserts extra padding between packed in-place transforms for
     * batched transforms with power-of-2 size. This is the default.
     */
    public static final int CUFFT_COMPATIBILITY_FFTW_PADDING    = 0x01;
    
    /**
     * Guarantees FFTW-compatible output for non-symmetric complex inputs
     * for transforms with power-of-2 size. This is only useful for
     * artificial (i.e. random) datasets as actual data will always be
     * symmetric if it has come from the real plane. If you don't
     * understand what this means, you probably don't have to use it.
     * 
     * @deprecated as of CUDA 6.5:  Asymmetric input is 
     * always treated as in FFTW.
     */
    public static final int CUFFT_COMPATIBILITY_FFTW_ASYMMETRIC = 0x02;

    /**
     * For convenience, enables all FFTW compatibility modes at once.
     */
    public static final int CUFFT_COMPATIBILITY_FFTW_ALL        = 0x03;

    /**
     * Returns the String identifying the given cufftCompatibility
     * 
     * @param m The cufftType
     * @return The String identifying the given cufftCompatibility
     */
    public static String stringFor(int m)
    {
        if (m == CUFFT_COMPATIBILITY_NATIVE)
        {
            return "CUFFT_COMPATIBILITY_NATIVE";
        }
        if ((m & CUFFT_COMPATIBILITY_FFTW_ALL) == CUFFT_COMPATIBILITY_FFTW_ALL)
        {
            return "CUFFT_COMPATIBILITY_FFTW_ALL";
        }
        StringBuilder sb = new StringBuilder();
        if ((m & CUFFT_COMPATIBILITY_FFTW_PADDING) != 0) 
        {
            sb.append("CUFFT_COMPATIBILITY_FFTW_PADDING ");
        }
        if ((m & CUFFT_COMPATIBILITY_FFTW_ASYMMETRIC) != 0) 
        {
            sb.append("CUFFT_COMPATIBILITY_FFTW_ASYMMETRIC ");
        }
        return sb.toString();
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private cufftCompatibility()
    {
    }

}
