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
 * Stream creation flags.<br />
 * <br />
 * Most comments are taken from the CUDA reference manual.<br />
 * <br />
 * @see JCudaDriver#cuStreamCreate(CUstream, int)
 */
public class CUstream_flags
{
    /**
     * Default stream flag
     */
    public static final int CU_STREAM_DEFAULT       = 0x0;

    /**
     * Stream does not synchronize with stream 0 (the NULL stream)
     */
    public static final int CU_STREAM_NON_BLOCKING = 0x1;

    /**
     * Returns the String identifying the given CUstream_flags
     *
     * @param n The CUstream_flags
     * @return The String identifying the given CUstream_flags
     */
    public static String stringFor(int n)
    {
        if (n == 0)
        {
            return "CU_STREAM_DEFAULT";
        }
        String result = "";
        if ((n & CU_STREAM_NON_BLOCKING) != 0) result += "CU_STREAM_NON_BLOCKING ";
        return result;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private CUstream_flags()
    {
    }

}
