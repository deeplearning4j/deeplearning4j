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

package jcuda.runtime;

/**
 * Channel formats.<br />
 * <br />
 * Most comments are taken from the CUDA reference manual.
 *
 * @see cudaChannelFormatDesc
 */
public class cudaChannelFormatKind
{
    /**
     * Signed channel format
     */
    public static final int cudaChannelFormatKindSigned = 0;

    /**
     * Unsigned channel format
     */
    public static final int cudaChannelFormatKindUnsigned = 1;

    /**
     * Float channel format
     */
    public static final int cudaChannelFormatKindFloat = 2;

    /**
     *  No channel format
     */
    public static final int cudaChannelFormatKindNone = 3;

    /**
     * Returns the String identifying the given cudaChannelFormatKind
     *
     * @param f The cudaChannelFormatKind
     * @return The String identifying the given cudaChannelFormatKind
     */
    public static String stringFor(int f)
    {
        switch (f)
        {
            case cudaChannelFormatKindSigned: return "cudaChannelFormatKindSigned";
            case cudaChannelFormatKindUnsigned: return "cudaChannelFormatKindUnsigned";
            case cudaChannelFormatKindFloat: return "cudaChannelFormatKindFloat";
            case cudaChannelFormatKindNone: return "cudaChannelFormatKindNone";
        }
        return "INVALID cudaChannelFormatKind: "+f;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private cudaChannelFormatKind()
    {
    }

}
