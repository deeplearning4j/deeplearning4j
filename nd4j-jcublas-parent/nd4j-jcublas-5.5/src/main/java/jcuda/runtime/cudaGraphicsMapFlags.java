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
 * CUDA graphics interop map flags
 */
public class cudaGraphicsMapFlags
{
    /**
     * Default; Assume resource can be read/written
     */
    public static final int cudaGraphicsMapFlagsNone         = 0;

    /**
     * CUDA will not write to this resource
     */
    public static final int cudaGraphicsMapFlagsReadOnly     = 1;

    /**
     * CUDA will only write to and will not read from this resource
     */
    public static final int cudaGraphicsMapFlagsWriteDiscard = 2;


    /**
     * Returns the String identifying the given cudaGraphicsMapFlags
     *
     * @param n The cudaGraphicsMapFlags
     * @return The String identifying the given cudaGraphicsMapFlags
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case cudaGraphicsMapFlagsNone: return "cudaGraphicsMapFlagsNone";
            case cudaGraphicsMapFlagsReadOnly: return "cudaGraphicsMapFlagsReadOnly";
            case cudaGraphicsMapFlagsWriteDiscard: return "cudaGraphicsMapFlagsWriteDiscard";
        }
        return "INVALID cudaGraphicsMapFlags: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private cudaGraphicsMapFlags()
    {
    }

}
