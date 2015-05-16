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
 * CUDA graphics interop array indices for cube maps
 */
public class cudaGraphicsCubeFace
{

    /**
     * Positive X face of cubemap
     */
    public static final int cudaGraphicsCubeFacePositiveX = 0x00;


    /**
     * Negative X face of cubemap
     */
    public static final int cudaGraphicsCubeFaceNegativeX = 0x01;


    /**
     * Positive Y face of cubemap
     */
    public static final int cudaGraphicsCubeFacePositiveY = 0x02;


    /**
     * Negative Y face of cubemap
     */
    public static final int cudaGraphicsCubeFaceNegativeY = 0x03;


    /**
     * Positive Z face of cubemap
     */
    public static final int cudaGraphicsCubeFacePositiveZ = 0x04;


    /**
     * Negative Z face of cubemap
     */
    public static final int cudaGraphicsCubeFaceNegativeZ = 0x05;

    /**
     * Returns the String identifying the given cudaGraphicsCubeFace
     *
     * @param n The cudaGraphicsCubeFace
     * @return The String identifying the given cudaGraphicsCubeFace
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case cudaGraphicsCubeFacePositiveX: return "cudaGraphicsCubeFacePositiveX";
            case cudaGraphicsCubeFaceNegativeX: return "cudaGraphicsCubeFaceNegativeX";
            case cudaGraphicsCubeFacePositiveY: return "cudaGraphicsCubeFacePositiveY";
            case cudaGraphicsCubeFaceNegativeY: return "cudaGraphicsCubeFaceNegativeY";
            case cudaGraphicsCubeFacePositiveZ: return "cudaGraphicsCubeFacePositiveZ";
            case cudaGraphicsCubeFaceNegativeZ: return "cudaGraphicsCubeFaceNegativeZ";
        }
        return "INVALID cudaGraphicsCubeFace: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private cudaGraphicsCubeFace()
    {
    }

}
