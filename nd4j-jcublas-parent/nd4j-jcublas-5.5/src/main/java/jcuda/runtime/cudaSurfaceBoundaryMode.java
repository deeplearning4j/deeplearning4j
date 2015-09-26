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
 * Surface boundary modes
 * 
 * @see surfaceReference
 */
public class cudaSurfaceBoundaryMode
{
    public static final int cudaBoundaryModeZero = 0;
    public static final int cudaBoundaryModeClamp = 1;
    public static final int cudaBoundaryModeTrap = 2;
    
    /**
     * Returns the String identifying the given cudaSurfaceBoundaryMode
     * 
     * @param m The cudaSurfaceBoundaryMode
     * @return The String identifying the given cudaSurfaceBoundaryMode
     */
    public static String stringFor(int m)
    {
        switch (m)
        {
            case cudaBoundaryModeZero: return "cudaBoundaryModeZero";
            case cudaBoundaryModeClamp: return "cudaBoundaryModeClamp";
            case cudaBoundaryModeTrap: return "cudaBoundaryModeTrap";
        }
        return "INVALID cudaSurfaceBoundaryMode: " + m;
    }
    
    /**
     * Private constructor to prevent instantiation.
     */
    private cudaSurfaceBoundaryMode()
    {
    }
    
};
