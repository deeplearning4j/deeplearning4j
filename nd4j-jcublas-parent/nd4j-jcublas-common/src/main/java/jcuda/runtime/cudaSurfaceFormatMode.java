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
 * Surface format modes
 * 
 * @see surfaceReference
 */
public class cudaSurfaceFormatMode
{
    public static final int cudaFormatModeForced = 0;
    public static final int cudaFormatModeAuto = 1;
    
    /**
     * Returns the String identifying the given cudaSurfaceFormatMode
     * 
     * @param m The cudaSurfaceFormatMode
     * @return The String identifying the given cudaSurfaceFormatMode
     */
    public static String stringFor(int m)
    {
        switch (m)
        {
            case cudaFormatModeForced: return "cudaFormatModeForced";
            case cudaFormatModeAuto: return "cudaFormatModeAuto";
        }
        return "INVALID cudaSurfaceFormatMode: " + m;
    }
    
    /**
     * Private constructor to prevent instantiation.
     */
    private cudaSurfaceFormatMode()
    {
    }
    
};
