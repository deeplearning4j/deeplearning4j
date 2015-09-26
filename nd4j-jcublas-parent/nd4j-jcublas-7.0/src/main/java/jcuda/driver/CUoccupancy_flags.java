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
 * Occupancy calculator flag
 */
public class CUoccupancy_flags
{
    /** 
     * Default behavior 
     */
    public static final int CU_OCCUPANCY_DEFAULT                  = 0x0;
    
    /** 
     * Assume global caching is enabled and cannot be automatically turned off 
     */
    public static final int CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE = 0x1;
    
    /**
     * Returns the String identifying the given CUoccupancy_flags
     *
     * @param n The CUoccupancy_flags
     * @return The String identifying the given CUoccupancy_flags
     */
    public static String stringFor(int n)
    {
        String result = "CU_OCCUPANCY_DEFAULT ";
        if ((n & CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE) != 0) result += "CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE ";
        return result;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private CUoccupancy_flags()
    {
    }
    
}
