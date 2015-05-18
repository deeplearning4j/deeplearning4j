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
 * Event creation flags.<br />
 * <br />
 * Most comments are taken from the CUDA reference manual.<br />
 * <br />
 * @see JCudaDriver#cuEventCreate(CUevent, int)
 */
public class CUevent_flags
{
    /**
     * Default event flag
     */
    public static final int CU_EVENT_DEFAULT       = 0x0;

    /**
     * Event uses blocking synchronization
     */
    public static final int CU_EVENT_BLOCKING_SYNC = 0x1;

    /** 
     * Event will not record timing data 
     */
    public static final int CU_EVENT_DISABLE_TIMING = 0x2; 
    
    /** 
     * Event is suitable for interprocess use. CU_EVENT_DISABLE_TIMING must be set 
     */
    public static final int CU_EVENT_INTERPROCESS   = 0x4;

    /**
     * Returns the String identifying the given CUevent_flags
     *
     * @param n The CUevent_flags
     * @return The String identifying the given CUevent_flags
     */
    public static String stringFor(int n)
    {
        if (n == 0)
        {
            return "CU_EVENT_DEFAULT";
        }
        String result = "";
        if ((n & CU_EVENT_BLOCKING_SYNC) != 0) result += "CU_EVENT_BLOCKING_SYNC ";
        if ((n & CU_EVENT_DISABLE_TIMING) != 0) result += "CU_EVENT_DISABLE_TIMING ";
        if ((n & CU_EVENT_INTERPROCESS) != 0) result += "CU_EVENT_INTERPROCESS ";
        return result;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private CUevent_flags()
    {
    }

}
