/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 *
 * Copyright (c) 2009-2015 Marco Hutter - http://www.jcuda.org
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
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
