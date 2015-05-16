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
package jcuda.jcusolver;

import jcuda.CudaException;
import jcuda.LibUtils;
import jcuda.LogLevel;

/**
 * Java bindings for CUSOLVER, the NVIDIA CUDA solver library. <br />
 * <br />
 * Note that this class is only intended for setting managing common
 * settings of the solver library, like logging and exception handling.
 * The actual implementations are in {@link JCusolverDn}, {@link JCusolverSp}
 * and {@link JCusolverRf}.  
 * <br />
 * The documentation is taken from the CUSOLVER header files.
 */
public class JCusolver
{
    /**
     * The flag that indicates whether the native library has been
     * loaded
     */
    private static boolean initialized = false;
    
    /**
     * Whether a CudaException should be thrown if a method is about
     * to return a result code that is not 
     * cusolverStatus.CUSOLVER_STATUS_SUCCESS
     */
    private static boolean exceptionsEnabled = false;

    /* Private constructor to prevent instantiation */
    private JCusolver()
    {
    }
    
    // Initialize the native library.
    static
    {
        initialize();
    }
    
    /**
     * Initializes the native library. Note that this method
     * does not have to be called explicitly, since it will
     * be called automatically when this class is loaded.
     */
    public static void initialize()
    {
        if (!initialized)
        {
            LibUtils.loadLibrary("JCusolver");
            initialized = true;
        }
    }

    /**
     * Set the specified log level for the JCusolver library.<br />
     * <br />
     * Currently supported log levels:
     * <br />
     * LOG_QUIET: Never print anything <br />
     * LOG_ERROR: Print error messages <br />
     * LOG_TRACE: Print a trace of all native function calls <br />
     *
     * @param logLevel The log level to use.
     */
    public static void setLogLevel(LogLevel logLevel)
    {
        setLogLevelNative(logLevel.ordinal());
    }

    private static native void setLogLevelNative(int logLevel);


    /**
     * Enables or disables exceptions. By default, the methods of this class
     * only set the {@link cusolverStatus} from the native methods. 
     * If exceptions are enabled, a CudaException with a detailed error 
     * message will be thrown if a method is about to set a result code 
     * that is not cusolverStatus.CUSOLVER_STATUS_SUCCESS
     * 
     * @param enabled Whether exceptions are enabled
     */
    public static void setExceptionsEnabled(boolean enabled)
    {
        exceptionsEnabled = enabled;
    }
    
    /**
     * If the given result is not cusolverStatus.CUSOLVER_STATUS_SUCCESS
     * and exceptions have been enabled, this method will throw a 
     * CudaException with an error message that corresponds to the
     * given result code. Otherwise, the given result is simply
     * returned.
     * 
     * @param result The result to check
     * @return The result that was given as the parameter
     * @throws CudaException If exceptions have been enabled and
     * the given result code is not cusolverStatus.CUSOLVER_STATUS_SUCCESS
     */
    static int checkResult(int result)
    {
        if (exceptionsEnabled && result != 
           cusolverStatus.CUSOLVER_STATUS_SUCCESS)
        {
            throw new CudaException(cusolverStatus.stringFor(result));
        }
        return result;
    }
    

}
