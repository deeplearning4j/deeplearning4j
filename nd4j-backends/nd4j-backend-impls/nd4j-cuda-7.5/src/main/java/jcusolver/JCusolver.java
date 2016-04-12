/*
 * JCusolver - Java bindings for CUSOLVER, the NVIDIA CUDA solver
 * library, to be used with JCuda
 *
 * Copyright (c) 2010-2015 Marco Hutter - http://www.jcuda.org
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
package jcusolver;

import jcuda.*;
import org.nd4j.linalg.api.buffer.util.LibUtils;

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
