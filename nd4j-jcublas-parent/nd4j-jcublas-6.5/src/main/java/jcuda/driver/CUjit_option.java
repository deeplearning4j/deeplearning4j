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
 * Online compiler and linker options.<br />
 * <br />
 * Most comments are taken from the CUDA reference manual.<br />
 *
 * @see JCudaDriver#cuModuleLoadDataEx
 */
public class CUjit_option
{
    /**
     * Max number of registers that a thread may use.<br />
     * Option type: unsigned int<br />
     * Applies to: compiler only
     */
    public static final int CU_JIT_MAX_REGISTERS = 0;

    /**
     * IN: Specifies minimum number of threads per block to target compilation
     * for<br />
     * OUT: Returns the number of threads the compiler actually targeted.
     * This restricts the resource utilization fo the compiler (e.g. max
     * registers) such that a block with the given number of threads should be
     * able to launch based on register limitations. Note, this option does not
     * currently take into account any other resource limitations, such as
     * shared memory utilization.<br />
     * Cannot be combined with ::CU_JIT_TARGET.<br />
     * Option type: unsigned int<br />
     * Applies to: compiler only
     */
    public static final int CU_JIT_THREADS_PER_BLOCK = 1;

    /**
     * Overwrites the option value with the total wall clock time, in
     * milliseconds, spent in the compiler and linker<br />
     * Option type: float<br />
     * Applies to: compiler and linker
     */
    public static final int CU_JIT_WALL_TIME = 2;

    /**
     * Pointer to a buffer in which to print any log messages
     * that are informational in nature (the buffer size is specified via
     * option ::CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES)<br />
     * Option type: char *<br />
     * Applies to: compiler and linker
     */
    public static final int CU_JIT_INFO_LOG_BUFFER = 3;

    /**
     * IN: Log buffer size in bytes.  Log messages will be capped at this size
     * (including null terminator)<br />
     * OUT: Amount of log buffer filled with messages<br />
     * Option type: unsigned int<br />
     * Applies to: compiler and linker
     */
    public static final int CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4;

    /**
     * Pointer to a buffer in which to print any log messages that
     * reflect errors (the buffer size is specified via option
     * ::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES)<br />
     * Option type: char *<br />
     * Applies to: compiler and linker
     */
    public static final int CU_JIT_ERROR_LOG_BUFFER = 5;

    /**
     * IN: Log buffer size in bytes.  Log messages will be capped at this size
     * (including null terminator)<br />
     * OUT: Amount of log buffer filled with messages<br />
     * Option type: unsigned int<br />
     * Applies to: compiler and linker
     */
    public static final int CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6;

    /**
     * Level of optimizations to apply to generated code (0 - 4), with 4
     * being the default and highest level of optimizations.<br />
     * Option type: unsigned int<br />
     * Applies to: compiler only
     */
    public static final int CU_JIT_OPTIMIZATION_LEVEL = 7;

    /**
     * No option value required. Determines the target based on the current
     * attached context (default)<br />
     * Option type: No option value needed<br />
     * Applies to: compiler and linker
     */
    public static final int CU_JIT_TARGET_FROM_CUCONTEXT = 8;

    /**
     * Target is chosen based on supplied ::CUjit_target.  Cannot be
     * combined with ::CU_JIT_THREADS_PER_BLOCK.<br />
     * Option type: unsigned int for enumerated type ::CUjit_target<br />
     * Applies to: compiler and linker
     */
    public static final int CU_JIT_TARGET = 9;

    /**
     * Specifies choice of fallback strategy if matching cubin is not found.
     * Choice is based on supplied ::CUjit_fallback.<br />
     * Option type: unsigned int for enumerated type ::CUjit_fallback<br />
     * Applies to: compiler only
     */
    public static final int CU_JIT_FALLBACK_STRATEGY = 10;

    /**
     * Specifies whether to create debug information in output (-g)
     * (0: false, default)<br />
     * Option type: int<br />
     * Applies to: compiler and linker
     */
    public static final int CU_JIT_GENERATE_DEBUG_INFO = 11;

    /**
     * Generate verbose log messages (0: false, default)<br />
     * Option type: int<br />
     * Applies to: compiler and linker
     */
    public static final int CU_JIT_LOG_VERBOSE = 12;

    /**
     * Generate line number information (-lineinfo) (0: false, default)<br />
     * Option type: int<br />
     * Applies to: compiler only
     */
    public static final int CU_JIT_GENERATE_LINE_INFO = 13;

    /**
     * Specifies whether to enable caching explicitly (-dlcm) <br />
     * Choice is based on supplied ::CUjit_cacheMode_enum.<br />
     * Option type: unsigned int for enumerated type ::CUjit_cacheMode_enum<br />
     * Applies to: compiler only
     */
    public static final int CU_JIT_CACHE_MODE = 14;


    /**
     * Returns the String identifying the given CUjit_option
     *
     * @param n The CUjit_option
     * @return The String identifying the given CUjit_option
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CU_JIT_MAX_REGISTERS: return "CU_JIT_MAX_REGISTERS";
            case CU_JIT_THREADS_PER_BLOCK: return "CU_JIT_THREADS_PER_BLOCK";
            case CU_JIT_WALL_TIME: return "CU_JIT_WALL_TIME";
            case CU_JIT_INFO_LOG_BUFFER: return "CU_JIT_INFO_LOG_BUFFER";
            case CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES: return "CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES";
            case CU_JIT_ERROR_LOG_BUFFER: return "CU_JIT_ERROR_LOG_BUFFER";
            case CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: return "CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES";
            case CU_JIT_OPTIMIZATION_LEVEL: return "CU_JIT_OPTIMIZATION_LEVEL";
            case CU_JIT_TARGET_FROM_CUCONTEXT: return "CU_JIT_TARGET_FROM_CUCONTEXT";
            case CU_JIT_TARGET: return "CU_JIT_TARGET";
            case CU_JIT_FALLBACK_STRATEGY: return "CU_JIT_FALLBACK_STRATEGY";
            case CU_JIT_GENERATE_DEBUG_INFO: return "CU_JIT_GENERATE_DEBUG_INFO";
            case CU_JIT_LOG_VERBOSE: return "CU_JIT_LOG_VERBOSE";
            case CU_JIT_GENERATE_LINE_INFO: return "CU_JIT_GENERATE_LINE_INFO";
            case CU_JIT_CACHE_MODE: return "CU_JIT_CACHE_MODE";
        }
        return "INVALID CUjit_option: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private CUjit_option()
    {
    }

}

