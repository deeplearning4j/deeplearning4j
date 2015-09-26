/*
 * JCurand - Java bindings for CURAND, the NVIDIA CUDA random
 * number generation library, to be used with JCuda
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

package jcuda.jcurand;

import jcuda.*;
import jcuda.runtime.cudaStream_t;

/**
 * Java bindings for CURAND, the NVIDIA CUDA random number
 * generation library. <br />
 * <br />
 * The documentation is taken from the CURAND header files.
 */
public class JCurand
{
    /**
     * The flag that indicates whether the native library has been
     * loaded
     */
    private static boolean initialized = false;

    /**
     * Whether a CudaException should be thrown if a method is about
     * to return a result code that is not
     * curandStatus.CURAND_STATUS_SUCCESS
     */
    private static boolean exceptionsEnabled = false;

    /* Private constructor to prevent instantiation */
    private JCurand()
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
            LibUtils.loadLibrary("JCurand");
            initialized = true;
        }
    }

    /**
     * Set the specified log level for the JCurand library.<br />
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
     * only set the {@link curandStatus} from the native methods.
     * If exceptions are enabled, a CudaException with a detailed error
     * message will be thrown if a method is about to set a result code
     * that is not curandStatus.CURAND_STATUS_SUCCESS
     *
     * @param enabled Whether exceptions are enabled
     */
    public static void setExceptionsEnabled(boolean enabled)
    {
        exceptionsEnabled = enabled;
    }

    /**
     * If the given result is not curandStatus.CURAND_STATUS_SUCCESS
     * and exceptions have been enabled, this method will throw a
     * CudaException with an error message that corresponds to the
     * given result code. Otherwise, the given result is simply
     * returned.
     *
     * @param result The result to check
     * @return The result that was given as the parameter
     * @throws CudaException If exceptions have been enabled and
     * the given result code is not curandStatus.CURAND_STATUS_SUCCESS
     */
    private static int checkResult(int result)
    {
        if (exceptionsEnabled && result != curandStatus.CURAND_STATUS_SUCCESS)
        {
            throw new CudaException(curandStatus.stringFor(result));
        }
        return result;
    }




    //=== Auto-generated part: ===============================================

    /**
     * <pre>
     * Create new random number generator.
     *
     * Creates a new random number generator of type rng_type
     * and returns it in *generator.
     *
     * Legal values for rng_type are:
     * - CURAND_RNG_PSEUDO_DEFAULT
     * - CURAND_RNG_PSEUDO_XORWOW
     * - CURAND_RNG_PSEUDO_MRG32K3A
     * - CURAND_RNG_PSEUDO_MTGP32
     * - CURAND_RNG_PSEUDO_MT19937
     * - CURAND_RNG_PSEUDO_PHILOX4_32_10
     * - CURAND_RNG_QUASI_DEFAULT
     * - CURAND_RNG_QUASI_SOBOL32
     * - CURAND_RNG_QUASI_SCRAMBLED_SOBOL32
     * - CURAND_RNG_QUASI_SOBOL64
     * - CURAND_RNG_QUASI_SCRAMBLED_SOBOL64
     *
     * When rng_type is CURAND_RNG_PSEUDO_DEFAULT, the type chosen
     * is CURAND_RNG_PSEUDO_XORWOW.
     * When rng_type is CURAND_RNG_QUASI_DEFAULT,
     * the type chosen is CURAND_RNG_QUASI_SOBOL32.
     *
     * The default values for rng_type = CURAND_RNG_PSEUDO_XORWOW are:
     * - seed = 0
     * - offset = 0
     * - ordering = CURAND_ORDERING_PSEUDO_DEFAULT
     *
     * The default values for rng_type = CURAND_RNG_PSEUDO_MRG32K3A are:
     * - seed = 0
     * - offset = 0
     * - ordering = CURAND_ORDERING_PSEUDO_DEFAULT
     *
     * The default values for rng_type = CURAND_RNG_PSEUDO_MTGP32 are:
     * - seed = 0
     * - offset = 0
     * - ordering = CURAND_ORDERING_PSEUDO_DEFAULT
     *
     * The default values for \p rng_type = CURAND_RNG_PSEUDO_MT19937 are:
     * - seed = 0
     * - offset = 0
     * - ordering = CURAND_ORDERING_PSEUDO_DEFAULT
     *
     * The default values for rng_type = CURAND_RNG_PSEUDO_PHILOX4_32_10 are:
     * - seed = 0
     *  - offset = 0
     *  - ordering = CURAND_ORDERING_PSEUDO_DEFAULT
     *
     * The default values for rng_type = CURAND_RNG_QUASI_SOBOL32 are:
     * - dimensions = 1
     * - offset = 0
     * - ordering = CURAND_ORDERING_QUASI_DEFAULT
     *
     * The default values for rng_type = CURAND_RNG_QUASI_SOBOL64 are:
     * - dimensions = 1
     * - offset = 0
     * - ordering = CURAND_ORDERING_QUASI_DEFAULT
     *
     * The default values for rng_type = CURAND_RNG_QUASI_SCRAMBBLED_SOBOL32 are:
     * - dimensions = 1
     * - offset = 0
     * - ordering = CURAND_ORDERING_QUASI_DEFAULT
     *
     * The default values for rng_type = CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 are:
     * - dimensions = 1
     * - offset = 0
     * - ordering = CURAND_ORDERING_QUASI_DEFAULT
     *
     * @param generator - Pointer to generator
     * @param rng_type - Type of generator to create
     *
     * @return
     *
     * CURAND_STATUS_ALLOCATION_FAILED if memory could not be allocated
     * CURAND_STATUS_INITIALIZATION_FAILED if there was a problem setting up the GPU
     * CURAND_STATUS_VERSION_MISMATCH if the header file version does not match the
     *   dynamically linked library version
     * CURAND_STATUS_TYPE_ERROR if the value for rng_type is invalid
     * CURAND_STATUS_SUCCESS if generator was created successfully
     * </pre>
     */
    public static int curandCreateGenerator(curandGenerator generator, int rng_type)
    {
        return checkResult(curandCreateGeneratorNative(generator, rng_type));
    }
    private native static int curandCreateGeneratorNative(curandGenerator generator, int rng_type);

    /**
     * <pre>
     * Create new host CPU random number generator.
     *
     * Creates a new host CPU random number generator of type rng_type
     * and returns it in *generator.
     *
     * Legal values for rng_type are:
     * - CURAND_RNG_PSEUDO_DEFAULT
     * - CURAND_RNG_PSEUDO_XORWOW
     * - CURAND_RNG_PSEUDO_MRG32K3A
     * - CURAND_RNG_PSEUDO_MTGP32
     * - CURAND_RNG_PSEUDO_MT19937
     * - CURAND_RNG_PSEUDO_PHILOX4_32_10
     * - CURAND_RNG_QUASI_DEFAULT
     * - CURAND_RNG_QUASI_SOBOL32
     *
     * When rng_type is CURAND_RNG_PSEUDO_DEFAULT, the type chosen
     * is CURAND_RNG_PSEUDO_XORWOW.
     * When rng_type is CURAND_RNG_QUASI_DEFAULT,
     * the type chosen is CURAND_RNG_QUASI_SOBOL32.
     *
     * The default values for rng_type = CURAND_RNG_PSEUDO_XORWOW are:
     * - seed = 0
     * - offset = 0
     * - ordering = CURAND_ORDERING_PSEUDO_DEFAULT
     *
     * The default values for rng_type = CURAND_RNG_PSEUDO_MRG32K3A are:
     * - seed = 0
     * - offset = 0
     * - ordering = CURAND_ORDERING_PSEUDO_DEFAULT
     *
     * The default values for rng_type = CURAND_RNG_PSEUDO_MTGP32 are:
     * - seed = 0
     * - offset = 0
     * - ordering = CURAND_ORDERING_PSEUDO_DEFAULT
     *
     * The default values for \p rng_type = CURAND_RNG_PSEUDO_MT19937 are:
     * - seed = 0
     * - offset = 0
     * - ordering = CURAND_ORDERING_PSEUDO_DEFAULT
     *
     * The default values for rng_type = CURAND_RNG_PSEUDO_PHILOX4_32_10 are:
     * - seed = 0
     * - offset = 0
     * - ordering = CURAND_ORDERING_PSEUDO_DEFAULT
     *
     * The default values for rng_type = CURAND_RNG_QUASI_SOBOL32 are:
     * - dimensions = 1
     * - offset = 0
     * - ordering = CURAND_ORDERING_QUASI_DEFAULT
     *
     * The default values for rng_type = CURAND_RNG_QUASI_SOBOL64 are:
     * - dimensions = 1
     * - offset = 0
     * - ordering = CURAND_ORDERING_QUASI_DEFAULT
     *
     * The default values for rng_type = CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 are:
     * - dimensions = 1
     * - offset = 0
     * - ordering = CURAND_ORDERING_QUASI_DEFAULT
     *
     * The default values for rng_type = CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 are:
     * - dimensions = 1
     * - offset = 0
     * - ordering = CURAND_ORDERING_QUASI_DEFAULT
     *
     * @param generator - Pointer to generator
     * @param rng_type - Type of generator to create
     *
     * @return
     *
     * CURAND_STATUS_ALLOCATION_FAILED if memory could not be allocated
     * CURAND_STATUS_INITIALIZATION_FAILED if there was a problem setting up the GPU
     * CURAND_STATUS_VERSION_MISMATCH if the header file version does not match the
     *   dynamically linked library version
     * CURAND_STATUS_TYPE_ERROR if the value for rng_type is invalid
     * CURAND_STATUS_SUCCESS if generator was created successfully
     * </pre>
     */
    public static int curandCreateGeneratorHost(curandGenerator generator, int rng_type)
    {
        return checkResult(curandCreateGeneratorHostNative(generator, rng_type));
    }
    private native static int curandCreateGeneratorHostNative(curandGenerator generator, int rng_type);

    /**
     * <pre>
     * Destroy an existing generator.
     *
     * Destroy an existing generator and free all memory associated with its state.
     *
     * @param generator - Generator to destroy
     *
     * @return
     *
     * CURAND_STATUS_NOT_INITIALIZED if the generator was never created
     * CURAND_STATUS_SUCCESS if generator was destroyed successfully
     * </pre>
     */
    public static int curandDestroyGenerator(curandGenerator generator)
    {
        return checkResult(curandDestroyGeneratorNative(generator));
    }
    private native static int curandDestroyGeneratorNative(curandGenerator generator);

    /**
     * <pre>
     * Return the version number of the library.
     *
     * Return in *version the version number of the dynamically linked CURAND
     * library.  The format is the same as CUDART_VERSION from the CUDA Runtime.
     * The only supported configuration is CURAND version equal to CUDA Runtime
     * version.
     *
     * @param version - CURAND library version
     *
     * @return
     *
     * CURAND_STATUS_SUCCESS if the version number was successfully returned
     * </pre>
     */
    public static int curandGetVersion(int version[])
    {
        return checkResult(curandGetVersionNative(version));
    }
    private native static int curandGetVersionNative(int version[]);

    /**
     * <pre>
     * Set the current stream for CURAND kernel launches.
     *
     * Set the current stream for CURAND kernel launches.  All library functions
     * will use this stream until set again.
     *
     * @param generator - Generator to modify
     * @param stream - Stream to use or ::NULL for null stream
     *
     * @return
     *
     * CURAND_STATUS_NOT_INITIALIZED if the generator was never created
     * CURAND_STATUS_SUCCESS if stream was set successfully
     * </pre>
     */
    public static int curandSetStream(curandGenerator generator, cudaStream_t stream)
    {
        return checkResult(curandSetStreamNative(generator, stream));
    }
    private native static int curandSetStreamNative(curandGenerator generator, cudaStream_t stream);

    /**
     * <pre>
     * Set the seed value of the pseudo-random number generator.
     *
     * Set the seed value of the pseudorandom number generator.
     * All values of seed are valid.  Different seeds will produce different sequences.
     * Different seeds will often not be statistically correlated with each other,
     * but some pairs of seed values may generate sequences which are statistically correlated.
     *
     * @param generator - Generator to modify
     * @param seed - Seed value
     *
     * @return
     *
     * CURAND_STATUS_NOT_INITIALIZED if the generator was never created
     * CURAND_STATUS_TYPE_ERROR if the generator is not a pseudorandom number generator
     * CURAND_STATUS_SUCCESS if generator seed was set successfully
     * </pre>
     */
    public static int curandSetPseudoRandomGeneratorSeed(curandGenerator generator, long seed)
    {
        return checkResult(curandSetPseudoRandomGeneratorSeedNative(generator, seed));
    }
    private native static int curandSetPseudoRandomGeneratorSeedNative(curandGenerator generator, long seed);

    /**
     * <pre>
     * Set the absolute offset of the pseudo or quasirandom number generator.
     *
     * Set the absolute offset of the pseudo or quasirandom number generator.
     *
     * All values of offset are valid.  The offset position is absolute, not
     * relative to the current position in the sequence.
     *
     * @param generator - Generator to modify
     * @param offset - Absolute offset position
     *
     * @return
     *
     * CURAND_STATUS_NOT_INITIALIZED if the generator was never created
     * CURAND_STATUS_SUCCESS if generator offset was set successfully
     * </pre>
     */
    public static int curandSetGeneratorOffset(curandGenerator generator, long offset)
    {
        return checkResult(curandSetGeneratorOffsetNative(generator, offset));
    }
    private native static int curandSetGeneratorOffsetNative(curandGenerator generator, long offset);

    /**
     * <pre>
     * Set the ordering of results of the pseudo or quasirandom number generator.
     *
     * Set the ordering of results of the pseudo or quasirandom number generator.
     *
     * Legal values of order for pseudorandom generators are:
     * - CURAND_ORDERING_PSEUDO_DEFAULT
     * - CURAND_ORDERING_PSEUDO_BEST
     * - CURAND_ORDERING_PSEUDO_SEEDED
     *
     * Legal values of order for quasirandom generators are:
     * - CURAND_ORDERING_QUASI_DEFAULT
     *
     * @param generator - Generator to modify
     * @param order - Ordering of results
     *
     * @return
     *
     * CURAND_STATUS_NOT_INITIALIZED if the generator was never created
     * CURAND_STATUS_OUT_OF_RANGE if the ordering is not valid
     * CURAND_STATUS_SUCCESS if generator ordering was set successfully
     * </pre>
     */
    public static int curandSetGeneratorOrdering(curandGenerator generator, int order)
    {
        return checkResult(curandSetGeneratorOrderingNative(generator, order));
    }
    private native static int curandSetGeneratorOrderingNative(curandGenerator generator, int order);

    /**
     * <pre>
     * Set the number of dimensions.
     *
     * Set the number of dimensions to be generated by the quasirandom number
     * generator.
     *
     * Legal values for num_dimensions are 1 to 20000.
     *
     * @param generator - Generator to modify
     * @param num_dimensions - Number of dimensions
     *
     * @return
     *
     * CURAND_STATUS_NOT_INITIALIZED if the generator was never created
     * CURAND_STATUS_OUT_OF_RANGE if num_dimensions is not valid
     * CURAND_STATUS_TYPE_ERROR if the generator is not a quasirandom number generator
     * CURAND_STATUS_SUCCESS if generator ordering was set successfully
     * </pre>
     */
    public static int curandSetQuasiRandomGeneratorDimensions(curandGenerator generator, int num_dimensions)
    {
        return checkResult(curandSetQuasiRandomGeneratorDimensionsNative(generator, num_dimensions));
    }
    private native static int curandSetQuasiRandomGeneratorDimensionsNative(curandGenerator generator, int num_dimensions);

    /**
     * <pre>
     * Generate 32-bit pseudo or quasirandom numbers.
     *
     * Use generator to generate num 32-bit results into the device memory at
     * outputPtr.  The device memory must have been previously allocated and be
     * large enough to hold all the results.  Launches are done with the stream
     * set using ::curandSetStream(), or the null stream if no stream has been set.
     *
     * Results are 32-bit values with every bit random.
     *
     * @param generator - Generator to use
     * @param outputPtr - Pointer to device memory to store CUDA-generated results, or
     *                 Pointer to host memory to store CPU-generated results
     * @param num - Number of random 32-bit values to generate
     *
     * @return
     *
     * CURAND_STATUS_NOT_INITIALIZED if the generator was never created
     * CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
     *     a previous kernel launch
     * CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is
     *    not a multiple of the quasirandom dimension
     * CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason
     * CURAND_STATUS_SUCCESS if the results were generated successfully
     * </pre>
     */
    public static int curandGenerate(curandGenerator generator, Pointer outputPtr, long num)
    {
        return checkResult(curandGenerateNative(generator, outputPtr, num));
    }
    private native static int curandGenerateNative(curandGenerator generator, Pointer outputPtr, long num);

    /**
     * <pre>
     * Generate 64-bit quasirandom numbers.
     *
     * Use generator to generate num 64-bit results into the device memory at
     * outputPtr.  The device memory must have been previously allocated and be
     * large enough to hold all the results.  Launches are done with the stream
     * set using ::curandSetStream(), or the null stream if no stream has been set.
     *
     * Results are 64-bit values with every bit random.
     *
     * @param generator - Generator to use
     * @param outputPtr - Pointer to device memory to store CUDA-generated results, or
     *                 Pointer to host memory to store CPU-generated results
     * @param num - Number of random 64-bit values to generate
     *
     * @return
     *
     * CURAND_STATUS_NOT_INITIALIZED if the generator was never created
     * CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
     *     a previous kernel launch
     * CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is
     *    not a multiple of the quasirandom dimension
     * CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason
     * CURAND_STATUS_SUCCESS if the results were generated successfully
     * </pre>
     */
    public static int curandGenerateLongLong(curandGenerator generator, Pointer outputPtr, long num)
    {
        return checkResult(curandGenerateLongLongNative(generator, outputPtr, num));
    }
    private native static int curandGenerateLongLongNative(curandGenerator generator, Pointer outputPtr, long num);

    /**
     * <pre>
     * Generate uniformly distributed floats.
     *
     * Use generator to generate num float results into the device memory at
     * outputPtr.  The device memory must have been previously allocated and be
     * large enough to hold all the results.  Launches are done with the stream
     * set using ::curandSetStream(), or the null stream if no stream has been set.
     *
     * Results are 32-bit floating point values between 0.0f and 1.0f,
     * excluding 0.0f and including 1.0f.
     *
     * @param generator - Generator to use
     * @param outputPtr - Pointer to device memory to store CUDA-generated results, or
     *                 Pointer to host memory to store CPU-generated results
     * @param num - Number of floats to generate
     *
     * @return
     *
     * CURAND_STATUS_NOT_INITIALIZED if the generator was never created
     * CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
     *    a previous kernel launch
     * CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason
     * CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is
     *    not a multiple of the quasirandom dimension
     * CURAND_STATUS_SUCCESS if the results were generated successfully
     * </pre>
     */
    public static int curandGenerateUniform(curandGenerator generator, Pointer outputPtr, long num)
    {
        return checkResult(curandGenerateUniformNative(generator, outputPtr, num));
    }
    private native static int curandGenerateUniformNative(curandGenerator generator, Pointer outputPtr, long num);

    /**
     * <pre>
     * Generate uniformly distributed doubles.
     *
     * Use generator to generate num double results into the device memory at
     * outputPtr.  The device memory must have been previously allocated and be
     * large enough to hold all the results.  Launches are done with the stream
     * set using ::curandSetStream(), or the null stream if no stream has been set.
     *
     * Results are 64-bit double precision floating point values between
     * 0.0 and 1.0, excluding 0.0 and including 1.0.
     *
     * @param generator - Generator to use
     * @param outputPtr - Pointer to device memory to store CUDA-generated results, or
     *                 Pointer to host memory to store CPU-generated results
     * @param num - Number of doubles to generate
     *
     * @return
     *
     * CURAND_STATUS_NOT_INITIALIZED if the generator was never created
     * CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
     *    a previous kernel launch
     * CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason
     * CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is
     *    not a multiple of the quasirandom dimension
     * CURAND_STATUS_DOUBLE_PRECISION_REQUIRED if the GPU does not support double precision
     * CURAND_STATUS_SUCCESS if the results were generated successfully
     * </pre>
     */
    public static int curandGenerateUniformDouble(curandGenerator generator, Pointer outputPtr, long num)
    {
        return checkResult(curandGenerateUniformDoubleNative(generator, outputPtr, num));
    }
    private native static int curandGenerateUniformDoubleNative(curandGenerator generator, Pointer outputPtr, long num);

    /**
     * <pre>
     * Generate normally distributed floats.
     *
     * Use generator to generate num float results into the device memory at
     * outputPtr.  The device memory must have been previously allocated and be
     * large enough to hold all the results.  Launches are done with the stream
     * set using ::curandSetStream(), or the null stream if no stream has been set.
     *
     * Results are 32-bit floating point values with mean mean and standard
     * deviation stddev.
     *
     * Normally distributed results are generated from pseudorandom generators
     * with a Box-Muller transform, and so require num to be even.
     * Quasirandom generators use an inverse cumulative distribution
     * function to preserve dimensionality.
     *
     * There may be slight numerical differences between results generated
     * on the GPU with generators created with ::curandCreateGenerator()
     * and results calculated on the CPU with generators created with
     * ::curandCreateGeneratorHost().  These differences arise because of
     * differences in results for transcendental functions.  In addition,
     * future versions of CURAND may use newer versions of the CUDA math
     * library, so different versions of CURAND may give slightly different
     * numerical values.
     *
     * @param generator - Generator to use
     * @param outputPtr - Pointer to device memory to store CUDA-generated results, or
     *                 Pointer to host memory to store CPU-generated results
     * @param n - Number of floats to generate
     * @param mean - Mean of normal distribution
     * @param stddev - Standard deviation of normal distribution
     *
     * @return
     *
     * CURAND_STATUS_NOT_INITIALIZED if the generator was never created
     * CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
     *    a previous kernel launch
     * CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason
     * CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is
     *    not a multiple of the quasirandom dimension, or is not a multiple
     *    of two for pseudorandom generators
     * CURAND_STATUS_SUCCESS if the results were generated successfully
     * </pre>
     */
    public static int curandGenerateNormal(curandGenerator generator, Pointer outputPtr, long n, float mean, float stddev)
    {
        return checkResult(curandGenerateNormalNative(generator, outputPtr, n, mean, stddev));
    }
    private native static int curandGenerateNormalNative(curandGenerator generator, Pointer outputPtr, long n, float mean, float stddev);

    /**
     * <pre>
     * Generate normally distributed doubles.
     *
     * Use generator to generate num double results into the device memory at
     * outputPtr.  The device memory must have been previously allocated and be
     * large enough to hold all the results.  Launches are done with the stream
     * set using ::curandSetStream(), or the null stream if no stream has been set.
     *
     * Results are 64-bit floating point values with mean mean and standard
     * deviation stddev.
     *
     * Normally distributed results are generated from pseudorandom generators
     * with a Box-Muller transform, and so require num to be even.
     * Quasirandom generators use an inverse cumulative distribution
     * function to preserve dimensionality.
     *
     * There may be slight numerical differences between results generated
     * on the GPU with generators created with ::curandCreateGenerator()
     * and results calculated on the CPU with generators created with
     * ::curandCreateGeneratorHost().  These differences arise because of
     * differences in results for transcendental functions.  In addition,
     * future versions of CURAND may use newer versions of the CUDA math
     * library, so different versions of CURAND may give slightly different
     * numerical values.
     *
     * @param generator - Generator to use
     * @param outputPtr - Pointer to device memory to store CUDA-generated results, or
     *                 Pointer to host memory to store CPU-generated results
     * @param n - Number of doubles to generate
     * @param mean - Mean of normal distribution
     * @param stddev - Standard deviation of normal distribution
     *
     * @return
     *
     * CURAND_STATUS_NOT_INITIALIZED if the generator was never created
     * CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
     *    a previous kernel launch
     * CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason
     * CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is
     *    not a multiple of the quasirandom dimension, or is not a multiple
     *    of two for pseudorandom generators
     * CURAND_STATUS_DOUBLE_PRECISION_REQUIRED if the GPU does not support double precision
     * CURAND_STATUS_SUCCESS if the results were generated successfully
     * </pre>
     */
    public static int curandGenerateNormalDouble(curandGenerator generator, Pointer outputPtr, long n, double mean, double stddev)
    {
        return checkResult(curandGenerateNormalDoubleNative(generator, outputPtr, n, mean, stddev));
    }
    private native static int curandGenerateNormalDoubleNative(curandGenerator generator, Pointer outputPtr, long n, double mean, double stddev);

    /**
     * <pre>
     * Generate log-normally distributed floats.
     *
     * Use generator to generate num float results into the device memory at
     * outputPtr.  The device memory must have been previously allocated and be
     * large enough to hold all the results.  Launches are done with the stream
     * set using ::curandSetStream(), or the null stream if no stream has been set.
     *
     * Results are 32-bit floating point values with log-normal distribution based on
     * an associated normal distribution with mean mean and standard deviation stddev.
     *
     * Normally distributed results are generated from pseudorandom generators
     * with a Box-Muller transform, and so require num to be even.
     * Quasirandom generators use an inverse cumulative distribution
     * function to preserve dimensionality.
     * The normally distributed results are transformed into log-normal distribution.
     *
     * There may be slight numerical differences between results generated
     * on the GPU with generators created with ::curandCreateGenerator()
     * and results calculated on the CPU with generators created with
     * ::curandCreateGeneratorHost().  These differences arise because of
     * differences in results for transcendental functions.  In addition,
     * future versions of CURAND may use newer versions of the CUDA math
     * library, so different versions of CURAND may give slightly different
     * numerical values.
     *
     * @param generator - Generator to use
     * @param outputPtr - Pointer to device memory to store CUDA-generated results, or
     *                 Pointer to host memory to store CPU-generated results
     * @param n - Number of floats to generate
     * @param mean - Mean of associated normal distribution
     * @param stddev - Standard deviation of associated normal distribution
     *
     * @return
     *
     * CURAND_STATUS_NOT_INITIALIZED if the generator was never created
     * CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
     *    a previous kernel launch
     * CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason
     * CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is
     *    not a multiple of the quasirandom dimension, or is not a multiple
     *    of two for pseudorandom generators
     * CURAND_STATUS_SUCCESS if the results were generated successfully
     * </pre>
     */
    public static int curandGenerateLogNormal(curandGenerator generator, Pointer outputPtr, long n, float mean, float stddev)
    {
        return checkResult(curandGenerateLogNormalNative(generator, outputPtr, n, mean, stddev));
    }
    private native static int curandGenerateLogNormalNative(curandGenerator generator, Pointer outputPtr, long n, float mean, float stddev);

    /**
     * <pre>
     * Generate log-normally distributed doubles.
     *
     * Use generator to generate num double results into the device memory at
     * outputPtr.  The device memory must have been previously allocated and be
     * large enough to hold all the results.  Launches are done with the stream
     * set using ::curandSetStream(), or the null stream if no stream has been set.
     *
     * Results are 64-bit floating point values with log-normal distribution based on
     * an associated normal distribution with mean mean and standard deviation stddev.
     *
     * Normally distributed results are generated from pseudorandom generators
     * with a Box-Muller transform, and so require num to be even.
     * Quasirandom generators use an inverse cumulative distribution
     * function to preserve dimensionality.
     * The normally distributed results are transformed into log-normal distribution.
     *
     * There may be slight numerical differences between results generated
     * on the GPU with generators created with ::curandCreateGenerator()
     * and results calculated on the CPU with generators created with
     * ::curandCreateGeneratorHost().  These differences arise because of
     * differences in results for transcendental functions.  In addition,
     * future versions of CURAND may use newer versions of the CUDA math
     * library, so different versions of CURAND may give slightly different
     * numerical values.
     *
     * @param generator - Generator to use
     * @param outputPtr - Pointer to device memory to store CUDA-generated results, or
     *                 Pointer to host memory to store CPU-generated results
     * @param n - Number of doubles to generate
     * @param mean - Mean of normal distribution
     * @param stddev - Standard deviation of normal distribution
     *
     * @return
     *
     * CURAND_STATUS_NOT_INITIALIZED if the generator was never created
     * CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
     *    a previous kernel launch
     * CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason
     * CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is
     *    not a multiple of the quasirandom dimension, or is not a multiple
     *    of two for pseudorandom generators
     * CURAND_STATUS_DOUBLE_PRECISION_REQUIRED if the GPU does not support double precision
     * CURAND_STATUS_SUCCESS if the results were generated successfully
     * </pre>
     */
    public static int curandGenerateLogNormalDouble(curandGenerator generator, Pointer outputPtr, long n, double mean, double stddev)
    {
        return checkResult(curandGenerateLogNormalDoubleNative(generator, outputPtr, n, mean, stddev));
    }
    private native static int curandGenerateLogNormalDoubleNative(curandGenerator generator, Pointer outputPtr, long n, double mean, double stddev);





    /**
     * <pre>
     * Construct the histogram array for a Poisson distribution.
     *
     * Construct the histogram array for the Poisson distribution with lambda lambda.
     * For lambda greater than 2000, an approximation with a normal distribution is used.
     *
     * @param lambda - lambda for the Poisson distribution
     * @param discrete_distribution - pointer to the histogram in device memory
     *
     * @return
     * - CURAND_STATUS_ALLOCATION_FAILED if memory could not be allocated
     * - CURAND_STATUS_DOUBLE_PRECISION_REQUIRED if the GPU does not support double precision
     * - CURAND_STATUS_INITIALIZATION_FAILED if there was a problem setting up the GPU
     * - CURAND_STATUS_NOT_INITIALIZED if the distribution pointer was null
     * - CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
     *    a previous kernel launch
     * - CURAND_STATUS_OUT_OF_RANGE if lambda is non-positive or greater than 400,000
     * - CURAND_STATUS_SUCCESS if the histogram was generated successfully
     * </pre>
     * (Note: This function actually belongs to the device API of CURAND,
     * and may not be used sensibly from Java)
     */
    public static int curandCreatePoissonDistribution(double lambda, curandDiscreteDistribution discrete_distribution)
    {
        return checkResult(curandCreatePoissonDistributionNative(lambda, discrete_distribution));
    }
    private static native int curandCreatePoissonDistributionNative(double lambda, curandDiscreteDistribution discrete_distribution);



    /**
     * <pre>
     * Destroy the histogram array for a discrete distribution (e.g. Poisson).
     *
     * Destroy the histogram array for a discrete distribution created by curandCreatePoissonDistribution.
     *
     * @param discrete_distribution - pointer to device memory where the histogram is stored
     * @return
     * - CURAND_STATUS_NOT_INITIALIZED if the histogram was never created
     * - CURAND_STATUS_SUCCESS if the histogram was destroyed successfully
     * </pre>
     * (Note: This function actually belongs to the device API of CURAND,
     * and may not be used sensibly from Java)
     */
    public static int curandDestroyDistribution(curandDiscreteDistribution discrete_distribution)
    {
        return checkResult(curandDestroyDistributionNative(discrete_distribution));
    }
    private static native int curandDestroyDistributionNative(curandDiscreteDistribution discrete_distribution);


    /**
     * <pre>
     * Generate Poisson-distributed unsigned ints.
     *
     * Use generator to generate n unsigned int results into device memory at
     * outputPtr.  The device memory must have been previously allocated and must be
     * large enough to hold all the results.  Launches are done with the stream
     * set using ::curandSetStream(), or the null stream if no stream has been set.
     *
     * Results are 32-bit unsigned int point values with Poisson distribution,
     * with lambda lambda.
     *
     * @param generator - Generator to use
     * @param outputPtr - Pointer to device memory to store CUDA-generated results, or
     *                 Pointer to host memory to store CPU-generated results
     * @param n - Number of unsigned ints to generate
     * @param lambda - lambda for the Poisson distribution
     *
     * @return
     * - CURAND_STATUS_NOT_INITIALIZED if the generator was never created
     * - CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
     *    a previous kernel launch
     * - CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason
     * - CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is
     *    not a multiple of the quasirandom dimension
     * - CURAND_STATUS_DOUBLE_PRECISION_REQUIRED if the GPU or sm does not support double precision
     * - CURAND_STATUS_OUT_OF_RANGE if lambda is non-positive or greater than 400,000
     * - CURAND_STATUS_SUCCESS if the results were generated successfully
     * </pre>
     */
    public static int curandGeneratePoisson(curandGenerator generator, Pointer outputPtr, long n, double lambda)
    {
        return checkResult(curandGeneratePoissonNative(generator, outputPtr, n, lambda));
    }
    private static native int curandGeneratePoissonNative(curandGenerator generator, Pointer outputPtr, long n, double lambda);


    /**
     * <pre>
     * Setup starting states.
     *
     * Generate the starting state of the generator.  This function is
     * automatically called by generation functions such as
     * ::curandGenerate() and ::curandGenerateUniform().
     * It can be called manually for performance testing reasons to separate
     * timings for starting state generation and random number generation.
     *
     * @param generator - Generator to update
     *
     * @return
     *
     * CURAND_STATUS_NOT_INITIALIZED if the generator was never created
     * CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
     *     a previous kernel launch
     * CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason
     * CURAND_STATUS_SUCCESS if the seeds were generated successfully
     * </pre>
     */
    public static int curandGenerateSeeds(curandGenerator generator)
    {
        return checkResult(curandGenerateSeedsNative(generator));
    }
    private native static int curandGenerateSeedsNative(curandGenerator generator);

    /**
     * <pre>
     * Get direction vectors for 32-bit quasirandom number generation.
     *
     * Get a pointer to an array of direction vectors that can be used
     * for quasirandom number generation.  The resulting pointer will
     * reference an array of direction vectors in host memory.
     *
     * The array contains vectors for many dimensions.  Each dimension
     * has 32 vectors.  Each individual vector is an unsigned int.
     *
     * Legal values for set are:
     * - CURAND_DIRECTION_VECTORS_32_JOEKUO6 (20,000 dimensions)
     * - CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 (20,000 dimensions)
     *
     * @param vectors - Address of pointer in which to return direction vectors
     * @param set - Which set of direction vectors to use
     *
     * @return
     *
     * CURAND_STATUS_OUT_OF_RANGE if the choice of set is invalid
     * CURAND_STATUS_SUCCESS if the pointer was set successfully
     * </pre>
     */
    public static int curandGetDirectionVectors32(int[][][] vectors, int set)
    {
        return checkResult(curandGetDirectionVectors32Native(vectors, set));
    }
    private native static int curandGetDirectionVectors32Native(int[][][] vectors, int set);

    /**
     * <pre>
     * Get scramble constants for 32-bit scrambled Sobol' .
     *
     * Get a pointer to an array of scramble constants that can be used
     * for quasirandom number generation.  The resulting pointer will
     * reference an array of unsinged ints in host memory.
     *
     * The array contains constants for many dimensions.  Each dimension
     * has a single unsigned int constant.
     *
     * @param constants - Address of pointer in which to return scramble constants
     *
     * @return
     *
     * CURAND_STATUS_SUCCESS if the pointer was set successfully
     * </pre>
     */
    public static int curandGetScrambleConstants32(int[][] constants)
    {
        return checkResult(curandGetScrambleConstants32Native(constants));
    }
    private native static int curandGetScrambleConstants32Native(int[][] constants);

    /**
     * <pre>
     * Get direction vectors for 64-bit quasirandom number generation.
     *
     * Get a pointer to an array of direction vectors that can be used
     * for quasirandom number generation.  The resulting pointer will
     * reference an array of direction vectors in host memory.
     *
     * The array contains vectors for many dimensions.  Each dimension
     * has 64 vectors.  Each individual vector is an unsigned long long.
     *
     * Legal values for set are:
     * - CURAND_DIRECTION_VECTORS_64_JOEKUO6 (20,000 dimensions)
     * - CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 (20,000 dimensions)
     *
     * @param vectors - Address of pointer in which to return direction vectors
     * @param set - Which set of direction vectors to use
     *
     * @return
     *
     * CURAND_STATUS_OUT_OF_RANGE if the choice of set is invalid
     * CURAND_STATUS_SUCCESS if the pointer was set successfully
     * </pre>
     */
    public static int curandGetDirectionVectors64(long[][][] vectors, int set)
    {
        return checkResult(curandGetDirectionVectors64Native(vectors, set));
    }
    private native static int curandGetDirectionVectors64Native(long[][][] vectors, int set);

    /**
     * <pre>
     * Get scramble constants for 64-bit scrambled Sobol' .
     *
     * Get a pointer to an array of scramble constants that can be used
     * for quasirandom number generation.  The resulting pointer will
     * reference an array of unsinged long longs in host memory.
     *
     * The array contains constants for many dimensions.  Each dimension
     * has a single unsigned long long constant.
     *
     * @param constants - Address of pointer in which to return scramble constants
     *
     * @return
     *
     * CURAND_STATUS_SUCCESS if the pointer was set successfully
     * </pre>
     */
    public static int curandGetScrambleConstants64(long[][] constants)
    {
        return checkResult(curandGetScrambleConstants64Native(constants));
    }
    private native static int curandGetScrambleConstants64Native(long[][] constants);


}
