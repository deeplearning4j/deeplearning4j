/*
 * JCublas - Java bindings for CUBLAS, the NVIDIA CUDA BLAS library,
 * to be used with JCuda
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

package jcuda.jcublas;

import jcuda.*;
import jcuda.runtime.cudaStream_t;

/**
 * Java bindings for CUBLAS, the NVIDIA CUDA BLAS library.
 * <br />
 * This class contains the new CUBLAS API that was introduced
 * with CUDA 4.0, defined in the C header "cublas_v2.h".<br />
 * <br />
 * Most comments are taken from the CUBLAS header file.
 * <br />
 */
public class JCublas2
{
    /**
     * The flag that indicates whether the native library has been
     * loaded
     */
    private static boolean initialized = false;

    /**
     * Whether a CudaException should be thrown if a method is about
     * to set a result code that is not cublasStatus.CUBLAS_STATUS_SUCCESS
     */
    private static boolean exceptionsEnabled = false;

    /* Private constructor to prevent instantiation */
    private JCublas2()
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
            LibUtils.loadLibrary("JCublas2");
            initialized = true;
        }
    }



    /**
     * Set the specified log level for the JCublas library.<br />
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
     * only return the {@link cublasStatus} from the native method.
     * If exceptions are enabled, a CudaException with a detailed error
     * message will be thrown if a method is about to return a result code
     * that is not cublasStatus.CUBLAS_STATUS_SUCCESS
     *
     * @param enabled Whether exceptions are enabled
     */
    public static void setExceptionsEnabled(boolean enabled)
    {
        exceptionsEnabled = enabled;
    }

    /**
     * If the given result is different to cublasStatus.CUBLAS_STATUS_SUCCESS
     * and exceptions have been enabled, this method will throw a
     * CudaException with an error message that corresponds to the
     * given result code. Otherwise, the given result is simply
     * returned.
     *
     * @param result The result to check
     * @return The result that was given as the parameter
     * @throws CudaException If exceptions have been enabled and
     * the given result code is not cublasStatus.CUBLAS_STATUS_SUCCESS
     */
    private static int checkResult(int result)
    {
        if (exceptionsEnabled && result != cublasStatus.CUBLAS_STATUS_SUCCESS)
        {
            throw new CudaException(cublasStatus.stringFor(result));
        }
        return result;
    }

    //=== Auto-generated part ================================================

    public static int cublasCreate(
        cublasHandle handle)
    {
        return checkResult(cublasCreateNative(handle));
    }
    private static native int cublasCreateNative(
        cublasHandle handle);


    public static int cublasDestroy(
        cublasHandle handle)
    {
        return checkResult(cublasDestroyNative(handle));
    }
    private static native int cublasDestroyNative(
        cublasHandle handle);


    public static int cublasGetVersion(
        cublasHandle handle,
        int[] version)
    {
        return checkResult(cublasGetVersionNative(handle, version));
    }
    private static native int cublasGetVersionNative(
        cublasHandle handle,
        int[] version);


    public static int cublasSetStream(
        cublasHandle handle,
        cudaStream_t streamId)
    {
        return checkResult(cublasSetStreamNative(handle, streamId));
    }
    private static native int cublasSetStreamNative(
        cublasHandle handle,
        cudaStream_t streamId);


    public static int cublasGetStream(
        cublasHandle handle,
        cudaStream_t streamId)
    {
        return checkResult(cublasGetStreamNative(handle, streamId));
    }
    private static native int cublasGetStreamNative(
        cublasHandle handle,
        cudaStream_t streamId);


    public static int cublasGetPointerMode(
        cublasHandle handle,
        int[] mode)
    {
        return checkResult(cublasGetPointerModeNative(handle, mode));
    }
    private static native int cublasGetPointerModeNative(
        cublasHandle handle,
        int[] mode);


    public static int cublasSetPointerMode(
        cublasHandle handle,
        int mode)
    {
        return checkResult(cublasSetPointerModeNative(handle, mode));
    }
    private static native int cublasSetPointerModeNative(
        cublasHandle handle,
        int mode);


    public static int cublasGetAtomicsMode(
        cublasHandle handle,
        int[] mode)
    {
        return checkResult(cublasGetAtomicsModeNative(handle, mode));
    }
    private static native int cublasGetAtomicsModeNative(
        cublasHandle handle,
        int[] mode);


    public static int cublasSetAtomicsMode(
        cublasHandle handle,
        int mode)
    {
        return checkResult(cublasSetAtomicsModeNative(handle, mode));
    }
    private static native int cublasSetAtomicsModeNative(
        cublasHandle handle,
        int mode);


    /**
     * <pre>
     * cublasStatus_t
     * cublasSetVector (int n, int elemSize, const void *x, int incx,
     *                  void *y, int incy)
     *
     * copies n elements from a vector x in CPU memory space to a vector y
     * in GPU memory space. Elements in both vectors are assumed to have a
     * size of elemSize bytes. Storage spacing between consecutive elements
     * is incx for the source vector x and incy for the destination vector
     * y. In general, y points to an object, or part of an object, allocated
     * via cublasAlloc(). Column major format for two-dimensional matrices
     * is assumed throughout CUBLAS. Therefore, if the increment for a vector
     * is equal to 1, this access a column vector while using an increment
     * equal to the leading dimension of the respective matrix accesses a
     * row vector.
     *
     * Return Values
     * -------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if incx, incy, or elemSize <= 0
     * CUBLAS_STATUS_MAPPING_ERROR    if an error occurred accessing GPU memory
     * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
     * </pre>
     */
    public static int cublasSetVector(
        int n,
        int elemSize,
        Pointer x,
        int incx,
        Pointer devicePtr,
        int incy)
    {
        return checkResult(cublasSetVectorNative(n, elemSize, x, incx, devicePtr, incy));
    }
    private static native int cublasSetVectorNative(
        int n,
        int elemSize,
        Pointer x,
        int incx,
        Pointer devicePtr,
        int incy);


    /**
     * <pre>
     * cublasStatus_t
     * cublasGetVector (int n, int elemSize, const void *x, int incx,
     *                  void *y, int incy)
     *
     * copies n elements from a vector x in GPU memory space to a vector y
     * in CPU memory space. Elements in both vectors are assumed to have a
     * size of elemSize bytes. Storage spacing between consecutive elements
     * is incx for the source vector x and incy for the destination vector
     * y. In general, x points to an object, or part of an object, allocated
     * via cublasAlloc(). Column major format for two-dimensional matrices
     * is assumed throughout CUBLAS. Therefore, if the increment for a vector
     * is equal to 1, this access a column vector while using an increment
     * equal to the leading dimension of the respective matrix accesses a
     * row vector.
     *
     * Return Values
     * -------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if incx, incy, or elemSize <= 0
     * CUBLAS_STATUS_MAPPING_ERROR    if an error occurred accessing GPU memory
     * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
     * </pre>
     */
    public static int cublasGetVector(
        int n,
        int elemSize,
        Pointer x,
        int incx,
        Pointer y,
        int incy)
    {
        return checkResult(cublasGetVectorNative(n, elemSize, x, incx, y, incy));
    }
    private static native int cublasGetVectorNative(
        int n,
        int elemSize,
        Pointer x,
        int incx,
        Pointer y,
        int incy);


    /**
     * <pre>
     * cublasStatus_t
     * cublasSetMatrix (int rows, int cols, int elemSize, const void *A,
     *                  int lda, void *B, int ldb)
     *
     * copies a tile of rows x cols elements from a matrix A in CPU memory
     * space to a matrix B in GPU memory space. Each element requires storage
     * of elemSize bytes. Both matrices are assumed to be stored in column
     * major format, with the leading dimension (i.e. number of rows) of
     * source matrix A provided in lda, and the leading dimension of matrix B
     * provided in ldb. In general, B points to an object, or part of an
     * object, that was allocated via cublasAlloc().
     *
     * Return Values
     * -------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if rows or cols < 0, or elemSize, lda, or
     *                                ldb <= 0
     * CUBLAS_STATUS_MAPPING_ERROR    if error occurred accessing GPU memory
     * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
     * </pre>
     */
    public static int cublasSetMatrix(
        int rows,
        int cols,
        int elemSize,
        Pointer A,
        int lda,
        Pointer B,
        int ldb)
    {
        return checkResult(cublasSetMatrixNative(rows, cols, elemSize, A, lda, B, ldb));
    }
    private static native int cublasSetMatrixNative(
        int rows,
        int cols,
        int elemSize,
        Pointer A,
        int lda,
        Pointer B,
        int ldb);


    /**
     * <pre>
     * cublasStatus_t
     * cublasGetMatrix (int rows, int cols, int elemSize, const void *A,
     *                  int lda, void *B, int ldb)
     *
     * copies a tile of rows x cols elements from a matrix A in GPU memory
     * space to a matrix B in CPU memory space. Each element requires storage
     * of elemSize bytes. Both matrices are assumed to be stored in column
     * major format, with the leading dimension (i.e. number of rows) of
     * source matrix A provided in lda, and the leading dimension of matrix B
     * provided in ldb. In general, A points to an object, or part of an
     * object, that was allocated via cublasAlloc().
     *
     * Return Values
     * -------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if rows, cols, eleSize, lda, or ldb <= 0
     * CUBLAS_STATUS_MAPPING_ERROR    if error occurred accessing GPU memory
     * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
     * </pre>
     */
    public static int cublasGetMatrix(
        int rows,
        int cols,
        int elemSize,
        Pointer A,
        int lda,
        Pointer B,
        int ldb)
    {
        return checkResult(cublasGetMatrixNative(rows, cols, elemSize, A, lda, B, ldb));
    }
    private static native int cublasGetMatrixNative(
        int rows,
        int cols,
        int elemSize,
        Pointer A,
        int lda,
        Pointer B,
        int ldb);


    /**
     * <pre>
     * cublasStatus
     * cublasSetVectorAsync ( int n, int elemSize, const void *x, int incx,
     *                       void *y, int incy, cudaStream_t stream );
     *
     * cublasSetVectorAsync has the same functionnality as cublasSetVector
     * but the transfer is done asynchronously within the CUDA stream passed
     * in parameter.
     *
     * Return Values
     * -------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if incx, incy, or elemSize <= 0
     * CUBLAS_STATUS_MAPPING_ERROR    if an error occurred accessing GPU memory
     * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
     * </pre>
     */
    public static int cublasSetVectorAsync(
        int n,
        int elemSize,
        Pointer hostPtr,
        int incx,
        Pointer devicePtr,
        int incy,
        cudaStream_t stream)
    {
        return checkResult(cublasSetVectorAsyncNative(n, elemSize, hostPtr, incx, devicePtr, incy, stream));
    }
    private static native int cublasSetVectorAsyncNative(
        int n,
        int elemSize,
        Pointer hostPtr,
        int incx,
        Pointer devicePtr,
        int incy,
        cudaStream_t stream);


    /**
     * <pre>
     * cublasStatus
     * cublasGetVectorAsync( int n, int elemSize, const void *x, int incx,
     *                       void *y, int incy, cudaStream_t stream)
     *
     * cublasGetVectorAsync has the same functionnality as cublasGetVector
     * but the transfer is done asynchronously within the CUDA stream passed
     * in parameter.
     *
     * Return Values
     * -------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if incx, incy, or elemSize <= 0
     * CUBLAS_STATUS_MAPPING_ERROR    if an error occurred accessing GPU memory
     * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
     * </pre>
     */
    public static int cublasGetVectorAsync(
        int n,
        int elemSize,
        Pointer devicePtr,
        int incx,
        Pointer hostPtr,
        int incy,
        cudaStream_t stream)
    {
        return checkResult(cublasGetVectorAsyncNative(n, elemSize, devicePtr, incx, hostPtr, incy, stream));
    }
    private static native int cublasGetVectorAsyncNative(
        int n,
        int elemSize,
        Pointer devicePtr,
        int incx,
        Pointer hostPtr,
        int incy,
        cudaStream_t stream);


    /**
     * <pre>
     * cublasStatus_t
     * cublasSetMatrixAsync (int rows, int cols, int elemSize, const void *A,
     *                       int lda, void *B, int ldb, cudaStream_t stream)
     *
     * cublasSetMatrixAsync has the same functionnality as cublasSetMatrix
     * but the transfer is done asynchronously within the CUDA stream passed
     * in parameter.
     *
     * Return Values
     * -------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if rows or cols < 0, or elemSize, lda, or
     *                                ldb <= 0
     * CUBLAS_STATUS_MAPPING_ERROR    if error occurred accessing GPU memory
     * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
     * </pre>
     */
    public static int cublasSetMatrixAsync(
        int rows,
        int cols,
        int elemSize,
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        cudaStream_t stream)
    {
        return checkResult(cublasSetMatrixAsyncNative(rows, cols, elemSize, A, lda, B, ldb, stream));
    }
    private static native int cublasSetMatrixAsyncNative(
        int rows,
        int cols,
        int elemSize,
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        cudaStream_t stream);


    /**
     * <pre>
     * cublasStatus_t
     * cublasGetMatrixAsync (int rows, int cols, int elemSize, const void *A,
     *                       int lda, void *B, int ldb, cudaStream_t stream)
     *
     * cublasGetMatrixAsync has the same functionnality as cublasGetMatrix
     * but the transfer is done asynchronously within the CUDA stream passed
     * in parameter.
     *
     * Return Values
     * -------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if rows, cols, eleSize, lda, or ldb <= 0
     * CUBLAS_STATUS_MAPPING_ERROR    if error occurred accessing GPU memory
     * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
     * </pre>
     */
    public static int cublasGetMatrixAsync(
        int rows,
        int cols,
        int elemSize,
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        cudaStream_t stream)
    {
        return checkResult(cublasGetMatrixAsyncNative(rows, cols, elemSize, A, lda, B, ldb, stream));
    }
    private static native int cublasGetMatrixAsyncNative(
        int rows,
        int cols,
        int elemSize,
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        cudaStream_t stream);


    public static int cublasSnrm2(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer result)/** host or device pointer */
    {
        return checkResult(cublasSnrm2Native(handle, n, x, incx, result));
    }
    private static native int cublasSnrm2Native(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer result);/** host or device pointer */


    public static int cublasDnrm2(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer result)/** host or device pointer */
    {
        return checkResult(cublasDnrm2Native(handle, n, x, incx, result));
    }
    private static native int cublasDnrm2Native(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer result);/** host or device pointer */


    public static int cublasScnrm2(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer result)/** host or device pointer */
    {
        return checkResult(cublasScnrm2Native(handle, n, x, incx, result));
    }
    private static native int cublasScnrm2Native(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer result);/** host or device pointer */


    public static int cublasDznrm2(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer result)/** host or device pointer */
    {
        return checkResult(cublasDznrm2Native(handle, n, x, incx, result));
    }
    private static native int cublasDznrm2Native(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer result);/** host or device pointer */


    public static int cublasSdot(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer result)/** host or device pointer */
    {
        return checkResult(cublasSdotNative(handle, n, x, incx, y, incy, result));
    }
    private static native int cublasSdotNative(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer result);/** host or device pointer */


    public static int cublasDdot(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer result)/** host or device pointer */
    {
        return checkResult(cublasDdotNative(handle, n, x, incx, y, incy, result));
    }
    private static native int cublasDdotNative(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer result);/** host or device pointer */


    public static int cublasCdotu(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer result)/** host or device pointer */
    {
        return checkResult(cublasCdotuNative(handle, n, x, incx, y, incy, result));
    }
    private static native int cublasCdotuNative(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer result);/** host or device pointer */


    public static int cublasCdotc(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer result)/** host or device pointer */
    {
        return checkResult(cublasCdotcNative(handle, n, x, incx, y, incy, result));
    }
    private static native int cublasCdotcNative(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer result);/** host or device pointer */


    public static int cublasZdotu(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer result)/** host or device pointer */
    {
        return checkResult(cublasZdotuNative(handle, n, x, incx, y, incy, result));
    }
    private static native int cublasZdotuNative(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer result);/** host or device pointer */


    public static int cublasZdotc(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer result)/** host or device pointer */
    {
        return checkResult(cublasZdotcNative(handle, n, x, incx, y, incy, result));
    }
    private static native int cublasZdotcNative(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer result);/** host or device pointer */


    public static int cublasSscal(
        cublasHandle handle,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx)
    {
        return checkResult(cublasSscalNative(handle, n, alpha, x, incx));
    }
    private static native int cublasSscalNative(
        cublasHandle handle,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx);


    public static int cublasDscal(
        cublasHandle handle,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx)
    {
        return checkResult(cublasDscalNative(handle, n, alpha, x, incx));
    }
    private static native int cublasDscalNative(
        cublasHandle handle,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx);


    public static int cublasCscal(
        cublasHandle handle,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx)
    {
        return checkResult(cublasCscalNative(handle, n, alpha, x, incx));
    }
    private static native int cublasCscalNative(
        cublasHandle handle,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx);


    public static int cublasCsscal(
        cublasHandle handle,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx)
    {
        return checkResult(cublasCsscalNative(handle, n, alpha, x, incx));
    }
    private static native int cublasCsscalNative(
        cublasHandle handle,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx);


    public static int cublasZscal(
        cublasHandle handle,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx)
    {
        return checkResult(cublasZscalNative(handle, n, alpha, x, incx));
    }
    private static native int cublasZscalNative(
        cublasHandle handle,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx);


    public static int cublasZdscal(
        cublasHandle handle,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx)
    {
        return checkResult(cublasZdscalNative(handle, n, alpha, x, incx));
    }
    private static native int cublasZdscalNative(
        cublasHandle handle,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx);


    public static int cublasSaxpy(
        cublasHandle handle,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy)
    {
        return checkResult(cublasSaxpyNative(handle, n, alpha, x, incx, y, incy));
    }
    private static native int cublasSaxpyNative(
        cublasHandle handle,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy);


    public static int cublasDaxpy(
        cublasHandle handle,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy)
    {
        return checkResult(cublasDaxpyNative(handle, n, alpha, x, incx, y, incy));
    }
    private static native int cublasDaxpyNative(
        cublasHandle handle,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy);


    public static int cublasCaxpy(
        cublasHandle handle,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy)
    {
        return checkResult(cublasCaxpyNative(handle, n, alpha, x, incx, y, incy));
    }
    private static native int cublasCaxpyNative(
        cublasHandle handle,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy);


    public static int cublasZaxpy(
        cublasHandle handle,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy)
    {
        return checkResult(cublasZaxpyNative(handle, n, alpha, x, incx, y, incy));
    }
    private static native int cublasZaxpyNative(
        cublasHandle handle,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy);


    public static int cublasScopy(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy)
    {
        return checkResult(cublasScopyNative(handle, n, x, incx, y, incy));
    }
    private static native int cublasScopyNative(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy);


    public static int cublasDcopy(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy)
    {
        return checkResult(cublasDcopyNative(handle, n, x, incx, y, incy));
    }
    private static native int cublasDcopyNative(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy);


    public static int cublasCcopy(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy)
    {
        return checkResult(cublasCcopyNative(handle, n, x, incx, y, incy));
    }
    private static native int cublasCcopyNative(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy);


    public static int cublasZcopy(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy)
    {
        return checkResult(cublasZcopyNative(handle, n, x, incx, y, incy));
    }
    private static native int cublasZcopyNative(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy);


    public static int cublasSswap(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy)
    {
        return checkResult(cublasSswapNative(handle, n, x, incx, y, incy));
    }
    private static native int cublasSswapNative(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy);


    public static int cublasDswap(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy)
    {
        return checkResult(cublasDswapNative(handle, n, x, incx, y, incy));
    }
    private static native int cublasDswapNative(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy);


    public static int cublasCswap(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy)
    {
        return checkResult(cublasCswapNative(handle, n, x, incx, y, incy));
    }
    private static native int cublasCswapNative(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy);


    public static int cublasZswap(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy)
    {
        return checkResult(cublasZswapNative(handle, n, x, incx, y, incy));
    }
    private static native int cublasZswapNative(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy);


    public static int cublasIsamax(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer result)/** host or device pointer */
    {
        return checkResult(cublasIsamaxNative(handle, n, x, incx, result));
    }
    private static native int cublasIsamaxNative(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer result);/** host or device pointer */


    public static int cublasIdamax(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer result)/** host or device pointer */
    {
        return checkResult(cublasIdamaxNative(handle, n, x, incx, result));
    }
    private static native int cublasIdamaxNative(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer result);/** host or device pointer */


    public static int cublasIcamax(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer result)/** host or device pointer */
    {
        return checkResult(cublasIcamaxNative(handle, n, x, incx, result));
    }
    private static native int cublasIcamaxNative(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer result);/** host or device pointer */


    public static int cublasIzamax(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer result)/** host or device pointer */
    {
        return checkResult(cublasIzamaxNative(handle, n, x, incx, result));
    }
    private static native int cublasIzamaxNative(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer result);/** host or device pointer */


    public static int cublasIsamin(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer result)/** host or device pointer */
    {
        return checkResult(cublasIsaminNative(handle, n, x, incx, result));
    }
    private static native int cublasIsaminNative(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer result);/** host or device pointer */


    public static int cublasIdamin(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer result)/** host or device pointer */
    {
        return checkResult(cublasIdaminNative(handle, n, x, incx, result));
    }
    private static native int cublasIdaminNative(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer result);/** host or device pointer */


    public static int cublasIcamin(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer result)/** host or device pointer */
    {
        return checkResult(cublasIcaminNative(handle, n, x, incx, result));
    }
    private static native int cublasIcaminNative(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer result);/** host or device pointer */


    public static int cublasIzamin(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer result)/** host or device pointer */
    {
        return checkResult(cublasIzaminNative(handle, n, x, incx, result));
    }
    private static native int cublasIzaminNative(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer result);/** host or device pointer */


    public static int cublasSasum(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer result)/** host or device pointer */
    {
        return checkResult(cublasSasumNative(handle, n, x, incx, result));
    }
    private static native int cublasSasumNative(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer result);/** host or device pointer */


    public static int cublasDasum(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer result)/** host or device pointer */
    {
        return checkResult(cublasDasumNative(handle, n, x, incx, result));
    }
    private static native int cublasDasumNative(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer result);/** host or device pointer */


    public static int cublasScasum(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer result)/** host or device pointer */
    {
        return checkResult(cublasScasumNative(handle, n, x, incx, result));
    }
    private static native int cublasScasumNative(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer result);/** host or device pointer */


    public static int cublasDzasum(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer result)/** host or device pointer */
    {
        return checkResult(cublasDzasumNative(handle, n, x, incx, result));
    }
    private static native int cublasDzasumNative(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer result);/** host or device pointer */


    public static int cublasSrot(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer c, /** host or device pointer */
        Pointer s)/** host or device pointer */
    {
        return checkResult(cublasSrotNative(handle, n, x, incx, y, incy, c, s));
    }
    private static native int cublasSrotNative(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer c, /** host or device pointer */
        Pointer s);/** host or device pointer */


    public static int cublasDrot(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer c, /** host or device pointer */
        Pointer s)/** host or device pointer */
    {
        return checkResult(cublasDrotNative(handle, n, x, incx, y, incy, c, s));
    }
    private static native int cublasDrotNative(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer c, /** host or device pointer */
        Pointer s);/** host or device pointer */


    public static int cublasCrot(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer c, /** host or device pointer */
        Pointer s)/** host or device pointer */
    {
        return checkResult(cublasCrotNative(handle, n, x, incx, y, incy, c, s));
    }
    private static native int cublasCrotNative(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer c, /** host or device pointer */
        Pointer s);/** host or device pointer */


    public static int cublasCsrot(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer c, /** host or device pointer */
        Pointer s)/** host or device pointer */
    {
        return checkResult(cublasCsrotNative(handle, n, x, incx, y, incy, c, s));
    }
    private static native int cublasCsrotNative(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer c, /** host or device pointer */
        Pointer s);/** host or device pointer */


    public static int cublasZrot(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer c, /** host or device pointer */
        Pointer s)/** host or device pointer */
    {
        return checkResult(cublasZrotNative(handle, n, x, incx, y, incy, c, s));
    }
    private static native int cublasZrotNative(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer c, /** host or device pointer */
        Pointer s);/** host or device pointer */


    public static int cublasZdrot(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer c, /** host or device pointer */
        Pointer s)/** host or device pointer */
    {
        return checkResult(cublasZdrotNative(handle, n, x, incx, y, incy, c, s));
    }
    private static native int cublasZdrotNative(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer c, /** host or device pointer */
        Pointer s);/** host or device pointer */


    public static int cublasSrotg(
        cublasHandle handle,
        Pointer a, /** host or device pointer */
        Pointer b, /** host or device pointer */
        Pointer c, /** host or device pointer */
        Pointer s)/** host or device pointer */
    {
        return checkResult(cublasSrotgNative(handle, a, b, c, s));
    }
    private static native int cublasSrotgNative(
        cublasHandle handle,
        Pointer a, /** host or device pointer */
        Pointer b, /** host or device pointer */
        Pointer c, /** host or device pointer */
        Pointer s);/** host or device pointer */


    public static int cublasDrotg(
        cublasHandle handle,
        Pointer a, /** host or device pointer */
        Pointer b, /** host or device pointer */
        Pointer c, /** host or device pointer */
        Pointer s)/** host or device pointer */
    {
        return checkResult(cublasDrotgNative(handle, a, b, c, s));
    }
    private static native int cublasDrotgNative(
        cublasHandle handle,
        Pointer a, /** host or device pointer */
        Pointer b, /** host or device pointer */
        Pointer c, /** host or device pointer */
        Pointer s);/** host or device pointer */


    public static int cublasCrotg(
        cublasHandle handle,
        Pointer a, /** host or device pointer */
        Pointer b, /** host or device pointer */
        Pointer c, /** host or device pointer */
        Pointer s)/** host or device pointer */
    {
        return checkResult(cublasCrotgNative(handle, a, b, c, s));
    }
    private static native int cublasCrotgNative(
        cublasHandle handle,
        Pointer a, /** host or device pointer */
        Pointer b, /** host or device pointer */
        Pointer c, /** host or device pointer */
        Pointer s);/** host or device pointer */


    public static int cublasZrotg(
        cublasHandle handle,
        Pointer a, /** host or device pointer */
        Pointer b, /** host or device pointer */
        Pointer c, /** host or device pointer */
        Pointer s)/** host or device pointer */
    {
        return checkResult(cublasZrotgNative(handle, a, b, c, s));
    }
    private static native int cublasZrotgNative(
        cublasHandle handle,
        Pointer a, /** host or device pointer */
        Pointer b, /** host or device pointer */
        Pointer c, /** host or device pointer */
        Pointer s);/** host or device pointer */


    public static int cublasSrotm(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer param)/** host or device pointer */
    {
        return checkResult(cublasSrotmNative(handle, n, x, incx, y, incy, param));
    }
    private static native int cublasSrotmNative(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer param);/** host or device pointer */


    public static int cublasDrotm(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer param)/** host or device pointer */
    {
        return checkResult(cublasDrotmNative(handle, n, x, incx, y, incy, param));
    }
    private static native int cublasDrotmNative(
        cublasHandle handle,
        int n,
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer param);/** host or device pointer */


    public static int cublasSrotmg(
        cublasHandle handle,
        Pointer d1, /** host or device pointer */
        Pointer d2, /** host or device pointer */
        Pointer x1, /** host or device pointer */
        Pointer y1, /** host or device pointer */
        Pointer param)/** host or device pointer */
    {
        return checkResult(cublasSrotmgNative(handle, d1, d2, x1, y1, param));
    }
    private static native int cublasSrotmgNative(
        cublasHandle handle,
        Pointer d1, /** host or device pointer */
        Pointer d2, /** host or device pointer */
        Pointer x1, /** host or device pointer */
        Pointer y1, /** host or device pointer */
        Pointer param);/** host or device pointer */


    public static int cublasDrotmg(
        cublasHandle handle,
        Pointer d1, /** host or device pointer */
        Pointer d2, /** host or device pointer */
        Pointer x1, /** host or device pointer */
        Pointer y1, /** host or device pointer */
        Pointer param)/** host or device pointer */
    {
        return checkResult(cublasDrotmgNative(handle, d1, d2, x1, y1, param));
    }
    private static native int cublasDrotmgNative(
        cublasHandle handle,
        Pointer d1, /** host or device pointer */
        Pointer d2, /** host or device pointer */
        Pointer x1, /** host or device pointer */
        Pointer y1, /** host or device pointer */
        Pointer param);/** host or device pointer */


    public static int cublasSgemv(
        cublasHandle handle,
        int trans,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy)
    {
        return checkResult(cublasSgemvNative(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasSgemvNative(
        cublasHandle handle,
        int trans,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy);


    public static int cublasDgemv(
        cublasHandle handle,
        int trans,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy)
    {
        return checkResult(cublasDgemvNative(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasDgemvNative(
        cublasHandle handle,
        int trans,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy);


    public static int cublasCgemv(
        cublasHandle handle,
        int trans,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy)
    {
        return checkResult(cublasCgemvNative(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasCgemvNative(
        cublasHandle handle,
        int trans,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy);


    public static int cublasZgemv(
        cublasHandle handle,
        int trans,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy)
    {
        return checkResult(cublasZgemvNative(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasZgemvNative(
        cublasHandle handle,
        int trans,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy);


    public static int cublasSgbmv(
        cublasHandle handle,
        int trans,
        int m,
        int n,
        int kl,
        int ku,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy)
    {
        return checkResult(cublasSgbmvNative(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasSgbmvNative(
        cublasHandle handle,
        int trans,
        int m,
        int n,
        int kl,
        int ku,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy);


    public static int cublasDgbmv(
        cublasHandle handle,
        int trans,
        int m,
        int n,
        int kl,
        int ku,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy)
    {
        return checkResult(cublasDgbmvNative(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasDgbmvNative(
        cublasHandle handle,
        int trans,
        int m,
        int n,
        int kl,
        int ku,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy);


    public static int cublasCgbmv(
        cublasHandle handle,
        int trans,
        int m,
        int n,
        int kl,
        int ku,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy)
    {
        return checkResult(cublasCgbmvNative(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasCgbmvNative(
        cublasHandle handle,
        int trans,
        int m,
        int n,
        int kl,
        int ku,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy);


    public static int cublasZgbmv(
        cublasHandle handle,
        int trans,
        int m,
        int n,
        int kl,
        int ku,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy)
    {
        return checkResult(cublasZgbmvNative(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasZgbmvNative(
        cublasHandle handle,
        int trans,
        int m,
        int n,
        int kl,
        int ku,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy);


    public static int cublasStrmv(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        Pointer A,
        int lda,
        Pointer x,
        int incx)
    {
        return checkResult(cublasStrmvNative(handle, uplo, trans, diag, n, A, lda, x, incx));
    }
    private static native int cublasStrmvNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        Pointer A,
        int lda,
        Pointer x,
        int incx);


    public static int cublasDtrmv(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        Pointer A,
        int lda,
        Pointer x,
        int incx)
    {
        return checkResult(cublasDtrmvNative(handle, uplo, trans, diag, n, A, lda, x, incx));
    }
    private static native int cublasDtrmvNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        Pointer A,
        int lda,
        Pointer x,
        int incx);


    public static int cublasCtrmv(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        Pointer A,
        int lda,
        Pointer x,
        int incx)
    {
        return checkResult(cublasCtrmvNative(handle, uplo, trans, diag, n, A, lda, x, incx));
    }
    private static native int cublasCtrmvNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        Pointer A,
        int lda,
        Pointer x,
        int incx);


    public static int cublasZtrmv(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        Pointer A,
        int lda,
        Pointer x,
        int incx)
    {
        return checkResult(cublasZtrmvNative(handle, uplo, trans, diag, n, A, lda, x, incx));
    }
    private static native int cublasZtrmvNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        Pointer A,
        int lda,
        Pointer x,
        int incx);


    public static int cublasStbmv(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        int k,
        Pointer A,
        int lda,
        Pointer x,
        int incx)
    {
        return checkResult(cublasStbmvNative(handle, uplo, trans, diag, n, k, A, lda, x, incx));
    }
    private static native int cublasStbmvNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        int k,
        Pointer A,
        int lda,
        Pointer x,
        int incx);


    public static int cublasDtbmv(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        int k,
        Pointer A,
        int lda,
        Pointer x,
        int incx)
    {
        return checkResult(cublasDtbmvNative(handle, uplo, trans, diag, n, k, A, lda, x, incx));
    }
    private static native int cublasDtbmvNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        int k,
        Pointer A,
        int lda,
        Pointer x,
        int incx);


    public static int cublasCtbmv(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        int k,
        Pointer A,
        int lda,
        Pointer x,
        int incx)
    {
        return checkResult(cublasCtbmvNative(handle, uplo, trans, diag, n, k, A, lda, x, incx));
    }
    private static native int cublasCtbmvNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        int k,
        Pointer A,
        int lda,
        Pointer x,
        int incx);


    public static int cublasZtbmv(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        int k,
        Pointer A,
        int lda,
        Pointer x,
        int incx)
    {
        return checkResult(cublasZtbmvNative(handle, uplo, trans, diag, n, k, A, lda, x, incx));
    }
    private static native int cublasZtbmvNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        int k,
        Pointer A,
        int lda,
        Pointer x,
        int incx);


    public static int cublasStpmv(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        Pointer AP,
        Pointer x,
        int incx)
    {
        return checkResult(cublasStpmvNative(handle, uplo, trans, diag, n, AP, x, incx));
    }
    private static native int cublasStpmvNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        Pointer AP,
        Pointer x,
        int incx);


    public static int cublasDtpmv(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        Pointer AP,
        Pointer x,
        int incx)
    {
        return checkResult(cublasDtpmvNative(handle, uplo, trans, diag, n, AP, x, incx));
    }
    private static native int cublasDtpmvNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        Pointer AP,
        Pointer x,
        int incx);


    public static int cublasCtpmv(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        Pointer AP,
        Pointer x,
        int incx)
    {
        return checkResult(cublasCtpmvNative(handle, uplo, trans, diag, n, AP, x, incx));
    }
    private static native int cublasCtpmvNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        Pointer AP,
        Pointer x,
        int incx);


    public static int cublasZtpmv(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        Pointer AP,
        Pointer x,
        int incx)
    {
        return checkResult(cublasZtpmvNative(handle, uplo, trans, diag, n, AP, x, incx));
    }
    private static native int cublasZtpmvNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        Pointer AP,
        Pointer x,
        int incx);


    public static int cublasStrsv(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        Pointer A,
        int lda,
        Pointer x,
        int incx)
    {
        return checkResult(cublasStrsvNative(handle, uplo, trans, diag, n, A, lda, x, incx));
    }
    private static native int cublasStrsvNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        Pointer A,
        int lda,
        Pointer x,
        int incx);


    public static int cublasDtrsv(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        Pointer A,
        int lda,
        Pointer x,
        int incx)
    {
        return checkResult(cublasDtrsvNative(handle, uplo, trans, diag, n, A, lda, x, incx));
    }
    private static native int cublasDtrsvNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        Pointer A,
        int lda,
        Pointer x,
        int incx);


    public static int cublasCtrsv(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        Pointer A,
        int lda,
        Pointer x,
        int incx)
    {
        return checkResult(cublasCtrsvNative(handle, uplo, trans, diag, n, A, lda, x, incx));
    }
    private static native int cublasCtrsvNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        Pointer A,
        int lda,
        Pointer x,
        int incx);


    public static int cublasZtrsv(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        Pointer A,
        int lda,
        Pointer x,
        int incx)
    {
        return checkResult(cublasZtrsvNative(handle, uplo, trans, diag, n, A, lda, x, incx));
    }
    private static native int cublasZtrsvNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        Pointer A,
        int lda,
        Pointer x,
        int incx);


    public static int cublasStpsv(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        Pointer AP,
        Pointer x,
        int incx)
    {
        return checkResult(cublasStpsvNative(handle, uplo, trans, diag, n, AP, x, incx));
    }
    private static native int cublasStpsvNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        Pointer AP,
        Pointer x,
        int incx);


    public static int cublasDtpsv(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        Pointer AP,
        Pointer x,
        int incx)
    {
        return checkResult(cublasDtpsvNative(handle, uplo, trans, diag, n, AP, x, incx));
    }
    private static native int cublasDtpsvNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        Pointer AP,
        Pointer x,
        int incx);


    public static int cublasCtpsv(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        Pointer AP,
        Pointer x,
        int incx)
    {
        return checkResult(cublasCtpsvNative(handle, uplo, trans, diag, n, AP, x, incx));
    }
    private static native int cublasCtpsvNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        Pointer AP,
        Pointer x,
        int incx);


    public static int cublasZtpsv(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        Pointer AP,
        Pointer x,
        int incx)
    {
        return checkResult(cublasZtpsvNative(handle, uplo, trans, diag, n, AP, x, incx));
    }
    private static native int cublasZtpsvNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        Pointer AP,
        Pointer x,
        int incx);


    public static int cublasStbsv(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        int k,
        Pointer A,
        int lda,
        Pointer x,
        int incx)
    {
        return checkResult(cublasStbsvNative(handle, uplo, trans, diag, n, k, A, lda, x, incx));
    }
    private static native int cublasStbsvNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        int k,
        Pointer A,
        int lda,
        Pointer x,
        int incx);


    public static int cublasDtbsv(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        int k,
        Pointer A,
        int lda,
        Pointer x,
        int incx)
    {
        return checkResult(cublasDtbsvNative(handle, uplo, trans, diag, n, k, A, lda, x, incx));
    }
    private static native int cublasDtbsvNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        int k,
        Pointer A,
        int lda,
        Pointer x,
        int incx);


    public static int cublasCtbsv(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        int k,
        Pointer A,
        int lda,
        Pointer x,
        int incx)
    {
        return checkResult(cublasCtbsvNative(handle, uplo, trans, diag, n, k, A, lda, x, incx));
    }
    private static native int cublasCtbsvNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        int k,
        Pointer A,
        int lda,
        Pointer x,
        int incx);


    public static int cublasZtbsv(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        int k,
        Pointer A,
        int lda,
        Pointer x,
        int incx)
    {
        return checkResult(cublasZtbsvNative(handle, uplo, trans, diag, n, k, A, lda, x, incx));
    }
    private static native int cublasZtbsvNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int diag,
        int n,
        int k,
        Pointer A,
        int lda,
        Pointer x,
        int incx);


    public static int cublasSsymv(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy)
    {
        return checkResult(cublasSsymvNative(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasSsymvNative(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy);


    public static int cublasDsymv(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy)
    {
        return checkResult(cublasDsymvNative(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasDsymvNative(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy);


    public static int cublasCsymv(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy)
    {
        return checkResult(cublasCsymvNative(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasCsymvNative(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy);


    public static int cublasZsymv(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy)
    {
        return checkResult(cublasZsymvNative(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasZsymvNative(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy);


    public static int cublasChemv(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy)
    {
        return checkResult(cublasChemvNative(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasChemvNative(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy);


    public static int cublasZhemv(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy)
    {
        return checkResult(cublasZhemvNative(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasZhemvNative(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy);


    public static int cublasSsbmv(
        cublasHandle handle,
        int uplo,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy)
    {
        return checkResult(cublasSsbmvNative(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasSsbmvNative(
        cublasHandle handle,
        int uplo,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy);


    public static int cublasDsbmv(
        cublasHandle handle,
        int uplo,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy)
    {
        return checkResult(cublasDsbmvNative(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasDsbmvNative(
        cublasHandle handle,
        int uplo,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy);


    public static int cublasChbmv(
        cublasHandle handle,
        int uplo,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy)
    {
        return checkResult(cublasChbmvNative(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasChbmvNative(
        cublasHandle handle,
        int uplo,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy);


    public static int cublasZhbmv(
        cublasHandle handle,
        int uplo,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy)
    {
        return checkResult(cublasZhbmvNative(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasZhbmvNative(
        cublasHandle handle,
        int uplo,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy);


    public static int cublasSspmv(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer AP,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy)
    {
        return checkResult(cublasSspmvNative(handle, uplo, n, alpha, AP, x, incx, beta, y, incy));
    }
    private static native int cublasSspmvNative(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer AP,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy);


    public static int cublasDspmv(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer AP,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy)
    {
        return checkResult(cublasDspmvNative(handle, uplo, n, alpha, AP, x, incx, beta, y, incy));
    }
    private static native int cublasDspmvNative(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer AP,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy);


    public static int cublasChpmv(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer AP,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy)
    {
        return checkResult(cublasChpmvNative(handle, uplo, n, alpha, AP, x, incx, beta, y, incy));
    }
    private static native int cublasChpmvNative(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer AP,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy);


    public static int cublasZhpmv(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer AP,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy)
    {
        return checkResult(cublasZhpmvNative(handle, uplo, n, alpha, AP, x, incx, beta, y, incy));
    }
    private static native int cublasZhpmvNative(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer AP,
        Pointer x,
        int incx,
        Pointer beta, /** host or device pointer */
        Pointer y,
        int incy);


    public static int cublasSger(
        cublasHandle handle,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer A,
        int lda)
    {
        return checkResult(cublasSgerNative(handle, m, n, alpha, x, incx, y, incy, A, lda));
    }
    private static native int cublasSgerNative(
        cublasHandle handle,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer A,
        int lda);


    public static int cublasDger(
        cublasHandle handle,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer A,
        int lda)
    {
        return checkResult(cublasDgerNative(handle, m, n, alpha, x, incx, y, incy, A, lda));
    }
    private static native int cublasDgerNative(
        cublasHandle handle,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer A,
        int lda);


    public static int cublasCgeru(
        cublasHandle handle,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer A,
        int lda)
    {
        return checkResult(cublasCgeruNative(handle, m, n, alpha, x, incx, y, incy, A, lda));
    }
    private static native int cublasCgeruNative(
        cublasHandle handle,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer A,
        int lda);


    public static int cublasCgerc(
        cublasHandle handle,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer A,
        int lda)
    {
        return checkResult(cublasCgercNative(handle, m, n, alpha, x, incx, y, incy, A, lda));
    }
    private static native int cublasCgercNative(
        cublasHandle handle,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer A,
        int lda);


    public static int cublasZgeru(
        cublasHandle handle,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer A,
        int lda)
    {
        return checkResult(cublasZgeruNative(handle, m, n, alpha, x, incx, y, incy, A, lda));
    }
    private static native int cublasZgeruNative(
        cublasHandle handle,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer A,
        int lda);


    public static int cublasZgerc(
        cublasHandle handle,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer A,
        int lda)
    {
        return checkResult(cublasZgercNative(handle, m, n, alpha, x, incx, y, incy, A, lda));
    }
    private static native int cublasZgercNative(
        cublasHandle handle,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer A,
        int lda);


    public static int cublasSsyr(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer A,
        int lda)
    {
        return checkResult(cublasSsyrNative(handle, uplo, n, alpha, x, incx, A, lda));
    }
    private static native int cublasSsyrNative(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer A,
        int lda);


    public static int cublasDsyr(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer A,
        int lda)
    {
        return checkResult(cublasDsyrNative(handle, uplo, n, alpha, x, incx, A, lda));
    }
    private static native int cublasDsyrNative(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer A,
        int lda);


    public static int cublasCsyr(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer A,
        int lda)
    {
        return checkResult(cublasCsyrNative(handle, uplo, n, alpha, x, incx, A, lda));
    }
    private static native int cublasCsyrNative(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer A,
        int lda);


    public static int cublasZsyr(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer A,
        int lda)
    {
        return checkResult(cublasZsyrNative(handle, uplo, n, alpha, x, incx, A, lda));
    }
    private static native int cublasZsyrNative(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer A,
        int lda);


    public static int cublasCher(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer A,
        int lda)
    {
        return checkResult(cublasCherNative(handle, uplo, n, alpha, x, incx, A, lda));
    }
    private static native int cublasCherNative(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer A,
        int lda);


    public static int cublasZher(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer A,
        int lda)
    {
        return checkResult(cublasZherNative(handle, uplo, n, alpha, x, incx, A, lda));
    }
    private static native int cublasZherNative(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer A,
        int lda);


    public static int cublasSspr(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer AP)
    {
        return checkResult(cublasSsprNative(handle, uplo, n, alpha, x, incx, AP));
    }
    private static native int cublasSsprNative(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer AP);


    public static int cublasDspr(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer AP)
    {
        return checkResult(cublasDsprNative(handle, uplo, n, alpha, x, incx, AP));
    }
    private static native int cublasDsprNative(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer AP);


    public static int cublasChpr(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer AP)
    {
        return checkResult(cublasChprNative(handle, uplo, n, alpha, x, incx, AP));
    }
    private static native int cublasChprNative(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer AP);


    public static int cublasZhpr(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer AP)
    {
        return checkResult(cublasZhprNative(handle, uplo, n, alpha, x, incx, AP));
    }
    private static native int cublasZhprNative(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer AP);


    public static int cublasSsyr2(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer A,
        int lda)
    {
        return checkResult(cublasSsyr2Native(handle, uplo, n, alpha, x, incx, y, incy, A, lda));
    }
    private static native int cublasSsyr2Native(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer A,
        int lda);


    public static int cublasDsyr2(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer A,
        int lda)
    {
        return checkResult(cublasDsyr2Native(handle, uplo, n, alpha, x, incx, y, incy, A, lda));
    }
    private static native int cublasDsyr2Native(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer A,
        int lda);


    public static int cublasCsyr2(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer A,
        int lda)
    {
        return checkResult(cublasCsyr2Native(handle, uplo, n, alpha, x, incx, y, incy, A, lda));
    }
    private static native int cublasCsyr2Native(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer A,
        int lda);


    public static int cublasZsyr2(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer A,
        int lda)
    {
        return checkResult(cublasZsyr2Native(handle, uplo, n, alpha, x, incx, y, incy, A, lda));
    }
    private static native int cublasZsyr2Native(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer A,
        int lda);


    public static int cublasCher2(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer A,
        int lda)
    {
        return checkResult(cublasCher2Native(handle, uplo, n, alpha, x, incx, y, incy, A, lda));
    }
    private static native int cublasCher2Native(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer A,
        int lda);


    public static int cublasZher2(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer A,
        int lda)
    {
        return checkResult(cublasZher2Native(handle, uplo, n, alpha, x, incx, y, incy, A, lda));
    }
    private static native int cublasZher2Native(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer A,
        int lda);


    public static int cublasSspr2(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer AP)
    {
        return checkResult(cublasSspr2Native(handle, uplo, n, alpha, x, incx, y, incy, AP));
    }
    private static native int cublasSspr2Native(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer AP);


    public static int cublasDspr2(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer AP)
    {
        return checkResult(cublasDspr2Native(handle, uplo, n, alpha, x, incx, y, incy, AP));
    }
    private static native int cublasDspr2Native(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer AP);


    public static int cublasChpr2(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer AP)
    {
        return checkResult(cublasChpr2Native(handle, uplo, n, alpha, x, incx, y, incy, AP));
    }
    private static native int cublasChpr2Native(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer AP);


    public static int cublasZhpr2(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer AP)
    {
        return checkResult(cublasZhpr2Native(handle, uplo, n, alpha, x, incx, y, incy, AP));
    }
    private static native int cublasZhpr2Native(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer x,
        int incx,
        Pointer y,
        int incy,
        Pointer AP);


    public static int cublasSgemm(
        cublasHandle handle,
        int transa,
        int transb,
        int m,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc)
    {
        return checkResult(cublasSgemmNative(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasSgemmNative(
        cublasHandle handle,
        int transa,
        int transb,
        int m,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc);


    public static int cublasDgemm(
        cublasHandle handle,
        int transa,
        int transb,
        int m,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc)
    {
        return checkResult(cublasDgemmNative(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasDgemmNative(
        cublasHandle handle,
        int transa,
        int transb,
        int m,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc);


    public static int cublasCgemm(
        cublasHandle handle,
        int transa,
        int transb,
        int m,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc)
    {
        return checkResult(cublasCgemmNative(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasCgemmNative(
        cublasHandle handle,
        int transa,
        int transb,
        int m,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc);


    public static int cublasZgemm(
        cublasHandle handle,
        int transa,
        int transb,
        int m,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc)
    {
        return checkResult(cublasZgemmNative(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasZgemmNative(
        cublasHandle handle,
        int transa,
        int transb,
        int m,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc);

    public static int cublasSgemmEx(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, /** host or device pointer */
        Pointer A, 
        int Atype, 
        int lda, 
        Pointer B, 
        int Btype, 
        int ldb, 
        Pointer beta, /** host or device pointer */
        Pointer C, 
        int Ctype, 
        int ldc)
    {
        return checkResult(cublasSgemmExNative(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc));
    }
    private static native int cublasSgemmExNative(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, /** host or device pointer */
        Pointer A, 
        int Atype, 
        int lda, 
        Pointer B, 
        int Btype, 
        int ldb, 
        Pointer beta, /** host or device pointer */
        Pointer C, 
        int Ctype, 
        int ldc);

    public static int cublasSsyrk(
        cublasHandle handle,
        int uplo,
        int trans,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc)
    {
        return checkResult(cublasSsyrkNative(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc));
    }
    private static native int cublasSsyrkNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc);


    public static int cublasDsyrk(
        cublasHandle handle,
        int uplo,
        int trans,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc)
    {
        return checkResult(cublasDsyrkNative(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc));
    }
    private static native int cublasDsyrkNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc);


    public static int cublasCsyrk(
        cublasHandle handle,
        int uplo,
        int trans,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc)
    {
        return checkResult(cublasCsyrkNative(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc));
    }
    private static native int cublasCsyrkNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc);


    public static int cublasZsyrk(
        cublasHandle handle,
        int uplo,
        int trans,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc)
    {
        return checkResult(cublasZsyrkNative(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc));
    }
    private static native int cublasZsyrkNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc);


    public static int cublasCherk(
        cublasHandle handle,
        int uplo,
        int trans,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc)
    {
        return checkResult(cublasCherkNative(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc));
    }
    private static native int cublasCherkNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc);


    public static int cublasZherk(
        cublasHandle handle,
        int uplo,
        int trans,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc)
    {
        return checkResult(cublasZherkNative(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc));
    }
    private static native int cublasZherkNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc);


    public static int cublasSsyr2k(
        cublasHandle handle,
        int uplo,
        int trans,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc)
    {
        return checkResult(cublasSsyr2kNative(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasSsyr2kNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc);


    public static int cublasDsyr2k(
        cublasHandle handle,
        int uplo,
        int trans,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc)
    {
        return checkResult(cublasDsyr2kNative(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasDsyr2kNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc);


    public static int cublasCsyr2k(
        cublasHandle handle,
        int uplo,
        int trans,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc)
    {
        return checkResult(cublasCsyr2kNative(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasCsyr2kNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc);


    public static int cublasZsyr2k(
        cublasHandle handle,
        int uplo,
        int trans,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc)
    {
        return checkResult(cublasZsyr2kNative(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasZsyr2kNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc);


    public static int cublasCher2k(
        cublasHandle handle,
        int uplo,
        int trans,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc)
    {
        return checkResult(cublasCher2kNative(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasCher2kNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc);


    public static int cublasZher2k(
        cublasHandle handle,
        int uplo,
        int trans,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc)
    {
        return checkResult(cublasZher2kNative(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasZher2kNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc);

    public static int cublasSsyrkx(
        cublasHandle handle,
        int uplo,
        int trans,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc)
    {
        return checkResult(cublasSsyrkxNative(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasSsyrkxNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc);


    public static int cublasDsyrkx(
        cublasHandle handle,
        int uplo,
        int trans,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc)
    {
        return checkResult(cublasDsyrkxNative(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasDsyrkxNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc);


    public static int cublasCsyrkx(
        cublasHandle handle,
        int uplo,
        int trans,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc)
    {
        return checkResult(cublasCsyrkxNative(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasCsyrkxNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc);


    public static int cublasZsyrkx(
        cublasHandle handle,
        int uplo,
        int trans,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc)
    {
        return checkResult(cublasZsyrkxNative(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasZsyrkxNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc);


    public static int cublasCherkx(
        cublasHandle handle,
        int uplo,
        int trans,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc)
    {
        return checkResult(cublasCherkxNative(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasCherkxNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc);


    public static int cublasZherkx(
        cublasHandle handle,
        int uplo,
        int trans,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc)
    {
        return checkResult(cublasZherkxNative(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasZherkxNative(
        cublasHandle handle,
        int uplo,
        int trans,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc);


    public static int cublasSsymm(
        cublasHandle handle,
        int side,
        int uplo,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc)
    {
        return checkResult(cublasSsymmNative(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasSsymmNative(
        cublasHandle handle,
        int side,
        int uplo,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc);


    public static int cublasDsymm(
        cublasHandle handle,
        int side,
        int uplo,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc)
    {
        return checkResult(cublasDsymmNative(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasDsymmNative(
        cublasHandle handle,
        int side,
        int uplo,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc);


    public static int cublasCsymm(
        cublasHandle handle,
        int side,
        int uplo,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc)
    {
        return checkResult(cublasCsymmNative(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasCsymmNative(
        cublasHandle handle,
        int side,
        int uplo,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc);


    public static int cublasZsymm(
        cublasHandle handle,
        int side,
        int uplo,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc)
    {
        return checkResult(cublasZsymmNative(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasZsymmNative(
        cublasHandle handle,
        int side,
        int uplo,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc);


    public static int cublasChemm(
        cublasHandle handle,
        int side,
        int uplo,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc)
    {
        return checkResult(cublasChemmNative(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasChemmNative(
        cublasHandle handle,
        int side,
        int uplo,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc);


    public static int cublasZhemm(
        cublasHandle handle,
        int side,
        int uplo,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc)
    {
        return checkResult(cublasZhemmNative(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasZhemmNative(
        cublasHandle handle,
        int side,
        int uplo,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer C,
        int ldc);


    public static int cublasStrsm(
        cublasHandle handle,
        int side,
        int uplo,
        int trans,
        int diag,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb)
    {
        return checkResult(cublasStrsmNative(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb));
    }
    private static native int cublasStrsmNative(
        cublasHandle handle,
        int side,
        int uplo,
        int trans,
        int diag,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb);


    public static int cublasDtrsm(
        cublasHandle handle,
        int side,
        int uplo,
        int trans,
        int diag,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb)
    {
        return checkResult(cublasDtrsmNative(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb));
    }
    private static native int cublasDtrsmNative(
        cublasHandle handle,
        int side,
        int uplo,
        int trans,
        int diag,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb);


    public static int cublasCtrsm(
        cublasHandle handle,
        int side,
        int uplo,
        int trans,
        int diag,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb)
    {
        return checkResult(cublasCtrsmNative(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb));
    }
    private static native int cublasCtrsmNative(
        cublasHandle handle,
        int side,
        int uplo,
        int trans,
        int diag,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb);


    public static int cublasZtrsm(
        cublasHandle handle,
        int side,
        int uplo,
        int trans,
        int diag,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb)
    {
        return checkResult(cublasZtrsmNative(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb));
    }
    private static native int cublasZtrsmNative(
        cublasHandle handle,
        int side,
        int uplo,
        int trans,
        int diag,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb);


    public static int cublasStrmm(
        cublasHandle handle,
        int side,
        int uplo,
        int trans,
        int diag,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer C,
        int ldc)
    {
        return checkResult(cublasStrmmNative(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc));
    }
    private static native int cublasStrmmNative(
        cublasHandle handle,
        int side,
        int uplo,
        int trans,
        int diag,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer C,
        int ldc);


    public static int cublasDtrmm(
        cublasHandle handle,
        int side,
        int uplo,
        int trans,
        int diag,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer C,
        int ldc)
    {
        return checkResult(cublasDtrmmNative(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc));
    }
    private static native int cublasDtrmmNative(
        cublasHandle handle,
        int side,
        int uplo,
        int trans,
        int diag,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer C,
        int ldc);


    public static int cublasCtrmm(
        cublasHandle handle,
        int side,
        int uplo,
        int trans,
        int diag,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer C,
        int ldc)
    {
        return checkResult(cublasCtrmmNative(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc));
    }
    private static native int cublasCtrmmNative(
        cublasHandle handle,
        int side,
        int uplo,
        int trans,
        int diag,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer C,
        int ldc);


    public static int cublasZtrmm(
        cublasHandle handle,
        int side,
        int uplo,
        int trans,
        int diag,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer C,
        int ldc)
    {
        return checkResult(cublasZtrmmNative(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc));
    }
    private static native int cublasZtrmmNative(
        cublasHandle handle,
        int side,
        int uplo,
        int trans,
        int diag,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        Pointer C,
        int ldc);


    public static int cublasSgemmBatched(
        cublasHandle handle,
        int transa,
        int transb,
        int m,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer Aarray,
        int lda,
        Pointer Barray,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer Carray,
        int ldc,
        int batchCount)
    {
        return checkResult(cublasSgemmBatchedNative(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount));
    }
    private static native int cublasSgemmBatchedNative(
        cublasHandle handle,
        int transa,
        int transb,
        int m,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer Aarray,
        int lda,
        Pointer Barray,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer Carray,
        int ldc,
        int batchCount);


    public static int cublasDgemmBatched(
        cublasHandle handle,
        int transa,
        int transb,
        int m,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer Aarray,
        int lda,
        Pointer Barray,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer Carray,
        int ldc,
        int batchCount)
    {
        return checkResult(cublasDgemmBatchedNative(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount));
    }
    private static native int cublasDgemmBatchedNative(
        cublasHandle handle,
        int transa,
        int transb,
        int m,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer Aarray,
        int lda,
        Pointer Barray,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer Carray,
        int ldc,
        int batchCount);


    public static int cublasCgemmBatched(
        cublasHandle handle,
        int transa,
        int transb,
        int m,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer Aarray,
        int lda,
        Pointer Barray,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer Carray,
        int ldc,
        int batchCount)
    {
        return checkResult(cublasCgemmBatchedNative(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount));
    }
    private static native int cublasCgemmBatchedNative(
        cublasHandle handle,
        int transa,
        int transb,
        int m,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer Aarray,
        int lda,
        Pointer Barray,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer Carray,
        int ldc,
        int batchCount);


    public static int cublasZgemmBatched(
        cublasHandle handle,
        int transa,
        int transb,
        int m,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer Aarray,
        int lda,
        Pointer Barray,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer Carray,
        int ldc,
        int batchCount)
    {
        return checkResult(cublasZgemmBatchedNative(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount));
    }
    private static native int cublasZgemmBatchedNative(
        cublasHandle handle,
        int transa,
        int transb,
        int m,
        int n,
        int k,
        Pointer alpha, /** host or device pointer */
        Pointer Aarray,
        int lda,
        Pointer Barray,
        int ldb,
        Pointer beta, /** host or device pointer */
        Pointer Carray,
        int ldc,
        int batchCount);


    public static int cublasSgeam(
        cublasHandle handle,
        int transa,
        int transb,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer beta, /** host or device pointer */
        Pointer B,
        int ldb,
        Pointer C,
        int ldc)
    {
        return checkResult(cublasSgeamNative(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc));
    }
    private static native int cublasSgeamNative(
        cublasHandle handle,
        int transa,
        int transb,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer beta, /** host or device pointer */
        Pointer B,
        int ldb,
        Pointer C,
        int ldc);


    public static int cublasDgeam(
        cublasHandle handle,
        int transa,
        int transb,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer beta, /** host or device pointer */
        Pointer B,
        int ldb,
        Pointer C,
        int ldc)
    {
        return checkResult(cublasDgeamNative(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc));
    }
    private static native int cublasDgeamNative(
        cublasHandle handle,
        int transa,
        int transb,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer beta, /** host or device pointer */
        Pointer B,
        int ldb,
        Pointer C,
        int ldc);


    public static int cublasCgeam(
        cublasHandle handle,
        int transa,
        int transb,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer beta, /** host or device pointer */
        Pointer B,
        int ldb,
        Pointer C,
        int ldc)
    {
        return checkResult(cublasCgeamNative(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc));
    }
    private static native int cublasCgeamNative(
        cublasHandle handle,
        int transa,
        int transb,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer beta, /** host or device pointer */
        Pointer B,
        int ldb,
        Pointer C,
        int ldc);


    public static int cublasZgeam(
        cublasHandle handle,
        int transa,
        int transb,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer beta, /** host or device pointer */
        Pointer B,
        int ldb,
        Pointer C,
        int ldc)
    {
        return checkResult(cublasZgeamNative(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc));
    }
    private static native int cublasZgeamNative(
        cublasHandle handle,
        int transa,
        int transb,
        int m,
        int n,
        Pointer alpha, /** host or device pointer */
        Pointer A,
        int lda,
        Pointer beta, /** host or device pointer */
        Pointer B,
        int ldb,
        Pointer C,
        int ldc);


    public static int cublasSgetrfBatched(
        cublasHandle handle,
        int n,
        Pointer A, /**Device pointer*/
        int lda,
        Pointer P, /**Device Pointer*/
        Pointer INFO, /**Device Pointer*/
        int batchSize)
    {
        return checkResult(cublasSgetrfBatchedNative(handle, n, A, lda, P, INFO, batchSize));
    }
    private static native int cublasSgetrfBatchedNative(
        cublasHandle handle,
        int n,
        Pointer A, /**Device pointer*/
        int lda,
        Pointer P, /**Device Pointer*/
        Pointer INFO, /**Device Pointer*/
        int batchSize);


    public static int cublasDgetrfBatched(
        cublasHandle handle,
        int n,
        Pointer A, /**Device pointer*/
        int lda,
        Pointer P, /**Device Pointer*/
        Pointer INFO, /**Device Pointer*/
        int batchSize)
    {
        return checkResult(cublasDgetrfBatchedNative(handle, n, A, lda, P, INFO, batchSize));
    }
    private static native int cublasDgetrfBatchedNative(
        cublasHandle handle,
        int n,
        Pointer A, /**Device pointer*/
        int lda,
        Pointer P, /**Device Pointer*/
        Pointer INFO, /**Device Pointer*/
        int batchSize);


    public static int cublasCgetrfBatched(
        cublasHandle handle,
        int n,
        Pointer A, /**Device pointer*/
        int lda,
        Pointer P, /**Device Pointer*/
        Pointer INFO, /**Device Pointer*/
        int batchSize)
    {
        return checkResult(cublasCgetrfBatchedNative(handle, n, A, lda, P, INFO, batchSize));
    }
    private static native int cublasCgetrfBatchedNative(
        cublasHandle handle,
        int n,
        Pointer A, /**Device pointer*/
        int lda,
        Pointer P, /**Device Pointer*/
        Pointer INFO, /**Device Pointer*/
        int batchSize);


    public static int cublasZgetrfBatched(
        cublasHandle handle,
        int n,
        Pointer A, /**Device pointer*/
        int lda,
        Pointer P, /**Device Pointer*/
        Pointer INFO, /**Device Pointer*/
        int batchSize)
    {
        return checkResult(cublasZgetrfBatchedNative(handle, n, A, lda, P, INFO, batchSize));
    }
    private static native int cublasZgetrfBatchedNative(
        cublasHandle handle,
        int n,
        Pointer A, /**Device pointer*/
        int lda,
        Pointer P, /**Device Pointer*/
        Pointer INFO, /**Device Pointer*/
        int batchSize);

    public static int cublasSgetriBatched(
        cublasHandle handle,
        int n,
        Pointer A, /**Device pointer*/
        int lda,
        Pointer P, /**Device pointer*/
        Pointer C, /**Device pointer*/
        int ldc,
        Pointer INFO,
        int batchSize)
    {
        return checkResult(cublasSgetriBatchedNative(handle, n, A, lda, P, C, ldc, INFO, batchSize));
    }
    private static native int cublasSgetriBatchedNative(
        cublasHandle handle,
        int n,
        Pointer A, /**Device pointer*/
        int lda,
        Pointer P, /**Device pointer*/
        Pointer C, /**Device pointer*/
        int ldc,
        Pointer INFO,
        int batchSize);


    public static int cublasDgetriBatched(
        cublasHandle handle,
        int n,
        Pointer A, /**Device pointer*/
        int lda,
        Pointer P, /**Device pointer*/
        Pointer C, /**Device pointer*/
        int ldc,
        Pointer INFO,
        int batchSize)
    {
        return checkResult(cublasDgetriBatchedNative(handle, n, A, lda, P, C, ldc, INFO, batchSize));
    }
    private static native int cublasDgetriBatchedNative(
        cublasHandle handle,
        int n,
        Pointer A, /**Device pointer*/
        int lda,
        Pointer P, /**Device pointer*/
        Pointer C, /**Device pointer*/
        int ldc,
        Pointer INFO,
        int batchSize);


    public static int cublasCgetriBatched(
        cublasHandle handle,
        int n,
        Pointer A, /**Device pointer*/
        int lda,
        Pointer P, /**Device pointer*/
        Pointer C, /**Device pointer*/
        int ldc,
        Pointer INFO,
        int batchSize)
    {
        return checkResult(cublasCgetriBatchedNative(handle, n, A, lda, P, C, ldc, INFO, batchSize));
    }
    private static native int cublasCgetriBatchedNative(
        cublasHandle handle,
        int n,
        Pointer A, /**Device pointer*/
        int lda,
        Pointer P, /**Device pointer*/
        Pointer C, /**Device pointer*/
        int ldc,
        Pointer INFO,
        int batchSize);


    public static int cublasZgetriBatched(
        cublasHandle handle,
        int n,
        Pointer A, /**Device pointer*/
        int lda,
        Pointer P, /**Device pointer*/
        Pointer C, /**Device pointer*/
        int ldc,
        Pointer INFO,
        int batchSize)
    {
        return checkResult(cublasZgetriBatchedNative(handle, n, A, lda, P, C, ldc, INFO, batchSize));
    }
    private static native int cublasZgetriBatchedNative(
        cublasHandle handle,
        int n,
        Pointer A, /**Device pointer*/
        int lda,
        Pointer P, /**Device pointer*/
        Pointer C, /**Device pointer*/
        int ldc,
        Pointer INFO,
        int batchSize);


    public static int cublasSgetrsBatched(
        cublasHandle handle,
        int trans,
        int n,
        int nrhs,
        Pointer Aarray,
        int lda,
        Pointer devIpiv,
        Pointer Barray,
        int ldb,
        Pointer info,
        int batchSize)
    {
        return checkResult(cublasSgetrsBatchedNative(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize));
    }
    private static native int cublasSgetrsBatchedNative(
        cublasHandle handle,
        int trans,
        int n,
        int nrhs,
        Pointer Aarray,
        int lda,
        Pointer devIpiv,
        Pointer Barray,
        int ldb,
        Pointer info,
        int batchSize);


    public static int cublasDgetrsBatched(
        cublasHandle handle,
        int trans,
        int n,
        int nrhs,
        Pointer Aarray,
        int lda,
        Pointer devIpiv,
        Pointer Barray,
        int ldb,
        Pointer info,
        int batchSize)
    {
        return checkResult(cublasDgetrsBatchedNative(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize));
    }
    private static native int cublasDgetrsBatchedNative(
        cublasHandle handle,
        int trans,
        int n,
        int nrhs,
        Pointer Aarray,
        int lda,
        Pointer devIpiv,
        Pointer Barray,
        int ldb,
        Pointer info,
        int batchSize);


    public static int cublasCgetrsBatched(
        cublasHandle handle,
        int trans,
        int n,
        int nrhs,
        Pointer Aarray,
        int lda,
        Pointer devIpiv,
        Pointer Barray,
        int ldb,
        Pointer info,
        int batchSize)
    {
        return checkResult(cublasCgetrsBatchedNative(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize));
    }
    private static native int cublasCgetrsBatchedNative(
        cublasHandle handle,
        int trans,
        int n,
        int nrhs,
        Pointer Aarray,
        int lda,
        Pointer devIpiv,
        Pointer Barray,
        int ldb,
        Pointer info,
        int batchSize);


    public static int cublasZgetrsBatched(
        cublasHandle handle,
        int trans,
        int n,
        int nrhs,
        Pointer Aarray,
        int lda,
        Pointer devIpiv,
        Pointer Barray,
        int ldb,
        Pointer info,
        int batchSize)
    {
        return checkResult(cublasZgetrsBatchedNative(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize));
    }
    private static native int cublasZgetrsBatchedNative(
        cublasHandle handle,
        int trans,
        int n,
        int nrhs,
        Pointer Aarray,
        int lda,
        Pointer devIpiv,
        Pointer Barray,
        int ldb,
        Pointer info,
        int batchSize);


    public static int cublasStrsmBatched(
        cublasHandle handle,
        int side,
        int uplo,
        int trans,
        int diag,
        int m,
        int n,
        Pointer alpha, /**Host or Device Pointer*/
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        int batchCount)
    {
        return checkResult(cublasStrsmBatchedNative(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount));
    }
    private static native int cublasStrsmBatchedNative(
        cublasHandle handle,
        int side,
        int uplo,
        int trans,
        int diag,
        int m,
        int n,
        Pointer alpha, /**Host or Device Pointer*/
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        int batchCount);


    public static int cublasDtrsmBatched(
        cublasHandle handle,
        int side,
        int uplo,
        int trans,
        int diag,
        int m,
        int n,
        Pointer alpha, /**Host or Device Pointer*/
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        int batchCount)
    {
        return checkResult(cublasDtrsmBatchedNative(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount));
    }
    private static native int cublasDtrsmBatchedNative(
        cublasHandle handle,
        int side,
        int uplo,
        int trans,
        int diag,
        int m,
        int n,
        Pointer alpha, /**Host or Device Pointer*/
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        int batchCount);


    public static int cublasCtrsmBatched(
        cublasHandle handle,
        int side,
        int uplo,
        int trans,
        int diag,
        int m,
        int n,
        Pointer alpha, /**Host or Device Pointer*/
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        int batchCount)
    {
        return checkResult(cublasCtrsmBatchedNative(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount));
    }
    private static native int cublasCtrsmBatchedNative(
        cublasHandle handle,
        int side,
        int uplo,
        int trans,
        int diag,
        int m,
        int n,
        Pointer alpha, /**Host or Device Pointer*/
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        int batchCount);


    public static int cublasZtrsmBatched(
        cublasHandle handle,
        int side,
        int uplo,
        int trans,
        int diag,
        int m,
        int n,
        Pointer alpha, /**Host or Device Pointer*/
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        int batchCount)
    {
        return checkResult(cublasZtrsmBatchedNative(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount));
    }
    private static native int cublasZtrsmBatchedNative(
        cublasHandle handle,
        int side,
        int uplo,
        int trans,
        int diag,
        int m,
        int n,
        Pointer alpha, /**Host or Device Pointer*/
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        int batchCount);

    public static int cublasSmatinvBatched(
        cublasHandle handle,
        int n,
        Pointer A, /**Device pointer*/
        int lda,
        Pointer Ainv, /**Device pointer*/
        int lda_inv,
        Pointer INFO, /**Device Pointer*/
        int batchSize)
    {
        return checkResult(cublasSmatinvBatchedNative(handle, n, A, lda, Ainv, lda_inv, INFO, batchSize));
    }
    private static native int cublasSmatinvBatchedNative(
        cublasHandle handle,
        int n,
        Pointer A, /**Device pointer*/
        int lda,
        Pointer Ainv, /**Device pointer*/
        int lda_inv,
        Pointer INFO, /**Device Pointer*/
        int batchSize);


    public static int cublasDmatinvBatched(
        cublasHandle handle,
        int n,
        Pointer A, /**Device pointer*/
        int lda,
        Pointer Ainv, /**Device pointer*/
        int lda_inv,
        Pointer INFO, /**Device Pointer*/
        int batchSize)
    {
        return checkResult(cublasDmatinvBatchedNative(handle, n, A, lda, Ainv, lda_inv, INFO, batchSize));
    }
    private static native int cublasDmatinvBatchedNative(
        cublasHandle handle,
        int n,
        Pointer A, /**Device pointer*/
        int lda,
        Pointer Ainv, /**Device pointer*/
        int lda_inv,
        Pointer INFO, /**Device Pointer*/
        int batchSize);


    public static int cublasCmatinvBatched(
        cublasHandle handle,
        int n,
        Pointer A, /**Device pointer*/
        int lda,
        Pointer Ainv, /**Device pointer*/
        int lda_inv,
        Pointer INFO, /**Device Pointer*/
        int batchSize)
    {
        return checkResult(cublasCmatinvBatchedNative(handle, n, A, lda, Ainv, lda_inv, INFO, batchSize));
    }
    private static native int cublasCmatinvBatchedNative(
        cublasHandle handle,
        int n,
        Pointer A, /**Device pointer*/
        int lda,
        Pointer Ainv, /**Device pointer*/
        int lda_inv,
        Pointer INFO, /**Device Pointer*/
        int batchSize);


    public static int cublasZmatinvBatched(
        cublasHandle handle,
        int n,
        Pointer A, /**Device pointer*/
        int lda,
        Pointer Ainv, /**Device pointer*/
        int lda_inv,
        Pointer INFO, /**Device Pointer*/
        int batchSize)
    {
        return checkResult(cublasZmatinvBatchedNative(handle, n, A, lda, Ainv, lda_inv, INFO, batchSize));
    }
    private static native int cublasZmatinvBatchedNative(
        cublasHandle handle,
        int n,
        Pointer A, /**Device pointer*/
        int lda,
        Pointer Ainv, /**Device pointer*/
        int lda_inv,
        Pointer INFO, /**Device Pointer*/
        int batchSize);


    public static int cublasSgeqrfBatched(
        cublasHandle handle,
        int m,
        int n,
        Pointer Aarray, /**Device pointer*/
        int lda,
        Pointer TauArray, /** Device pointer*/
        Pointer info,
        int batchSize)
    {
        return checkResult(cublasSgeqrfBatchedNative(handle, m, n, Aarray, lda, TauArray, info, batchSize));
    }
    private static native int cublasSgeqrfBatchedNative(
        cublasHandle handle,
        int m,
        int n,
        Pointer Aarray, /**Device pointer*/
        int lda,
        Pointer TauArray, /** Device pointer*/
        Pointer info,
        int batchSize);


    public static int cublasDgeqrfBatched(
        cublasHandle handle,
        int m,
        int n,
        Pointer Aarray, /**Device pointer*/
        int lda,
        Pointer TauArray, /** Device pointer*/
        Pointer info,
        int batchSize)
    {
        return checkResult(cublasDgeqrfBatchedNative(handle, m, n, Aarray, lda, TauArray, info, batchSize));
    }
    private static native int cublasDgeqrfBatchedNative(
        cublasHandle handle,
        int m,
        int n,
        Pointer Aarray, /**Device pointer*/
        int lda,
        Pointer TauArray, /** Device pointer*/
        Pointer info,
        int batchSize);


    public static int cublasCgeqrfBatched(
        cublasHandle handle,
        int m,
        int n,
        Pointer Aarray, /**Device pointer*/
        int lda,
        Pointer TauArray, /** Device pointer*/
        Pointer info,
        int batchSize)
    {
        return checkResult(cublasCgeqrfBatchedNative(handle, m, n, Aarray, lda, TauArray, info, batchSize));
    }
    private static native int cublasCgeqrfBatchedNative(
        cublasHandle handle,
        int m,
        int n,
        Pointer Aarray, /**Device pointer*/
        int lda,
        Pointer TauArray, /** Device pointer*/
        Pointer info,
        int batchSize);


    public static int cublasZgeqrfBatched(
        cublasHandle handle,
        int m,
        int n,
        Pointer Aarray, /**Device pointer*/
        int lda,
        Pointer TauArray, /** Device pointer*/
        Pointer info,
        int batchSize)
    {
        return checkResult(cublasZgeqrfBatchedNative(handle, m, n, Aarray, lda, TauArray, info, batchSize));
    }
    private static native int cublasZgeqrfBatchedNative(
        cublasHandle handle,
        int m,
        int n,
        Pointer Aarray, /**Device pointer*/
        int lda,
        Pointer TauArray, /** Device pointer*/
        Pointer info,
        int batchSize);


    public static int cublasSgelsBatched(
        cublasHandle handle,
        int trans,
        int m,
        int n,
        int nrhs,
        Pointer Aarray, /**Device pointer*/
        int lda,
        Pointer Carray, /** Device pointer*/
        int ldc,
        Pointer info,
        Pointer devInfoArray, /** Device pointer*/
        int batchSize)
    {
        return checkResult(cublasSgelsBatchedNative(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize));
    }
    private static native int cublasSgelsBatchedNative(
        cublasHandle handle,
        int trans,
        int m,
        int n,
        int nrhs,
        Pointer Aarray, /**Device pointer*/
        int lda,
        Pointer Carray, /** Device pointer*/
        int ldc,
        Pointer info,
        Pointer devInfoArray, /** Device pointer*/
        int batchSize);


    public static int cublasDgelsBatched(
        cublasHandle handle,
        int trans,
        int m,
        int n,
        int nrhs,
        Pointer Aarray, /**Device pointer*/
        int lda,
        Pointer Carray, /** Device pointer*/
        int ldc,
        Pointer info,
        Pointer devInfoArray, /** Device pointer*/
        int batchSize)
    {
        return checkResult(cublasDgelsBatchedNative(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize));
    }
    private static native int cublasDgelsBatchedNative(
        cublasHandle handle,
        int trans,
        int m,
        int n,
        int nrhs,
        Pointer Aarray, /**Device pointer*/
        int lda,
        Pointer Carray, /** Device pointer*/
        int ldc,
        Pointer info,
        Pointer devInfoArray, /** Device pointer*/
        int batchSize);


    public static int cublasCgelsBatched(
        cublasHandle handle,
        int trans,
        int m,
        int n,
        int nrhs,
        Pointer Aarray, /**Device pointer*/
        int lda,
        Pointer Carray, /** Device pointer*/
        int ldc,
        Pointer info,
        Pointer devInfoArray,
        int batchSize)
    {
        return checkResult(cublasCgelsBatchedNative(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize));
    }
    private static native int cublasCgelsBatchedNative(
        cublasHandle handle,
        int trans,
        int m,
        int n,
        int nrhs,
        Pointer Aarray, /**Device pointer*/
        int lda,
        Pointer Carray, /** Device pointer*/
        int ldc,
        Pointer info,
        Pointer devInfoArray,
        int batchSize);


    public static int cublasZgelsBatched(
        cublasHandle handle,
        int trans,
        int m,
        int n,
        int nrhs,
        Pointer Aarray, /**Device pointer*/
        int lda,
        Pointer Carray, /** Device pointer*/
        int ldc,
        Pointer info,
        Pointer devInfoArray,
        int batchSize)
    {
        return checkResult(cublasZgelsBatchedNative(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize));
    }
    private static native int cublasZgelsBatchedNative(
        cublasHandle handle,
        int trans,
        int m,
        int n,
        int nrhs,
        Pointer Aarray, /**Device pointer*/
        int lda,
        Pointer Carray, /** Device pointer*/
        int ldc,
        Pointer info,
        Pointer devInfoArray,
        int batchSize);


    public static int cublasSdgmm(
        cublasHandle handle,
        int mode,
        int m,
        int n,
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer C,
        int ldc)
    {
        return checkResult(cublasSdgmmNative(handle, mode, m, n, A, lda, x, incx, C, ldc));
    }
    private static native int cublasSdgmmNative(
        cublasHandle handle,
        int mode,
        int m,
        int n,
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer C,
        int ldc);


    public static int cublasDdgmm(
        cublasHandle handle,
        int mode,
        int m,
        int n,
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer C,
        int ldc)
    {
        return checkResult(cublasDdgmmNative(handle, mode, m, n, A, lda, x, incx, C, ldc));
    }
    private static native int cublasDdgmmNative(
        cublasHandle handle,
        int mode,
        int m,
        int n,
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer C,
        int ldc);


    public static int cublasCdgmm(
        cublasHandle handle,
        int mode,
        int m,
        int n,
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer C,
        int ldc)
    {
        return checkResult(cublasCdgmmNative(handle, mode, m, n, A, lda, x, incx, C, ldc));
    }
    private static native int cublasCdgmmNative(
        cublasHandle handle,
        int mode,
        int m,
        int n,
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer C,
        int ldc);


    public static int cublasZdgmm(
        cublasHandle handle,
        int mode,
        int m,
        int n,
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer C,
        int ldc)
    {
        return checkResult(cublasZdgmmNative(handle, mode, m, n, A, lda, x, incx, C, ldc));
    }
    private static native int cublasZdgmmNative(
        cublasHandle handle,
        int mode,
        int m,
        int n,
        Pointer A,
        int lda,
        Pointer x,
        int incx,
        Pointer C,
        int ldc);





    public static int cublasStpttr(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer AP,
        Pointer A,
        int lda)
    {
        return checkResult(cublasStpttrNative(handle, uplo, n, AP, A, lda));
    }
    private static native int cublasStpttrNative(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer AP,
        Pointer A,
        int lda);


    public static int cublasDtpttr(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer AP,
        Pointer A,
        int lda)
    {
        return checkResult(cublasDtpttrNative(handle, uplo, n, AP, A, lda));
    }
    private static native int cublasDtpttrNative(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer AP,
        Pointer A,
        int lda);


    public static int cublasCtpttr(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer AP,
        Pointer A,
        int lda)
    {
        return checkResult(cublasCtpttrNative(handle, uplo, n, AP, A, lda));
    }
    private static native int cublasCtpttrNative(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer AP,
        Pointer A,
        int lda);


    public static int cublasZtpttr(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer AP,
        Pointer A,
        int lda)
    {
        return checkResult(cublasZtpttrNative(handle, uplo, n, AP, A, lda));
    }
    private static native int cublasZtpttrNative(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer AP,
        Pointer A,
        int lda);


    public static int cublasStrttp(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer A,
        int lda,
        Pointer AP)
    {
        return checkResult(cublasStrttpNative(handle, uplo, n, A, lda, AP));
    }
    private static native int cublasStrttpNative(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer A,
        int lda,
        Pointer AP);


    public static int cublasDtrttp(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer A,
        int lda,
        Pointer AP)
    {
        return checkResult(cublasDtrttpNative(handle, uplo, n, A, lda, AP));
    }
    private static native int cublasDtrttpNative(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer A,
        int lda,
        Pointer AP);


    public static int cublasCtrttp(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer A,
        int lda,
        Pointer AP)
    {
        return checkResult(cublasCtrttpNative(handle, uplo, n, A, lda, AP));
    }
    private static native int cublasCtrttpNative(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer A,
        int lda,
        Pointer AP);


    public static int cublasZtrttp(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer A,
        int lda,
        Pointer AP)
    {
        return checkResult(cublasZtrttpNative(handle, uplo, n, A, lda, AP));
    }
    private static native int cublasZtrttpNative(
        cublasHandle handle,
        int uplo,
        int n,
        Pointer A,
        int lda,
        Pointer AP);


}

