/*
 * JCublas - Java bindings for CUBLAS, the NVIDIA CUDA BLAS library,
 * to be used with JCuda
 *
 * Copyright (c) 2008-2015 Marco Hutter - http://www.jcuda.org
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

import java.nio.*;
import jcuda.*;
import jcuda.runtime.*;

/**
 * Java bindings for CUBLAS, the NVIDIA CUDA BLAS library.
 * <br />
 * Most comments are taken from the cublas.h header file.
 * <br />
 * <b>Note:</b>: This class mimics the original CUBLAS API. 
 * With CUDA 4.0, a new API for CUBLAS has been introduced.
 * This is referred to as the "new CUBLAS API", and is 
 * defined in the C header "cublas_v2.h". Consequently,
 * the new CUBLAS API is offered via the {@link JCublas2} 
 * class.<br>
 * <br>
 * New applications should generally use the new CUBLAS API.
 * This class is only maintained for backward compatibility
 * of existing applications. For more information, see
 * http://docs.nvidia.com/cuda/cublas/#new-and-legacy-cublas-api 
 */
public class JCublas
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

    /**
     * The last result code that was set by any of the BLAS functions.
     * This will be stored in the checkResultBLAS() method if
     * exceptions are enabled.
     */
    private static int lastResult = cublasStatus.CUBLAS_STATUS_SUCCESS;

    /* Private constructor to prevent instantiation */
    private JCublas()
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
            LibUtils.loadLibrary("JCublas");
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
     * only set the result status which may be queried with
     * {@link JCublas#cublasGetError()}.
     * If exceptions are enabled, a CudaException with a detailed error
     * message will be thrown if a method is about to set a result code
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

    /**
     * Obtain the current CUBLAS status by calling cublasGetErrorNative,
     * and store the result as the lastResult. If the obtained result
     * code is not cublasStatus.CUBLAS_STATUS_SUCCESS and exceptions
     * have been enabled, an CudaException will be thrown.
     */
    private static void checkResultBLAS()
    {
        if (exceptionsEnabled)
        {
            lastResult = cublasGetErrorNative();
            if (lastResult != cublasStatus.CUBLAS_STATUS_SUCCESS)
            {
                throw new CudaException(cublasStatus.stringFor(lastResult));
            }
        }
    }





    /**
     * Wrapper for CUBLAS function.<br />
     * <br />
     * cublasStatus
     * cublasInit()<br />
     *<br />
     * initializes the CUBLAS library and must be called before any other
     * CUBLAS API function is invoked. It allocates hardware resources
     * necessary for accessing the GPU.<br />
     *<br />
     * Return Values<br />
     * -------------<br />
     * CUBLAS_STATUS_ALLOC_FAILED     if resources could not be allocated<br />
     * CUBLAS_STATUS_SUCCESS          if CUBLAS library initialized successfully<br />
     */
    public static int cublasInit()
    {
        return checkResult(cublasInitNative());
    }
    private static native int cublasInitNative();

    /**
     * Wrapper for CUBLAS function.<br />
     * <br />
     * cublasStatus
     * cublasShutdown()<br />
     *<br />
     * releases CPU-side resources used by the CUBLAS library. The release of
     * GPU-side resources may be deferred until the application shuts down.<br />
     *<br />
     * Return Values<br />
     * -------------<br />
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized<br />
     * CUBLAS_STATUS_SUCCESS          if CUBLAS library shut down successfully<br />
     */
    public static int cublasShutdown()
    {
        return checkResult(cublasShutdownNative());
    }
    private static native int cublasShutdownNative();



    /**
     * Wrapper for CUBLAS function.<br />
     * <br />
     * cublasStatus
     * cublasGetError()<br />
     *<br />
     * returns the last error that occurred on invocation of any of the
     * CUBLAS BLAS functions. While the CUBLAS helper functions return status
     * directly, the BLAS functions do not do so for improved
     * compatibility with existing environments that do not expect BLAS
     * functions to return status. Reading the error status via
     * cublasGetError() resets the internal error state to
     * CUBLAS_STATUS_SUCCESS.
     */
    public static int cublasGetError()
    {
        if (exceptionsEnabled)
        {
            int returnedResult = lastResult;
            lastResult = cublasStatus.CUBLAS_STATUS_SUCCESS;
            return returnedResult;
        }
        return cublasGetErrorNative();
    }
    private static native int cublasGetErrorNative();

    /**
     * Wrapper for CUBLAS function.<br />
     * <br />
     * cublasStatus
     * cublasAlloc (int n, int elemSize, void **devicePtr)<br />
     *<br />
     * creates an object in GPU memory space capable of holding an array of
     * n elements, where each element requires elemSize bytes of storage. If
     * the function call is successful, a pointer to the object in GPU memory
     * space is placed in devicePtr. Note that this is a device pointer that
     * cannot be dereferenced in host code.<br />
     *<br />
     * Return Values<br />
     * -------------<br />
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized<br />
     * CUBLAS_STATUS_INVALID_VALUE    if n <= 0, or elemSize <= 0<br />
     * CUBLAS_STATUS_ALLOC_FAILED     if the object could not be allocated due to
     *                                lack of resources.<br />
     * CUBLAS_STATUS_SUCCESS          if storage was successfully allocated<br />
     */
    public static int cublasAlloc(int n, int elemSize, Pointer ptr)
    {
        return checkResult(cublasAllocNative(n, elemSize, ptr));
    }
    private static native int cublasAllocNative(int n, int elemSize, Pointer ptr);

    /**
     * Wrapper for CUBLAS function.<br />
     * <br />
     * cublasStatus
     * cublasFree (const void *devicePtr)<br />
     *<br />
     * destroys the object in GPU memory space pointed to by devicePtr.<br />
     *<br />
     * Return Values<br />
     * -------------<br />
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized<br />
     * CUBLAS_STATUS_INTERNAL_ERROR   if the object could not be deallocated<br />
     * CUBLAS_STATUS_SUCCESS          if object was destroyed successfully<br />
     */
    public static int cublasFree(Pointer ptr)
    {
        return checkResult(cublasFreeNative(ptr));
    }
    private static native int cublasFreeNative(Pointer ptr);


    // Debug method
    public static native void printVector(int n, Pointer x);

    // Debug method
    public static native void printMatrix(int cols, Pointer A, int lda);


    /*
     * Internal method to which all calls to an implementation of
     * cublasSetVector are finally delegated
     */
    private static native int cublasSetVectorNative(int n, int elemSize, Pointer x, int incx, Pointer y, int incy);

    /*
     * Internal method to which all calls to an implementation of
     * cublasGetVector are finally delegated
     */
    private static native int cublasGetVectorNative(int n, int elemSize, Pointer x, int incx, Pointer y, int incy);

    /*
     * Internal method to which all calls to an implementation of
     * cublasSetMatrix are finally delegated
     */
    private static native int cublasSetMatrixNative(int rows, int cols, int elemSize, Pointer A, int lda, Pointer B, int ldb);

    /*
     * Internal method to which all calls to an implementation of
     * cublasGetMatrix are finally delegated
     */
    private static native int cublasGetMatrixNative(int rows, int cols, int elemSize, Pointer A, int lda, Pointer B, int ldb);


    /*
     * Internal method to which all calls to an implementation of
     * cublasSetVectorAsync are finally delegated
     */
    private static native int cublasSetVectorAsyncNative(int n, int elemSize, Pointer x, int incx, Pointer y, int incy, cudaStream_t stream);

    /*
     * Internal method to which all calls to an implementation of
     * cublasGetVectorAsync are finally delegated
     */
    private static native int cublasGetVectorAsyncNative(int n, int elemSize, Pointer x, int incx, Pointer y, int incy, cudaStream_t stream);

    /*
     * Internal method to which all calls to an implementation of
     * cublasSetMatrixAsync are finally delegated
     */
    private static native int cublasSetMatrixAsyncNative(int rows, int cols, int elemSize, Pointer A, int lda, Pointer B, int ldb, cudaStream_t stream);

    /*
     * Internal method to which all calls to an implementation of
     * cublasGetMatrix are finally delegated
     */
    private static native int cublasGetMatrixAsyncNative(int rows, int cols, int elemSize, Pointer A, int lda, Pointer B, int ldb, cudaStream_t stream);








    //============================================================================
    // Memory management methods for single precision data:


    /**
     * Wrapper for CUBLAS function.<br />
     * <br />
     * cublasStatus<br />
     * cublasSetVector (int n, int elemSize, const void *x, int incx,
     *                  void *y, int incy)<br />
     *<br />
     * copies n elements from a vector x in CPU memory space to a vector y
     * in GPU memory space. Elements in both vectors are assumed to have a
     * size of elemSize bytes. Storage spacing between consecutive elements
     * is incx for the source vector x and incy for the destination vector
     * y. In general, y points to an object, or part of an object, allocated
     * via cublasAlloc(). Column major format for two-dimensional matrices
     * is assumed throughout CUBLAS. Therefore, if the increment for a vector
     * is equal to 1, this access a column vector while using an increment
     * equal to the leading dimension of the respective matrix accesses a
     * row vector.<br />
     *<br />
     * Return Values<br />
     * -------------<br />
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library not been initialized<br />
     * CUBLAS_STATUS_INVALID_VALUE    if incx, incy, or elemSize <= 0<br />
     * CUBLAS_STATUS_MAPPING_ERROR    if an error occurred accessing GPU memory<br />
     * CUBLAS_STATUS_SUCCESS          if the operation completed successfully<br />
     */
    public static int cublasSetVector(int n, int elemSize, Pointer x, int incx, Pointer y, int incy)
    {
        return checkResult(cublasSetVectorNative(n, elemSize, x, incx, y, incy));
    }


    /**
     * Extended wrapper for arrays of cuComplex values. Note that this method
     * only exists for convenience and compatibility with native C code. It
     * is much more efficient to provide a Pointer to a float array containing
     * the complex numbers, where each pair of consecutive numbers in the array
     * describes the real- and imaginary part of one complex number.
     *
     * @see JCublas#cublasSetVector(int, int, Pointer, int, Pointer, int)
     */
    public static int cublasSetVector (int n, cuComplex x[], int offsetx, int incx, Pointer y, int incy)
    {
        ByteBuffer byteBufferx = ByteBuffer.allocateDirect(x.length * 4 * 2);
        byteBufferx.order(ByteOrder.nativeOrder());
        FloatBuffer floatBufferx = byteBufferx.asFloatBuffer();

        int indexx = offsetx;
        for (int i=0; i<n; i++, indexx+=incx)
        {
            floatBufferx.put(indexx*2+0, x[indexx].x);
            floatBufferx.put(indexx*2+1, x[indexx].y);
        }
        return checkResult(cublasSetVectorNative(n, 8, Pointer.to(floatBufferx).withByteOffset(offsetx * 4 * 2), incx, y, incy));
    }






    /**
     * Wrapper for CUBLAS function.<br />
     * <br />
     * cublasStatus<br />
     * cublasGetVector (int n, int elemSize, const void *x, int incx,
     *                  void *y, int incy)<br />
     *<br />
     * copies n elements from a vector x in GPU memory space to a vector y
     * in CPU memory space. Elements in both vectors are assumed to have a
     * size of elemSize bytes. Storage spacing between consecutive elements
     * is incx for the source vector x and incy for the destination vector
     * y. In general, x points to an object, or part of an object, allocated
     * via cublasAlloc(). Column major format for two-dimensional matrices
     * is assumed throughout CUBLAS. Therefore, if the increment for a vector
     * is equal to 1, this access a column vector while using an increment
     * equal to the leading dimension of the respective matrix accesses a
     * row vector.<br />
     *<br />
     * Return Values<br />
     * -------------<br />
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library not been initialized<br />
     * CUBLAS_STATUS_INVALID_VALUE    if incx, incy, or elemSize <= 0<br />
     * CUBLAS_STATUS_MAPPING_ERROR    if an error occurred accessing GPU memory<br />
     * CUBLAS_STATUS_SUCCESS          if the operation completed successfully<br />
     */
    public static int cublasGetVector (int n, int elemSize, Pointer x, int incx, Pointer y, int incy)
    {
        return checkResult(cublasGetVectorNative(n, elemSize, x, incx, y, incy));
    }

    /**
     * Extended wrapper for arrays of cuComplex values. Note that this method
     * only exists for convenience and compatibility with native C code. It
     * is much more efficient to provide a Pointer to a float array that may
     * store the complex numbers, where each pair of consecutive numbers in
     * the array describes the real- and imaginary part of one complex number.
     *
     * @see JCublas#cublasGetVector(int, int, Pointer, int, Pointer, int)
     */
    public static int cublasGetVector (int n, Pointer x, int incx, cuComplex y[], int offsety, int incy)
    {
        ByteBuffer byteBuffery = ByteBuffer.allocateDirect(y.length * 4 * 2);
        byteBuffery.order(ByteOrder.nativeOrder());
        FloatBuffer floatBuffery = byteBuffery.asFloatBuffer();
        int status = cublasGetVectorNative(n, 8, x, incx, Pointer.to(floatBuffery).withByteOffset(offsety * 4 * 2), incy);
        if (status == cublasStatus.CUBLAS_STATUS_SUCCESS)
        {
            floatBuffery.rewind();
            int indexy = offsety;
            for (int i=0; i<n; i++, indexy+=incy)
            {
                y[indexy].x = floatBuffery.get(indexy*2+0);
                y[indexy].y = floatBuffery.get(indexy*2+1);
            }
        }
        return checkResult(status);
    }





    /**
     * Wrapper for CUBLAS function.<br />
     * <br />
     * cublasStatus
     * cublasSetMatrix (int rows, int cols, int elemSize, const void *A,
     *                  int lda, void *B, int ldb)<br />
     *<br />
     * copies a tile of rows x cols elements from a matrix A in CPU memory
     * space to a matrix B in GPU memory space. Each element requires storage
     * of elemSize bytes. Both matrices are assumed to be stored in column
     * major format, with the leading dimension (i.e. number of rows) of
     * source matrix A provided in lda, and the leading dimension of matrix B
     * provided in ldb. In general, B points to an object, or part of an
     * object, that was allocated via cublasAlloc().<br />
     *<br />
     * Return Values<br />
     * -------------<br />
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized<br />
     * CUBLAS_STATUS_INVALID_VALUE    if rows or cols < 0, or elemSize, lda, or
     *                                ldb <= 0<br />
     * CUBLAS_STATUS_MAPPING_ERROR    if error occurred accessing GPU memory<br />
     * CUBLAS_STATUS_SUCCESS          if the operation completed successfully<br />
     */
    public static int cublasSetMatrix (int rows, int cols, int elemSize, Pointer A, int lda, Pointer B, int ldb)
    {
        return checkResult(cublasSetMatrixNative(rows, elemSize, cols, A, lda, B, ldb));
    }


    /**
     * Extended wrapper for arrays of cuComplex values. Note that this method
     * only exists for convenience and compatibility with native C code. It
     * is much more efficient to provide a Pointer to a float array containing
     * the complex numbers, where each pair of consecutive numbers in the array
     * describes the real- and imaginary part of one complex number.
     *
     * @see JCublas#cublasSetMatrix(int, int, int, Pointer, int, Pointer, int)
     */
    public static int cublasSetMatrix (int rows, int cols, cuComplex A[], int offsetA, int lda, Pointer B, int ldb)
    {
        ByteBuffer byteBufferA = ByteBuffer.allocateDirect(A.length * 4 * 2);
        byteBufferA.order(ByteOrder.nativeOrder());
        FloatBuffer floatBufferA = byteBufferA.asFloatBuffer();
        for (int i=0; i<A.length; i++)
        {
            floatBufferA.put(A[i].x);
            floatBufferA.put(A[i].y);
        }
        return checkResult(cublasSetMatrixNative(rows, cols, 8, Pointer.to(floatBufferA).withByteOffset(offsetA * 4 * 2), lda, B, ldb));
    }







    /**
     * Wrapper for CUBLAS function.<br />
     * <br />
     * cublasStatus
     * cublasGetMatrix (int rows, int cols, int elemSize, const void *A,
     *                  int lda, void *B, int ldb)<br />
     *<br />
     * copies a tile of rows x cols elements from a matrix A in GPU memory
     * space to a matrix B in CPU memory space. Each element requires storage
     * of elemSize bytes. Both matrices are assumed to be stored in column
     * major format, with the leading dimension (i.e. number of rows) of
     * source matrix A provided in lda, and the leading dimension of matrix B
     * provided in ldb. In general, A points to an object, or part of an
     * object, that was allocated via cublasAlloc().<br />
     *<br />
     * Return Values<br />
     * -------------<br />
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized<br />
     * CUBLAS_STATUS_INVALID_VALUE    if rows, cols, eleSize, lda, or ldb <= 0<br />
     * CUBLAS_STATUS_MAPPING_ERROR    if error occurred accessing GPU memory<br />
     * CUBLAS_STATUS_SUCCESS          if the operation completed successfully<br />
     */
    public static int cublasGetMatrix (int rows, int cols, int elemSize, Pointer A, int lda, Pointer B, int ldb)
    {
        return checkResult(cublasGetMatrixNative(rows, cols, elemSize, A, lda, B, ldb));
    }

    /**
     * Extended wrapper for arrays of cuComplex values. Note that this method
     * only exists for convenience and compatibility with native C code. It
     * is much more efficient to provide a Pointer to a float array that may
     * store the complex numbers, where each pair of consecutive numbers in
     * the array describes the real- and imaginary part of one complex number.
     *
     * @see JCublas#cublasGetMatrix(int, int, int, Pointer, int, Pointer, int)
     */
    public static int cublasGetMatrix (int rows, int cols, Pointer A, int lda, cuComplex B[], int offsetB, int ldb)
    {
        ByteBuffer byteBufferB = ByteBuffer.allocateDirect(B.length * 4 * 2);
        byteBufferB.order(ByteOrder.nativeOrder());
        FloatBuffer floatBufferB = byteBufferB.asFloatBuffer();
        int status = cublasGetMatrixNative(rows, cols, 8, A, lda, Pointer.to(floatBufferB).withByteOffset(offsetB * 4 * 2), ldb);
        if (status == cublasStatus.CUBLAS_STATUS_SUCCESS)
        {
            floatBufferB.rewind();
            for (int c=0; c<cols; c++)
            {
                for (int r=0; r<rows; r++)
                {
                    int index = c * ldb + r + offsetB;
                    B[index].x = floatBufferB.get(index*2+0);
                    B[index].y = floatBufferB.get(index*2+1);
                }
            }
        }
        return checkResult(status);
    }























    //============================================================================
    // Memory management methods for double precision data:

    /**
     * Extended wrapper for arrays of cuDoubleComplex values. Note that this method
     * only exists for convenience and compatibility with native C code. It
     * is much more efficient to provide a Pointer to a double array containing
     * the complex numbers, where each pair of consecutive numbers in the array
     * describes the real- and imaginary part of one complex number.
     *
     *  @see JCublas#cublasSetVector(int, int, Pointer, int, Pointer, int)
     */
    public static int cublasSetVector (int n, cuDoubleComplex x[], int offsetx, int incx, Pointer y, int incy)
    {
        ByteBuffer byteBufferx = ByteBuffer.allocateDirect(x.length * 8 * 2);
        byteBufferx.order(ByteOrder.nativeOrder());
        DoubleBuffer doubleBufferx = byteBufferx.asDoubleBuffer();

        int indexx = offsetx;
        for (int i=0; i<n; i++, indexx+=incx)
        {
            doubleBufferx.put(indexx*2+0, x[indexx].x);
            doubleBufferx.put(indexx*2+1, x[indexx].y);
        }
        return checkResult(cublasSetVectorNative(n, 16, Pointer.to(doubleBufferx).withByteOffset(offsetx * 8 * 2), incx, y, incy));
    }




    /**
     * Extended wrapper for arrays of cuDoubleComplex values. Note that this method
     * only exists for convenience and compatibility with native C code. It
     * is much more efficient to provide a Pointer to a double array that may
     * store the complex numbers, where each pair of consecutive numbers in
     * the array describes the real- and imaginary part of one complex number.
     *
     * @see JCublas#cublasGetVector(int, int, Pointer, int, Pointer, int)
     */
    public static int cublasGetVector (int n, Pointer x, int incx, cuDoubleComplex y[], int offsety, int incy)
    {
        ByteBuffer byteBuffery = ByteBuffer.allocateDirect(y.length * 8 * 2);
        byteBuffery.order(ByteOrder.nativeOrder());
        DoubleBuffer doubleBuffery = byteBuffery.asDoubleBuffer();
        int status = cublasGetVectorNative(n, 16, x, incx, Pointer.to(doubleBuffery).withByteOffset(offsety * 8 * 2), incy);
        if (status == cublasStatus.CUBLAS_STATUS_SUCCESS)
        {
            doubleBuffery.rewind();
            int indexy = offsety;
            for (int i=0; i<n; i++, indexy+=incy)
            {
                y[indexy].x = doubleBuffery.get(indexy*2+0);
                y[indexy].y = doubleBuffery.get(indexy*2+1);
            }
        }
        return checkResult(status);
    }




    /**
     * Extended wrapper for arrays of cuDoubleComplex values. Note that this method
     * only exists for convenience and compatibility with native C code. It
     * is much more efficient to provide a Pointer to a double array containing
     * the complex numbers, where each pair of consecutive numbers in the array
     * describes the real- and imaginary part of one complex number.
     *
     * @see JCublas#cublasSetMatrix(int, int, int, Pointer, int, Pointer, int)
     */
    public static int cublasSetMatrix (int rows, int cols, cuDoubleComplex A[], int offsetA, int lda, Pointer B, int ldb)
    {
        ByteBuffer byteBufferA = ByteBuffer.allocateDirect(A.length * 8 * 2);
        byteBufferA.order(ByteOrder.nativeOrder());
        DoubleBuffer doubleBufferA = byteBufferA.asDoubleBuffer();
        for (int i=0; i<A.length; i++)
        {
            doubleBufferA.put(A[i].x);
            doubleBufferA.put(A[i].y);
        }
        return checkResult(cublasSetMatrixNative(rows, cols, 16, Pointer.to(doubleBufferA).withByteOffset(offsetA * 8 * 2), lda, B, ldb));
    }


    /**
     * Extended wrapper for arrays of cuDoubleComplex values. Note that this method
     * only exists for convenience and compatibility with native C code. It
     * is much more efficient to provide a Pointer to a double array that may
     * store the complex numbers, where each pair of consecutive numbers in
     * the array describes the real- and imaginary part of one complex number.
     *
     * @see JCublas#cublasGetMatrix(int, int, int, Pointer, int, Pointer, int)
     */
    public static int cublasGetMatrix (int rows, int cols, Pointer A, int lda, cuDoubleComplex B[], int offsetB, int ldb)
    {
        ByteBuffer byteBufferB = ByteBuffer.allocateDirect(B.length * 8 * 2);
        byteBufferB.order(ByteOrder.nativeOrder());
        DoubleBuffer doubleBufferB = byteBufferB.asDoubleBuffer();
        int status = cublasGetMatrixNative(rows, cols, 16, A, lda, Pointer.to(doubleBufferB).withByteOffset(offsetB * 8 * 2), ldb);
        if (status == cublasStatus.CUBLAS_STATUS_SUCCESS)
        {
            doubleBufferB.rewind();
            for (int c=0; c<cols; c++)
            {
                for (int r=0; r<rows; r++)
                {
                    int index = c * ldb + r + offsetB;
                    B[index].x = doubleBufferB.get(index*2+0);
                    B[index].y = doubleBufferB.get(index*2+1);
                }
            }
        }
        return checkResult(status);
    }






    /*
     * <pre>
     * Set the CUBLAS stream in which all subsequent CUBLAS kernel launches will run.
     *
     * cublasStatus
     * cublasSetKernelStream ( cudaStream_t stream )
     *
     * set the CUBLAS stream in which all subsequent CUBLAS kernel launches will run.
     * By default, if the CUBLAS stream is not set, all kernels will use the NULL
     * stream. This routine can be used to change the stream between kernels launches
     * and can be used also to set the CUBLAS stream back to NULL.
     *
     * Return Values
     * -------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_SUCCESS          if stream set successfully
     * </pre>
     */
    public static int cublasSetKernelStream (cudaStream_t stream)
    {
        return checkResult(cublasSetKernelStreamNative(stream));
    }
    private static native int cublasSetKernelStreamNative(cudaStream_t stream);


    /*
     * Wrapper for CUBLAS function.
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
    public static int cublasSetVectorAsync (int n, int elemSize, Pointer hostPtr, int incx, Pointer devicePtr, int incy, cudaStream_t stream)
    {
        return checkResult(cublasSetVectorAsyncNative(n, elemSize, hostPtr, incx, devicePtr, incy, stream));
    }
    /*
     * Wrapper for CUBLAS function.
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
    public static int cublasGetVectorAsync(int n, int elemSize, Pointer devicePtr, int incx, Pointer hostPtr, int incy, cudaStream_t stream)
    {
        return checkResult(cublasGetVectorAsyncNative(n, elemSize, devicePtr, incx, hostPtr, incy, stream));
    }

    /*
     * Wrapper for CUBLAS function.
     * <pre>
     * cublasStatus
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
    public static int cublasSetMatrixAsync (int rows, int cols, int elemSize, Pointer A, int lda, Pointer B, int ldb, cudaStream_t stream)
    {
        return checkResult(cublasSetMatrixAsyncNative(rows, cols, elemSize, A, lda, B, ldb, stream));
    }

    /*
     * Wrapper for CUBLAS function.
     * <pre>
     * cublasStatus
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
    public static int cublasGetMatrixAsync (int rows, int cols, int elemSize, Pointer A, int lda,  Pointer B, int ldb, cudaStream_t stream)
    {
        return checkResult(cublasGetMatrixAsyncNative(rows, cols, elemSize, A, lda, B, ldb, stream));
    }










    //============================================================================
    // Methods that are not handled by the code generator:

    /**
     * Wrapper for CUBLAS function.
     * <pre>
     * void
     * cublasSrotm (int n, float *x, int incx, float *y, int incy,
     *              const float* sparam)
     *
     * applies the modified Givens transformation, h, to the 2 x n matrix
     *
     *    ( transpose(x) )
     *    ( transpose(y) )
     *
     * The elements of x are in x[lx + i * incx], i = 0 to n-1, where lx = 1 if
     * incx >= 0, else lx = 1 + (1 - n) * incx, and similarly for y using ly and
     * incy. With sparam[0] = sflag, h has one of the following forms:
     *
     *        sflag = -1.0f   sflag = 0.0f    sflag = 1.0f    sflag = -2.0f
     *
     *        (sh00  sh01)    (1.0f  sh01)    (sh00  1.0f)    (1.0f  0.0f)
     *    h = (          )    (          )    (          )    (          )
     *        (sh10  sh11)    (sh10  1.0f)    (-1.0f sh11)    (0.0f  1.0f)
     *
     * Input
     * -----
     * n      number of elements in input vectors
     * x      single precision vector with n elements
     * incx   storage spacing between elements of x
     * y      single precision vector with n elements
     * incy   storage spacing between elements of y
     * sparam 5-element vector. sparam[0] is sflag described above. sparam[1]
     *        through sparam[4] contain the 2x2 rotation matrix h: sparam[1]
     *        contains sh00, sparam[2] contains sh10, sparam[3] contains sh01,
     *        and sprams[4] contains sh11.
     *
     * Output
     * ------
     * x     rotated vector x (unchanged if n <= 0)
     * y     rotated vector y (unchanged if n <= 0)
     *
     * Reference: http://www.netlib.org/blas/srotm.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */
    public static void cublasSrotm(int n, Pointer x, int incx, Pointer y, int incy, float sparam[])
    {
        cublasSrotmNative(n, x, incx, y, incy, sparam);
        checkResultBLAS();
    }
    private static native void cublasSrotmNative(int n, Pointer x, int incx, Pointer y, int incy, float sparam[]);

    /**
     * Wrapper for CUBLAS function.
     * <pre>
     * void
     * cublasSrotmg (float *psd1, float *psd2, float *psx1, const float *psy1,
     *                float *sparam)
     *
     * constructs the modified Givens transformation matrix h which zeros
     * the second component of the 2-vector transpose(sqrt(sd1)*sx1,sqrt(sd2)*sy1).
     * With sparam[0] = sflag, h has one of the following forms:
     *
     *        sflag = -1.0f   sflag = 0.0f    sflag = 1.0f    sflag = -2.0f
     *
     *        (sh00  sh01)    (1.0f  sh01)    (sh00  1.0f)    (1.0f  0.0f)
     *    h = (          )    (          )    (          )    (          )
     *        (sh10  sh11)    (sh10  1.0f)    (-1.0f sh11)    (0.0f  1.0f)
     *
     * sparam[1] through sparam[4] contain sh00, sh10, sh01, sh11,
     * respectively. Values of 1.0f, -1.0f, or 0.0f implied by the value
     * of sflag are not stored in sparam.
     *
     * Input
     * -----
     * sd1    single precision scalar
     * sd2    single precision scalar
     * sx1    single precision scalar
     * sy1    single precision scalar
     *
     * Output
     * ------
     * sd1    changed to represent the effect of the transformation
     * sd2    changed to represent the effect of the transformation
     * sx1    changed to represent the effect of the transformation
     * sparam 5-element vector. sparam[0] is sflag described above. sparam[1]
     *        through sparam[4] contain the 2x2 rotation matrix h: sparam[1]
     *        contains sh00, sparam[2] contains sh10, sparam[3] contains sh01,
     *        and sprams[4] contains sh11.
     *
     * Reference: http://www.netlib.org/blas/srotmg.f
     *
     * This functions does not set any error status.
     * </pre>
     */
    public static void cublasSrotmg(float sd1[], float sd2[], float sx1[], float sy1, float sparam[])
    {
        cublasSrotmgNative(sd1, sd2, sx1, sy1, sparam);
        checkResultBLAS();
    }
    private static native void cublasSrotmgNative(float sd1[], float sd2[], float sx1[], float sy1, float sparam[]);





    /**
     * Wrapper for CUBLAS function.
     * <pre>
     * void
     * cublasDrotm (int n, double *x, int incx, double *y, int incy,
     *              const double* sparam)
     *
     * applies the modified Givens transformation, h, to the 2 x n matrix
     *
     *    ( transpose(x) )
     *    ( transpose(y) )
     *
     * The elements of x are in x[lx + i * incx], i = 0 to n-1, where lx = 1 if
     * incx >= 0, else lx = 1 + (1 - n) * incx, and similarly for y using ly and
     * incy. With sparam[0] = sflag, h has one of the following forms:
     *
     *        sflag = -1.0    sflag = 0.0     sflag = 1.0     sflag = -2.0
     *
     *        (sh00  sh01)    (1.0   sh01)    (sh00   1.0)    (1.0    0.0)
     *    h = (          )    (          )    (          )    (          )
     *        (sh10  sh11)    (sh10   1.0)    (-1.0  sh11)    (0.0    1.0)
     *
     * Input
     * -----
     * n      number of elements in input vectors
     * x      double-precision vector with n elements
     * incx   storage spacing between elements of x
     * y      double-precision vector with n elements
     * incy   storage spacing between elements of y
     * sparam 5-element vector. sparam[0] is sflag described above. sparam[1]
     *        through sparam[4] contain the 2x2 rotation matrix h: sparam[1]
     *        contains sh00, sparam[2] contains sh10, sparam[3] contains sh01,
     *        and sprams[4] contains sh11.
     *
     * Output
     * ------
     * x     rotated vector x (unchanged if n <= 0)
     * y     rotated vector y (unchanged if n <= 0)
     *
     * Reference: http://www.netlib.org/blas/drotm.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */
    public static void cublasDrotm(int n, Pointer x, int incx, Pointer y, int incy, double sparam[])
    {
        cublasDrotmNative(n, x, incx, y, incy, sparam);
        checkResultBLAS();
    }
    private static native void cublasDrotmNative(int n, Pointer x, int incx, Pointer y, int incy, double sparam[]);


    /**
     * Wrapper for CUBLAS function.
     * <pre>
     * void
     * cublasDrotmg (double *psd1, double *psd2, double *psx1, const double *psy1,
     *               double *sparam)
     *
     * constructs the modified Givens transformation matrix h which zeros
     * the second component of the 2-vector transpose(sqrt(sd1)*sx1,sqrt(sd2)*sy1).
     * With sparam[0] = sflag, h has one of the following forms:
     *
     *        sflag = -1.0    sflag = 0.0     sflag = 1.0     sflag = -2.0
     *
     *        (sh00  sh01)    (1.0   sh01)    (sh00   1.0)    (1.0    0.0)
     *    h = (          )    (          )    (          )    (          )
     *        (sh10  sh11)    (sh10   1.0)    (-1.0  sh11)    (0.0    1.0)
     *
     * sparam[1] through sparam[4] contain sh00, sh10, sh01, sh11,
     * respectively. Values of 1.0, -1.0, or 0.0 implied by the value
     * of sflag are not stored in sparam.
     *
     * Input
     * -----
     * sd1    single precision scalar
     * sd2    single precision scalar
     * sx1    single precision scalar
     * sy1    single precision scalar
     *
     * Output
     * ------
     * sd1    changed to represent the effect of the transformation
     * sd2    changed to represent the effect of the transformation
     * sx1    changed to represent the effect of the transformation
     * sparam 5-element vector. sparam[0] is sflag described above. sparam[1]
     *        through sparam[4] contain the 2x2 rotation matrix h: sparam[1]
     *        contains sh00, sparam[2] contains sh10, sparam[3] contains sh01,
     *        and sprams[4] contains sh11.
     *
     * Reference: http://www.netlib.org/blas/drotmg.f
     *
     * This functions does not set any error status.
     *
     * </pre>
     */
    public static void cublasDrotmg(double sd1[], double sd2[], double sx1[], double sy1, double sparam[])
    {
        cublasDrotmgNative(sd1, sd2, sx1, sy1, sparam);
        checkResultBLAS();
    }
    private static native void cublasDrotmgNative(double sd1[], double sd2[], double sx1[], double sy1, double sparam[]);






















    //============================================================================
    // Auto-generated part:

    /**
     * <pre>
     * int
     * cublasIsamax (int n, const float *x, int incx)
     *
     * finds the smallest index of the maximum magnitude element of single
     * precision vector x; that is, the result is the first i, i = 0 to n - 1,
     * that maximizes abs(x[1 + i * incx])).
     *
     * Input
     * -----
     * n      number of elements in input vector
     * x      single precision vector with n elements
     * incx   storage spacing between elements of x
     *
     * Output
     * ------
     * returns the smallest index (0 if n <= 0 or incx <= 0)
     *
     * Reference: http://www.netlib.org/blas/isamax.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static int cublasIsamax(int n, Pointer x, int incx)
    {
        int result = cublasIsamaxNative(n, x, incx);
        checkResultBLAS();
        return result;
    }
    private static native int cublasIsamaxNative(int n, Pointer x, int incx);





    /**
     * <pre>
     * int
     * cublasIsamin (int n, const float *x, int incx)
     *
     * finds the smallest index of the minimum magnitude element of single
     * precision vector x; that is, the result is the first i, i = 0 to n - 1,
     * that minimizes abs(x[1 + i * incx])).
     *
     * Input
     * -----
     * n      number of elements in input vector
     * x      single precision vector with n elements
     * incx   storage spacing between elements of x
     *
     * Output
     * ------
     * returns the smallest index (0 if n <= 0 or incx <= 0)
     *
     * Reference: http://www.netlib.org/scilib/blass.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static int cublasIsamin(int n, Pointer x, int incx)
    {
        int result = cublasIsaminNative(n, x, incx);
        checkResultBLAS();
        return result;
    }
    private static native int cublasIsaminNative(int n, Pointer x, int incx);





    /**
     * <pre>
     * float
     * cublasSasum (int n, const float *x, int incx)
     *
     * computes the sum of the absolute values of the elements of single
     * precision vector x; that is, the result is the sum from i = 0 to n - 1 of
     * abs(x[1 + i * incx]).
     *
     * Input
     * -----
     * n      number of elements in input vector
     * x      single precision vector with n elements
     * incx   storage spacing between elements of x
     *
     * Output
     * ------
     * returns the single precision sum of absolute values
     * (0 if n <= 0 or incx <= 0, or if an error occurs)
     *
     * Reference: http://www.netlib.org/blas/sasum.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static float cublasSasum(int n, Pointer x, int incx)
    {
        float result = cublasSasumNative(n, x, incx);
        checkResultBLAS();
        return result;
    }
    private static native float cublasSasumNative(int n, Pointer x, int incx);





    /**
     * <pre>
     * void
     * cublasSaxpy (int n, float alpha, const float *x, int incx, float *y,
     *              int incy)
     *
     * multiplies single precision vector x by single precision scalar alpha
     * and adds the result to single precision vector y; that is, it overwrites
     * single precision y with single precision alpha * x + y. For i = 0 to n - 1,
     * it replaces y[ly + i * incy] with alpha * x[lx + i * incx] + y[ly + i *
     * incy], where lx = 1 if incx >= 0, else lx = 1 +(1 - n) * incx, and ly is
     * defined in a similar way using incy.
     *
     * Input
     * -----
     * n      number of elements in input vectors
     * alpha  single precision scalar multiplier
     * x      single precision vector with n elements
     * incx   storage spacing between elements of x
     * y      single precision vector with n elements
     * incy   storage spacing between elements of y
     *
     * Output
     * ------
     * y      single precision result (unchanged if n <= 0)
     *
     * Reference: http://www.netlib.org/blas/saxpy.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasSaxpy(int n, float alpha, Pointer x, int incx, Pointer y, int incy)
    {
        cublasSaxpyNative(n, alpha, x, incx, y, incy);
        checkResultBLAS();
    }
    private static native void cublasSaxpyNative(int n, float alpha, Pointer x, int incx, Pointer y, int incy);





    /**
     * <pre>
     * void
     * cublasScopy (int n, const float *x, int incx, float *y, int incy)
     *
     * copies the single precision vector x to the single precision vector y. For
     * i = 0 to n-1, copies x[lx + i * incx] to y[ly + i * incy], where lx = 1 if
     * incx >= 0, else lx = 1 + (1 - n) * incx, and ly is defined in a similar
     * way using incy.
     *
     * Input
     * -----
     * n      number of elements in input vectors
     * x      single precision vector with n elements
     * incx   storage spacing between elements of x
     * y      single precision vector with n elements
     * incy   storage spacing between elements of y
     *
     * Output
     * ------
     * y      contains single precision vector x
     *
     * Reference: http://www.netlib.org/blas/scopy.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasScopy(int n, Pointer x, int incx, Pointer y, int incy)
    {
        cublasScopyNative(n, x, incx, y, incy);
        checkResultBLAS();
    }
    private static native void cublasScopyNative(int n, Pointer x, int incx, Pointer y, int incy);





    /**
     * <pre>
     * float
     * cublasSdot (int n, const float *x, int incx, const float *y, int incy)
     *
     * computes the dot product of two single precision vectors. It returns the
     * dot product of the single precision vectors x and y if successful, and
     * 0.0f otherwise. It computes the sum for i = 0 to n - 1 of x[lx + i *
     * incx] * y[ly + i * incy], where lx = 1 if incx >= 0, else lx = 1 + (1 - n)
     * *incx, and ly is defined in a similar way using incy.
     *
     * Input
     * -----
     * n      number of elements in input vectors
     * x      single precision vector with n elements
     * incx   storage spacing between elements of x
     * y      single precision vector with n elements
     * incy   storage spacing between elements of y
     *
     * Output
     * ------
     * returns single precision dot product (zero if n <= 0)
     *
     * Reference: http://www.netlib.org/blas/sdot.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has nor been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to execute on GPU
     * </pre>
     */

    public static float cublasSdot(int n, Pointer x, int incx, Pointer y, int incy)
    {
        float result = cublasSdotNative(n, x, incx, y, incy);
        checkResultBLAS();
        return result;
    }
    private static native float cublasSdotNative(int n, Pointer x, int incx, Pointer y, int incy);





    /**
     * <pre>
     * float
     * cublasSnrm2 (int n, const float *x, int incx)
     *
     * computes the Euclidean norm of the single precision n-vector x (with
     * storage increment incx). This code uses a multiphase model of
     * accumulation to avoid intermediate underflow and overflow.
     *
     * Input
     * -----
     * n      number of elements in input vector
     * x      single precision vector with n elements
     * incx   storage spacing between elements of x
     *
     * Output
     * ------
     * returns Euclidian norm (0 if n <= 0 or incx <= 0, or if an error occurs)
     *
     * Reference: http://www.netlib.org/blas/snrm2.f
     * Reference: http://www.netlib.org/slatec/lin/snrm2.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static float cublasSnrm2(int n, Pointer x, int incx)
    {
        float result = cublasSnrm2Native(n, x, incx);
        checkResultBLAS();
        return result;
    }
    private static native float cublasSnrm2Native(int n, Pointer x, int incx);





    /**
     * <pre>
     * void
     * cublasSrot (int n, float *x, int incx, float *y, int incy, float sc,
     *             float ss)
     *
     * multiplies a 2x2 matrix ( sc ss) with the 2xn matrix ( transpose(x) )
     *                         (-ss sc)                     ( transpose(y) )
     *
     * The elements of x are in x[lx + i * incx], i = 0 ... n - 1, where lx = 1 if
     * incx >= 0, else lx = 1 + (1 - n) * incx, and similarly for y using ly and
     * incy.
     *
     * Input
     * -----
     * n      number of elements in input vectors
     * x      single precision vector with n elements
     * incx   storage spacing between elements of x
     * y      single precision vector with n elements
     * incy   storage spacing between elements of y
     * sc     element of rotation matrix
     * ss     element of rotation matrix
     *
     * Output
     * ------
     * x      rotated vector x (unchanged if n <= 0)
     * y      rotated vector y (unchanged if n <= 0)
     *
     * Reference  http://www.netlib.org/blas/srot.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasSrot(int n, Pointer x, int incx, Pointer y, int incy, float sc, float ss)
    {
        cublasSrotNative(n, x, incx, y, incy, sc, ss);
        checkResultBLAS();
    }
    private static native void cublasSrotNative(int n, Pointer x, int incx, Pointer y, int incy, float sc, float ss);





    /**
     * <pre>
     * void
     * cublasSrotg (float *host_sa, float *host_sb, float *host_sc, float *host_ss)
     *
     * constructs the Givens tranformation
     *
     *        ( sc  ss )
     *    G = (        ) ,  sc^2 + ss^2 = 1,
     *        (-ss  sc )
     *
     * which zeros the second entry of the 2-vector transpose(sa, sb).
     *
     * The quantity r = (+/-) sqrt (sa^2 + sb^2) overwrites sa in storage. The
     * value of sb is overwritten by a value z which allows sc and ss to be
     * recovered by the following algorithm:
     *
     *    if z=1          set sc = 0.0 and ss = 1.0
     *    if abs(z) < 1   set sc = sqrt(1-z^2) and ss = z
     *    if abs(z) > 1   set sc = 1/z and ss = sqrt(1-sc^2)
     *
     * The function srot (n, x, incx, y, incy, sc, ss) normally is called next
     * to apply the transformation to a 2 x n matrix.
     * Note that is function is provided for completeness and run exclusively
     * on the Host.
     *
     * Input
     * -----
     * sa     single precision scalar
     * sb     single precision scalar
     *
     * Output
     * ------
     * sa     single precision r
     * sb     single precision z
     * sc     single precision result
     * ss     single precision result
     *
     * Reference: http://www.netlib.org/blas/srotg.f
     *
     * This function does not set any error status.
     * </pre>
     */

    public static void cublasSrotg(Pointer host_sa, Pointer host_sb, Pointer host_sc, Pointer host_ss)
    {
        cublasSrotgNative(host_sa, host_sb, host_sc, host_ss);
        checkResultBLAS();
    }
    private static native void cublasSrotgNative(Pointer host_sa, Pointer host_sb, Pointer host_sc, Pointer host_ss);





    /**
     * <pre>
     * void
     * sscal (int n, float alpha, float *x, int incx)
     *
     * replaces single precision vector x with single precision alpha * x. For i
     * = 0 to n - 1, it replaces x[ix + i * incx] with alpha * x[ix + i * incx],
     * where ix = 1 if incx >= 0, else ix = 1 + (1 - n) * incx.
     *
     * Input
     * -----
     * n      number of elements in input vectors
     * alpha  single precision scalar multiplier
     * x      single precision vector with n elements
     * incx   storage spacing between elements of x
     *
     * Output
     * ------
     * x      single precision result (unchanged if n <= 0 or incx <= 0)
     *
     * Reference: http://www.netlib.org/blas/sscal.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasSscal(int n, float alpha, Pointer x, int incx)
    {
        cublasSscalNative(n, alpha, x, incx);
        checkResultBLAS();
    }
    private static native void cublasSscalNative(int n, float alpha, Pointer x, int incx);





    /**
     * <pre>
     * void
     * cublasSswap (int n, float *x, int incx, float *y, int incy)
     *
     * interchanges the single-precision vector x with the single-precision vector y.
     * For i = 0 to n-1, interchanges x[lx + i * incx] with y[ly + i * incy], where
     * lx = 1 if incx >= 0, else lx = 1 + (1 - n) * incx, and ly is defined in a
     * similar way using incy.
     *
     * Input
     * -----
     * n      number of elements in input vectors
     * x      single precision vector with n elements
     * incx   storage spacing between elements of x
     * y      single precision vector with n elements
     * incy   storage spacing between elements of y
     *
     * Output
     * ------
     * x      contains single precision vector y
     * y      contains single precision vector x
     *
     * Reference: http://www.netlib.org/blas/sscal.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasSswap(int n, Pointer x, int incx, Pointer y, int incy)
    {
        cublasSswapNative(n, x, incx, y, incy);
        checkResultBLAS();
    }
    private static native void cublasSswapNative(int n, Pointer x, int incx, Pointer y, int incy);





    /**
     * <pre>
     * void
     * cublasCaxpy (int n, cuComplex alpha, const cuComplex *x, int incx,
     *              cuComplex *y, int incy)
     *
     * multiplies single-complex vector x by single-complex scalar alpha and adds
     * the result to single-complex vector y; that is, it overwrites single-complex
     * y with single-complex alpha * x + y. For i = 0 to n - 1, it replaces
     * y[ly + i * incy] with alpha * x[lx + i * incx] + y[ly + i * incy], where
     * lx = 0 if incx >= 0, else lx = 1 + (1 - n) * incx, and ly is defined in a
     * similar way using incy.
     *
     * Input
     * -----
     * n      number of elements in input vectors
     * alpha  single-complex scalar multiplier
     * x      single-complex vector with n elements
     * incx   storage spacing between elements of x
     * y      single-complex vector with n elements
     * incy   storage spacing between elements of y
     *
     * Output
     * ------
     * y      single-complex result (unchanged if n <= 0)
     *
     * Reference: http://www.netlib.org/blas/caxpy.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasCaxpy(int n, cuComplex alpha, Pointer x, int incx, Pointer y, int incy)
    {
        cublasCaxpyNative(n, alpha, x, incx, y, incy);
        checkResultBLAS();
    }
    private static native void cublasCaxpyNative(int n, cuComplex alpha, Pointer x, int incx, Pointer y, int incy);





    /**
     * <pre>
     * void
     * cublasCcopy (int n, const cuComplex *x, int incx, cuComplex *y, int incy)
     *
     * copies the single-complex vector x to the single-complex vector y. For
     * i = 0 to n-1, copies x[lx + i * incx] to y[ly + i * incy], where lx = 1 if
     * incx >= 0, else lx = 1 + (1 - n) * incx, and ly is defined in a similar
     * way using incy.
     *
     * Input
     * -----
     * n      number of elements in input vectors
     * x      single-complex vector with n elements
     * incx   storage spacing between elements of x
     * y      single-complex vector with n elements
     * incy   storage spacing between elements of y
     *
     * Output
     * ------
     * y      contains single complex vector x
     *
     * Reference: http://www.netlib.org/blas/ccopy.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasCcopy(int n, Pointer x, int incx, Pointer y, int incy)
    {
        cublasCcopyNative(n, x, incx, y, incy);
        checkResultBLAS();
    }
    private static native void cublasCcopyNative(int n, Pointer x, int incx, Pointer y, int incy);





    /**
     * <pre>
     * void
     * cublasZcopy (int n, const cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy)
     *
     * copies the double-complex vector x to the double-complex vector y. For
     * i = 0 to n-1, copies x[lx + i * incx] to y[ly + i * incy], where lx = 1 if
     * incx >= 0, else lx = 1 + (1 - n) * incx, and ly is defined in a similar
     * way using incy.
     *
     * Input
     * -----
     * n      number of elements in input vectors
     * x      double-complex vector with n elements
     * incx   storage spacing between elements of x
     * y      double-complex vector with n elements
     * incy   storage spacing between elements of y
     *
     * Output
     * ------
     * y      contains double complex vector x
     *
     * Reference: http://www.netlib.org/blas/zcopy.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasZcopy(int n, Pointer x, int incx, Pointer y, int incy)
    {
        cublasZcopyNative(n, x, incx, y, incy);
        checkResultBLAS();
    }
    private static native void cublasZcopyNative(int n, Pointer x, int incx, Pointer y, int incy);





    /**
     * <pre>
     * void
     * cublasCscal (int n, cuComplex alpha, cuComplex *x, int incx)
     *
     * replaces single-complex vector x with single-complex alpha * x. For i
     * = 0 to n - 1, it replaces x[ix + i * incx] with alpha * x[ix + i * incx],
     * where ix = 1 if incx >= 0, else ix = 1 + (1 - n) * incx.
     *
     * Input
     * -----
     * n      number of elements in input vectors
     * alpha  single-complex scalar multiplier
     * x      single-complex vector with n elements
     * incx   storage spacing between elements of x
     *
     * Output
     * ------
     * x      single-complex result (unchanged if n <= 0 or incx <= 0)
     *
     * Reference: http://www.netlib.org/blas/cscal.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasCscal(int n, cuComplex alpha, Pointer x, int incx)
    {
        cublasCscalNative(n, alpha, x, incx);
        checkResultBLAS();
    }
    private static native void cublasCscalNative(int n, cuComplex alpha, Pointer x, int incx);





    /**
     * <pre>
     * void
     * cublasCrotg (cuComplex *host_ca, cuComplex cb, float *host_sc, cuComplex *host_cs)
     *
     * constructs the complex Givens tranformation
     *
     *        ( sc  cs )
     *    G = (        ) ,  sc^2 + cabs(cs)^2 = 1,
     *        (-cs  sc )
     *
     * which zeros the second entry of the complex 2-vector transpose(ca, cb).
     *
     * The quantity ca/cabs(ca)*norm(ca,cb) overwrites ca in storage. The
     * function crot (n, x, incx, y, incy, sc, cs) is normally called next
     * to apply the transformation to a 2 x n matrix.
     * Note that is function is provided for completeness and run exclusively
     * on the Host.
     *
     * Input
     * -----
     * ca     single-precision complex precision scalar
     * cb     single-precision complex scalar
     *
     * Output
     * ------
     * ca     single-precision complex ca/cabs(ca)*norm(ca,cb)
     * sc     single-precision cosine component of rotation matrix
     * cs     single-precision complex sine component of rotation matrix
     *
     * Reference: http://www.netlib.org/blas/crotg.f
     *
     * This function does not set any error status.
     * </pre>
     */

    public static void cublasCrotg(Pointer host_ca, cuComplex cb, Pointer host_sc, Pointer host_cs)
    {
        cublasCrotgNative(host_ca, cb, host_sc, host_cs);
        checkResultBLAS();
    }
    private static native void cublasCrotgNative(Pointer host_ca, cuComplex cb, Pointer host_sc, Pointer host_cs);





    /**
     * <pre>
     * void
     * cublasCrot (int n, cuComplex *x, int incx, cuComplex *y, int incy, float sc,
     *             cuComplex cs)
     *
     * multiplies a 2x2 matrix ( sc       cs) with the 2xn matrix ( transpose(x) )
     *                         (-conj(cs) sc)                     ( transpose(y) )
     *
     * The elements of x are in x[lx + i * incx], i = 0 ... n - 1, where lx = 1 if
     * incx >= 0, else lx = 1 + (1 - n) * incx, and similarly for y using ly and
     * incy.
     *
     * Input
     * -----
     * n      number of elements in input vectors
     * x      single-precision complex vector with n elements
     * incx   storage spacing between elements of x
     * y      single-precision complex vector with n elements
     * incy   storage spacing between elements of y
     * sc     single-precision cosine component of rotation matrix
     * cs     single-precision complex sine component of rotation matrix
     *
     * Output
     * ------
     * x      rotated single-precision complex vector x (unchanged if n <= 0)
     * y      rotated single-precision complex vector y (unchanged if n <= 0)
     *
     * Reference: http://netlib.org/lapack/explore-html/crot.f.html
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasCrot(int n, Pointer x, int incx, Pointer y, int incy, float c, cuComplex s)
    {
        cublasCrotNative(n, x, incx, y, incy, c, s);
        checkResultBLAS();
    }
    private static native void cublasCrotNative(int n, Pointer x, int incx, Pointer y, int incy, float c, cuComplex s);





    /**
     * <pre>
     * void
     * csrot (int n, cuComplex *x, int incx, cuCumplex *y, int incy, float c,
     *        float s)
     *
     * multiplies a 2x2 rotation matrix ( c s) with a 2xn matrix ( transpose(x) )
     *                                  (-s c)                   ( transpose(y) )
     *
     * The elements of x are in x[lx + i * incx], i = 0 ... n - 1, where lx = 1 if
     * incx >= 0, else lx = 1 + (1 - n) * incx, and similarly for y using ly and
     * incy.
     *
     * Input
     * -----
     * n      number of elements in input vectors
     * x      single-precision complex vector with n elements
     * incx   storage spacing between elements of x
     * y      single-precision complex vector with n elements
     * incy   storage spacing between elements of y
     * c      cosine component of rotation matrix
     * s      sine component of rotation matrix
     *
     * Output
     * ------
     * x      rotated vector x (unchanged if n <= 0)
     * y      rotated vector y (unchanged if n <= 0)
     *
     * Reference  http://www.netlib.org/blas/csrot.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasCsrot(int n, Pointer x, int incx, Pointer y, int incy, float c, float s)
    {
        cublasCsrotNative(n, x, incx, y, incy, c, s);
        checkResultBLAS();
    }
    private static native void cublasCsrotNative(int n, Pointer x, int incx, Pointer y, int incy, float c, float s);





    /**
     * <pre>
     * void
     * cublasCsscal (int n, float alpha, cuComplex *x, int incx)
     *
     * replaces single-complex vector x with single-complex alpha * x. For i
     * = 0 to n - 1, it replaces x[ix + i * incx] with alpha * x[ix + i * incx],
     * where ix = 1 if incx >= 0, else ix = 1 + (1 - n) * incx.
     *
     * Input
     * -----
     * n      number of elements in input vectors
     * alpha  single precision scalar multiplier
     * x      single-complex vector with n elements
     * incx   storage spacing between elements of x
     *
     * Output
     * ------
     * x      single-complex result (unchanged if n <= 0 or incx <= 0)
     *
     * Reference: http://www.netlib.org/blas/csscal.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasCsscal(int n, float alpha, Pointer x, int incx)
    {
        cublasCsscalNative(n, alpha, x, incx);
        checkResultBLAS();
    }
    private static native void cublasCsscalNative(int n, float alpha, Pointer x, int incx);





    /**
     * <pre>
     * void
     * cublasCswap (int n, const cuComplex *x, int incx, cuComplex *y, int incy)
     *
     * interchanges the single-complex vector x with the single-complex vector y.
     * For i = 0 to n-1, interchanges x[lx + i * incx] with y[ly + i * incy], where
     * lx = 1 if incx >= 0, else lx = 1 + (1 - n) * incx, and ly is defined in a
     * similar way using incy.
     *
     * Input
     * -----
     * n      number of elements in input vectors
     * x      single-complex vector with n elements
     * incx   storage spacing between elements of x
     * y      single-complex vector with n elements
     * incy   storage spacing between elements of y
     *
     * Output
     * ------
     * x      contains-single complex vector y
     * y      contains-single complex vector x
     *
     * Reference: http://www.netlib.org/blas/cswap.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasCswap(int n, Pointer x, int incx, Pointer y, int incy)
    {
        cublasCswapNative(n, x, incx, y, incy);
        checkResultBLAS();
    }
    private static native void cublasCswapNative(int n, Pointer x, int incx, Pointer y, int incy);





    /**
     * <pre>
     * void
     * cublasZswap (int n, const cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy)
     *
     * interchanges the double-complex vector x with the double-complex vector y.
     * For i = 0 to n-1, interchanges x[lx + i * incx] with y[ly + i * incy], where
     * lx = 1 if incx >= 0, else lx = 1 + (1 - n) * incx, and ly is defined in a
     * similar way using incy.
     *
     * Input
     * -----
     * n      number of elements in input vectors
     * x      double-complex vector with n elements
     * incx   storage spacing between elements of x
     * y      double-complex vector with n elements
     * incy   storage spacing between elements of y
     *
     * Output
     * ------
     * x      contains-double complex vector y
     * y      contains-double complex vector x
     *
     * Reference: http://www.netlib.org/blas/zswap.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasZswap(int n, Pointer x, int incx, Pointer y, int incy)
    {
        cublasZswapNative(n, x, incx, y, incy);
        checkResultBLAS();
    }
    private static native void cublasZswapNative(int n, Pointer x, int incx, Pointer y, int incy);





    /**
     * <pre>
     * cuComplex
     * cdotu (int n, const cuComplex *x, int incx, const cuComplex *y, int incy)
     *
     * computes the dot product of two single-complex vectors. It returns the
     * dot product of the single-complex vectors x and y if successful, and complex
     * zero otherwise. It computes the sum for i = 0 to n - 1 of x[lx + i * incx] *
     * y[ly + i * incy], where lx = 1 if incx >= 0, else lx = 1 + (1 - n) * incx;
     * ly is defined in a similar way using incy.
     *
     * Input
     * -----
     * n      number of elements in input vectors
     * x      single-complex vector with n elements
     * incx   storage spacing between elements of x
     * y      single-complex vector with n elements
     * incy   storage spacing between elements of y
     *
     * Output
     * ------
     * returns single-complex dot product (zero if n <= 0)
     *
     * Reference: http://www.netlib.org/blas/cdotu.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has nor been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to execute on GPU
     * </pre>
     */

    public static cuComplex cublasCdotu(int n, Pointer x, int incx, Pointer y, int incy)
    {
        cuComplex result = cublasCdotuNative(n, x, incx, y, incy);
        checkResultBLAS();
        return result;
    }
    private static native cuComplex cublasCdotuNative(int n, Pointer x, int incx, Pointer y, int incy);





    /**
     * <pre>
     * cuComplex
     * cublasCdotc (int n, const cuComplex *x, int incx, const cuComplex *y,
     *              int incy)
     *
     * computes the dot product of two single-complex vectors. It returns the
     * dot product of the single-complex vectors x and y if successful, and complex
     * zero otherwise. It computes the sum for i = 0 to n - 1 of x[lx + i * incx] *
     * y[ly + i * incy], where lx = 1 if incx >= 0, else lx = 1 + (1 - n) * incx;
     * ly is defined in a similar way using incy.
     *
     * Input
     * -----
     * n      number of elements in input vectors
     * x      single-complex vector with n elements
     * incx   storage spacing between elements of x
     * y      single-complex vector with n elements
     * incy   storage spacing between elements of y
     *
     * Output
     * ------
     * returns single-complex dot product (zero if n <= 0)
     *
     * Reference: http://www.netlib.org/blas/cdotc.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has nor been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to execute on GPU
     * </pre>
     */

    public static cuComplex cublasCdotc(int n, Pointer x, int incx, Pointer y, int incy)
    {
        cuComplex result = cublasCdotcNative(n, x, incx, y, incy);
        checkResultBLAS();
        return result;
    }
    private static native cuComplex cublasCdotcNative(int n, Pointer x, int incx, Pointer y, int incy);





    /**
     * <pre>
     * int
     * cublasIcamax (int n, const float *x, int incx)
     *
     * finds the smallest index of the element having maximum absolute value
     * in single-complex vector x; that is, the result is the first i, i = 0
     * to n - 1 that maximizes abs(real(x[1+i*incx]))+abs(imag(x[1 + i * incx])).
     *
     * Input
     * -----
     * n      number of elements in input vector
     * x      single-complex vector with n elements
     * incx   storage spacing between elements of x
     *
     * Output
     * ------
     * returns the smallest index (0 if n <= 0 or incx <= 0)
     *
     * Reference: http://www.netlib.org/blas/icamax.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static int cublasIcamax(int n, Pointer x, int incx)
    {
        int result = cublasIcamaxNative(n, x, incx);
        checkResultBLAS();
        return result;
    }
    private static native int cublasIcamaxNative(int n, Pointer x, int incx);





    /**
     * <pre>
     * int
     * cublasIcamin (int n, const float *x, int incx)
     *
     * finds the smallest index of the element having minimum absolute value
     * in single-complex vector x; that is, the result is the first i, i = 0
     * to n - 1 that minimizes abs(real(x[1+i*incx]))+abs(imag(x[1 + i * incx])).
     *
     * Input
     * -----
     * n      number of elements in input vector
     * x      single-complex vector with n elements
     * incx   storage spacing between elements of x
     *
     * Output
     * ------
     * returns the smallest index (0 if n <= 0 or incx <= 0)
     *
     * Reference: see ICAMAX.
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static int cublasIcamin(int n, Pointer x, int incx)
    {
        int result = cublasIcaminNative(n, x, incx);
        checkResultBLAS();
        return result;
    }
    private static native int cublasIcaminNative(int n, Pointer x, int incx);





    /**
     * <pre>
     * float
     * cublasScasum (int n, const cuDouble *x, int incx)
     *
     * takes the sum of the absolute values of a complex vector and returns a
     * single precision result. Note that this is not the L1 norm of the vector.
     * The result is the sum from 0 to n-1 of abs(real(x[ix+i*incx])) +
     * abs(imag(x(ix+i*incx))), where ix = 1 if incx <= 0, else ix = 1+(1-n)*incx.
     *
     * Input
     * -----
     * n      number of elements in input vector
     * x      single-complex vector with n elements
     * incx   storage spacing between elements of x
     *
     * Output
     * ------
     * returns the single precision sum of absolute values of real and imaginary
     * parts (0 if n <= 0 or incx <= 0, or if an error occurs)
     *
     * Reference: http://www.netlib.org/blas/scasum.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static float cublasScasum(int n, Pointer x, int incx)
    {
        float result = cublasScasumNative(n, x, incx);
        checkResultBLAS();
        return result;
    }
    private static native float cublasScasumNative(int n, Pointer x, int incx);





    /**
     * <pre>
     * float
     * cublasScnrm2 (int n, const cuComplex *x, int incx)
     *
     * computes the Euclidean norm of the single-complex n-vector x. This code
     * uses simple scaling to avoid intermediate underflow and overflow.
     *
     * Input
     * -----
     * n      number of elements in input vector
     * x      single-complex vector with n elements
     * incx   storage spacing between elements of x
     *
     * Output
     * ------
     * returns Euclidian norm (0 if n <= 0 or incx <= 0, or if an error occurs)
     *
     * Reference: http://www.netlib.org/blas/scnrm2.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static float cublasScnrm2(int n, Pointer x, int incx)
    {
        float result = cublasScnrm2Native(n, x, incx);
        checkResultBLAS();
        return result;
    }
    private static native float cublasScnrm2Native(int n, Pointer x, int incx);





    /**
     * <pre>
     * void
     * cublasZaxpy (int n, cuDoubleComplex alpha, const cuDoubleComplex *x, int incx,
     *              cuDoubleComplex *y, int incy)
     *
     * multiplies double-complex vector x by double-complex scalar alpha and adds
     * the result to double-complex vector y; that is, it overwrites double-complex
     * y with double-complex alpha * x + y. For i = 0 to n - 1, it replaces
     * y[ly + i * incy] with alpha * x[lx + i * incx] + y[ly + i * incy], where
     * lx = 0 if incx >= 0, else lx = 1 + (1 - n) * incx, and ly is defined in a
     * similar way using incy.
     *
     * Input
     * -----
     * n      number of elements in input vectors
     * alpha  double-complex scalar multiplier
     * x      double-complex vector with n elements
     * incx   storage spacing between elements of x
     * y      double-complex vector with n elements
     * incy   storage spacing between elements of y
     *
     * Output
     * ------
     * y      double-complex result (unchanged if n <= 0)
     *
     * Reference: http://www.netlib.org/blas/zaxpy.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasZaxpy(int n, cuDoubleComplex alpha, Pointer x, int incx, Pointer y, int incy)
    {
        cublasZaxpyNative(n, alpha, x, incx, y, incy);
        checkResultBLAS();
    }
    private static native void cublasZaxpyNative(int n, cuDoubleComplex alpha, Pointer x, int incx, Pointer y, int incy);





    /**
     * <pre>
     * cuDoubleComplex
     * zdotu (int n, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy)
     *
     * computes the dot product of two double-complex vectors. It returns the
     * dot product of the double-complex vectors x and y if successful, and double-complex
     * zero otherwise. It computes the sum for i = 0 to n - 1 of x[lx + i * incx] *
     * y[ly + i * incy], where lx = 1 if incx >= 0, else lx = 1 + (1 - n) * incx;
     * ly is defined in a similar way using incy.
     *
     * Input
     * -----
     * n      number of elements in input vectors
     * x      double-complex vector with n elements
     * incx   storage spacing between elements of x
     * y      double-complex vector with n elements
     * incy   storage spacing between elements of y
     *
     * Output
     * ------
     * returns double-complex dot product (zero if n <= 0)
     *
     * Reference: http://www.netlib.org/blas/zdotu.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has nor been initialized
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to execute on GPU
     * </pre>
     */

    public static cuDoubleComplex cublasZdotu(int n, Pointer x, int incx, Pointer y, int incy)
    {
        cuDoubleComplex result = cublasZdotuNative(n, x, incx, y, incy);
        checkResultBLAS();
        return result;
    }
    private static native cuDoubleComplex cublasZdotuNative(int n, Pointer x, int incx, Pointer y, int incy);





    /**
     * <pre>
     * cuDoubleComplex
     * cublasZdotc (int n, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy)
     *
     * computes the dot product of two double-precision complex vectors. It returns the
     * dot product of the double-precision complex vectors conjugate(x) and y if successful,
     * and double-precision complex zero otherwise. It computes the
     * sum for i = 0 to n - 1 of conjugate(x[lx + i * incx]) *  y[ly + i * incy],
     * where lx = 1 if incx >= 0, else lx = 1 + (1 - n) * incx;
     * ly is defined in a similar way using incy.
     *
     * Input
     * -----
     * n      number of elements in input vectors
     * x      double-precision complex vector with n elements
     * incx   storage spacing between elements of x
     * y      double-precision complex vector with n elements
     * incy   storage spacing between elements of y
     *
     * Output
     * ------
     * returns double-complex dot product (zero if n <= 0)
     *
     * Reference: http://www.netlib.org/blas/zdotc.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has nor been initialized
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to execute on GPU
     * </pre>
     */

    public static cuDoubleComplex cublasZdotc(int n, Pointer x, int incx, Pointer y, int incy)
    {
        cuDoubleComplex result = cublasZdotcNative(n, x, incx, y, incy);
        checkResultBLAS();
        return result;
    }
    private static native cuDoubleComplex cublasZdotcNative(int n, Pointer x, int incx, Pointer y, int incy);





    /**
     * <pre>
     * void
     * cublasZscal (int n, cuComplex alpha, cuComplex *x, int incx)
     *
     * replaces double-complex vector x with double-complex alpha * x. For i
     * = 0 to n - 1, it replaces x[ix + i * incx] with alpha * x[ix + i * incx],
     * where ix = 1 if incx >= 0, else ix = 1 + (1 - n) * incx.
     *
     * Input
     * -----
     * n      number of elements in input vectors
     * alpha  double-complex scalar multiplier
     * x      double-complex vector with n elements
     * incx   storage spacing between elements of x
     *
     * Output
     * ------
     * x      double-complex result (unchanged if n <= 0 or incx <= 0)
     *
     * Reference: http://www.netlib.org/blas/zscal.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasZscal(int n, cuDoubleComplex alpha, Pointer x, int incx)
    {
        cublasZscalNative(n, alpha, x, incx);
        checkResultBLAS();
    }
    private static native void cublasZscalNative(int n, cuDoubleComplex alpha, Pointer x, int incx);





    /**
     * <pre>
     * void
     * cublasZdscal (int n, double alpha, cuDoubleComplex *x, int incx)
     *
     * replaces double-complex vector x with double-complex alpha * x. For i
     * = 0 to n - 1, it replaces x[ix + i * incx] with alpha * x[ix + i * incx],
     * where ix = 1 if incx >= 0, else ix = 1 + (1 - n) * incx.
     *
     * Input
     * -----
     * n      number of elements in input vectors
     * alpha  double precision scalar multiplier
     * x      double-complex vector with n elements
     * incx   storage spacing between elements of x
     *
     * Output
     * ------
     * x      double-complex result (unchanged if n <= 0 or incx <= 0)
     *
     * Reference: http://www.netlib.org/blas/zdscal.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasZdscal(int n, double alpha, Pointer x, int incx)
    {
        cublasZdscalNative(n, alpha, x, incx);
        checkResultBLAS();
    }
    private static native void cublasZdscalNative(int n, double alpha, Pointer x, int incx);





    /**
     * <pre>
     * double
     * cublasDznrm2 (int n, const cuDoubleComplex *x, int incx)
     *
     * computes the Euclidean norm of the double precision complex n-vector x. This code
     * uses simple scaling to avoid intermediate underflow and overflow.
     *
     * Input
     * -----
     * n      number of elements in input vector
     * x      double-complex vector with n elements
     * incx   storage spacing between elements of x
     *
     * Output
     * ------
     * returns Euclidian norm (0 if n <= 0 or incx <= 0, or if an error occurs)
     *
     * Reference: http://www.netlib.org/blas/dznrm2.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static double cublasDznrm2(int n, Pointer x, int incx)
    {
        double result = cublasDznrm2Native(n, x, incx);
        checkResultBLAS();
        return result;
    }
    private static native double cublasDznrm2Native(int n, Pointer x, int incx);





    /**
     * <pre>
     * void
     * cublasZrotg (cuDoubleComplex *host_ca, cuDoubleComplex cb, double *host_sc, double *host_cs)
     *
     * constructs the complex Givens tranformation
     *
     *        ( sc  cs )
     *    G = (        ) ,  sc^2 + cabs(cs)^2 = 1,
     *        (-cs  sc )
     *
     * which zeros the second entry of the complex 2-vector transpose(ca, cb).
     *
     * The quantity ca/cabs(ca)*norm(ca,cb) overwrites ca in storage. The
     * function crot (n, x, incx, y, incy, sc, cs) is normally called next
     * to apply the transformation to a 2 x n matrix.
     * Note that is function is provided for completeness and run exclusively
     * on the Host.
     *
     * Input
     * -----
     * ca     double-precision complex precision scalar
     * cb     double-precision complex scalar
     *
     * Output
     * ------
     * ca     double-precision complex ca/cabs(ca)*norm(ca,cb)
     * sc     double-precision cosine component of rotation matrix
     * cs     double-precision complex sine component of rotation matrix
     *
     * Reference: http://www.netlib.org/blas/zrotg.f
     *
     * This function does not set any error status.
     * </pre>
     */

    public static void cublasZrotg(Pointer host_ca, cuDoubleComplex cb, Pointer host_sc, Pointer host_cs)
    {
        cublasZrotgNative(host_ca, cb, host_sc, host_cs);
        checkResultBLAS();
    }
    private static native void cublasZrotgNative(Pointer host_ca, cuDoubleComplex cb, Pointer host_sc, Pointer host_cs);





    /**
     * <pre>
     * cublasZrot (int n, cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy, double sc,
     *             cuDoubleComplex cs)
     *
     * multiplies a 2x2 matrix ( sc       cs) with the 2xn matrix ( transpose(x) )
     *                         (-conj(cs) sc)                     ( transpose(y) )
     *
     * The elements of x are in x[lx + i * incx], i = 0 ... n - 1, where lx = 1 if
     * incx >= 0, else lx = 1 + (1 - n) * incx, and similarly for y using ly and
     * incy.
     *
     * Input
     * -----
     * n      number of elements in input vectors
     * x      double-precision complex vector with n elements
     * incx   storage spacing between elements of x
     * y      double-precision complex vector with n elements
     * incy   storage spacing between elements of y
     * sc     double-precision cosine component of rotation matrix
     * cs     double-precision complex sine component of rotation matrix
     *
     * Output
     * ------
     * x      rotated double-precision complex vector x (unchanged if n <= 0)
     * y      rotated double-precision complex vector y (unchanged if n <= 0)
     *
     * Reference: http://netlib.org/lapack/explore-html/zrot.f.html
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasZrot(int n, Pointer x, int incx, Pointer y, int incy, double sc, cuDoubleComplex cs)
    {
        cublasZrotNative(n, x, incx, y, incy, sc, cs);
        checkResultBLAS();
    }
    private static native void cublasZrotNative(int n, Pointer x, int incx, Pointer y, int incy, double sc, cuDoubleComplex cs);





    /**
     * <pre>
     * void
     * zdrot (int n, cuDoubleComplex *x, int incx, cuCumplex *y, int incy, double c,
     *        double s)
     *
     * multiplies a 2x2 matrix ( c s) with the 2xn matrix ( transpose(x) )
     *                         (-s c)                     ( transpose(y) )
     *
     * The elements of x are in x[lx + i * incx], i = 0 ... n - 1, where lx = 1 if
     * incx >= 0, else lx = 1 + (1 - n) * incx, and similarly for y using ly and
     * incy.
     *
     * Input
     * -----
     * n      number of elements in input vectors
     * x      double-precision complex vector with n elements
     * incx   storage spacing between elements of x
     * y      double-precision complex vector with n elements
     * incy   storage spacing between elements of y
     * c      cosine component of rotation matrix
     * s      sine component of rotation matrix
     *
     * Output
     * ------
     * x      rotated vector x (unchanged if n <= 0)
     * y      rotated vector y (unchanged if n <= 0)
     *
     * Reference  http://www.netlib.org/blas/zdrot.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasZdrot(int n, Pointer x, int incx, Pointer y, int incy, double c, double s)
    {
        cublasZdrotNative(n, x, incx, y, incy, c, s);
        checkResultBLAS();
    }
    private static native void cublasZdrotNative(int n, Pointer x, int incx, Pointer y, int incy, double c, double s);





    /**
     * <pre>
     * int
     * cublasIzamax (int n, const double *x, int incx)
     *
     * finds the smallest index of the element having maximum absolute value
     * in double-complex vector x; that is, the result is the first i, i = 0
     * to n - 1 that maximizes abs(real(x[1+i*incx]))+abs(imag(x[1 + i * incx])).
     *
     * Input
     * -----
     * n      number of elements in input vector
     * x      double-complex vector with n elements
     * incx   storage spacing between elements of x
     *
     * Output
     * ------
     * returns the smallest index (0 if n <= 0 or incx <= 0)
     *
     * Reference: http://www.netlib.org/blas/izamax.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static int cublasIzamax(int n, Pointer x, int incx)
    {
        int result = cublasIzamaxNative(n, x, incx);
        checkResultBLAS();
        return result;
    }
    private static native int cublasIzamaxNative(int n, Pointer x, int incx);





    /**
     * <pre>
     * int
     * cublasIzamin (int n, const cuDoubleComplex *x, int incx)
     *
     * finds the smallest index of the element having minimum absolute value
     * in double-complex vector x; that is, the result is the first i, i = 0
     * to n - 1 that minimizes abs(real(x[1+i*incx]))+abs(imag(x[1 + i * incx])).
     *
     * Input
     * -----
     * n      number of elements in input vector
     * x      double-complex vector with n elements
     * incx   storage spacing between elements of x
     *
     * Output
     * ------
     * returns the smallest index (0 if n <= 0 or incx <= 0)
     *
     * Reference: Analogous to IZAMAX, see there.
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static int cublasIzamin(int n, Pointer x, int incx)
    {
        int result = cublasIzaminNative(n, x, incx);
        checkResultBLAS();
        return result;
    }
    private static native int cublasIzaminNative(int n, Pointer x, int incx);





    /**
     * <pre>
     * double
     * cublasDzasum (int n, const cuDoubleComplex *x, int incx)
     *
     * takes the sum of the absolute values of a complex vector and returns a
     * double precision result. Note that this is not the L1 norm of the vector.
     * The result is the sum from 0 to n-1 of abs(real(x[ix+i*incx])) +
     * abs(imag(x(ix+i*incx))), where ix = 1 if incx <= 0, else ix = 1+(1-n)*incx.
     *
     * Input
     * -----
     * n      number of elements in input vector
     * x      double-complex vector with n elements
     * incx   storage spacing between elements of x
     *
     * Output
     * ------
     * returns the double precision sum of absolute values of real and imaginary
     * parts (0 if n <= 0 or incx <= 0, or if an error occurs)
     *
     * Reference: http://www.netlib.org/blas/dzasum.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static double cublasDzasum(int n, Pointer x, int incx)
    {
        double result = cublasDzasumNative(n, x, incx);
        checkResultBLAS();
        return result;
    }
    private static native double cublasDzasumNative(int n, Pointer x, int incx);





    /**
     * <pre>
     * void
     * cublasSgbmv (char trans, int m, int n, int kl, int ku, float alpha,
     *              const float *A, int lda, const float *x, int incx, float beta,
     *              float *y, int incy)
     *
     * performs one of the matrix-vector operations
     *
     *    y = alpha*op(A)*x + beta*y,  op(A)=A or op(A) = transpose(A)
     *
     * alpha and beta are single precision scalars. x and y are single precision
     * vectors. A is an m by n band matrix consisting of single precision elements
     * with kl sub-diagonals and ku super-diagonals.
     *
     * Input
     * -----
     * trans  specifies op(A). If trans == 'N' or 'n', op(A) = A. If trans == 'T',
     *        't', 'C', or 'c', op(A) = transpose(A)
     * m      specifies the number of rows of the matrix A. m must be at least
     *        zero.
     * n      specifies the number of columns of the matrix A. n must be at least
     *        zero.
     * kl     specifies the number of sub-diagonals of matrix A. It must be at
     *        least zero.
     * ku     specifies the number of super-diagonals of matrix A. It must be at
     *        least zero.
     * alpha  single precision scalar multiplier applied to op(A).
     * A      single precision array of dimensions (lda, n). The leading
     *        (kl + ku + 1) x n part of the array A must contain the band matrix A,
     *        supplied column by column, with the leading diagonal of the matrix
     *        in row (ku + 1) of the array, the first super-diagonal starting at
     *        position 2 in row ku, the first sub-diagonal starting at position 1
     *        in row (ku + 2), and so on. Elements in the array A that do not
     *        correspond to elements in the band matrix (such as the top left
     *        ku x ku triangle) are not referenced.
     * lda    leading dimension of A. lda must be at least (kl + ku + 1).
     * x      single precision array of length at least (1+(n-1)*abs(incx)) when
     *        trans == 'N' or 'n' and at least (1+(m-1)*abs(incx)) otherwise.
     * incx   storage spacing between elements of x. incx must not be zero.
     * beta   single precision scalar multiplier applied to vector y. If beta is
     *        zero, y is not read.
     * y      single precision array of length at least (1+(m-1)*abs(incy)) when
     *        trans == 'N' or 'n' and at least (1+(n-1)*abs(incy)) otherwise. If
     *        beta is zero, y is not read.
     * incy   storage spacing between elements of y. incy must not be zero.
     *
     * Output
     * ------
     * y      updated according to y = alpha*op(A)*x + beta*y
     *
     * Reference: http://www.netlib.org/blas/sgbmv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n, kl, or ku < 0; if incx or incy == 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasSgbmv(char trans, int m, int n, int kl, int ku, float alpha, Pointer A, int lda, Pointer x, int incx, float beta, Pointer y, int incy)
    {
        cublasSgbmvNative(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
        checkResultBLAS();
    }
    private static native void cublasSgbmvNative(char trans, int m, int n, int kl, int ku, float alpha, Pointer A, int lda, Pointer x, int incx, float beta, Pointer y, int incy);





    /**
     * <pre>
     * cublasSgemv (char trans, int m, int n, float alpha, const float *A, int lda,
     *              const float *x, int incx, float beta, float *y, int incy)
     *
     * performs one of the matrix-vector operations
     *
     *    y = alpha * op(A) * x + beta * y,
     *
     * where op(A) is one of
     *
     *    op(A) = A   or   op(A) = transpose(A)
     *
     * where alpha and beta are single precision scalars, x and y are single
     * precision vectors, and A is an m x n matrix consisting of single precision
     * elements. Matrix A is stored in column major format, and lda is the leading
     * dimension of the two-dimensional array in which A is stored.
     *
     * Input
     * -----
     * trans  specifies op(A). If transa = 'n' or 'N', op(A) = A. If trans =
     *        trans = 't', 'T', 'c', or 'C', op(A) = transpose(A)
     * m      specifies the number of rows of the matrix A. m must be at least
     *        zero.
     * n      specifies the number of columns of the matrix A. n must be at least
     *        zero.
     * alpha  single precision scalar multiplier applied to op(A).
     * A      single precision array of dimensions (lda, n) if trans = 'n' or
     *        'N'), and of dimensions (lda, m) otherwise. lda must be at least
     *        max(1, m) and at least max(1, n) otherwise.
     * lda    leading dimension of two-dimensional array used to store matrix A
     * x      single precision array of length at least (1 + (n - 1) * abs(incx))
     *        when trans = 'N' or 'n' and at least (1 + (m - 1) * abs(incx))
     *        otherwise.
     * incx   specifies the storage spacing between elements of x. incx must not
     *        be zero.
     * beta   single precision scalar multiplier applied to vector y. If beta
     *        is zero, y is not read.
     * y      single precision array of length at least (1 + (m - 1) * abs(incy))
     *        when trans = 'N' or 'n' and at least (1 + (n - 1) * abs(incy))
     *        otherwise.
     * incy   specifies the storage spacing between elements of x. incx must not
     *        be zero.
     *
     * Output
     * ------
     * y      updated according to alpha * op(A) * x + beta * y
     *
     * Reference: http://www.netlib.org/blas/sgemv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if m or n are < 0, or if incx or incy == 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasSgemv(char trans, int m, int n, float alpha, Pointer A, int lda, Pointer x, int incx, float beta, Pointer y, int incy)
    {
        cublasSgemvNative(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
        checkResultBLAS();
    }
    private static native void cublasSgemvNative(char trans, int m, int n, float alpha, Pointer A, int lda, Pointer x, int incx, float beta, Pointer y, int incy);





    /**
     * <pre>
     * cublasSger (int m, int n, float alpha, const float *x, int incx,
     *             const float *y, int incy, float *A, int lda)
     *
     * performs the symmetric rank 1 operation
     *
     *    A = alpha * x * transpose(y) + A,
     *
     * where alpha is a single precision scalar, x is an m element single
     * precision vector, y is an n element single precision vector, and A
     * is an m by n matrix consisting of single precision elements. Matrix A
     * is stored in column major format, and lda is the leading dimension of
     * the two-dimensional array used to store A.
     *
     * Input
     * -----
     * m      specifies the number of rows of the matrix A. It must be at least
     *        zero.
     * n      specifies the number of columns of the matrix A. It must be at
     *        least zero.
     * alpha  single precision scalar multiplier applied to x * transpose(y)
     * x      single precision array of length at least (1 + (m - 1) * abs(incx))
     * incx   specifies the storage spacing between elements of x. incx must not
     *        be zero.
     * y      single precision array of length at least (1 + (n - 1) * abs(incy))
     * incy   specifies the storage spacing between elements of y. incy must not
     *        be zero.
     * A      single precision array of dimensions (lda, n).
     * lda    leading dimension of two-dimensional array used to store matrix A
     *
     * Output
     * ------
     * A      updated according to A = alpha * x * transpose(y) + A
     *
     * Reference: http://www.netlib.org/blas/sger.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0, incx == 0, incy == 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasSger(int m, int n, float alpha, Pointer x, int incx, Pointer y, int incy, Pointer A, int lda)
    {
        cublasSgerNative(m, n, alpha, x, incx, y, incy, A, lda);
        checkResultBLAS();
    }
    private static native void cublasSgerNative(int m, int n, float alpha, Pointer x, int incx, Pointer y, int incy, Pointer A, int lda);





    /**
     * <pre>
     * void
     * cublasSsbmv (char uplo, int n, int k, float alpha, const float *A, int lda,
     *              const float *x, int incx, float beta, float *y, int incy)
     *
     * performs the matrix-vector operation
     *
     *     y := alpha*A*x + beta*y
     *
     * alpha and beta are single precision scalars. x and y are single precision
     * vectors with n elements. A is an n x n symmetric band matrix consisting
     * of single precision elements, with k super-diagonals and the same number
     * of sub-diagonals.
     *
     * Input
     * -----
     * uplo   specifies whether the upper or lower triangular part of the symmetric
     *        band matrix A is being supplied. If uplo == 'U' or 'u', the upper
     *        triangular part is being supplied. If uplo == 'L' or 'l', the lower
     *        triangular part is being supplied.
     * n      specifies the number of rows and the number of columns of the
     *        symmetric matrix A. n must be at least zero.
     * k      specifies the number of super-diagonals of matrix A. Since the matrix
     *        is symmetric, this is also the number of sub-diagonals. k must be at
     *        least zero.
     * alpha  single precision scalar multiplier applied to A*x.
     * A      single precision array of dimensions (lda, n). When uplo == 'U' or
     *        'u', the leading (k + 1) x n part of array A must contain the upper
     *        triangular band of the symmetric matrix, supplied column by column,
     *        with the leading diagonal of the matrix in row (k+1) of the array,
     *        the first super-diagonal starting at position 2 in row k, and so on.
     *        The top left k x k triangle of the array A is not referenced. When
     *        uplo == 'L' or 'l', the leading (k + 1) x n part of the array A must
     *        contain the lower triangular band part of the symmetric matrix,
     *        supplied column by column, with the leading diagonal of the matrix in
     *        row 1 of the array, the first sub-diagonal starting at position 1 in
     *        row 2, and so on. The bottom right k x k triangle of the array A is
     *        not referenced.
     * lda    leading dimension of A. lda must be at least (k + 1).
     * x      single precision array of length at least (1 + (n - 1) * abs(incx)).
     * incx   storage spacing between elements of x. incx must not be zero.
     * beta   single precision scalar multiplier applied to vector y. If beta is
     *        zero, y is not read.
     * y      single precision array of length at least (1 + (n - 1) * abs(incy)).
     *        If beta is zero, y is not read.
     * incy   storage spacing between elements of y. incy must not be zero.
     *
     * Output
     * ------
     * y      updated according to alpha*A*x + beta*y
     *
     * Reference: http://www.netlib.org/blas/ssbmv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_INVALID_VALUE    if k or n < 0, or if incx or incy == 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasSsbmv(char uplo, int n, int k, float alpha, Pointer A, int lda, Pointer x, int incx, float beta, Pointer y, int incy)
    {
        cublasSsbmvNative(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
        checkResultBLAS();
    }
    private static native void cublasSsbmvNative(char uplo, int n, int k, float alpha, Pointer A, int lda, Pointer x, int incx, float beta, Pointer y, int incy);





    /**
     * <pre>
     * void
     * cublasSspmv (char uplo, int n, float alpha, const float *AP, const float *x,
     *              int incx, float beta, float *y, int incy)
     *
     * performs the matrix-vector operation
     *
     *    y = alpha * A * x + beta * y
     *
     * Alpha and beta are single precision scalars, and x and y are single
     * precision vectors with n elements. A is a symmetric n x n matrix
     * consisting of single precision elements that is supplied in packed form.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix data is stored in the upper or the lower
     *        triangular part of array AP. If uplo == 'U' or 'u', then the upper
     *        triangular part of A is supplied in AP. If uplo == 'L' or 'l', then
     *        the lower triangular part of A is supplied in AP.
     * n      specifies the number of rows and columns of the matrix A. It must be
     *        at least zero.
     * alpha  single precision scalar multiplier applied to A*x.
     * AP     single precision array with at least ((n * (n + 1)) / 2) elements. If
     *        uplo == 'U' or 'u', the array AP contains the upper triangular part
     *        of the symmetric matrix A, packed sequentially, column by column;
     *        that is, if i <= j, then A[i,j] is stored is AP[i+(j*(j+1)/2)]. If
     *        uplo == 'L' or 'L', the array AP contains the lower triangular part
     *        of the symmetric matrix A, packed sequentially, column by column;
     *        that is, if i >= j, then A[i,j] is stored in AP[i+((2*n-j+1)*j)/2].
     * x      single precision array of length at least (1 + (n - 1) * abs(incx)).
     * incx   storage spacing between elements of x. incx must not be zero.
     * beta   single precision scalar multiplier applied to vector y;
     * y      single precision array of length at least (1 + (n - 1) * abs(incy)).
     *        If beta is zero, y is not read.
     * incy   storage spacing between elements of y. incy must not be zero.
     *
     * Output
     * ------
     * y      updated according to y = alpha*A*x + beta*y
     *
     * Reference: http://www.netlib.org/blas/sspmv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or if incx or incy == 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasSspmv(char uplo, int n, float alpha, Pointer AP, Pointer x, int incx, float beta, Pointer y, int incy)
    {
        cublasSspmvNative(uplo, n, alpha, AP, x, incx, beta, y, incy);
        checkResultBLAS();
    }
    private static native void cublasSspmvNative(char uplo, int n, float alpha, Pointer AP, Pointer x, int incx, float beta, Pointer y, int incy);





    /**
     * <pre>
     * void
     * cublasSspr (char uplo, int n, float alpha, const float *x, int incx,
     *             float *AP)
     *
     * performs the symmetric rank 1 operation
     *
     *    A = alpha * x * transpose(x) + A,
     *
     * where alpha is a single precision scalar and x is an n element single
     * precision vector. A is a symmetric n x n matrix consisting of single
     * precision elements that is supplied in packed form.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix data is stored in the upper or the lower
     *        triangular part of array AP. If uplo == 'U' or 'u', then the upper
     *        triangular part of A is supplied in AP. If uplo == 'L' or 'l', then
     *        the lower triangular part of A is supplied in AP.
     * n      specifies the number of rows and columns of the matrix A. It must be
     *        at least zero.
     * alpha  single precision scalar multiplier applied to x * transpose(x).
     * x      single precision array of length at least (1 + (n - 1) * abs(incx)).
     * incx   storage spacing between elements of x. incx must not be zero.
     * AP     single precision array with at least ((n * (n + 1)) / 2) elements. If
     *        uplo == 'U' or 'u', the array AP contains the upper triangular part
     *        of the symmetric matrix A, packed sequentially, column by column;
     *        that is, if i <= j, then A[i,j] is stored is AP[i+(j*(j+1)/2)]. If
     *        uplo == 'L' or 'L', the array AP contains the lower triangular part
     *        of the symmetric matrix A, packed sequentially, column by column;
     *        that is, if i >= j, then A[i,j] is stored in AP[i+((2*n-j+1)*j)/2].
     *
     * Output
     * ------
     * A      updated according to A = alpha * x * transpose(x) + A
     *
     * Reference: http://www.netlib.org/blas/sspr.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or incx == 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasSspr(char uplo, int n, float alpha, Pointer x, int incx, Pointer AP)
    {
        cublasSsprNative(uplo, n, alpha, x, incx, AP);
        checkResultBLAS();
    }
    private static native void cublasSsprNative(char uplo, int n, float alpha, Pointer x, int incx, Pointer AP);





    /**
     * <pre>
     * void
     * cublasSspr2 (char uplo, int n, float alpha, const float *x, int incx,
     *              const float *y, int incy, float *AP)
     *
     * performs the symmetric rank 2 operation
     *
     *    A = alpha*x*transpose(y) + alpha*y*transpose(x) + A,
     *
     * where alpha is a single precision scalar, and x and y are n element single
     * precision vectors. A is a symmetric n x n matrix consisting of single
     * precision elements that is supplied in packed form.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix data is stored in the upper or the lower
     *        triangular part of array A. If uplo == 'U' or 'u', then only the
     *        upper triangular part of A may be referenced and the lower triangular
     *        part of A is inferred. If uplo == 'L' or 'l', then only the lower
     *        triangular part of A may be referenced and the upper triangular part
     *        of A is inferred.
     * n      specifies the number of rows and columns of the matrix A. It must be
     *        at least zero.
     * alpha  single precision scalar multiplier applied to x * transpose(y) +
     *        y * transpose(x).
     * x      single precision array of length at least (1 + (n - 1) * abs (incx)).
     * incx   storage spacing between elements of x. incx must not be zero.
     * y      single precision array of length at least (1 + (n - 1) * abs (incy)).
     * incy   storage spacing between elements of y. incy must not be zero.
     * AP     single precision array with at least ((n * (n + 1)) / 2) elements. If
     *        uplo == 'U' or 'u', the array AP contains the upper triangular part
     *        of the symmetric matrix A, packed sequentially, column by column;
     *        that is, if i <= j, then A[i,j] is stored is AP[i+(j*(j+1)/2)]. If
     *        uplo == 'L' or 'L', the array AP contains the lower triangular part
     *        of the symmetric matrix A, packed sequentially, column by column;
     *        that is, if i >= j, then A[i,j] is stored in AP[i+((2*n-j+1)*j)/2].
     *
     * Output
     * ------
     * A      updated according to A = alpha*x*transpose(y)+alpha*y*transpose(x)+A
     *
     * Reference: http://www.netlib.org/blas/sspr2.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0, incx == 0, incy == 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasSspr2(char uplo, int n, float alpha, Pointer x, int incx, Pointer y, int incy, Pointer AP)
    {
        cublasSspr2Native(uplo, n, alpha, x, incx, y, incy, AP);
        checkResultBLAS();
    }
    private static native void cublasSspr2Native(char uplo, int n, float alpha, Pointer x, int incx, Pointer y, int incy, Pointer AP);





    /**
     * <pre>
     * void
     * cublasSsymv (char uplo, int n, float alpha, const float *A, int lda,
     *              const float *x, int incx, float beta, float *y, int incy)
     *
     * performs the matrix-vector operation
     *
     *     y = alpha*A*x + beta*y
     *
     * Alpha and beta are single precision scalars, and x and y are single
     * precision vectors, each with n elements. A is a symmetric n x n matrix
     * consisting of single precision elements that is stored in either upper or
     * lower storage mode.
     *
     * Input
     * -----
     * uplo   specifies whether the upper or lower triangular part of the array A
     *        is to be referenced. If uplo == 'U' or 'u', the symmetric matrix A
     *        is stored in upper storage mode, i.e. only the upper triangular part
     *        of A is to be referenced while the lower triangular part of A is to
     *        be inferred. If uplo == 'L' or 'l', the symmetric matrix A is stored
     *        in lower storage mode, i.e. only the lower triangular part of A is
     *        to be referenced while the upper triangular part of A is to be
     *        inferred.
     * n      specifies the number of rows and the number of columns of the
     *        symmetric matrix A. n must be at least zero.
     * alpha  single precision scalar multiplier applied to A*x.
     * A      single precision array of dimensions (lda, n). If uplo == 'U' or 'u',
     *        the leading n x n upper triangular part of the array A must contain
     *        the upper triangular part of the symmetric matrix and the strictly
     *        lower triangular part of A is not referenced. If uplo == 'L' or 'l',
     *        the leading n x n lower triangular part of the array A must contain
     *        the lower triangular part of the symmetric matrix and the strictly
     *        upper triangular part of A is not referenced.
     * lda    leading dimension of A. It must be at least max (1, n).
     * x      single precision array of length at least (1 + (n - 1) * abs(incx)).
     * incx   storage spacing between elements of x. incx must not be zero.
     * beta   single precision scalar multiplier applied to vector y.
     * y      single precision array of length at least (1 + (n - 1) * abs(incy)).
     *        If beta is zero, y is not read.
     * incy   storage spacing between elements of y. incy must not be zero.
     *
     * Output
     * ------
     * y      updated according to y = alpha*A*x + beta*y
     *
     * Reference: http://www.netlib.org/blas/ssymv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or if incx or incy == 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasSsymv(char uplo, int n, float alpha, Pointer A, int lda, Pointer x, int incx, float beta, Pointer y, int incy)
    {
        cublasSsymvNative(uplo, n, alpha, A, lda, x, incx, beta, y, incy);
        checkResultBLAS();
    }
    private static native void cublasSsymvNative(char uplo, int n, float alpha, Pointer A, int lda, Pointer x, int incx, float beta, Pointer y, int incy);





    /**
     * <pre>
     * void
     * cublasSsyr (char uplo, int n, float alpha, const float *x, int incx,
     *             float *A, int lda)
     *
     * performs the symmetric rank 1 operation
     *
     *    A = alpha * x * transpose(x) + A,
     *
     * where alpha is a single precision scalar, x is an n element single
     * precision vector and A is an n x n symmetric matrix consisting of
     * single precision elements. Matrix A is stored in column major format,
     * and lda is the leading dimension of the two-dimensional array
     * containing A.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix data is stored in the upper or
     *        the lower triangular part of array A. If uplo = 'U' or 'u',
     *        then only the upper triangular part of A may be referenced.
     *        If uplo = 'L' or 'l', then only the lower triangular part of
     *        A may be referenced.
     * n      specifies the number of rows and columns of the matrix A. It
     *        must be at least 0.
     * alpha  single precision scalar multiplier applied to x * transpose(x)
     * x      single precision array of length at least (1 + (n - 1) * abs(incx))
     * incx   specifies the storage spacing between elements of x. incx must
     *        not be zero.
     * A      single precision array of dimensions (lda, n). If uplo = 'U' or
     *        'u', then A must contain the upper triangular part of a symmetric
     *        matrix, and the strictly lower triangular part is not referenced.
     *        If uplo = 'L' or 'l', then A contains the lower triangular part
     *        of a symmetric matrix, and the strictly upper triangular part is
     *        not referenced.
     * lda    leading dimension of the two-dimensional array containing A. lda
     *        must be at least max(1, n).
     *
     * Output
     * ------
     * A      updated according to A = alpha * x * transpose(x) + A
     *
     * Reference: http://www.netlib.org/blas/ssyr.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or incx == 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasSsyr(char uplo, int n, float alpha, Pointer x, int incx, Pointer A, int lda)
    {
        cublasSsyrNative(uplo, n, alpha, x, incx, A, lda);
        checkResultBLAS();
    }
    private static native void cublasSsyrNative(char uplo, int n, float alpha, Pointer x, int incx, Pointer A, int lda);





    /**
     * <pre>
     * void
     * cublasSsyr2 (char uplo, int n, float alpha, const float *x, int incx,
     *              const float *y, int incy, float *A, int lda)
     *
     * performs the symmetric rank 2 operation
     *
     *    A = alpha*x*transpose(y) + alpha*y*transpose(x) + A,
     *
     * where alpha is a single precision scalar, x and y are n element single
     * precision vector and A is an n by n symmetric matrix consisting of single
     * precision elements.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix data is stored in the upper or the lower
     *        triangular part of array A. If uplo == 'U' or 'u', then only the
     *        upper triangular part of A may be referenced and the lower triangular
     *        part of A is inferred. If uplo == 'L' or 'l', then only the lower
     *        triangular part of A may be referenced and the upper triangular part
     *        of A is inferred.
     * n      specifies the number of rows and columns of the matrix A. It must be
     *        at least zero.
     * alpha  single precision scalar multiplier applied to x * transpose(y) +
     *        y * transpose(x).
     * x      single precision array of length at least (1 + (n - 1) * abs (incx)).
     * incx   storage spacing between elements of x. incx must not be zero.
     * y      single precision array of length at least (1 + (n - 1) * abs (incy)).
     * incy   storage spacing between elements of y. incy must not be zero.
     * A      single precision array of dimensions (lda, n). If uplo == 'U' or 'u',
     *        then A must contains the upper triangular part of a symmetric matrix,
     *        and the strictly lower triangular parts is not referenced. If uplo ==
     *        'L' or 'l', then A contains the lower triangular part of a symmetric
     *        matrix, and the strictly upper triangular part is not referenced.
     * lda    leading dimension of A. It must be at least max(1, n).
     *
     * Output
     * ------
     * A      updated according to A = alpha*x*transpose(y)+alpha*y*transpose(x)+A
     *
     * Reference: http://www.netlib.org/blas/ssyr2.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0, incx == 0, incy == 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasSsyr2(char uplo, int n, float alpha, Pointer x, int incx, Pointer y, int incy, Pointer A, int lda)
    {
        cublasSsyr2Native(uplo, n, alpha, x, incx, y, incy, A, lda);
        checkResultBLAS();
    }
    private static native void cublasSsyr2Native(char uplo, int n, float alpha, Pointer x, int incx, Pointer y, int incy, Pointer A, int lda);





    /**
     * <pre>
     * void
     * cublasStbmv (char uplo, char trans, char diag, int n, int k, const float *A,
     *              int lda, float *x, int incx)
     *
     * performs one of the matrix-vector operations x = op(A) * x, where op(A) = A
     * or op(A) = transpose(A). x is an n-element single precision vector, and A is
     * an n x n, unit or non-unit upper or lower triangular band matrix consisting
     * of single precision elements.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix A is an upper or lower triangular band
     *        matrix. If uplo == 'U' or 'u', A is an upper triangular band matrix.
     *        If uplo == 'L' or 'l', A is a lower triangular band matrix.
     * trans  specifies op(A). If transa == 'N' or 'n', op(A) = A. If trans == 'T',
     *        't', 'C', or 'c', op(A) = transpose(A).
     * diag   specifies whether or not matrix A is unit triangular. If diag == 'U'
     *        or 'u', A is assumed to be unit triangular. If diag == 'N' or 'n', A
     *        is not assumed to be unit triangular.
     * n      specifies the number of rows and columns of the matrix A. n must be
     *        at least zero. In the current implementation n must not exceed 4070.
     * k      specifies the number of super- or sub-diagonals. If uplo == 'U' or
     *        'u', k specifies the number of super-diagonals. If uplo == 'L' or
     *        'l', k specifies the number of sub-diagonals. k must at least be
     *        zero.
     * A      single precision array of dimension (lda, n). If uplo == 'U' or 'u',
     *        the leading (k + 1) x n part of the array A must contain the upper
     *        triangular band matrix, supplied column by column, with the leading
     *        diagonal of the matrix in row (k + 1) of the array, the first
     *        super-diagonal starting at position 2 in row k, and so on. The top
     *        left k x k triangle of the array A is not referenced. If uplo == 'L'
     *        or 'l', the leading (k + 1) x n part of the array A must constain the
     *        lower triangular band matrix, supplied column by column, with the
     *        leading diagonal of the matrix in row 1 of the array, the first
     *        sub-diagonal startingat position 1 in row 2, and so on. The bottom
     *        right k x k triangle of the array is not referenced.
     * lda    is the leading dimension of A. It must be at least (k + 1).
     * x      single precision array of length at least (1 + (n - 1) * abs(incx)).
     *        On entry, x contains the source vector. On exit, x is overwritten
     *        with the result vector.
     * incx   specifies the storage spacing for elements of x. incx must not be
     *        zero.
     *
     * Output
     * ------
     * x      updated according to x = op(A) * x
     *
     * Reference: http://www.netlib.org/blas/stbmv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0, k < 0, or incx == 0
     * CUBLAS_STATUS_ALLOC_FAILED     if function cannot allocate enough internal scratch vector memory
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasStbmv(char uplo, char trans, char diag, int n, int k, Pointer A, int lda, Pointer x, int incx)
    {
        cublasStbmvNative(uplo, trans, diag, n, k, A, lda, x, incx);
        checkResultBLAS();
    }
    private static native void cublasStbmvNative(char uplo, char trans, char diag, int n, int k, Pointer A, int lda, Pointer x, int incx);





    /**
     * <pre>
     * void cublasStbsv (char uplo, char trans, char diag, int n, int k,
     *                   const float *A, int lda, float *X, int incx)
     *
     * solves one of the systems of equations op(A)*x = b, where op(A) is either
     * op(A) = A or op(A) = transpose(A). b and x are n-element vectors, and A is
     * an n x n unit or non-unit, upper or lower triangular band matrix with k + 1
     * diagonals. No test for singularity or near-singularity is included in this
     * function. Such tests must be performed before calling this function.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix is an upper or lower triangular band
     *        matrix as follows: If uplo == 'U' or 'u', A is an upper triangular
     *        band matrix. If uplo == 'L' or 'l', A is a lower triangular band
     *        matrix.
     * trans  specifies op(A). If trans == 'N' or 'n', op(A) = A. If trans == 'T',
     *        't', 'C', or 'c', op(A) = transpose(A).
     * diag   specifies whether A is unit triangular. If diag == 'U' or 'u', A is
     *        assumed to be unit triangular; thas is, diagonal elements are not
     *        read and are assumed to be unity. If diag == 'N' or 'n', A is not
     *        assumed to be unit triangular.
     * n      specifies the number of rows and columns of the matrix A. n must be
     *        at least zero.
     * k      specifies the number of super- or sub-diagonals. If uplo == 'U' or
     *        'u', k specifies the number of super-diagonals. If uplo == 'L' or
     *        'l', k specifies the number of sub-diagonals. k must be at least
     *        zero.
     * A      single precision array of dimension (lda, n). If uplo == 'U' or 'u',
     *        the leading (k + 1) x n part of the array A must contain the upper
     *        triangular band matrix, supplied column by column, with the leading
     *        diagonal of the matrix in row (k + 1) of the array, the first super-
     *        diagonal starting at position 2 in row k, and so on. The top left
     *        k x k triangle of the array A is not referenced. If uplo == 'L' or
     *        'l', the leading (k + 1) x n part of the array A must constain the
     *        lower triangular band matrix, supplied column by column, with the
     *        leading diagonal of the matrix in row 1 of the array, the first
     *        sub-diagonal starting at position 1 in row 2, and so on. The bottom
     *        right k x k triangle of the array is not referenced.
     * x      single precision array of length at least (1 + (n - 1) * abs(incx)).
     *        On entry, x contains the n-element right-hand side vector b. On exit,
     *        it is overwritten with the solution vector x.
     * incx   storage spacing between elements of x. incx must not be zero.
     *
     * Output
     * ------
     * x      updated to contain the solution vector x that solves op(A) * x = b.
     *
     * Reference: http://www.netlib.org/blas/stbsv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if incx == 0, n < 0 or n > 4070
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasStbsv(char uplo, char trans, char diag, int n, int k, Pointer A, int lda, Pointer x, int incx)
    {
        cublasStbsvNative(uplo, trans, diag, n, k, A, lda, x, incx);
        checkResultBLAS();
    }
    private static native void cublasStbsvNative(char uplo, char trans, char diag, int n, int k, Pointer A, int lda, Pointer x, int incx);





    /**
     * <pre>
     * void
     * cublasStpmv (char uplo, char trans, char diag, int n, const float *AP,
     *              float *x, int incx);
     *
     * performs one of the matrix-vector operations x = op(A) * x, where op(A) = A,
     * or op(A) = transpose(A). x is an n element single precision vector, and A
     * is an n x n, unit or non-unit, upper or lower triangular matrix composed
     * of single precision elements.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix A is an upper or lower triangular
     *        matrix. If uplo == 'U' or 'u', then A is an upper triangular matrix.
     *        If uplo == 'L' or 'l', then A is a lower triangular matrix.
     * trans  specifies op(A). If transa == 'N' or 'n', op(A) = A. If trans == 'T',
     *        't', 'C', or 'c', op(A) = transpose(A)
     * diag   specifies whether or not matrix A is unit triangular. If diag == 'U'
     *        or 'u', A is assumed to be unit triangular. If diag == 'N' or 'n', A
     *        is not assumed to be unit triangular.
     * n      specifies the number of rows and columns of the matrix A. n must be
     *        at least zero.
     * AP     single precision array with at least ((n * (n + 1)) / 2) elements. If
     *        uplo == 'U' or 'u', the array AP contains the upper triangular part
     *        of the symmetric matrix A, packed sequentially, column by column;
     *        that is, if i <= j, then A[i,j] is stored in AP[i+(j*(j+1)/2)]. If
     *        uplo == 'L' or 'L', the array AP contains the lower triangular part
     *        of the symmetric matrix A, packed sequentially, column by column;
     *        that is, if i >= j, then A[i,j] is stored in AP[i+((2*n-j+1)*j)/2].
     * x      single precision array of length at least (1 + (n - 1) * abs(incx)).
     *        On entry, x contains the source vector. On exit, x is overwritten
     *        with the result vector.
     * incx   specifies the storage spacing for elements of x. incx must not be
     *        zero.
     *
     * Output
     * ------
     * x      updated according to x = op(A) * x,
     *
     * Reference: http://www.netlib.org/blas/stpmv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if incx == 0 or if n < 0
     * CUBLAS_STATUS_ALLOC_FAILED     if function cannot allocate enough internal scratch vector memory
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasStpmv(char uplo, char trans, char diag, int n, Pointer AP, Pointer x, int incx)
    {
        cublasStpmvNative(uplo, trans, diag, n, AP, x, incx);
        checkResultBLAS();
    }
    private static native void cublasStpmvNative(char uplo, char trans, char diag, int n, Pointer AP, Pointer x, int incx);





    /**
     * <pre>
     * void
     * cublasStpsv (char uplo, char trans, char diag, int n, const float *AP,
     *              float *X, int incx)
     *
     * solves one of the systems of equations op(A)*x = b, where op(A) is either
     * op(A) = A or op(A) = transpose(A). b and x are n element vectors, and A is
     * an n x n unit or non-unit, upper or lower triangular matrix. No test for
     * singularity or near-singularity is included in this function. Such tests
     * must be performed before calling this function.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix is an upper or lower triangular matrix
     *        as follows: If uplo == 'U' or 'u', A is an upper triangluar matrix.
     *        If uplo == 'L' or 'l', A is a lower triangular matrix.
     * trans  specifies op(A). If trans == 'N' or 'n', op(A) = A. If trans == 'T',
     *        't', 'C', or 'c', op(A) = transpose(A).
     * diag   specifies whether A is unit triangular. If diag == 'U' or 'u', A is
     *        assumed to be unit triangular; thas is, diagonal elements are not
     *        read and are assumed to be unity. If diag == 'N' or 'n', A is not
     *        assumed to be unit triangular.
     * n      specifies the number of rows and columns of the matrix A. n must be
     *        at least zero. In the current implementation n must not exceed 4070.
     * AP     single precision array with at least ((n*(n+1))/2) elements. If uplo
     *        == 'U' or 'u', the array AP contains the upper triangular matrix A,
     *        packed sequentially, column by column; that is, if i <= j, then
     *        A[i,j] is stored is AP[i+(j*(j+1)/2)]. If uplo == 'L' or 'L', the
     *        array AP contains the lower triangular matrix A, packed sequentially,
     *        column by column; that is, if i >= j, then A[i,j] is stored in
     *        AP[i+((2*n-j+1)*j)/2]. When diag = 'U' or 'u', the diagonal elements
     *        of A are not referenced and are assumed to be unity.
     * x      single precision array of length at least (1 + (n - 1) * abs(incx)).
     *        On entry, x contains the n-element right-hand side vector b. On exit,
     *        it is overwritten with the solution vector x.
     * incx   storage spacing between elements of x. It must not be zero.
     *
     * Output
     * ------
     * x      updated to contain the solution vector x that solves op(A) * x = b.
     *
     * Reference: http://www.netlib.org/blas/stpsv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if incx == 0, n < 0, or n > 4070
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
    * </pre>
     */

    public static void cublasStpsv(char uplo, char trans, char diag, int n, Pointer AP, Pointer x, int incx)
    {
        cublasStpsvNative(uplo, trans, diag, n, AP, x, incx);
        checkResultBLAS();
    }
    private static native void cublasStpsvNative(char uplo, char trans, char diag, int n, Pointer AP, Pointer x, int incx);





    /**
     * <pre>
     * void
     * cublasStrmv (char uplo, char trans, char diag, int n, const float *A,
     *              int lda, float *x, int incx);
     *
     * performs one of the matrix-vector operations x = op(A) * x, where op(A) =
     = A, or op(A) = transpose(A). x is an n-element single precision vector, and
     * A is an n x n, unit or non-unit, upper or lower, triangular matrix composed
     * of single precision elements.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix A is an upper or lower triangular
     *        matrix. If uplo = 'U' or 'u', then A is an upper triangular matrix.
     *        If uplo = 'L' or 'l', then A is a lower triangular matrix.
     * trans  specifies op(A). If transa = 'N' or 'n', op(A) = A. If trans = 'T',
     *        't', 'C', or 'c', op(A) = transpose(A)
     * diag   specifies whether or not matrix A is unit triangular. If diag = 'U'
     *        or 'u', A is assumed to be unit triangular. If diag = 'N' or 'n', A
     *        is not assumed to be unit triangular.
     * n      specifies the number of rows and columns of the matrix A. n must be
     *        at least zero.
     * A      single precision array of dimension (lda, n). If uplo = 'U' or 'u',
     *        the leading n x n upper triangular part of the array A must contain
     *        the upper triangular matrix and the strictly lower triangular part
     *        of A is not referenced. If uplo = 'L' or 'l', the leading n x n lower
     *        triangular part of the array A must contain the lower triangular
     *        matrix and the strictly upper triangular part of A is not referenced.
     *        When diag = 'U' or 'u', the diagonal elements of A are not referenced
     *        either, but are are assumed to be unity.
     * lda    is the leading dimension of A. It must be at least max (1, n).
     * x      single precision array of length at least (1 + (n - 1) * abs(incx) ).
     *        On entry, x contains the source vector. On exit, x is overwritten
     *        with the result vector.
     * incx   specifies the storage spacing for elements of x. incx must not be
     *        zero.
     *
     * Output
     * ------
     * x      updated according to x = op(A) * x,
     *
     * Reference: http://www.netlib.org/blas/strmv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if incx == 0 or if n < 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasStrmv(char uplo, char trans, char diag, int n, Pointer A, int lda, Pointer x, int incx)
    {
        cublasStrmvNative(uplo, trans, diag, n, A, lda, x, incx);
        checkResultBLAS();
    }
    private static native void cublasStrmvNative(char uplo, char trans, char diag, int n, Pointer A, int lda, Pointer x, int incx);





    /**
     * <pre>
     * void
     * cublasStrsv (char uplo, char trans, char diag, int n, const float *A,
     *              int lda, float *x, int incx)
     *
     * solves a system of equations op(A) * x = b, where op(A) is either A or
     * transpose(A). b and x are single precision vectors consisting of n
     * elements, and A is an n x n matrix composed of a unit or non-unit, upper
     * or lower triangular matrix. Matrix A is stored in column major format,
     * and lda is the leading dimension of the two-dimensional array containing
     * A.
     *
     * No test for singularity or near-singularity is included in this function.
     * Such tests must be performed before calling this function.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix data is stored in the upper or the
     *        lower triangular part of array A. If uplo = 'U' or 'u', then only
     *        the upper triangular part of A may be referenced. If uplo = 'L' or
     *        'l', then only the lower triangular part of A may be referenced.
     * trans  specifies op(A). If transa = 'n' or 'N', op(A) = A. If transa = 't',
     *        'T', 'c', or 'C', op(A) = transpose(A)
     * diag   specifies whether or not A is a unit triangular matrix like so:
     *        if diag = 'U' or 'u', A is assumed to be unit triangular. If
     *        diag = 'N' or 'n', then A is not assumed to be unit triangular.
     * n      specifies the number of rows and columns of the matrix A. It
     *        must be at least 0.
     * A      is a single precision array of dimensions (lda, n). If uplo = 'U'
     *        or 'u', then A must contains the upper triangular part of a symmetric
     *        matrix, and the strictly lower triangular parts is not referenced.
     *        If uplo = 'L' or 'l', then A contains the lower triangular part of
     *        a symmetric matrix, and the strictly upper triangular part is not
     *        referenced.
     * lda    is the leading dimension of the two-dimensional array containing A.
     *        lda must be at least max(1, n).
     * x      single precision array of length at least (1 + (n - 1) * abs(incx)).
     *        On entry, x contains the n element right-hand side vector b. On exit,
     *        it is overwritten with the solution vector x.
     * incx   specifies the storage spacing between elements of x. incx must not
     *        be zero.
     *
     * Output
     * ------
     * x      updated to contain the solution vector x that solves op(A) * x = b.
     *
     * Reference: http://www.netlib.org/blas/strsv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if incx == 0 or if n < 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasStrsv(char uplo, char trans, char diag, int n, Pointer A, int lda, Pointer x, int incx)
    {
        cublasStrsvNative(uplo, trans, diag, n, A, lda, x, incx);
        checkResultBLAS();
    }
    private static native void cublasStrsvNative(char uplo, char trans, char diag, int n, Pointer A, int lda, Pointer x, int incx);





    /**
     * <pre>
     * void
     * cublasZtrmv (char uplo, char trans, char diag, int n, const cuDoubleComplex *A,
     *              int lda, cuDoubleComplex *x, int incx);
     *
     * performs one of the matrix-vector operations x = op(A) * x,
     * where op(A) = A, or op(A) = transpose(A) or op(A) = conjugate(transpose(A)).
     * x is an n-element double precision complex vector, and
     * A is an n x n, unit or non-unit, upper or lower, triangular matrix composed
     * of double precision complex elements.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix A is an upper or lower triangular
     *        matrix. If uplo = 'U' or 'u', then A is an upper triangular matrix.
     *        If uplo = 'L' or 'l', then A is a lower triangular matrix.
     * trans  specifies op(A). If trans = 'n' or 'N', op(A) = A. If trans = 't' or
     *        'T', op(A) = transpose(A).  If trans = 'c' or 'C', op(A) =
     *        conjugate(transpose(A)).
     * diag   specifies whether or not matrix A is unit triangular. If diag = 'U'
     *        or 'u', A is assumed to be unit triangular. If diag = 'N' or 'n', A
     *        is not assumed to be unit triangular.
     * n      specifies the number of rows and columns of the matrix A. n must be
     *        at least zero.
     * A      double precision array of dimension (lda, n). If uplo = 'U' or 'u',
     *        the leading n x n upper triangular part of the array A must contain
     *        the upper triangular matrix and the strictly lower triangular part
     *        of A is not referenced. If uplo = 'L' or 'l', the leading n x n lower
     *        triangular part of the array A must contain the lower triangular
     *        matrix and the strictly upper triangular part of A is not referenced.
     *        When diag = 'U' or 'u', the diagonal elements of A are not referenced
     *        either, but are are assumed to be unity.
     * lda    is the leading dimension of A. It must be at least max (1, n).
     * x      double precision array of length at least (1 + (n - 1) * abs(incx) ).
     *        On entry, x contains the source vector. On exit, x is overwritten
     *        with the result vector.
     * incx   specifies the storage spacing for elements of x. incx must not be
     *        zero.
     *
     * Output
     * ------
     * x      updated according to x = op(A) * x,
     *
     * Reference: http://www.netlib.org/blas/ztrmv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if incx == 0 or if n < 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasZtrmv(char uplo, char trans, char diag, int n, Pointer A, int lda, Pointer x, int incx)
    {
        cublasZtrmvNative(uplo, trans, diag, n, A, lda, x, incx);
        checkResultBLAS();
    }
    private static native void cublasZtrmvNative(char uplo, char trans, char diag, int n, Pointer A, int lda, Pointer x, int incx);





    /**
     * <pre>
     * void
     * cublasZgbmv (char trans, int m, int n, int kl, int ku, cuDoubleComplex alpha,
     *              const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, cuDoubleComplex beta,
     *              cuDoubleComplex *y, int incy);
     *
     * performs one of the matrix-vector operations
     *
     *    y = alpha*op(A)*x + beta*y,  op(A)=A or op(A) = transpose(A)
     *
     * alpha and beta are double precision complex scalars. x and y are double precision
     * complex vectors. A is an m by n band matrix consisting of double precision complex elements
     * with kl sub-diagonals and ku super-diagonals.
     *
     * Input
     * -----
     * trans  specifies op(A). If trans == 'N' or 'n', op(A) = A. If trans == 'T',
     *        or 't', op(A) = transpose(A). If trans == 'C' or 'c',
     *        op(A) = conjugate(transpose(A)).
     * m      specifies the number of rows of the matrix A. m must be at least
     *        zero.
     * n      specifies the number of columns of the matrix A. n must be at least
     *        zero.
     * kl     specifies the number of sub-diagonals of matrix A. It must be at
     *        least zero.
     * ku     specifies the number of super-diagonals of matrix A. It must be at
     *        least zero.
     * alpha  double precision complex scalar multiplier applied to op(A).
     * A      double precision complex array of dimensions (lda, n). The leading
     *        (kl + ku + 1) x n part of the array A must contain the band matrix A,
     *        supplied column by column, with the leading diagonal of the matrix
     *        in row (ku + 1) of the array, the first super-diagonal starting at
     *        position 2 in row ku, the first sub-diagonal starting at position 1
     *        in row (ku + 2), and so on. Elements in the array A that do not
     *        correspond to elements in the band matrix (such as the top left
     *        ku x ku triangle) are not referenced.
     * lda    leading dimension of A. lda must be at least (kl + ku + 1).
     * x      double precision complex array of length at least (1+(n-1)*abs(incx)) when
     *        trans == 'N' or 'n' and at least (1+(m-1)*abs(incx)) otherwise.
     * incx   specifies the increment for the elements of x. incx must not be zero.
     * beta   double precision complex scalar multiplier applied to vector y. If beta is
     *        zero, y is not read.
     * y      double precision complex array of length at least (1+(m-1)*abs(incy)) when
     *        trans == 'N' or 'n' and at least (1+(n-1)*abs(incy)) otherwise. If
     *        beta is zero, y is not read.
     * incy   On entry, incy specifies the increment for the elements of y. incy
     *        must not be zero.
     *
     * Output
     * ------
     * y      updated according to y = alpha*op(A)*x + beta*y
     *
     * Reference: http://www.netlib.org/blas/zgbmv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or if incx or incy == 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasZgbmv(char trans, int m, int n, int kl, int ku, cuDoubleComplex alpha, Pointer A, int lda, Pointer x, int incx, cuDoubleComplex beta, Pointer y, int incy)
    {
        cublasZgbmvNative(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
        checkResultBLAS();
    }
    private static native void cublasZgbmvNative(char trans, int m, int n, int kl, int ku, cuDoubleComplex alpha, Pointer A, int lda, Pointer x, int incx, cuDoubleComplex beta, Pointer y, int incy);





    /**
     * <pre>
     * void
     * cublasZtbmv (char uplo, char trans, char diag, int n, int k, const cuDoubleComplex *A,
     *              int lda, cuDoubleComplex *x, int incx)
     *
     * performs one of the matrix-vector operations x = op(A) * x, where op(A) = A,
     * op(A) = transpose(A) or op(A) = conjugate(transpose(A)). x is an n-element
     * double precision complex vector, and A is an n x n, unit or non-unit, upper
     * or lower triangular band matrix composed of double precision complex elements.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix A is an upper or lower triangular band
     *        matrix. If uplo == 'U' or 'u', A is an upper triangular band matrix.
     *        If uplo == 'L' or 'l', A is a lower triangular band matrix.
     * trans  specifies op(A). If transa == 'N' or 'n', op(A) = A. If trans == 'T',
     *        or 't', op(A) = transpose(A). If trans == 'C' or 'c',
     *        op(A) = conjugate(transpose(A)).
     * diag   specifies whether or not matrix A is unit triangular. If diag == 'U'
     *        or 'u', A is assumed to be unit triangular. If diag == 'N' or 'n', A
     *        is not assumed to be unit triangular.
     * n      specifies the number of rows and columns of the matrix A. n must be
     *        at least zero.
     * k      specifies the number of super- or sub-diagonals. If uplo == 'U' or
     *        'u', k specifies the number of super-diagonals. If uplo == 'L' or
     *        'l', k specifies the number of sub-diagonals. k must at least be
     *        zero.
     * A      double precision complex array of dimension (lda, n). If uplo == 'U' or 'u',
     *        the leading (k + 1) x n part of the array A must contain the upper
     *        triangular band matrix, supplied column by column, with the leading
     *        diagonal of the matrix in row (k + 1) of the array, the first
     *        super-diagonal starting at position 2 in row k, and so on. The top
     *        left k x k triangle of the array A is not referenced. If uplo == 'L'
     *        or 'l', the leading (k + 1) x n part of the array A must constain the
     *        lower triangular band matrix, supplied column by column, with the
     *        leading diagonal of the matrix in row 1 of the array, the first
     *        sub-diagonal startingat position 1 in row 2, and so on. The bottom
     *        right k x k triangle of the array is not referenced.
     * lda    is the leading dimension of A. It must be at least (k + 1).
     * x      double precision complex array of length at least (1 + (n - 1) * abs(incx)).
     *        On entry, x contains the source vector. On exit, x is overwritten
     *        with the result vector.
     * incx   specifies the storage spacing for elements of x. incx must not be
     *        zero.
     *
     * Output
     * ------
     * x      updated according to x = op(A) * x
     *
     * Reference: http://www.netlib.org/blas/ztbmv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n or k < 0, or if incx == 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasZtbmv(char uplo, char trans, char diag, int n, int k, Pointer A, int lda, Pointer x, int incx)
    {
        cublasZtbmvNative(uplo, trans, diag, n, k, A, lda, x, incx);
        checkResultBLAS();
    }
    private static native void cublasZtbmvNative(char uplo, char trans, char diag, int n, int k, Pointer A, int lda, Pointer x, int incx);





    /**
     * <pre>
     * void cublasZtbsv (char uplo, char trans, char diag, int n, int k,
     *                   const cuDoubleComplex *A, int lda, cuDoubleComplex *X, int incx)
     *
     * solves one of the systems of equations op(A)*x = b, where op(A) is either
     * op(A) = A , op(A) = transpose(A) or op(A) = conjugate(transpose(A)).
     * b and x are n element vectors, and A is an n x n unit or non-unit,
     * upper or lower triangular band matrix with k + 1 diagonals. No test
     * for singularity or near-singularity is included in this function.
     * Such tests must be performed before calling this function.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix is an upper or lower triangular band
     *        matrix as follows: If uplo == 'U' or 'u', A is an upper triangular
     *        band matrix. If uplo == 'L' or 'l', A is a lower triangular band
     *        matrix.
     * trans  specifies op(A). If trans == 'N' or 'n', op(A) = A. If trans == 'T',
     *        't', op(A) = transpose(A). If trans == 'C' or 'c',
     *        op(A) = conjugate(transpose(A)).
     * diag   specifies whether A is unit triangular. If diag == 'U' or 'u', A is
     *        assumed to be unit triangular; thas is, diagonal elements are not
     *        read and are assumed to be unity. If diag == 'N' or 'n', A is not
     *        assumed to be unit triangular.
     * n      specifies the number of rows and columns of the matrix A. n must be
     *        at least zero.
     * k      specifies the number of super- or sub-diagonals. If uplo == 'U' or
     *        'u', k specifies the number of super-diagonals. If uplo == 'L' or
     *        'l', k specifies the number of sub-diagonals. k must at least be
     *        zero.
     * A      double precision complex array of dimension (lda, n). If uplo == 'U' or 'u',
     *        the leading (k + 1) x n part of the array A must contain the upper
     *        triangular band matrix, supplied column by column, with the leading
     *        diagonal of the matrix in row (k + 1) of the array, the first super-
     *        diagonal starting at position 2 in row k, and so on. The top left
     *        k x k triangle of the array A is not referenced. If uplo == 'L' or
     *        'l', the leading (k + 1) x n part of the array A must constain the
     *        lower triangular band matrix, supplied column by column, with the
     *        leading diagonal of the matrix in row 1 of the array, the first
     *        sub-diagonal starting at position 1 in row 2, and so on. The bottom
     *        right k x k triangle of the array is not referenced.
     * x      double precision complex array of length at least (1+(n-1)*abs(incx)).
     * incx   storage spacing between elements of x. It must not be zero.
     *
     * Output
     * ------
     * x      updated to contain the solution vector x that solves op(A) * x = b.
     *
     * Reference: http://www.netlib.org/blas/ztbsv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if incx == 0, n < 0 or n > 1016
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasZtbsv(char uplo, char trans, char diag, int n, int k, Pointer A, int lda, Pointer x, int incx)
    {
        cublasZtbsvNative(uplo, trans, diag, n, k, A, lda, x, incx);
        checkResultBLAS();
    }
    private static native void cublasZtbsvNative(char uplo, char trans, char diag, int n, int k, Pointer A, int lda, Pointer x, int incx);





    /**
     * <pre>
     * void
     * cublasZhemv (char uplo, int n, cuDoubleComplex alpha, const cuDoubleComplex *A, int lda,
     *              const cuDoubleComplex *x, int incx, cuDoubleComplex beta, cuDoubleComplex *y, int incy)
     *
     * performs the matrix-vector operation
     *
     *     y = alpha*A*x + beta*y
     *
     * Alpha and beta are double precision complex scalars, and x and y are double
     * precision complex vectors, each with n elements. A is a hermitian n x n matrix
     * consisting of double precision complex elements that is stored in either upper or
     * lower storage mode.
     *
     * Input
     * -----
     * uplo   specifies whether the upper or lower triangular part of the array A
     *        is to be referenced. If uplo == 'U' or 'u', the hermitian matrix A
     *        is stored in upper storage mode, i.e. only the upper triangular part
     *        of A is to be referenced while the lower triangular part of A is to
     *        be inferred. If uplo == 'L' or 'l', the hermitian matrix A is stored
     *        in lower storage mode, i.e. only the lower triangular part of A is
     *        to be referenced while the upper triangular part of A is to be
     *        inferred.
     * n      specifies the number of rows and the number of columns of the
     *        hermitian matrix A. n must be at least zero.
     * alpha  double precision complex scalar multiplier applied to A*x.
     * A      double precision complex array of dimensions (lda, n). If uplo == 'U' or 'u',
     *        the leading n x n upper triangular part of the array A must contain
     *        the upper triangular part of the hermitian matrix and the strictly
     *        lower triangular part of A is not referenced. If uplo == 'L' or 'l',
     *        the leading n x n lower triangular part of the array A must contain
     *        the lower triangular part of the hermitian matrix and the strictly
     *        upper triangular part of A is not referenced. The imaginary parts
     *        of the diagonal elements need not be set, they are assumed to be zero.
     * lda    leading dimension of A. It must be at least max (1, n).
     * x      double precision complex array of length at least (1 + (n - 1) * abs(incx)).
     * incx   storage spacing between elements of x. incx must not be zero.
     * beta   double precision complex scalar multiplier applied to vector y.
     * y      double precision complex array of length at least (1 + (n - 1) * abs(incy)).
     *        If beta is zero, y is not read.
     * incy   storage spacing between elements of y. incy must not be zero.
     *
     * Output
     * ------
     * y      updated according to y = alpha*A*x + beta*y
     *
     * Reference: http://www.netlib.org/blas/zhemv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or if incx or incy == 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasZhemv(char uplo, int n, cuDoubleComplex alpha, Pointer A, int lda, Pointer x, int incx, cuDoubleComplex beta, Pointer y, int incy)
    {
        cublasZhemvNative(uplo, n, alpha, A, lda, x, incx, beta, y, incy);
        checkResultBLAS();
    }
    private static native void cublasZhemvNative(char uplo, int n, cuDoubleComplex alpha, Pointer A, int lda, Pointer x, int incx, cuDoubleComplex beta, Pointer y, int incy);





    /**
     * <pre>
     * void
     * cublasZhpmv (char uplo, int n, cuDoubleComplex alpha, const cuDoubleComplex *AP, const cuDoubleComplex *x,
     *              int incx, cuDoubleComplex beta, cuDoubleComplex *y, int incy)
     *
     * performs the matrix-vector operation
     *
     *    y = alpha * A * x + beta * y
     *
     * Alpha and beta are double precision complex scalars, and x and y are double
     * precision complex vectors with n elements. A is an hermitian n x n matrix
     * consisting of double precision complex elements that is supplied in packed form.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix data is stored in the upper or the lower
     *        triangular part of array AP. If uplo == 'U' or 'u', then the upper
     *        triangular part of A is supplied in AP. If uplo == 'L' or 'l', then
     *        the lower triangular part of A is supplied in AP.
     * n      specifies the number of rows and columns of the matrix A. It must be
     *        at least zero.
     * alpha  double precision complex scalar multiplier applied to A*x.
     * AP     double precision complex array with at least ((n * (n + 1)) / 2) elements. If
     *        uplo == 'U' or 'u', the array AP contains the upper triangular part
     *        of the hermitian matrix A, packed sequentially, column by column;
     *        that is, if i <= j, then A[i,j] is stored is AP[i+(j*(j+1)/2)]. If
     *        uplo == 'L' or 'L', the array AP contains the lower triangular part
     *        of the hermitian matrix A, packed sequentially, column by column;
     *        that is, if i >= j, then A[i,j] is stored in AP[i+((2*n-j+1)*j)/2].
     *        The imaginary parts of the diagonal elements need not be set, they
     *        are assumed to be zero.
     * x      double precision complex array of length at least (1 + (n - 1) * abs(incx)).
     * incx   storage spacing between elements of x. incx must not be zero.
     * beta   double precision complex scalar multiplier applied to vector y;
     * y      double precision array of length at least (1 + (n - 1) * abs(incy)).
     *        If beta is zero, y is not read.
     * incy   storage spacing between elements of y. incy must not be zero.
     *
     * Output
     * ------
     * y      updated according to y = alpha*A*x + beta*y
     *
     * Reference: http://www.netlib.org/blas/zhpmv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or if incx or incy == 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasZhpmv(char uplo, int n, cuDoubleComplex alpha, Pointer AP, Pointer x, int incx, cuDoubleComplex beta, Pointer y, int incy)
    {
        cublasZhpmvNative(uplo, n, alpha, AP, x, incx, beta, y, incy);
        checkResultBLAS();
    }
    private static native void cublasZhpmvNative(char uplo, int n, cuDoubleComplex alpha, Pointer AP, Pointer x, int incx, cuDoubleComplex beta, Pointer y, int incy);





    /**
     * <pre>
     * cublasZgemv (char trans, int m, int n, cuDoubleComplex alpha, const cuDoubleComplex *A, int lda,
     *              const cuDoubleComplex *x, int incx, cuDoubleComplex beta, cuDoubleComplex *y, int incy)
     *
     * performs one of the matrix-vector operations
     *
     *    y = alpha * op(A) * x + beta * y,
     *
     * where op(A) is one of
     *
     *    op(A) = A   or   op(A) = transpose(A)
     *
     * where alpha and beta are double precision scalars, x and y are double
     * precision vectors, and A is an m x n matrix consisting of double precision
     * elements. Matrix A is stored in column major format, and lda is the leading
     * dimension of the two-dimensional array in which A is stored.
     *
     * Input
     * -----
     * trans  specifies op(A). If transa = 'n' or 'N', op(A) = A. If trans =
     *        trans = 't', 'T', 'c', or 'C', op(A) = transpose(A)
     * m      specifies the number of rows of the matrix A. m must be at least
     *        zero.
     * n      specifies the number of columns of the matrix A. n must be at least
     *        zero.
     * alpha  double precision scalar multiplier applied to op(A).
     * A      double precision array of dimensions (lda, n) if trans = 'n' or
     *        'N'), and of dimensions (lda, m) otherwise. lda must be at least
     *        max(1, m) and at least max(1, n) otherwise.
     * lda    leading dimension of two-dimensional array used to store matrix A
     * x      double precision array of length at least (1 + (n - 1) * abs(incx))
     *        when trans = 'N' or 'n' and at least (1 + (m - 1) * abs(incx))
     *        otherwise.
     * incx   specifies the storage spacing between elements of x. incx must not
     *        be zero.
     * beta   double precision scalar multiplier applied to vector y. If beta
     *        is zero, y is not read.
     * y      double precision array of length at least (1 + (m - 1) * abs(incy))
     *        when trans = 'N' or 'n' and at least (1 + (n - 1) * abs(incy))
     *        otherwise.
     * incy   specifies the storage spacing between elements of x. incx must not
     *        be zero.
     *
     * Output
     * ------
     * y      updated according to alpha * op(A) * x + beta * y
     *
     * Reference: http://www.netlib.org/blas/zgemv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if m or n are < 0, or if incx or incy == 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasZgemv(char trans, int m, int n, cuDoubleComplex alpha, Pointer A, int lda, Pointer x, int incx, cuDoubleComplex beta, Pointer y, int incy)
    {
        cublasZgemvNative(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
        checkResultBLAS();
    }
    private static native void cublasZgemvNative(char trans, int m, int n, cuDoubleComplex alpha, Pointer A, int lda, Pointer x, int incx, cuDoubleComplex beta, Pointer y, int incy);





    /**
     * <pre>
     * void
     * cublasZtpmv (char uplo, char trans, char diag, int n, const cuDoubleComplex *AP,
     *              cuDoubleComplex *x, int incx);
     *
     * performs one of the matrix-vector operations x = op(A) * x, where op(A) = A,
     * op(A) = transpose(A) or op(A) = conjugate(transpose(A)) . x is an n element
     * double precision complex vector, and A is an n x n, unit or non-unit, upper
     * or lower triangular matrix composed of double precision complex elements.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix A is an upper or lower triangular
     *        matrix. If uplo == 'U' or 'u', then A is an upper triangular matrix.
     *        If uplo == 'L' or 'l', then A is a lower triangular matrix.
     * trans  specifies op(A). If transa == 'N' or 'n', op(A) = A. If trans == 'T',
     *        or 't', op(A) = transpose(A). If trans == 'C' or 'c',
     *        op(A) = conjugate(transpose(A)).
     *
     * diag   specifies whether or not matrix A is unit triangular. If diag == 'U'
     *        or 'u', A is assumed to be unit triangular. If diag == 'N' or 'n', A
     *        is not assumed to be unit triangular.
     * n      specifies the number of rows and columns of the matrix A. n must be
     *        at least zero. In the current implementation n must not exceed 4070.
     * AP     double precision complex array with at least ((n * (n + 1)) / 2) elements. If
     *        uplo == 'U' or 'u', the array AP contains the upper triangular part
     *        of the symmetric matrix A, packed sequentially, column by column;
     *        that is, if i <= j, then A[i,j] is stored in AP[i+(j*(j+1)/2)]. If
     *        uplo == 'L' or 'L', the array AP contains the lower triangular part
     *        of the symmetric matrix A, packed sequentially, column by column;
     *        that is, if i >= j, then A[i,j] is stored in AP[i+((2*n-j+1)*j)/2].
     * x      double precision complex array of length at least (1 + (n - 1) * abs(incx)).
     *        On entry, x contains the source vector. On exit, x is overwritten
     *        with the result vector.
     * incx   specifies the storage spacing for elements of x. incx must not be
     *        zero.
     *
     * Output
     * ------
     * x      updated according to x = op(A) * x,
     *
     * Reference: http://www.netlib.org/blas/ztpmv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if incx == 0 or n < 0
     * CUBLAS_STATUS_ALLOC_FAILED     if function cannot allocate enough internal scratch vector memory
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasZtpmv(char uplo, char trans, char diag, int n, Pointer AP, Pointer x, int incx)
    {
        cublasZtpmvNative(uplo, trans, diag, n, AP, x, incx);
        checkResultBLAS();
    }
    private static native void cublasZtpmvNative(char uplo, char trans, char diag, int n, Pointer AP, Pointer x, int incx);





    /**
     * <pre>
     * void
     * cublasZtpsv (char uplo, char trans, char diag, int n, const cuDoubleComplex *AP,
     *              cuDoubleComplex *X, int incx)
     *
     * solves one of the systems of equations op(A)*x = b, where op(A) is either
     * op(A) = A , op(A) = transpose(A) or op(A) = conjugate(transpose)). b and
     * x are n element complex vectors, and A is an n x n unit or non-unit,
     * upper or lower triangular matrix. No test for singularity or near-singularity
     * is included in this routine. Such tests must be performed before calling this routine.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix is an upper or lower triangular matrix
     *        as follows: If uplo == 'U' or 'u', A is an upper triangluar matrix.
     *        If uplo == 'L' or 'l', A is a lower triangular matrix.
     * trans  specifies op(A). If trans == 'N' or 'n', op(A) = A. If trans == 'T'
     *        or 't', op(A) = transpose(A). If trans == 'C' or 'c', op(A) =
     *        conjugate(transpose(A)).
     * diag   specifies whether A is unit triangular. If diag == 'U' or 'u', A is
     *        assumed to be unit triangular; thas is, diagonal elements are not
     *        read and are assumed to be unity. If diag == 'N' or 'n', A is not
     *        assumed to be unit triangular.
     * n      specifies the number of rows and columns of the matrix A. n must be
     *        at least zero.
     * AP     double precision complex array with at least ((n*(n+1))/2) elements.
     *        If uplo == 'U' or 'u', the array AP contains the upper triangular
     *        matrix A, packed sequentially, column by column; that is, if i <= j, then
     *        A[i,j] is stored is AP[i+(j*(j+1)/2)]. If uplo == 'L' or 'L', the
     *        array AP contains the lower triangular matrix A, packed sequentially,
     *        column by column; that is, if i >= j, then A[i,j] is stored in
     *        AP[i+((2*n-j+1)*j)/2]. When diag = 'U' or 'u', the diagonal elements
     *        of A are not referenced and are assumed to be unity.
     * x      double precision complex array of length at least (1+(n-1)*abs(incx)).
     * incx   storage spacing between elements of x. It must not be zero.
     *
     * Output
     * ------
     * x      updated to contain the solution vector x that solves op(A) * x = b.
     *
     * Reference: http://www.netlib.org/blas/ztpsv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if incx == 0 or if n < 0 or n > 2035
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasZtpsv(char uplo, char trans, char diag, int n, Pointer AP, Pointer x, int incx)
    {
        cublasZtpsvNative(uplo, trans, diag, n, AP, x, incx);
        checkResultBLAS();
    }
    private static native void cublasZtpsvNative(char uplo, char trans, char diag, int n, Pointer AP, Pointer x, int incx);





    /**
     * <pre>
     * cublasCgemv (char trans, int m, int n, cuComplex alpha, const cuComplex *A,
     *              int lda, const cuComplex *x, int incx, cuComplex beta, cuComplex *y,
     *              int incy)
     *
     * performs one of the matrix-vector operations
     *
     *    y = alpha * op(A) * x + beta * y,
     *
     * where op(A) is one of
     *
     *    op(A) = A   or   op(A) = transpose(A) or op(A) = conjugate(transpose(A))
     *
     * where alpha and beta are single precision scalars, x and y are single
     * precision vectors, and A is an m x n matrix consisting of single precision
     * elements. Matrix A is stored in column major format, and lda is the leading
     * dimension of the two-dimensional array in which A is stored.
     *
     * Input
     * -----
     * trans  specifies op(A). If transa = 'n' or 'N', op(A) = A. If trans =
     *        trans = 't' or 'T', op(A) = transpose(A). If trans = 'c' or 'C',
     *        op(A) = conjugate(transpose(A))
     * m      specifies the number of rows of the matrix A. m must be at least
     *        zero.
     * n      specifies the number of columns of the matrix A. n must be at least
     *        zero.
     * alpha  single precision scalar multiplier applied to op(A).
     * A      single precision array of dimensions (lda, n) if trans = 'n' or
     *        'N'), and of dimensions (lda, m) otherwise. lda must be at least
     *        max(1, m) and at least max(1, n) otherwise.
     * lda    leading dimension of two-dimensional array used to store matrix A
     * x      single precision array of length at least (1 + (n - 1) * abs(incx))
     *        when trans = 'N' or 'n' and at least (1 + (m - 1) * abs(incx))
     *        otherwise.
     * incx   specifies the storage spacing between elements of x. incx must not
     *        be zero.
     * beta   single precision scalar multiplier applied to vector y. If beta
     *        is zero, y is not read.
     * y      single precision array of length at least (1 + (m - 1) * abs(incy))
     *        when trans = 'N' or 'n' and at least (1 + (n - 1) * abs(incy))
     *        otherwise.
     * incy   specifies the storage spacing between elements of y. incy must not
     *        be zero.
     *
     * Output
     * ------
     * y      updated according to alpha * op(A) * x + beta * y
     *
     * Reference: http://www.netlib.org/blas/cgemv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if m or n are < 0, or if incx or incy == 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasCgemv(char trans, int m, int n, cuComplex alpha, Pointer A, int lda, Pointer x, int incx, cuComplex beta, Pointer y, int incy)
    {
        cublasCgemvNative(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
        checkResultBLAS();
    }
    private static native void cublasCgemvNative(char trans, int m, int n, cuComplex alpha, Pointer A, int lda, Pointer x, int incx, cuComplex beta, Pointer y, int incy);





    /**
     * <pre>
     * void
     * cublasCgbmv (char trans, int m, int n, int kl, int ku, cuComplex alpha,
     *              const cuComplex *A, int lda, const cuComplex *x, int incx, cuComplex beta,
     *              cuComplex *y, int incy);
     *
     * performs one of the matrix-vector operations
     *
     *    y = alpha*op(A)*x + beta*y,  op(A)=A or op(A) = transpose(A)
     *
     * alpha and beta are single precision complex scalars. x and y are single precision
     * complex vectors. A is an m by n band matrix consisting of single precision complex elements
     * with kl sub-diagonals and ku super-diagonals.
     *
     * Input
     * -----
     * trans  specifies op(A). If trans == 'N' or 'n', op(A) = A. If trans == 'T',
     *        or 't', op(A) = transpose(A). If trans == 'C' or 'c',
     *        op(A) = conjugate(transpose(A)).
     * m      specifies the number of rows of the matrix A. m must be at least
     *        zero.
     * n      specifies the number of columns of the matrix A. n must be at least
     *        zero.
     * kl     specifies the number of sub-diagonals of matrix A. It must be at
     *        least zero.
     * ku     specifies the number of super-diagonals of matrix A. It must be at
     *        least zero.
     * alpha  single precision complex scalar multiplier applied to op(A).
     * A      single precision complex array of dimensions (lda, n). The leading
     *        (kl + ku + 1) x n part of the array A must contain the band matrix A,
     *        supplied column by column, with the leading diagonal of the matrix
     *        in row (ku + 1) of the array, the first super-diagonal starting at
     *        position 2 in row ku, the first sub-diagonal starting at position 1
     *        in row (ku + 2), and so on. Elements in the array A that do not
     *        correspond to elements in the band matrix (such as the top left
     *        ku x ku triangle) are not referenced.
     * lda    leading dimension of A. lda must be at least (kl + ku + 1).
     * x      single precision complex array of length at least (1+(n-1)*abs(incx)) when
     *        trans == 'N' or 'n' and at least (1+(m-1)*abs(incx)) otherwise.
     * incx   specifies the increment for the elements of x. incx must not be zero.
     * beta   single precision complex scalar multiplier applied to vector y. If beta is
     *        zero, y is not read.
     * y      single precision complex array of length at least (1+(m-1)*abs(incy)) when
     *        trans == 'N' or 'n' and at least (1+(n-1)*abs(incy)) otherwise. If
     *        beta is zero, y is not read.
     * incy   On entry, incy specifies the increment for the elements of y. incy
     *        must not be zero.
     *
     * Output
     * ------
     * y      updated according to y = alpha*op(A)*x + beta*y
     *
     * Reference: http://www.netlib.org/blas/cgbmv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or if incx or incy == 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasCgbmv(char trans, int m, int n, int kl, int ku, cuComplex alpha, Pointer A, int lda, Pointer x, int incx, cuComplex beta, Pointer y, int incy)
    {
        cublasCgbmvNative(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
        checkResultBLAS();
    }
    private static native void cublasCgbmvNative(char trans, int m, int n, int kl, int ku, cuComplex alpha, Pointer A, int lda, Pointer x, int incx, cuComplex beta, Pointer y, int incy);





    /**
     * <pre>
     * void
     * cublasChemv (char uplo, int n, cuComplex alpha, const cuComplex *A, int lda,
     *              const cuComplex *x, int incx, cuComplex beta, cuComplex *y, int incy)
     *
     * performs the matrix-vector operation
     *
     *     y = alpha*A*x + beta*y
     *
     * Alpha and beta are single precision complex scalars, and x and y are single
     * precision complex vectors, each with n elements. A is a hermitian n x n matrix
     * consisting of single precision complex elements that is stored in either upper or
     * lower storage mode.
     *
     * Input
     * -----
     * uplo   specifies whether the upper or lower triangular part of the array A
     *        is to be referenced. If uplo == 'U' or 'u', the hermitian matrix A
     *        is stored in upper storage mode, i.e. only the upper triangular part
     *        of A is to be referenced while the lower triangular part of A is to
     *        be inferred. If uplo == 'L' or 'l', the hermitian matrix A is stored
     *        in lower storage mode, i.e. only the lower triangular part of A is
     *        to be referenced while the upper triangular part of A is to be
     *        inferred.
     * n      specifies the number of rows and the number of columns of the
     *        hermitian matrix A. n must be at least zero.
     * alpha  single precision complex scalar multiplier applied to A*x.
     * A      single precision complex array of dimensions (lda, n). If uplo == 'U' or 'u',
     *        the leading n x n upper triangular part of the array A must contain
     *        the upper triangular part of the hermitian matrix and the strictly
     *        lower triangular part of A is not referenced. If uplo == 'L' or 'l',
     *        the leading n x n lower triangular part of the array A must contain
     *        the lower triangular part of the hermitian matrix and the strictly
     *        upper triangular part of A is not referenced. The imaginary parts
     *        of the diagonal elements need not be set, they are assumed to be zero.
     * lda    leading dimension of A. It must be at least max (1, n).
     * x      single precision complex array of length at least (1 + (n - 1) * abs(incx)).
     * incx   storage spacing between elements of x. incx must not be zero.
     * beta   single precision complex scalar multiplier applied to vector y.
     * y      single precision complex array of length at least (1 + (n - 1) * abs(incy)).
     *        If beta is zero, y is not read.
     * incy   storage spacing between elements of y. incy must not be zero.
     *
     * Output
     * ------
     * y      updated according to y = alpha*A*x + beta*y
     *
     * Reference: http://www.netlib.org/blas/chemv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or if incx or incy == 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasChemv(char uplo, int n, cuComplex alpha, Pointer A, int lda, Pointer x, int incx, cuComplex beta, Pointer y, int incy)
    {
        cublasChemvNative(uplo, n, alpha, A, lda, x, incx, beta, y, incy);
        checkResultBLAS();
    }
    private static native void cublasChemvNative(char uplo, int n, cuComplex alpha, Pointer A, int lda, Pointer x, int incx, cuComplex beta, Pointer y, int incy);





    /**
     * <pre>
     * void
     * cublasChbmv (char uplo, int n, int k, cuComplex alpha, const cuComplex *A, int lda,
     *              const cuComplex *x, int incx, cuComplex beta, cuComplex *y, int incy)
     *
     * performs the matrix-vector operation
     *
     *     y := alpha*A*x + beta*y
     *
     * alpha and beta are single precision complex scalars. x and y are single precision
     * complex vectors with n elements. A is an n by n hermitian band matrix consisting
     * of single precision complex elements, with k super-diagonals and the same number
     * of subdiagonals.
     *
     * Input
     * -----
     * uplo   specifies whether the upper or lower triangular part of the hermitian
     *        band matrix A is being supplied. If uplo == 'U' or 'u', the upper
     *        triangular part is being supplied. If uplo == 'L' or 'l', the lower
     *        triangular part is being supplied.
     * n      specifies the number of rows and the number of columns of the
     *        hermitian matrix A. n must be at least zero.
     * k      specifies the number of super-diagonals of matrix A. Since the matrix
     *        is hermitian, this is also the number of sub-diagonals. k must be at
     *        least zero.
     * alpha  single precision complex scalar multiplier applied to A*x.
     * A      single precision complex array of dimensions (lda, n). When uplo == 'U' or
     *        'u', the leading (k + 1) x n part of array A must contain the upper
     *        triangular band of the hermitian matrix, supplied column by column,
     *        with the leading diagonal of the matrix in row (k+1) of the array,
     *        the first super-diagonal starting at position 2 in row k, and so on.
     *        The top left k x k triangle of the array A is not referenced. When
     *        uplo == 'L' or 'l', the leading (k + 1) x n part of the array A must
     *        contain the lower triangular band part of the hermitian matrix,
     *        supplied column by column, with the leading diagonal of the matrix in
     *        row 1 of the array, the first sub-diagonal starting at position 1 in
     *        row 2, and so on. The bottom right k x k triangle of the array A is
     *        not referenced. The imaginary parts of the diagonal elements need
     *        not be set, they are assumed to be zero.
     * lda    leading dimension of A. lda must be at least (k + 1).
     * x      single precision complex array of length at least (1 + (n - 1) * abs(incx)).
     * incx   storage spacing between elements of x. incx must not be zero.
     * beta   single precision complex scalar multiplier applied to vector y. If beta is
     *        zero, y is not read.
     * y      single precision complex array of length at least (1 + (n - 1) * abs(incy)).
     *        If beta is zero, y is not read.
     * incy   storage spacing between elements of y. incy must not be zero.
     *
     * Output
     * ------
     * y      updated according to alpha*A*x + beta*y
     *
     * Reference: http://www.netlib.org/blas/chbmv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if k or n < 0, or if incx or incy == 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasChbmv(char uplo, int n, int k, cuComplex alpha, Pointer A, int lda, Pointer x, int incx, cuComplex beta, Pointer y, int incy)
    {
        cublasChbmvNative(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
        checkResultBLAS();
    }
    private static native void cublasChbmvNative(char uplo, int n, int k, cuComplex alpha, Pointer A, int lda, Pointer x, int incx, cuComplex beta, Pointer y, int incy);





    /**
     * <pre>
     *
     * cublasCtrmv (char uplo, char trans, char diag, int n, const cuComplex *A,
     *              int lda, cuComplex *x, int incx);
     *
     * performs one of the matrix-vector operations x = op(A) * x,
     * where op(A) = A, or op(A) = transpose(A) or op(A) = conjugate(transpose(A)).
     * x is an n-element signle precision complex vector, and
     * A is an n x n, unit or non-unit, upper or lower, triangular matrix composed
     * of single precision complex elements.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix A is an upper or lower triangular
     *        matrix. If uplo = 'U' or 'u', then A is an upper triangular matrix.
     *        If uplo = 'L' or 'l', then A is a lower triangular matrix.
     * trans  specifies op(A). If trans = 'n' or 'N', op(A) = A. If trans = 't' or
     *        'T', op(A) = transpose(A).  If trans = 'c' or 'C', op(A) =
     *        conjugate(transpose(A)).
     * diag   specifies whether or not matrix A is unit triangular. If diag = 'U'
     *        or 'u', A is assumed to be unit triangular. If diag = 'N' or 'n', A
     *        is not assumed to be unit triangular.
     * n      specifies the number of rows and columns of the matrix A. n must be
     *        at least zero.
     * A      single precision array of dimension (lda, n). If uplo = 'U' or 'u',
     *        the leading n x n upper triangular part of the array A must contain
     *        the upper triangular matrix and the strictly lower triangular part
     *        of A is not referenced. If uplo = 'L' or 'l', the leading n x n lower
     *        triangular part of the array A must contain the lower triangular
     *        matrix and the strictly upper triangular part of A is not referenced.
     *        When diag = 'U' or 'u', the diagonal elements of A are not referenced
     *        either, but are are assumed to be unity.
     * lda    is the leading dimension of A. It must be at least max (1, n).
     * x      single precision array of length at least (1 + (n - 1) * abs(incx) ).
     *        On entry, x contains the source vector. On exit, x is overwritten
     *        with the result vector.
     * incx   specifies the storage spacing for elements of x. incx must not be
     *        zero.
     *
     * Output
     * ------
     * x      updated according to x = op(A) * x,
     *
     * Reference: http://www.netlib.org/blas/ctrmv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if incx == 0 or if n < 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasCtrmv(char uplo, char trans, char diag, int n, Pointer A, int lda, Pointer x, int incx)
    {
        cublasCtrmvNative(uplo, trans, diag, n, A, lda, x, incx);
        checkResultBLAS();
    }
    private static native void cublasCtrmvNative(char uplo, char trans, char diag, int n, Pointer A, int lda, Pointer x, int incx);





    /**
     * <pre>
     * void
     * cublasCtbmv (char uplo, char trans, char diag, int n, int k, const cuComplex *A,
     *              int lda, cuComplex *x, int incx)
     *
     * performs one of the matrix-vector operations x = op(A) * x, where op(A) = A,
     * op(A) = transpose(A) or op(A) = conjugate(transpose(A)). x is an n-element
     * single precision complex vector, and A is an n x n, unit or non-unit, upper
     * or lower triangular band matrix composed of single precision complex elements.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix A is an upper or lower triangular band
     *        matrix. If uplo == 'U' or 'u', A is an upper triangular band matrix.
     *        If uplo == 'L' or 'l', A is a lower triangular band matrix.
     * trans  specifies op(A). If transa == 'N' or 'n', op(A) = A. If trans == 'T',
     *        or 't', op(A) = transpose(A). If trans == 'C' or 'c',
     *        op(A) = conjugate(transpose(A)).
     * diag   specifies whether or not matrix A is unit triangular. If diag == 'U'
     *        or 'u', A is assumed to be unit triangular. If diag == 'N' or 'n', A
     *        is not assumed to be unit triangular.
     * n      specifies the number of rows and columns of the matrix A. n must be
     *        at least zero.
     * k      specifies the number of super- or sub-diagonals. If uplo == 'U' or
     *        'u', k specifies the number of super-diagonals. If uplo == 'L' or
     *        'l', k specifies the number of sub-diagonals. k must at least be
     *        zero.
     * A      single precision complex array of dimension (lda, n). If uplo == 'U' or 'u',
     *        the leading (k + 1) x n part of the array A must contain the upper
     *        triangular band matrix, supplied column by column, with the leading
     *        diagonal of the matrix in row (k + 1) of the array, the first
     *        super-diagonal starting at position 2 in row k, and so on. The top
     *        left k x k triangle of the array A is not referenced. If uplo == 'L'
     *        or 'l', the leading (k + 1) x n part of the array A must constain the
     *        lower triangular band matrix, supplied column by column, with the
     *        leading diagonal of the matrix in row 1 of the array, the first
     *        sub-diagonal startingat position 1 in row 2, and so on. The bottom
     *        right k x k triangle of the array is not referenced.
     * lda    is the leading dimension of A. It must be at least (k + 1).
     * x      single precision complex array of length at least (1 + (n - 1) * abs(incx)).
     *        On entry, x contains the source vector. On exit, x is overwritten
     *        with the result vector.
     * incx   specifies the storage spacing for elements of x. incx must not be
     *        zero.
     *
     * Output
     * ------
     * x      updated according to x = op(A) * x
     *
     * Reference: http://www.netlib.org/blas/ctbmv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n or k < 0, or if incx == 0
     * CUBLAS_STATUS_ALLOC_FAILED     if function cannot allocate enough internal scratch vector memory
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasCtbmv(char uplo, char trans, char diag, int n, int k, Pointer A, int lda, Pointer x, int incx)
    {
        cublasCtbmvNative(uplo, trans, diag, n, k, A, lda, x, incx);
        checkResultBLAS();
    }
    private static native void cublasCtbmvNative(char uplo, char trans, char diag, int n, int k, Pointer A, int lda, Pointer x, int incx);





    /**
     * <pre>
     * void
     * cublasCtpmv (char uplo, char trans, char diag, int n, const cuComplex *AP,
     *              cuComplex *x, int incx);
     *
     * performs one of the matrix-vector operations x = op(A) * x, where op(A) = A,
     * op(A) = transpose(A) or op(A) = conjugate(transpose(A)) . x is an n element
     * single precision complex vector, and A is an n x n, unit or non-unit, upper
     * or lower triangular matrix composed of single precision complex elements.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix A is an upper or lower triangular
     *        matrix. If uplo == 'U' or 'u', then A is an upper triangular matrix.
     *        If uplo == 'L' or 'l', then A is a lower triangular matrix.
     * trans  specifies op(A). If transa == 'N' or 'n', op(A) = A. If trans == 'T',
     *        or 't', op(A) = transpose(A). If trans == 'C' or 'c',
     *        op(A) = conjugate(transpose(A)).
     *
     * diag   specifies whether or not matrix A is unit triangular. If diag == 'U'
     *        or 'u', A is assumed to be unit triangular. If diag == 'N' or 'n', A
     *        is not assumed to be unit triangular.
     * n      specifies the number of rows and columns of the matrix A. n must be
     *        at least zero. In the current implementation n must not exceed 4070.
     * AP     single precision complex array with at least ((n * (n + 1)) / 2) elements. If
     *        uplo == 'U' or 'u', the array AP contains the upper triangular part
     *        of the symmetric matrix A, packed sequentially, column by column;
     *        that is, if i <= j, then A[i,j] is stored in AP[i+(j*(j+1)/2)]. If
     *        uplo == 'L' or 'L', the array AP contains the lower triangular part
     *        of the symmetric matrix A, packed sequentially, column by column;
     *        that is, if i >= j, then A[i,j] is stored in AP[i+((2*n-j+1)*j)/2].
     * x      single precision complex array of length at least (1 + (n - 1) * abs(incx)).
     *        On entry, x contains the source vector. On exit, x is overwritten
     *        with the result vector.
     * incx   specifies the storage spacing for elements of x. incx must not be
     *        zero.
     *
     * Output
     * ------
     * x      updated according to x = op(A) * x,
     *
     * Reference: http://www.netlib.org/blas/ctpmv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if incx == 0 or n < 0
     * CUBLAS_STATUS_ALLOC_FAILED     if function cannot allocate enough internal scratch vector memory
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasCtpmv(char uplo, char trans, char diag, int n, Pointer AP, Pointer x, int incx)
    {
        cublasCtpmvNative(uplo, trans, diag, n, AP, x, incx);
        checkResultBLAS();
    }
    private static native void cublasCtpmvNative(char uplo, char trans, char diag, int n, Pointer AP, Pointer x, int incx);





    /**
     * <pre>
     * void
     * cublasCtrsv (char uplo, char trans, char diag, int n, const cuComplex *A,
     *              int lda, cuComplex *x, int incx)
     *
     * solves a system of equations op(A) * x = b, where op(A) is either A,
     * transpose(A) or conjugate(transpose(A)). b and x are single precision
     * complex vectors consisting of n elements, and A is an n x n matrix
     * composed of a unit or non-unit, upper or lower triangular matrix.
     * Matrix A is stored in column major format, and lda is the leading
     * dimension of the two-dimensional array containing A.
     *
     * No test for singularity or near-singularity is included in this function.
     * Such tests must be performed before calling this function.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix data is stored in the upper or the
     *        lower triangular part of array A. If uplo = 'U' or 'u', then only
     *        the upper triangular part of A may be referenced. If uplo = 'L' or
     *        'l', then only the lower triangular part of A may be referenced.
     * trans  specifies op(A). If transa = 'n' or 'N', op(A) = A. If transa = 't',
     *        'T', 'c', or 'C', op(A) = transpose(A)
     * diag   specifies whether or not A is a unit triangular matrix like so:
     *        if diag = 'U' or 'u', A is assumed to be unit triangular. If
     *        diag = 'N' or 'n', then A is not assumed to be unit triangular.
     * n      specifies the number of rows and columns of the matrix A. It
     *        must be at least 0.
     * A      is a single precision complex array of dimensions (lda, n). If uplo = 'U'
     *        or 'u', then A must contains the upper triangular part of a symmetric
     *        matrix, and the strictly lower triangular parts is not referenced.
     *        If uplo = 'L' or 'l', then A contains the lower triangular part of
     *        a symmetric matrix, and the strictly upper triangular part is not
     *        referenced.
     * lda    is the leading dimension of the two-dimensional array containing A.
     *        lda must be at least max(1, n).
     * x      single precision complex array of length at least (1 + (n - 1) * abs(incx)).
     *        On entry, x contains the n element right-hand side vector b. On exit,
     *        it is overwritten with the solution vector x.
     * incx   specifies the storage spacing between elements of x. incx must not
     *        be zero.
     *
     * Output
     * ------
     * x      updated to contain the solution vector x that solves op(A) * x = b.
     *
     * Reference: http://www.netlib.org/blas/ctrsv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if incx == 0 or if n < 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasCtrsv(char uplo, char trans, char diag, int n, Pointer A, int lda, Pointer x, int incx)
    {
        cublasCtrsvNative(uplo, trans, diag, n, A, lda, x, incx);
        checkResultBLAS();
    }
    private static native void cublasCtrsvNative(char uplo, char trans, char diag, int n, Pointer A, int lda, Pointer x, int incx);





    /**
     * <pre>
     * void cublasCtbsv (char uplo, char trans, char diag, int n, int k,
     *                   const cuComplex *A, int lda, cuComplex *X, int incx)
     *
     * solves one of the systems of equations op(A)*x = b, where op(A) is either
     * op(A) = A , op(A) = transpose(A) or op(A) = conjugate(transpose(A)).
     * b and x are n element vectors, and A is an n x n unit or non-unit,
     * upper or lower triangular band matrix with k + 1 diagonals. No test
     * for singularity or near-singularity is included in this function.
     * Such tests must be performed before calling this function.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix is an upper or lower triangular band
     *        matrix as follows: If uplo == 'U' or 'u', A is an upper triangular
     *        band matrix. If uplo == 'L' or 'l', A is a lower triangular band
     *        matrix.
     * trans  specifies op(A). If trans == 'N' or 'n', op(A) = A. If trans == 'T',
     *        't', op(A) = transpose(A). If trans == 'C' or 'c',
     *        op(A) = conjugate(transpose(A)).
     * diag   specifies whether A is unit triangular. If diag == 'U' or 'u', A is
     *        assumed to be unit triangular; thas is, diagonal elements are not
     *        read and are assumed to be unity. If diag == 'N' or 'n', A is not
     *        assumed to be unit triangular.
     * n      specifies the number of rows and columns of the matrix A. n must be
     *        at least zero.
     * k      specifies the number of super- or sub-diagonals. If uplo == 'U' or
     *        'u', k specifies the number of super-diagonals. If uplo == 'L' or
     *        'l', k specifies the number of sub-diagonals. k must at least be
     *        zero.
     * A      single precision complex array of dimension (lda, n). If uplo == 'U' or 'u',
     *        the leading (k + 1) x n part of the array A must contain the upper
     *        triangular band matrix, supplied column by column, with the leading
     *        diagonal of the matrix in row (k + 1) of the array, the first super-
     *        diagonal starting at position 2 in row k, and so on. The top left
     *        k x k triangle of the array A is not referenced. If uplo == 'L' or
     *        'l', the leading (k + 1) x n part of the array A must constain the
     *        lower triangular band matrix, supplied column by column, with the
     *        leading diagonal of the matrix in row 1 of the array, the first
     *        sub-diagonal starting at position 1 in row 2, and so on. The bottom
     *        right k x k triangle of the array is not referenced.
     * x      single precision complex array of length at least (1+(n-1)*abs(incx)).
     * incx   storage spacing between elements of x. It must not be zero.
     *
     * Output
     * ------
     * x      updated to contain the solution vector x that solves op(A) * x = b.
     *
     * Reference: http://www.netlib.org/blas/ctbsv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if incx == 0, n < 0 or n > 2035
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasCtbsv(char uplo, char trans, char diag, int n, int k, Pointer A, int lda, Pointer x, int incx)
    {
        cublasCtbsvNative(uplo, trans, diag, n, k, A, lda, x, incx);
        checkResultBLAS();
    }
    private static native void cublasCtbsvNative(char uplo, char trans, char diag, int n, int k, Pointer A, int lda, Pointer x, int incx);





    /**
     * <pre>
     * void
     * cublasCtpsv (char uplo, char trans, char diag, int n, const cuComplex *AP,
     *              cuComplex *X, int incx)
     *
     * solves one of the systems of equations op(A)*x = b, where op(A) is either
     * op(A) = A , op(A) = transpose(A) or op(A) = conjugate(transpose)). b and
     * x are n element complex vectors, and A is an n x n unit or non-unit,
     * upper or lower triangular matrix. No test for singularity or near-singularity
     * is included in this routine. Such tests must be performed before calling this routine.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix is an upper or lower triangular matrix
     *        as follows: If uplo == 'U' or 'u', A is an upper triangluar matrix.
     *        If uplo == 'L' or 'l', A is a lower triangular matrix.
     * trans  specifies op(A). If trans == 'N' or 'n', op(A) = A. If trans == 'T'
     *        or 't', op(A) = transpose(A). If trans == 'C' or 'c', op(A) =
     *        conjugate(transpose(A)).
     * diag   specifies whether A is unit triangular. If diag == 'U' or 'u', A is
     *        assumed to be unit triangular; thas is, diagonal elements are not
     *        read and are assumed to be unity. If diag == 'N' or 'n', A is not
     *        assumed to be unit triangular.
     * n      specifies the number of rows and columns of the matrix A. n must be
     *        at least zero.
     * AP     single precision complex array with at least ((n*(n+1))/2) elements.
     *        If uplo == 'U' or 'u', the array AP contains the upper triangular
     *        matrix A, packed sequentially, column by column; that is, if i <= j, then
     *        A[i,j] is stored is AP[i+(j*(j+1)/2)]. If uplo == 'L' or 'L', the
     *        array AP contains the lower triangular matrix A, packed sequentially,
     *        column by column; that is, if i >= j, then A[i,j] is stored in
     *        AP[i+((2*n-j+1)*j)/2]. When diag = 'U' or 'u', the diagonal elements
     *        of A are not referenced and are assumed to be unity.
     * x      single precision complex array of length at least (1+(n-1)*abs(incx)).
     * incx   storage spacing between elements of x. It must not be zero.
     *
     * Output
     * ------
     * x      updated to contain the solution vector x that solves op(A) * x = b.
     *
     * Reference: http://www.netlib.org/blas/ctpsv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if incx == 0 or if n < 0 or n > 2035
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasCtpsv(char uplo, char trans, char diag, int n, Pointer AP, Pointer x, int incx)
    {
        cublasCtpsvNative(uplo, trans, diag, n, AP, x, incx);
        checkResultBLAS();
    }
    private static native void cublasCtpsvNative(char uplo, char trans, char diag, int n, Pointer AP, Pointer x, int incx);





    /**
     * <pre>
     * cublasCgeru (int m, int n, cuComplex alpha, const cuComplex *x, int incx,
     *             const cuComplex *y, int incy, cuComplex *A, int lda)
     *
     * performs the symmetric rank 1 operation
     *
     *    A = alpha * x * transpose(y) + A,
     *
     * where alpha is a single precision complex scalar, x is an m element single
     * precision complex vector, y is an n element single precision complex vector, and A
     * is an m by n matrix consisting of single precision complex elements. Matrix A
     * is stored in column major format, and lda is the leading dimension of
     * the two-dimensional array used to store A.
     *
     * Input
     * -----
     * m      specifies the number of rows of the matrix A. It must be at least
     *        zero.
     * n      specifies the number of columns of the matrix A. It must be at
     *        least zero.
     * alpha  single precision complex scalar multiplier applied to x * transpose(y)
     * x      single precision complex array of length at least (1 + (m - 1) * abs(incx))
     * incx   specifies the storage spacing between elements of x. incx must not
     *        be zero.
     * y      single precision complex array of length at least (1 + (n - 1) * abs(incy))
     * incy   specifies the storage spacing between elements of y. incy must not
     *        be zero.
     * A      single precision complex array of dimensions (lda, n).
     * lda    leading dimension of two-dimensional array used to store matrix A
     *
     * Output
     * ------
     * A      updated according to A = alpha * x * transpose(y) + A
     *
     * Reference: http://www.netlib.org/blas/cgeru.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if m <0, n < 0, incx == 0, incy == 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasCgeru(int m, int n, cuComplex alpha, Pointer x, int incx, Pointer y, int incy, Pointer A, int lda)
    {
        cublasCgeruNative(m, n, alpha, x, incx, y, incy, A, lda);
        checkResultBLAS();
    }
    private static native void cublasCgeruNative(int m, int n, cuComplex alpha, Pointer x, int incx, Pointer y, int incy, Pointer A, int lda);





    /**
     * <pre>
     * cublasCgerc (int m, int n, cuComplex alpha, const cuComplex *x, int incx,
     *             const cuComplex *y, int incy, cuComplex *A, int lda)
     *
     * performs the symmetric rank 1 operation
     *
     *    A = alpha * x * conjugate(transpose(y)) + A,
     *
     * where alpha is a single precision complex scalar, x is an m element single
     * precision complex vector, y is an n element single precision complex vector, and A
     * is an m by n matrix consisting of single precision complex elements. Matrix A
     * is stored in column major format, and lda is the leading dimension of
     * the two-dimensional array used to store A.
     *
     * Input
     * -----
     * m      specifies the number of rows of the matrix A. It must be at least
     *        zero.
     * n      specifies the number of columns of the matrix A. It must be at
     *        least zero.
     * alpha  single precision complex scalar multiplier applied to x * transpose(y)
     * x      single precision complex array of length at least (1 + (m - 1) * abs(incx))
     * incx   specifies the storage spacing between elements of x. incx must not
     *        be zero.
     * y      single precision complex array of length at least (1 + (n - 1) * abs(incy))
     * incy   specifies the storage spacing between elements of y. incy must not
     *        be zero.
     * A      single precision complex array of dimensions (lda, n).
     * lda    leading dimension of two-dimensional array used to store matrix A
     *
     * Output
     * ------
     * A      updated according to A = alpha * x * conjugate(transpose(y)) + A
     *
     * Reference: http://www.netlib.org/blas/cgerc.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if m <0, n < 0, incx == 0, incy == 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasCgerc(int m, int n, cuComplex alpha, Pointer x, int incx, Pointer y, int incy, Pointer A, int lda)
    {
        cublasCgercNative(m, n, alpha, x, incx, y, incy, A, lda);
        checkResultBLAS();
    }
    private static native void cublasCgercNative(int m, int n, cuComplex alpha, Pointer x, int incx, Pointer y, int incy, Pointer A, int lda);





    /**
     * <pre>
     * void
     * cublasCher (char uplo, int n, float alpha, const cuComplex *x, int incx,
     *             cuComplex *A, int lda)
     *
     * performs the hermitian rank 1 operation
     *
     *    A = alpha * x * conjugate(transpose(x)) + A,
     *
     * where alpha is a single precision real scalar, x is an n element single
     * precision complex vector and A is an n x n hermitian matrix consisting of
     * single precision complex elements. Matrix A is stored in column major format,
     * and lda is the leading dimension of the two-dimensional array
     * containing A.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix data is stored in the upper or
     *        the lower triangular part of array A. If uplo = 'U' or 'u',
     *        then only the upper triangular part of A may be referenced.
     *        If uplo = 'L' or 'l', then only the lower triangular part of
     *        A may be referenced.
     * n      specifies the number of rows and columns of the matrix A. It
     *        must be at least 0.
     * alpha  single precision real scalar multiplier applied to
     *        x * conjugate(transpose(x))
     * x      single precision complex array of length at least (1 + (n - 1) * abs(incx))
     * incx   specifies the storage spacing between elements of x. incx must
     *        not be zero.
     * A      single precision complex array of dimensions (lda, n). If uplo = 'U' or
     *        'u', then A must contain the upper triangular part of a hermitian
     *        matrix, and the strictly lower triangular part is not referenced.
     *        If uplo = 'L' or 'l', then A contains the lower triangular part
     *        of a hermitian matrix, and the strictly upper triangular part is
     *        not referenced. The imaginary parts of the diagonal elements need
     *        not be set, they are assumed to be zero, and on exit they
     *        are set to zero.
     * lda    leading dimension of the two-dimensional array containing A. lda
     *        must be at least max(1, n).
     *
     * Output
     * ------
     * A      updated according to A = alpha * x * conjugate(transpose(x)) + A
     *
     * Reference: http://www.netlib.org/blas/cher.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or incx == 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasCher(char uplo, int n, float alpha, Pointer x, int incx, Pointer A, int lda)
    {
        cublasCherNative(uplo, n, alpha, x, incx, A, lda);
        checkResultBLAS();
    }
    private static native void cublasCherNative(char uplo, int n, float alpha, Pointer x, int incx, Pointer A, int lda);





    /**
     * <pre>
     * void
     * cublasChpr (char uplo, int n, float alpha, const cuComplex *x, int incx,
     *             cuComplex *AP)
     *
     * performs the hermitian rank 1 operation
     *
     *    A = alpha * x * conjugate(transpose(x)) + A,
     *
     * where alpha is a single precision real scalar and x is an n element single
     * precision complex vector. A is a hermitian n x n matrix consisting of single
     * precision complex elements that is supplied in packed form.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix data is stored in the upper or the lower
     *        triangular part of array AP. If uplo == 'U' or 'u', then the upper
     *        triangular part of A is supplied in AP. If uplo == 'L' or 'l', then
     *        the lower triangular part of A is supplied in AP.
     * n      specifies the number of rows and columns of the matrix A. It must be
     *        at least zero.
     * alpha  single precision real scalar multiplier applied to x * conjugate(transpose(x)).
     * x      single precision array of length at least (1 + (n - 1) * abs(incx)).
     * incx   storage spacing between elements of x. incx must not be zero.
     * AP     single precision complex array with at least ((n * (n + 1)) / 2) elements. If
     *        uplo == 'U' or 'u', the array AP contains the upper triangular part
     *        of the hermitian matrix A, packed sequentially, column by column;
     *        that is, if i <= j, then A[i,j] is stored is AP[i+(j*(j+1)/2)]. If
     *        uplo == 'L' or 'L', the array AP contains the lower triangular part
     *        of the hermitian matrix A, packed sequentially, column by column;
     *        that is, if i >= j, then A[i,j] is stored in AP[i+((2*n-j+1)*j)/2].
     *        The imaginary parts of the diagonal elements need not be set, they
     *        are assumed to be zero, and on exit they are set to zero.
     *
     * Output
     * ------
     * A      updated according to A = alpha * x * conjugate(transpose(x)) + A
     *
     * Reference: http://www.netlib.org/blas/chpr.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or incx == 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasChpr(char uplo, int n, float alpha, Pointer x, int incx, Pointer AP)
    {
        cublasChprNative(uplo, n, alpha, x, incx, AP);
        checkResultBLAS();
    }
    private static native void cublasChprNative(char uplo, int n, float alpha, Pointer x, int incx, Pointer AP);





    /**
     * <pre>
     * void
     * cublasChpr2 (char uplo, int n, cuComplex alpha, const cuComplex *x, int incx,
     *              const cuComplex *y, int incy, cuComplex *AP)
     *
     * performs the hermitian rank 2 operation
     *
     *    A = alpha*x*conjugate(transpose(y)) + conjugate(alpha)*y*conjugate(transpose(x)) + A,
     *
     * where alpha is a single precision complex scalar, and x and y are n element single
     * precision complex vectors. A is a hermitian n x n matrix consisting of single
     * precision complex elements that is supplied in packed form.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix data is stored in the upper or the lower
     *        triangular part of array A. If uplo == 'U' or 'u', then only the
     *        upper triangular part of A may be referenced and the lower triangular
     *        part of A is inferred. If uplo == 'L' or 'l', then only the lower
     *        triangular part of A may be referenced and the upper triangular part
     *        of A is inferred.
     * n      specifies the number of rows and columns of the matrix A. It must be
     *        at least zero.
     * alpha  single precision complex scalar multiplier applied to x * conjugate(transpose(y)) +
     *        y * conjugate(transpose(x)).
     * x      single precision complex array of length at least (1 + (n - 1) * abs (incx)).
     * incx   storage spacing between elements of x. incx must not be zero.
     * y      single precision complex array of length at least (1 + (n - 1) * abs (incy)).
     * incy   storage spacing between elements of y. incy must not be zero.
     * AP     single precision complex array with at least ((n * (n + 1)) / 2) elements. If
     *        uplo == 'U' or 'u', the array AP contains the upper triangular part
     *        of the hermitian matrix A, packed sequentially, column by column;
     *        that is, if i <= j, then A[i,j] is stored is AP[i+(j*(j+1)/2)]. If
     *        uplo == 'L' or 'L', the array AP contains the lower triangular part
     *        of the hermitian matrix A, packed sequentially, column by column;
     *        that is, if i >= j, then A[i,j] is stored in AP[i+((2*n-j+1)*j)/2].
     *        The imaginary parts of the diagonal elements need not be set, they
     *        are assumed to be zero, and on exit they are set to zero.
     *
     * Output
     * ------
     * A      updated according to A = alpha*x*conjugate(transpose(y))
     *                               + conjugate(alpha)*y*conjugate(transpose(x))+A
     *
     * Reference: http://www.netlib.org/blas/chpr2.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0, incx == 0, incy == 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasChpr2(char uplo, int n, cuComplex alpha, Pointer x, int incx, Pointer y, int incy, Pointer AP)
    {
        cublasChpr2Native(uplo, n, alpha, x, incx, y, incy, AP);
        checkResultBLAS();
    }
    private static native void cublasChpr2Native(char uplo, int n, cuComplex alpha, Pointer x, int incx, Pointer y, int incy, Pointer AP);





    /**
     * <pre>
     * void cublasCher2 (char uplo, int n, cuComplex alpha, const cuComplex *x, int incx,
     *                   const cuComplex *y, int incy, cuComplex *A, int lda)
     *
     * performs the hermitian rank 2 operation
     *
     *    A = alpha*x*conjugate(transpose(y)) + conjugate(alpha)*y*conjugate(transpose(x)) + A,
     *
     * where alpha is a single precision complex scalar, x and y are n element single
     * precision complex vector and A is an n by n hermitian matrix consisting of single
     * precision complex elements.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix data is stored in the upper or the lower
     *        triangular part of array A. If uplo == 'U' or 'u', then only the
     *        upper triangular part of A may be referenced and the lower triangular
     *        part of A is inferred. If uplo == 'L' or 'l', then only the lower
     *        triangular part of A may be referenced and the upper triangular part
     *        of A is inferred.
     * n      specifies the number of rows and columns of the matrix A. It must be
     *        at least zero.
     * alpha  single precision complex scalar multiplier applied to x * conjugate(transpose(y)) +
     *        y * conjugate(transpose(x)).
     * x      single precision array of length at least (1 + (n - 1) * abs (incx)).
     * incx   storage spacing between elements of x. incx must not be zero.
     * y      single precision array of length at least (1 + (n - 1) * abs (incy)).
     * incy   storage spacing between elements of y. incy must not be zero.
     * A      single precision complex array of dimensions (lda, n). If uplo == 'U' or 'u',
     *        then A must contains the upper triangular part of a hermitian matrix,
     *        and the strictly lower triangular parts is not referenced. If uplo ==
     *        'L' or 'l', then A contains the lower triangular part of a hermitian
     *        matrix, and the strictly upper triangular part is not referenced.
     *        The imaginary parts of the diagonal elements need not be set,
     *        they are assumed to be zero, and on exit they are set to zero.
     *
     * lda    leading dimension of A. It must be at least max(1, n).
     *
     * Output
     * ------
     * A      updated according to A = alpha*x*conjugate(transpose(y))
     *                               + conjugate(alpha)*y*conjugate(transpose(x))+A
     *
     * Reference: http://www.netlib.org/blas/cher2.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0, incx == 0, incy == 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasCher2(char uplo, int n, cuComplex alpha, Pointer x, int incx, Pointer y, int incy, Pointer A, int lda)
    {
        cublasCher2Native(uplo, n, alpha, x, incx, y, incy, A, lda);
        checkResultBLAS();
    }
    private static native void cublasCher2Native(char uplo, int n, cuComplex alpha, Pointer x, int incx, Pointer y, int incy, Pointer A, int lda);





    /**
     * <pre>
     * void
     * cublasSgemm (char transa, char transb, int m, int n, int k, float alpha,
     *              const float *A, int lda, const float *B, int ldb, float beta,
     *              float *C, int ldc)
     *
     * computes the product of matrix A and matrix B, multiplies the result
     * by a scalar alpha, and adds the sum to the product of matrix C and
     * scalar beta. sgemm() performs one of the matrix-matrix operations:
     *
     *     C = alpha * op(A) * op(B) + beta * C,
     *
     * where op(X) is one of
     *
     *     op(X) = X   or   op(X) = transpose(X)
     *
     * alpha and beta are single precision scalars, and A, B and C are
     * matrices consisting of single precision elements, with op(A) an m x k
     * matrix, op(B) a k x n matrix, and C an m x n matrix. Matrices A, B,
     * and C are stored in column major format, and lda, ldb, and ldc are
     * the leading dimensions of the two-dimensional arrays containing A,
     * B, and C.
     *
     * Input
     * -----
     * transa specifies op(A). If transa = 'n' or 'N', op(A) = A. If
     *        transa = 't', 'T', 'c', or 'C', op(A) = transpose(A)
     * transb specifies op(B). If transb = 'n' or 'N', op(B) = B. If
     *        transb = 't', 'T', 'c', or 'C', op(B) = transpose(B)
     * m      number of rows of matrix op(A) and rows of matrix C
     * n      number of columns of matrix op(B) and number of columns of C
     * k      number of columns of matrix op(A) and number of rows of op(B)
     * alpha  single precision scalar multiplier applied to op(A)op(B)
     * A      single precision array of dimensions (lda, k) if transa =
     *        'n' or 'N'), and of dimensions (lda, m) otherwise. When transa =
     *        'N' or 'n' then lda must be at least  max( 1, m ), otherwise lda
     *        must be at least max(1, k).
     * lda    leading dimension of two-dimensional array used to store matrix A
     * B      single precision array of dimensions  (ldb, n) if transb =
     *        'n' or 'N'), and of dimensions (ldb, k) otherwise. When transb =
     *        'N' or 'n' then ldb must be at least  max (1, k), otherwise ldb
     *        must be at least max (1, n).
     * ldb    leading dimension of two-dimensional array used to store matrix B
     * beta   single precision scalar multiplier applied to C. If 0, C does
     *        not have to be a valid input
     * C      single precision array of dimensions (ldc, n). ldc must be at
     *        least max (1, m).
     * ldc    leading dimension of two-dimensional array used to store matrix C
     *
     * Output
     * ------
     * C      updated based on C = alpha * op(A)*op(B) + beta * C
     *
     * Reference: http://www.netlib.org/blas/sgemm.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if any of m, n, or k are < 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasSgemm(char transa, char transb, int m, int n, int k, float alpha, Pointer A, int lda, Pointer B, int ldb, float beta, Pointer C, int ldc)
    {
        cublasSgemmNative(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        checkResultBLAS();
    }
    private static native void cublasSgemmNative(char transa, char transb, int m, int n, int k, float alpha, Pointer A, int lda, Pointer B, int ldb, float beta, Pointer C, int ldc);





    /**
     * <pre>
     * void
     * cublasSsymm (char side, char uplo, int m, int n, float alpha,
     *              const float *A, int lda, const float *B, int ldb,
     *              float beta, float *C, int ldc);
     *
     * performs one of the matrix-matrix operations
     *
     *   C = alpha * A * B + beta * C, or
     *   C = alpha * B * A + beta * C,
     *
     * where alpha and beta are single precision scalars, A is a symmetric matrix
     * consisting of single precision elements and stored in either lower or upper
     * storage mode, and B and C are m x n matrices consisting of single precision
     * elements.
     *
     * Input
     * -----
     * side   specifies whether the symmetric matrix A appears on the left side
     *        hand side or right hand side of matrix B, as follows. If side == 'L'
     *        or 'l', then C = alpha * A * B + beta * C. If side = 'R' or 'r',
     *        then C = alpha * B * A + beta * C.
     * uplo   specifies whether the symmetric matrix A is stored in upper or lower
     *        storage mode, as follows. If uplo == 'U' or 'u', only the upper
     *        triangular part of the symmetric matrix is to be referenced, and the
     *        elements of the strictly lower triangular part are to be infered from
     *        those in the upper triangular part. If uplo == 'L' or 'l', only the
     *        lower triangular part of the symmetric matrix is to be referenced,
     *        and the elements of the strictly upper triangular part are to be
     *        infered from those in the lower triangular part.
     * m      specifies the number of rows of the matrix C, and the number of rows
     *        of matrix B. It also specifies the dimensions of symmetric matrix A
     *        when side == 'L' or 'l'. m must be at least zero.
     * n      specifies the number of columns of the matrix C, and the number of
     *        columns of matrix B. It also specifies the dimensions of symmetric
     *        matrix A when side == 'R' or 'r'. n must be at least zero.
     * alpha  single precision scalar multiplier applied to A * B, or B * A
     * A      single precision array of dimensions (lda, ka), where ka is m when
     *        side == 'L' or 'l' and is n otherwise. If side == 'L' or 'l' the
     *        leading m x m part of array A must contain the symmetric matrix,
     *        such that when uplo == 'U' or 'u', the leading m x m part stores the
     *        upper triangular part of the symmetric matrix, and the strictly lower
     *        triangular part of A is not referenced, and when uplo == 'U' or 'u',
     *        the leading m x m part stores the lower triangular part of the
     *        symmetric matrix and the strictly upper triangular part is not
     *        referenced. If side == 'R' or 'r' the leading n x n part of array A
     *        must contain the symmetric matrix, such that when uplo == 'U' or 'u',
     *        the leading n x n part stores the upper triangular part of the
     *        symmetric matrix and the strictly lower triangular part of A is not
     *        referenced, and when uplo == 'U' or 'u', the leading n x n part
     *        stores the lower triangular part of the symmetric matrix and the
     *        strictly upper triangular part is not referenced.
     * lda    leading dimension of A. When side == 'L' or 'l', it must be at least
     *        max(1, m) and at least max(1, n) otherwise.
     * B      single precision array of dimensions (ldb, n). On entry, the leading
     *        m x n part of the array contains the matrix B.
     * ldb    leading dimension of B. It must be at least max (1, m).
     * beta   single precision scalar multiplier applied to C. If beta is zero, C
     *        does not have to be a valid input
     * C      single precision array of dimensions (ldc, n)
     * ldc    leading dimension of C. Must be at least max(1, m)
     *
     * Output
     * ------
     * C      updated according to C = alpha * A * B + beta * C, or C = alpha *
     *        B * A + beta * C
     *
     * Reference: http://www.netlib.org/blas/ssymm.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if m or n are < 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasSsymm(char side, char uplo, int m, int n, float alpha, Pointer A, int lda, Pointer B, int ldb, float beta, Pointer C, int ldc)
    {
        cublasSsymmNative(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
        checkResultBLAS();
    }
    private static native void cublasSsymmNative(char side, char uplo, int m, int n, float alpha, Pointer A, int lda, Pointer B, int ldb, float beta, Pointer C, int ldc);





    /**
     * <pre>
     * void
     * cublasSsyrk (char uplo, char trans, int n, int k, float alpha,
     *              const float *A, int lda, float beta, float *C, int ldc)
     *
     * performs one of the symmetric rank k operations
     *
     *   C = alpha * A * transpose(A) + beta * C, or
     *   C = alpha * transpose(A) * A + beta * C.
     *
     * Alpha and beta are single precision scalars. C is an n x n symmetric matrix
     * consisting of single precision elements and stored in either lower or
     * upper storage mode. A is a matrix consisting of single precision elements
     * with dimension of n x k in the first case, and k x n in the second case.
     *
     * Input
     * -----
     * uplo   specifies whether the symmetric matrix C is stored in upper or lower
     *        storage mode as follows. If uplo == 'U' or 'u', only the upper
     *        triangular part of the symmetric matrix is to be referenced, and the
     *        elements of the strictly lower triangular part are to be infered from
     *        those in the upper triangular part. If uplo == 'L' or 'l', only the
     *        lower triangular part of the symmetric matrix is to be referenced,
     *        and the elements of the strictly upper triangular part are to be
     *        infered from those in the lower triangular part.
     * trans  specifies the operation to be performed. If trans == 'N' or 'n', C =
     *        alpha * transpose(A) + beta * C. If trans == 'T', 't', 'C', or 'c',
     *        C = transpose(A) * A + beta * C.
     * n      specifies the number of rows and the number columns of matrix C. If
     *        trans == 'N' or 'n', n specifies the number of rows of matrix A. If
     *        trans == 'T', 't', 'C', or 'c', n specifies the columns of matrix A.
     *        n must be at least zero.
     * k      If trans == 'N' or 'n', k specifies the number of rows of matrix A.
     *        If trans == 'T', 't', 'C', or 'c', k specifies the number of rows of
     *        matrix A. k must be at least zero.
     * alpha  single precision scalar multiplier applied to A * transpose(A) or
     *        transpose(A) * A.
     * A      single precision array of dimensions (lda, ka), where ka is k when
     *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n',
     *        the leading n x k part of array A must contain the matrix A,
     *        otherwise the leading k x n part of the array must contains the
     *        matrix A.
     * lda    leading dimension of A. When trans == 'N' or 'n' then lda must be at
     *        least max(1, n). Otherwise lda must be at least max(1, k).
     * beta   single precision scalar multiplier applied to C. If beta izs zero, C
     *        does not have to be a valid input
     * C      single precision array of dimensions (ldc, n). If uplo == 'U' or 'u',
     *        the leading n x n triangular part of the array C must contain the
     *        upper triangular part of the symmetric matrix C and the strictly
     *        lower triangular part of C is not referenced. On exit, the upper
     *        triangular part of C is overwritten by the upper triangular part of
     *        the updated matrix. If uplo == 'L' or 'l', the leading n x n
     *        triangular part of the array C must contain the lower triangular part
     *        of the symmetric matrix C and the strictly upper triangular part of C
     *        is not referenced. On exit, the lower triangular part of C is
     *        overwritten by the lower triangular part of the updated matrix.
     * ldc    leading dimension of C. It must be at least max(1, n).
     *
     * Output
     * ------
     * C      updated according to C = alpha * A * transpose(A) + beta * C, or C =
     *        alpha * transpose(A) * A + beta * C
     *
     * Reference: http://www.netlib.org/blas/ssyrk.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0 or k < 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasSsyrk(char uplo, char trans, int n, int k, float alpha, Pointer A, int lda, float beta, Pointer C, int ldc)
    {
        cublasSsyrkNative(uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
        checkResultBLAS();
    }
    private static native void cublasSsyrkNative(char uplo, char trans, int n, int k, float alpha, Pointer A, int lda, float beta, Pointer C, int ldc);





    /**
     * <pre>
     * void
     * cublasSsyr2k (char uplo, char trans, int n, int k, float alpha,
     *               const float *A, int lda, const float *B, int ldb,
     *               float beta, float *C, int ldc)
     *
     * performs one of the symmetric rank 2k operations
     *
     *    C = alpha * A * transpose(B) + alpha * B * transpose(A) + beta * C, or
     *    C = alpha * transpose(A) * B + alpha * transpose(B) * A + beta * C.
     *
     * Alpha and beta are single precision scalars. C is an n x n symmetric matrix
     * consisting of single precision elements and stored in either lower or upper
     * storage mode. A and B are matrices consisting of single precision elements
     * with dimension of n x k in the first case, and k x n in the second case.
     *
     * Input
     * -----
     * uplo   specifies whether the symmetric matrix C is stored in upper or lower
     *        storage mode, as follows. If uplo == 'U' or 'u', only the upper
     *        triangular part of the symmetric matrix is to be referenced, and the
     *        elements of the strictly lower triangular part are to be infered from
     *        those in the upper triangular part. If uplo == 'L' or 'l', only the
     *        lower triangular part of the symmetric matrix is to be references,
     *        and the elements of the strictly upper triangular part are to be
     *        infered from those in the lower triangular part.
     * trans  specifies the operation to be performed. If trans == 'N' or 'n',
     *        C = alpha * A * transpose(B) + alpha * B * transpose(A) + beta * C,
     *        If trans == 'T', 't', 'C', or 'c', C = alpha * transpose(A) * B +
     *        alpha * transpose(B) * A + beta * C.
     * n      specifies the number of rows and the number columns of matrix C. If
     *        trans == 'N' or 'n', n specifies the number of rows of matrix A. If
     *        trans == 'T', 't', 'C', or 'c', n specifies the columns of matrix A.
     *        n must be at least zero.
     * k      If trans == 'N' or 'n', k specifies the number of rows of matrix A.
     *        If trans == 'T', 't', 'C', or 'c', k specifies the number of rows of
     *        matrix A. k must be at least zero.
     * alpha  single precision scalar multiplier.
     * A      single precision array of dimensions (lda, ka), where ka is k when
     *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n',
     *        the leading n x k part of array A must contain the matrix A,
     *        otherwise the leading k x n part of the array must contain the matrix
     *        A.
     * lda    leading dimension of A. When trans == 'N' or 'n' then lda must be at
     *        least max(1, n). Otherwise lda must be at least max(1,k).
     * B      single precision array of dimensions (lda, kb), where kb is k when
     *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n',
     *        the leading n x k part of array B must contain the matrix B,
     *        otherwise the leading k x n part of the array must contain the matrix
     *        B.
     * ldb    leading dimension of N. When trans == 'N' or 'n' then ldb must be at
     *        least max(1, n). Otherwise ldb must be at least max(1, k).
     * beta   single precision scalar multiplier applied to C. If beta is zero, C
     *        does not have to be a valid input.
     * C      single precision array of dimensions (ldc, n). If uplo == 'U' or 'u',
     *        the leading n x n triangular part of the array C must contain the
     *        upper triangular part of the symmetric matrix C and the strictly
     *        lower triangular part of C is not referenced. On exit, the upper
     *        triangular part of C is overwritten by the upper triangular part of
     *        the updated matrix. If uplo == 'L' or 'l', the leading n x n
     *        triangular part of the array C must contain the lower triangular part
     *        of the symmetric matrix C and the strictly upper triangular part of C
     *        is not referenced. On exit, the lower triangular part of C is
     *        overwritten by the lower triangular part of the updated matrix.
     * ldc    leading dimension of C. Must be at least max(1, n).
     *
     * Output
     * ------
     * C      updated according to alpha*A*transpose(B) + alpha*B*transpose(A) +
     *        beta*C or alpha*transpose(A)*B + alpha*transpose(B)*A + beta*C
     *
     * Reference:   http://www.netlib.org/blas/ssyr2k.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0 or k < 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasSsyr2k(char uplo, char trans, int n, int k, float alpha, Pointer A, int lda, Pointer B, int ldb, float beta, Pointer C, int ldc)
    {
        cublasSsyr2kNative(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        checkResultBLAS();
    }
    private static native void cublasSsyr2kNative(char uplo, char trans, int n, int k, float alpha, Pointer A, int lda, Pointer B, int ldb, float beta, Pointer C, int ldc);





    /**
     * <pre>
     * void
     * cublasStrmm (char side, char uplo, char transa, char diag, int m, int n,
     *              float alpha, const float *A, int lda, const float *B, int ldb)
     *
     * performs one of the matrix-matrix operations
     *
     *   B = alpha * op(A) * B,  or  B = alpha * B * op(A)
     *
     * where alpha is a single-precision scalar, B is an m x n matrix composed
     * of single precision elements, and A is a unit or non-unit, upper or lower,
     * triangular matrix composed of single precision elements. op(A) is one of
     *
     *   op(A) = A  or  op(A) = transpose(A)
     *
     * Matrices A and B are stored in column major format, and lda and ldb are
     * the leading dimensions of the two-dimensonials arrays that contain A and
     * B, respectively.
     *
     * Input
     * -----
     * side   specifies whether op(A) multiplies B from the left or right.
     *        If side = 'L' or 'l', then B = alpha * op(A) * B. If side =
     *        'R' or 'r', then B = alpha * B * op(A).
     * uplo   specifies whether the matrix A is an upper or lower triangular
     *        matrix. If uplo = 'U' or 'u', A is an upper triangular matrix.
     *        If uplo = 'L' or 'l', A is a lower triangular matrix.
     * transa specifies the form of op(A) to be used in the matrix
     *        multiplication. If transa = 'N' or 'n', then op(A) = A. If
     *        transa = 'T', 't', 'C', or 'c', then op(A) = transpose(A).
     * diag   specifies whether or not A is unit triangular. If diag = 'U'
     *        or 'u', A is assumed to be unit triangular. If diag = 'N' or
     *        'n', A is not assumed to be unit triangular.
     * m      the number of rows of matrix B. m must be at least zero.
     * n      the number of columns of matrix B. n must be at least zero.
     * alpha  single precision scalar multiplier applied to op(A)*B, or
     *        B*op(A), respectively. If alpha is zero no accesses are made
     *        to matrix A, and no read accesses are made to matrix B.
     * A      single precision array of dimensions (lda, k). k = m if side =
     *        'L' or 'l', k = n if side = 'R' or 'r'. If uplo = 'U' or 'u'
     *        the leading k x k upper triangular part of the array A must
     *        contain the upper triangular matrix, and the strictly lower
     *        triangular part of A is not referenced. If uplo = 'L' or 'l'
     *        the leading k x k lower triangular part of the array A must
     *        contain the lower triangular matrix, and the strictly upper
     *        triangular part of A is not referenced. When diag = 'U' or 'u'
     *        the diagonal elements of A are no referenced and are assumed
     *        to be unity.
     * lda    leading dimension of A. When side = 'L' or 'l', it must be at
     *        least max(1,m) and at least max(1,n) otherwise
     * B      single precision array of dimensions (ldb, n). On entry, the
     *        leading m x n part of the array contains the matrix B. It is
     *        overwritten with the transformed matrix on exit.
     * ldb    leading dimension of B. It must be at least max (1, m).
     *
     * Output
     * ------
     * B      updated according to B = alpha * op(A) * B  or B = alpha * B * op(A)
     *
     * Reference: http://www.netlib.org/blas/strmm.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if m or n < 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasStrmm(char side, char uplo, char transa, char diag, int m, int n, float alpha, Pointer A, int lda, Pointer B, int ldb)
    {
        cublasStrmmNative(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
        checkResultBLAS();
    }
    private static native void cublasStrmmNative(char side, char uplo, char transa, char diag, int m, int n, float alpha, Pointer A, int lda, Pointer B, int ldb);





    /**
     * <pre>
     * void
     * cublasStrsm (char side, char uplo, char transa, char diag, int m, int n,
     *              float alpha, const float *A, int lda, float *B, int ldb)
     *
     * solves one of the matrix equations
     *
     *    op(A) * X = alpha * B,   or   X * op(A) = alpha * B,
     *
     * where alpha is a single precision scalar, and X and B are m x n matrices
     * that are composed of single precision elements. A is a unit or non-unit,
     * upper or lower triangular matrix, and op(A) is one of
     *
     *    op(A) = A  or  op(A) = transpose(A)
     *
     * The result matrix X overwrites input matrix B; that is, on exit the result
     * is stored in B. Matrices A and B are stored in column major format, and
     * lda and ldb are the leading dimensions of the two-dimensonials arrays that
     * contain A and B, respectively.
     *
     * Input
     * -----
     * side   specifies whether op(A) appears on the left or right of X as
     *        follows: side = 'L' or 'l' indicates solve op(A) * X = alpha * B.
     *        side = 'R' or 'r' indicates solve X * op(A) = alpha * B.
     * uplo   specifies whether the matrix A is an upper or lower triangular
     *        matrix as follows: uplo = 'U' or 'u' indicates A is an upper
     *        triangular matrix. uplo = 'L' or 'l' indicates A is a lower
     *        triangular matrix.
     * transa specifies the form of op(A) to be used in matrix multiplication
     *        as follows: If transa = 'N' or 'N', then op(A) = A. If transa =
     *        'T', 't', 'C', or 'c', then op(A) = transpose(A).
     * diag   specifies whether or not A is a unit triangular matrix like so:
     *        if diag = 'U' or 'u', A is assumed to be unit triangular. If
     *        diag = 'N' or 'n', then A is not assumed to be unit triangular.
     * m      specifies the number of rows of B. m must be at least zero.
     * n      specifies the number of columns of B. n must be at least zero.
     * alpha  is a single precision scalar to be multiplied with B. When alpha is
     *        zero, then A is not referenced and B need not be set before entry.
     * A      is a single precision array of dimensions (lda, k), where k is
     *        m when side = 'L' or 'l', and is n when side = 'R' or 'r'. If
     *        uplo = 'U' or 'u', the leading k x k upper triangular part of
     *        the array A must contain the upper triangular matrix and the
     *        strictly lower triangular matrix of A is not referenced. When
     *        uplo = 'L' or 'l', the leading k x k lower triangular part of
     *        the array A must contain the lower triangular matrix and the
     *        strictly upper triangular part of A is not referenced. Note that
     *        when diag = 'U' or 'u', the diagonal elements of A are not
     *        referenced, and are assumed to be unity.
     * lda    is the leading dimension of the two dimensional array containing A.
     *        When side = 'L' or 'l' then lda must be at least max(1, m), when
     *        side = 'R' or 'r' then lda must be at least max(1, n).
     * B      is a single precision array of dimensions (ldb, n). ldb must be
     *        at least max (1,m). The leading m x n part of the array B must
     *        contain the right-hand side matrix B. On exit B is overwritten
     *        by the solution matrix X.
     * ldb    is the leading dimension of the two dimensional array containing B.
     *        ldb must be at least max(1, m).
     *
     * Output
     * ------
     * B      contains the solution matrix X satisfying op(A) * X = alpha * B,
     *        or X * op(A) = alpha * B
     *
     * Reference: http://www.netlib.org/blas/strsm.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if m or n < 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasStrsm(char side, char uplo, char transa, char diag, int m, int n, float alpha, Pointer A, int lda, Pointer B, int ldb)
    {
        cublasStrsmNative(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
        checkResultBLAS();
    }
    private static native void cublasStrsmNative(char side, char uplo, char transa, char diag, int m, int n, float alpha, Pointer A, int lda, Pointer B, int ldb);





    /**
     * <pre>
     * void cublasCgemm (char transa, char transb, int m, int n, int k,
     *                   cuComplex alpha, const cuComplex *A, int lda,
     *                   const cuComplex *B, int ldb, cuComplex beta,
     *                   cuComplex *C, int ldc)
     *
     * performs one of the matrix-matrix operations
     *
     *    C = alpha * op(A) * op(B) + beta*C,
     *
     * where op(X) is one of
     *
     *    op(X) = X   or   op(X) = transpose  or  op(X) = conjg(transpose(X))
     *
     * alpha and beta are single-complex scalars, and A, B and C are matrices
     * consisting of single-complex elements, with op(A) an m x k matrix, op(B)
     * a k x n matrix and C an m x n matrix.
     *
     * Input
     * -----
     * transa specifies op(A). If transa == 'N' or 'n', op(A) = A. If transa ==
     *        'T' or 't', op(A) = transpose(A). If transa == 'C' or 'c', op(A) =
     *        conjg(transpose(A)).
     * transb specifies op(B). If transa == 'N' or 'n', op(B) = B. If transb ==
     *        'T' or 't', op(B) = transpose(B). If transb == 'C' or 'c', op(B) =
     *        conjg(transpose(B)).
     * m      number of rows of matrix op(A) and rows of matrix C. It must be at
     *        least zero.
     * n      number of columns of matrix op(B) and number of columns of C. It
     *        must be at least zero.
     * k      number of columns of matrix op(A) and number of rows of op(B). It
     *        must be at least zero.
     * alpha  single-complex scalar multiplier applied to op(A)op(B)
     * A      single-complex array of dimensions (lda, k) if transa ==  'N' or
     *        'n'), and of dimensions (lda, m) otherwise.
     * lda    leading dimension of A. When transa == 'N' or 'n', it must be at
     *        least max(1, m) and at least max(1, k) otherwise.
     * B      single-complex array of dimensions (ldb, n) if transb == 'N' or 'n',
     *        and of dimensions (ldb, k) otherwise
     * ldb    leading dimension of B. When transb == 'N' or 'n', it must be at
     *        least max(1, k) and at least max(1, n) otherwise.
     * beta   single-complex scalar multiplier applied to C. If beta is zero, C
     *        does not have to be a valid input.
     * C      single precision array of dimensions (ldc, n)
     * ldc    leading dimension of C. Must be at least max(1, m).
     *
     * Output
     * ------
     * C      updated according to C = alpha*op(A)*op(B) + beta*C
     *
     * Reference: http://www.netlib.org/blas/cgemm.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if any of m, n, or k are < 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasCgemm(char transa, char transb, int m, int n, int k, cuComplex alpha, Pointer A, int lda, Pointer B, int ldb, cuComplex beta, Pointer C, int ldc)
    {
        cublasCgemmNative(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        checkResultBLAS();
    }
    private static native void cublasCgemmNative(char transa, char transb, int m, int n, int k, cuComplex alpha, Pointer A, int lda, Pointer B, int ldb, cuComplex beta, Pointer C, int ldc);





    /**
     * <pre>
     * void
     * cublasCsymm (char side, char uplo, int m, int n, cuComplex alpha,
     *              const cuComplex *A, int lda, const cuComplex *B, int ldb,
     *              cuComplex beta, cuComplex *C, int ldc);
     *
     * performs one of the matrix-matrix operations
     *
     *   C = alpha * A * B + beta * C, or
     *   C = alpha * B * A + beta * C,
     *
     * where alpha and beta are single precision complex scalars, A is a symmetric matrix
     * consisting of single precision complex elements and stored in either lower or upper
     * storage mode, and B and C are m x n matrices consisting of single precision
     * complex elements.
     *
     * Input
     * -----
     * side   specifies whether the symmetric matrix A appears on the left side
     *        hand side or right hand side of matrix B, as follows. If side == 'L'
     *        or 'l', then C = alpha * A * B + beta * C. If side = 'R' or 'r',
     *        then C = alpha * B * A + beta * C.
     * uplo   specifies whether the symmetric matrix A is stored in upper or lower
     *        storage mode, as follows. If uplo == 'U' or 'u', only the upper
     *        triangular part of the symmetric matrix is to be referenced, and the
     *        elements of the strictly lower triangular part are to be infered from
     *        those in the upper triangular part. If uplo == 'L' or 'l', only the
     *        lower triangular part of the symmetric matrix is to be referenced,
     *        and the elements of the strictly upper triangular part are to be
     *        infered from those in the lower triangular part.
     * m      specifies the number of rows of the matrix C, and the number of rows
     *        of matrix B. It also specifies the dimensions of symmetric matrix A
     *        when side == 'L' or 'l'. m must be at least zero.
     * n      specifies the number of columns of the matrix C, and the number of
     *        columns of matrix B. It also specifies the dimensions of symmetric
     *        matrix A when side == 'R' or 'r'. n must be at least zero.
     * alpha  single precision scalar multiplier applied to A * B, or B * A
     * A      single precision array of dimensions (lda, ka), where ka is m when
     *        side == 'L' or 'l' and is n otherwise. If side == 'L' or 'l' the
     *        leading m x m part of array A must contain the symmetric matrix,
     *        such that when uplo == 'U' or 'u', the leading m x m part stores the
     *        upper triangular part of the symmetric matrix, and the strictly lower
     *        triangular part of A is not referenced, and when uplo == 'U' or 'u',
     *        the leading m x m part stores the lower triangular part of the
     *        symmetric matrix and the strictly upper triangular part is not
     *        referenced. If side == 'R' or 'r' the leading n x n part of array A
     *        must contain the symmetric matrix, such that when uplo == 'U' or 'u',
     *        the leading n x n part stores the upper triangular part of the
     *        symmetric matrix and the strictly lower triangular part of A is not
     *        referenced, and when uplo == 'U' or 'u', the leading n x n part
     *        stores the lower triangular part of the symmetric matrix and the
     *        strictly upper triangular part is not referenced.
     * lda    leading dimension of A. When side == 'L' or 'l', it must be at least
     *        max(1, m) and at least max(1, n) otherwise.
     * B      single precision array of dimensions (ldb, n). On entry, the leading
     *        m x n part of the array contains the matrix B.
     * ldb    leading dimension of B. It must be at least max (1, m).
     * beta   single precision scalar multiplier applied to C. If beta is zero, C
     *        does not have to be a valid input
     * C      single precision array of dimensions (ldc, n)
     * ldc    leading dimension of C. Must be at least max(1, m)
     *
     * Output
     * ------
     * C      updated according to C = alpha * A * B + beta * C, or C = alpha *
     *        B * A + beta * C
     *
     * Reference: http://www.netlib.org/blas/csymm.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if m or n are < 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasCsymm(char side, char uplo, int m, int n, cuComplex alpha, Pointer A, int lda, Pointer B, int ldb, cuComplex beta, Pointer C, int ldc)
    {
        cublasCsymmNative(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
        checkResultBLAS();
    }
    private static native void cublasCsymmNative(char side, char uplo, int m, int n, cuComplex alpha, Pointer A, int lda, Pointer B, int ldb, cuComplex beta, Pointer C, int ldc);





    /**
     * <pre>
     * void
     * cublasChemm (char side, char uplo, int m, int n, cuComplex alpha,
     *              const cuComplex *A, int lda, const cuComplex *B, int ldb,
     *              cuComplex beta, cuComplex *C, int ldc);
     *
     * performs one of the matrix-matrix operations
     *
     *   C = alpha * A * B + beta * C, or
     *   C = alpha * B * A + beta * C,
     *
     * where alpha and beta are single precision complex scalars, A is a hermitian matrix
     * consisting of single precision complex elements and stored in either lower or upper
     * storage mode, and B and C are m x n matrices consisting of single precision
     * complex elements.
     *
     * Input
     * -----
     * side   specifies whether the hermitian matrix A appears on the left side
     *        hand side or right hand side of matrix B, as follows. If side == 'L'
     *        or 'l', then C = alpha * A * B + beta * C. If side = 'R' or 'r',
     *        then C = alpha * B * A + beta * C.
     * uplo   specifies whether the hermitian matrix A is stored in upper or lower
     *        storage mode, as follows. If uplo == 'U' or 'u', only the upper
     *        triangular part of the hermitian matrix is to be referenced, and the
     *        elements of the strictly lower triangular part are to be infered from
     *        those in the upper triangular part. If uplo == 'L' or 'l', only the
     *        lower triangular part of the hermitian matrix is to be referenced,
     *        and the elements of the strictly upper triangular part are to be
     *        infered from those in the lower triangular part.
     * m      specifies the number of rows of the matrix C, and the number of rows
     *        of matrix B. It also specifies the dimensions of hermitian matrix A
     *        when side == 'L' or 'l'. m must be at least zero.
     * n      specifies the number of columns of the matrix C, and the number of
     *        columns of matrix B. It also specifies the dimensions of hermitian
     *        matrix A when side == 'R' or 'r'. n must be at least zero.
     * alpha  single precision complex scalar multiplier applied to A * B, or B * A
     * A      single precision complex array of dimensions (lda, ka), where ka is m when
     *        side == 'L' or 'l' and is n otherwise. If side == 'L' or 'l' the
     *        leading m x m part of array A must contain the hermitian matrix,
     *        such that when uplo == 'U' or 'u', the leading m x m part stores the
     *        upper triangular part of the hermitian matrix, and the strictly lower
     *        triangular part of A is not referenced, and when uplo == 'U' or 'u',
     *        the leading m x m part stores the lower triangular part of the
     *        hermitian matrix and the strictly upper triangular part is not
     *        referenced. If side == 'R' or 'r' the leading n x n part of array A
     *        must contain the hermitian matrix, such that when uplo == 'U' or 'u',
     *        the leading n x n part stores the upper triangular part of the
     *        hermitian matrix and the strictly lower triangular part of A is not
     *        referenced, and when uplo == 'U' or 'u', the leading n x n part
     *        stores the lower triangular part of the hermitian matrix and the
     *        strictly upper triangular part is not referenced. The imaginary parts
     *        of the diagonal elements need not be set, they are assumed to be zero.
     * lda    leading dimension of A. When side == 'L' or 'l', it must be at least
     *        max(1, m) and at least max(1, n) otherwise.
     * B      single precision complex array of dimensions (ldb, n). On entry, the leading
     *        m x n part of the array contains the matrix B.
     * ldb    leading dimension of B. It must be at least max (1, m).
     * beta   single precision complex scalar multiplier applied to C. If beta is zero, C
     *        does not have to be a valid input
     * C      single precision complex array of dimensions (ldc, n)
     * ldc    leading dimension of C. Must be at least max(1, m)
     *
     * Output
     * ------
     * C      updated according to C = alpha * A * B + beta * C, or C = alpha *
     *        B * A + beta * C
     *
     * Reference: http://www.netlib.org/blas/chemm.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if m or n are < 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasChemm(char side, char uplo, int m, int n, cuComplex alpha, Pointer A, int lda, Pointer B, int ldb, cuComplex beta, Pointer C, int ldc)
    {
        cublasChemmNative(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
        checkResultBLAS();
    }
    private static native void cublasChemmNative(char side, char uplo, int m, int n, cuComplex alpha, Pointer A, int lda, Pointer B, int ldb, cuComplex beta, Pointer C, int ldc);





    /**
     * <pre>
     * void
     * cublasCsyrk (char uplo, char trans, int n, int k, cuComplex alpha,
     *              const cuComplex *A, int lda, cuComplex beta, cuComplex *C, int ldc)
     *
     * performs one of the symmetric rank k operations
     *
     *   C = alpha * A * transpose(A) + beta * C, or
     *   C = alpha * transpose(A) * A + beta * C.
     *
     * Alpha and beta are single precision complex scalars. C is an n x n symmetric matrix
     * consisting of single precision complex elements and stored in either lower or
     * upper storage mode. A is a matrix consisting of single precision complex elements
     * with dimension of n x k in the first case, and k x n in the second case.
     *
     * Input
     * -----
     * uplo   specifies whether the symmetric matrix C is stored in upper or lower
     *        storage mode as follows. If uplo == 'U' or 'u', only the upper
     *        triangular part of the symmetric matrix is to be referenced, and the
     *        elements of the strictly lower triangular part are to be infered from
     *        those in the upper triangular part. If uplo == 'L' or 'l', only the
     *        lower triangular part of the symmetric matrix is to be referenced,
     *        and the elements of the strictly upper triangular part are to be
     *        infered from those in the lower triangular part.
     * trans  specifies the operation to be performed. If trans == 'N' or 'n', C =
     *        alpha * transpose(A) + beta * C. If trans == 'T', 't', 'C', or 'c',
     *        C = transpose(A) * A + beta * C.
     * n      specifies the number of rows and the number columns of matrix C. If
     *        trans == 'N' or 'n', n specifies the number of rows of matrix A. If
     *        trans == 'T', 't', 'C', or 'c', n specifies the columns of matrix A.
     *        n must be at least zero.
     * k      If trans == 'N' or 'n', k specifies the number of rows of matrix A.
     *        If trans == 'T', 't', 'C', or 'c', k specifies the number of rows of
     *        matrix A. k must be at least zero.
     * alpha  single precision complex scalar multiplier applied to A * transpose(A) or
     *        transpose(A) * A.
     * A      single precision complex array of dimensions (lda, ka), where ka is k when
     *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n',
     *        the leading n x k part of array A must contain the matrix A,
     *        otherwise the leading k x n part of the array must contains the
     *        matrix A.
     * lda    leading dimension of A. When trans == 'N' or 'n' then lda must be at
     *        least max(1, n). Otherwise lda must be at least max(1, k).
     * beta   single precision complex scalar multiplier applied to C. If beta izs zero, C
     *        does not have to be a valid input
     * C      single precision complex array of dimensions (ldc, n). If uplo = 'U' or 'u',
     *        the leading n x n triangular part of the array C must contain the
     *        upper triangular part of the symmetric matrix C and the strictly
     *        lower triangular part of C is not referenced. On exit, the upper
     *        triangular part of C is overwritten by the upper triangular part of
     *        the updated matrix. If uplo = 'L' or 'l', the leading n x n
     *        triangular part of the array C must contain the lower triangular part
     *        of the symmetric matrix C and the strictly upper triangular part of C
     *        is not referenced. On exit, the lower triangular part of C is
     *        overwritten by the lower triangular part of the updated matrix.
     * ldc    leading dimension of C. It must be at least max(1, n).
     *
     * Output
     * ------
     * C      updated according to C = alpha * A * transpose(A) + beta * C, or C =
     *        alpha * transpose(A) * A + beta * C
     *
     * Reference: http://www.netlib.org/blas/csyrk.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0 or k < 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasCsyrk(char uplo, char trans, int n, int k, cuComplex alpha, Pointer A, int lda, cuComplex beta, Pointer C, int ldc)
    {
        cublasCsyrkNative(uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
        checkResultBLAS();
    }
    private static native void cublasCsyrkNative(char uplo, char trans, int n, int k, cuComplex alpha, Pointer A, int lda, cuComplex beta, Pointer C, int ldc);





    /**
     * <pre>
     * void
     * cublasCherk (char uplo, char trans, int n, int k, float alpha,
     *              const cuComplex *A, int lda, float beta, cuComplex *C, int ldc)
     *
     * performs one of the hermitian rank k operations
     *
     *   C = alpha * A * conjugate(transpose(A)) + beta * C, or
     *   C = alpha * conjugate(transpose(A)) * A + beta * C.
     *
     * Alpha and beta are single precision real scalars. C is an n x n hermitian matrix
     * consisting of single precision complex elements and stored in either lower or
     * upper storage mode. A is a matrix consisting of single precision complex elements
     * with dimension of n x k in the first case, and k x n in the second case.
     *
     * Input
     * -----
     * uplo   specifies whether the hermitian matrix C is stored in upper or lower
     *        storage mode as follows. If uplo == 'U' or 'u', only the upper
     *        triangular part of the hermitian matrix is to be referenced, and the
     *        elements of the strictly lower triangular part are to be infered from
     *        those in the upper triangular part. If uplo == 'L' or 'l', only the
     *        lower triangular part of the hermitian matrix is to be referenced,
     *        and the elements of the strictly upper triangular part are to be
     *        infered from those in the lower triangular part.
     * trans  specifies the operation to be performed. If trans == 'N' or 'n', C =
     *        alpha * A * conjugate(transpose(A)) + beta * C. If trans == 'T', 't', 'C', or 'c',
     *        C = alpha * conjugate(transpose(A)) * A + beta * C.
     * n      specifies the number of rows and the number columns of matrix C. If
     *        trans == 'N' or 'n', n specifies the number of rows of matrix A. If
     *        trans == 'T', 't', 'C', or 'c', n specifies the columns of matrix A.
     *        n must be at least zero.
     * k      If trans == 'N' or 'n', k specifies the number of columns of matrix A.
     *        If trans == 'T', 't', 'C', or 'c', k specifies the number of rows of
     *        matrix A. k must be at least zero.
     * alpha  single precision scalar multiplier applied to A * conjugate(transpose(A)) or
     *        conjugate(transpose(A)) * A.
     * A      single precision complex array of dimensions (lda, ka), where ka is k when
     *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n',
     *        the leading n x k part of array A must contain the matrix A,
     *        otherwise the leading k x n part of the array must contains the
     *        matrix A.
     * lda    leading dimension of A. When trans == 'N' or 'n' then lda must be at
     *        least max(1, n). Otherwise lda must be at least max(1, k).
     * beta   single precision scalar multiplier applied to C. If beta is zero, C
     *        does not have to be a valid input.
     * C      single precision complex array of dimensions (ldc, n). If uplo = 'U' or 'u',
     *        the leading n x n triangular part of the array C must contain the
     *        upper triangular part of the hermitian matrix C and the strictly
     *        lower triangular part of C is not referenced. On exit, the upper
     *        triangular part of C is overwritten by the upper triangular part of
     *        the updated matrix. If uplo = 'L' or 'l', the leading n x n
     *        triangular part of the array C must contain the lower triangular part
     *        of the hermitian matrix C and the strictly upper triangular part of C
     *        is not referenced. On exit, the lower triangular part of C is
     *        overwritten by the lower triangular part of the updated matrix.
     *        The imaginary parts of the diagonal elements need
     *        not be set,  they are assumed to be zero,  and on exit they
     *        are set to zero.
     * ldc    leading dimension of C. It must be at least max(1, n).
     *
     * Output
     * ------
     * C      updated according to C = alpha * A * conjugate(transpose(A)) + beta * C, or C =
     *        alpha * conjugate(transpose(A)) * A + beta * C
     *
     * Reference: http://www.netlib.org/blas/cherk.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0 or k < 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasCherk(char uplo, char trans, int n, int k, float alpha, Pointer A, int lda, float beta, Pointer C, int ldc)
    {
        cublasCherkNative(uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
        checkResultBLAS();
    }
    private static native void cublasCherkNative(char uplo, char trans, int n, int k, float alpha, Pointer A, int lda, float beta, Pointer C, int ldc);





    /**
     * <pre>
     * void
     * cublasCsyr2k (char uplo, char trans, int n, int k, cuComplex alpha,
     *               const cuComplex *A, int lda, const cuComplex *B, int ldb,
     *               cuComplex beta, cuComplex *C, int ldc)
     *
     * performs one of the symmetric rank 2k operations
     *
     *    C = alpha * A * transpose(B) + alpha * B * transpose(A) + beta * C, or
     *    C = alpha * transpose(A) * B + alpha * transpose(B) * A + beta * C.
     *
     * Alpha and beta are single precision complex scalars. C is an n x n symmetric matrix
     * consisting of single precision complex elements and stored in either lower or upper
     * storage mode. A and B are matrices consisting of single precision complex elements
     * with dimension of n x k in the first case, and k x n in the second case.
     *
     * Input
     * -----
     * uplo   specifies whether the symmetric matrix C is stored in upper or lower
     *        storage mode, as follows. If uplo == 'U' or 'u', only the upper
     *        triangular part of the symmetric matrix is to be referenced, and the
     *        elements of the strictly lower triangular part are to be infered from
     *        those in the upper triangular part. If uplo == 'L' or 'l', only the
     *        lower triangular part of the symmetric matrix is to be references,
     *        and the elements of the strictly upper triangular part are to be
     *        infered from those in the lower triangular part.
     * trans  specifies the operation to be performed. If trans == 'N' or 'n',
     *        C = alpha * A * transpose(B) + alpha * B * transpose(A) + beta * C,
     *        If trans == 'T', 't', 'C', or 'c', C = alpha * transpose(A) * B +
     *        alpha * transpose(B) * A + beta * C.
     * n      specifies the number of rows and the number columns of matrix C. If
     *        trans == 'N' or 'n', n specifies the number of rows of matrix A. If
     *        trans == 'T', 't', 'C', or 'c', n specifies the columns of matrix A.
     *        n must be at least zero.
     * k      If trans == 'N' or 'n', k specifies the number of rows of matrix A.
     *        If trans == 'T', 't', 'C', or 'c', k specifies the number of rows of
     *        matrix A. k must be at least zero.
     * alpha  single precision complex scalar multiplier.
     * A      single precision complex array of dimensions (lda, ka), where ka is k when
     *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n',
     *        the leading n x k part of array A must contain the matrix A,
     *        otherwise the leading k x n part of the array must contain the matrix
     *        A.
     * lda    leading dimension of A. When trans == 'N' or 'n' then lda must be at
     *        least max(1, n). Otherwise lda must be at least max(1,k).
     * B      single precision complex array of dimensions (lda, kb), where kb is k when
     *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n',
     *        the leading n x k part of array B must contain the matrix B,
     *        otherwise the leading k x n part of the array must contain the matrix
     *        B.
     * ldb    leading dimension of N. When trans == 'N' or 'n' then ldb must be at
     *        least max(1, n). Otherwise ldb must be at least max(1, k).
     * beta   single precision complex scalar multiplier applied to C. If beta is zero, C
     *        does not have to be a valid input.
     * C      single precision complex array of dimensions (ldc, n). If uplo == 'U' or 'u',
     *        the leading n x n triangular part of the array C must contain the
     *        upper triangular part of the symmetric matrix C and the strictly
     *        lower triangular part of C is not referenced. On exit, the upper
     *        triangular part of C is overwritten by the upper triangular part of
     *        the updated matrix. If uplo == 'L' or 'l', the leading n x n
     *        triangular part of the array C must contain the lower triangular part
     *        of the symmetric matrix C and the strictly upper triangular part of C
     *        is not referenced. On exit, the lower triangular part of C is
     *        overwritten by the lower triangular part of the updated matrix.
     * ldc    leading dimension of C. Must be at least max(1, n).
     *
     * Output
     * ------
     * C      updated according to alpha*A*transpose(B) + alpha*B*transpose(A) +
     *        beta*C or alpha*transpose(A)*B + alpha*transpose(B)*A + beta*C
     *
     * Reference:   http://www.netlib.org/blas/csyr2k.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0 or k < 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasCsyr2k(char uplo, char trans, int n, int k, cuComplex alpha, Pointer A, int lda, Pointer B, int ldb, cuComplex beta, Pointer C, int ldc)
    {
        cublasCsyr2kNative(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        checkResultBLAS();
    }
    private static native void cublasCsyr2kNative(char uplo, char trans, int n, int k, cuComplex alpha, Pointer A, int lda, Pointer B, int ldb, cuComplex beta, Pointer C, int ldc);





    /**
     * <pre>
     * void
     * cublasCher2k (char uplo, char trans, int n, int k, cuComplex alpha,
     *               const cuComplex *A, int lda, const cuComplex *B, int ldb,
     *               float beta, cuComplex *C, int ldc)
     *
     * performs one of the hermitian rank 2k operations
     *
     *    C =   alpha * A * conjugate(transpose(B))
     *        + conjugate(alpha) * B * conjugate(transpose(A))
     *        + beta * C ,
     *    or
     *    C =  alpha * conjugate(transpose(A)) * B
     *       + conjugate(alpha) * conjugate(transpose(B)) * A
     *       + beta * C.
     *
     * Alpha is single precision complex scalar whereas Beta is a single preocision real scalar.
     * C is an n x n hermitian matrix consisting of single precision complex elements
     * and stored in either lower or upper storage mode. A and B are matrices consisting
     * of single precision complex elements with dimension of n x k in the first case,
     * and k x n in the second case.
     *
     * Input
     * -----
     * uplo   specifies whether the hermitian matrix C is stored in upper or lower
     *        storage mode, as follows. If uplo == 'U' or 'u', only the upper
     *        triangular part of the hermitian matrix is to be referenced, and the
     *        elements of the strictly lower triangular part are to be infered from
     *        those in the upper triangular part. If uplo == 'L' or 'l', only the
     *        lower triangular part of the hermitian matrix is to be references,
     *        and the elements of the strictly upper triangular part are to be
     *        infered from those in the lower triangular part.
     * trans  specifies the operation to be performed. If trans == 'N' or 'n',
     *        C =   alpha * A * conjugate(transpose(B))
     *            + conjugate(alpha) * B * conjugate(transpose(A))
     *            + beta * C .
     *        If trans == 'T', 't', 'C', or 'c',
     *        C =  alpha * conjugate(transpose(A)) * B
     *          + conjugate(alpha) * conjugate(transpose(B)) * A
     *          + beta * C.
     * n      specifies the number of rows and the number columns of matrix C. If
     *        trans == 'N' or 'n', n specifies the number of rows of matrix A. If
     *        trans == 'T', 't', 'C', or 'c', n specifies the columns of matrix A.
     *        n must be at least zero.
     * k      If trans == 'N' or 'n', k specifies the number of rows of matrix A.
     *        If trans == 'T', 't', 'C', or 'c', k specifies the number of rows of
     *        matrix A. k must be at least zero.
     * alpha  single precision complex scalar multiplier.
     * A      single precision complex array of dimensions (lda, ka), where ka is k when
     *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n',
     *        the leading n x k part of array A must contain the matrix A,
     *        otherwise the leading k x n part of the array must contain the matrix
     *        A.
     * lda    leading dimension of A. When trans == 'N' or 'n' then lda must be at
     *        least max(1, n). Otherwise lda must be at least max(1,k).
     * B      single precision complex array of dimensions (lda, kb), where kb is k when
     *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n',
     *        the leading n x k part of array B must contain the matrix B,
     *        otherwise the leading k x n part of the array must contain the matrix
     *        B.
     * ldb    leading dimension of N. When trans == 'N' or 'n' then ldb must be at
     *        least max(1, n). Otherwise ldb must be at least max(1, k).
     * beta   single precision scalar multiplier applied to C. If beta is zero, C
     *        does not have to be a valid input.
     * C      single precision complex array of dimensions (ldc, n). If uplo == 'U' or 'u',
     *        the leading n x n triangular part of the array C must contain the
     *        upper triangular part of the hermitian matrix C and the strictly
     *        lower triangular part of C is not referenced. On exit, the upper
     *        triangular part of C is overwritten by the upper triangular part of
     *        the updated matrix. If uplo == 'L' or 'l', the leading n x n
     *        triangular part of the array C must contain the lower triangular part
     *        of the hermitian matrix C and the strictly upper triangular part of C
     *        is not referenced. On exit, the lower triangular part of C is
     *        overwritten by the lower triangular part of the updated matrix.
     *        The imaginary parts of the diagonal elements need
     *        not be set,  they are assumed to be zero,  and on exit they
     *        are set to zero.
     * ldc    leading dimension of C. Must be at least max(1, n).
     *
     * Output
     * ------
     * C      updated according to alpha*A*conjugate(transpose(B)) +
     *        + conjugate(alpha)*B*conjugate(transpose(A)) + beta*C or
     *        alpha*conjugate(transpose(A))*B + conjugate(alpha)*conjugate(transpose(B))*A
     *        + beta*C.
     *
     * Reference:   http://www.netlib.org/blas/cher2k.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0 or k < 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasCher2k(char uplo, char trans, int n, int k, cuComplex alpha, Pointer A, int lda, Pointer B, int ldb, float beta, Pointer C, int ldc)
    {
        cublasCher2kNative(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        checkResultBLAS();
    }
    private static native void cublasCher2kNative(char uplo, char trans, int n, int k, cuComplex alpha, Pointer A, int lda, Pointer B, int ldb, float beta, Pointer C, int ldc);





    /**
     * <pre>
     * void
     * cublasCtrmm (char side, char uplo, char transa, char diag, int m, int n,
     *              cuComplex alpha, const cuComplex *A, int lda, const cuComplex *B,
     *              int ldb)
     *
     * performs one of the matrix-matrix operations
     *
     *   B = alpha * op(A) * B,  or  B = alpha * B * op(A)
     *
     * where alpha is a single-precision complex scalar, B is an m x n matrix composed
     * of single precision complex elements, and A is a unit or non-unit, upper or lower,
     * triangular matrix composed of single precision complex elements. op(A) is one of
     *
     *   op(A) = A  , op(A) = transpose(A) or op(A) = conjugate(transpose(A))
     *
     * Matrices A and B are stored in column major format, and lda and ldb are
     * the leading dimensions of the two-dimensonials arrays that contain A and
     * B, respectively.
     *
     * Input
     * -----
     * side   specifies whether op(A) multiplies B from the left or right.
     *        If side = 'L' or 'l', then B = alpha * op(A) * B. If side =
     *        'R' or 'r', then B = alpha * B * op(A).
     * uplo   specifies whether the matrix A is an upper or lower triangular
     *        matrix. If uplo = 'U' or 'u', A is an upper triangular matrix.
     *        If uplo = 'L' or 'l', A is a lower triangular matrix.
     * transa specifies the form of op(A) to be used in the matrix
     *        multiplication. If transa = 'N' or 'n', then op(A) = A. If
     *        transa = 'T' or 't', then op(A) = transpose(A).
     *        If transa = 'C' or 'c', then op(A) = conjugate(transpose(A)).
     * diag   specifies whether or not A is unit triangular. If diag = 'U'
     *        or 'u', A is assumed to be unit triangular. If diag = 'N' or
     *        'n', A is not assumed to be unit triangular.
     * m      the number of rows of matrix B. m must be at least zero.
     * n      the number of columns of matrix B. n must be at least zero.
     * alpha  single precision complex scalar multiplier applied to op(A)*B, or
     *        B*op(A), respectively. If alpha is zero no accesses are made
     *        to matrix A, and no read accesses are made to matrix B.
     * A      single precision complex array of dimensions (lda, k). k = m if side =
     *        'L' or 'l', k = n if side = 'R' or 'r'. If uplo = 'U' or 'u'
     *        the leading k x k upper triangular part of the array A must
     *        contain the upper triangular matrix, and the strictly lower
     *        triangular part of A is not referenced. If uplo = 'L' or 'l'
     *        the leading k x k lower triangular part of the array A must
     *        contain the lower triangular matrix, and the strictly upper
     *        triangular part of A is not referenced. When diag = 'U' or 'u'
     *        the diagonal elements of A are no referenced and are assumed
     *        to be unity.
     * lda    leading dimension of A. When side = 'L' or 'l', it must be at
     *        least max(1,m) and at least max(1,n) otherwise
     * B      single precision complex array of dimensions (ldb, n). On entry, the
     *        leading m x n part of the array contains the matrix B. It is
     *        overwritten with the transformed matrix on exit.
     * ldb    leading dimension of B. It must be at least max (1, m).
     *
     * Output
     * ------
     * B      updated according to B = alpha * op(A) * B  or B = alpha * B * op(A)
     *
     * Reference: http://www.netlib.org/blas/ctrmm.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if m or n < 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasCtrmm(char side, char uplo, char transa, char diag, int m, int n, cuComplex alpha, Pointer A, int lda, Pointer B, int ldb)
    {
        cublasCtrmmNative(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
        checkResultBLAS();
    }
    private static native void cublasCtrmmNative(char side, char uplo, char transa, char diag, int m, int n, cuComplex alpha, Pointer A, int lda, Pointer B, int ldb);





    /**
     * <pre>
     * void
     * cublasCtrsm (char side, char uplo, char transa, char diag, int m, int n,
     *              cuComplex alpha, const cuComplex *A, int lda,
     *              cuComplex *B, int ldb)
     *
     * solves one of the matrix equations
     *
     *    op(A) * X = alpha * B,   or   X * op(A) = alpha * B,
     *
     * where alpha is a single precision complex scalar, and X and B are m x n matrices
     * that are composed of single precision complex elements. A is a unit or non-unit,
     * upper or lower triangular matrix, and op(A) is one of
     *
     *    op(A) = A  or  op(A) = transpose(A)  or  op( A ) = conj( A' ).
     *
     * The result matrix X overwrites input matrix B; that is, on exit the result
     * is stored in B. Matrices A and B are stored in column major format, and
     * lda and ldb are the leading dimensions of the two-dimensonials arrays that
     * contain A and B, respectively.
     *
     * Input
     * -----
     * side   specifies whether op(A) appears on the left or right of X as
     *        follows: side = 'L' or 'l' indicates solve op(A) * X = alpha * B.
     *        side = 'R' or 'r' indicates solve X * op(A) = alpha * B.
     * uplo   specifies whether the matrix A is an upper or lower triangular
     *        matrix as follows: uplo = 'U' or 'u' indicates A is an upper
     *        triangular matrix. uplo = 'L' or 'l' indicates A is a lower
     *        triangular matrix.
     * transa specifies the form of op(A) to be used in matrix multiplication
     *        as follows: If transa = 'N' or 'N', then op(A) = A. If transa =
     *        'T', 't', 'C', or 'c', then op(A) = transpose(A).
     * diag   specifies whether or not A is a unit triangular matrix like so:
     *        if diag = 'U' or 'u', A is assumed to be unit triangular. If
     *        diag = 'N' or 'n', then A is not assumed to be unit triangular.
     * m      specifies the number of rows of B. m must be at least zero.
     * n      specifies the number of columns of B. n must be at least zero.
     * alpha  is a single precision complex scalar to be multiplied with B. When alpha is
     *        zero, then A is not referenced and B need not be set before entry.
     * A      is a single precision complex array of dimensions (lda, k), where k is
     *        m when side = 'L' or 'l', and is n when side = 'R' or 'r'. If
     *        uplo = 'U' or 'u', the leading k x k upper triangular part of
     *        the array A must contain the upper triangular matrix and the
     *        strictly lower triangular matrix of A is not referenced. When
     *        uplo = 'L' or 'l', the leading k x k lower triangular part of
     *        the array A must contain the lower triangular matrix and the
     *        strictly upper triangular part of A is not referenced. Note that
     *        when diag = 'U' or 'u', the diagonal elements of A are not
     *        referenced, and are assumed to be unity.
     * lda    is the leading dimension of the two dimensional array containing A.
     *        When side = 'L' or 'l' then lda must be at least max(1, m), when
     *        side = 'R' or 'r' then lda must be at least max(1, n).
     * B      is a single precision complex array of dimensions (ldb, n). ldb must be
     *        at least max (1,m). The leading m x n part of the array B must
     *        contain the right-hand side matrix B. On exit B is overwritten
     *        by the solution matrix X.
     * ldb    is the leading dimension of the two dimensional array containing B.
     *        ldb must be at least max(1, m).
     *
     * Output
     * ------
     * B      contains the solution matrix X satisfying op(A) * X = alpha * B,
     *        or X * op(A) = alpha * B
     *
     * Reference: http://www.netlib.org/blas/ctrsm.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if m or n < 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasCtrsm(char side, char uplo, char transa, char diag, int m, int n, cuComplex alpha, Pointer A, int lda, Pointer B, int ldb)
    {
        cublasCtrsmNative(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
        checkResultBLAS();
    }
    private static native void cublasCtrsmNative(char side, char uplo, char transa, char diag, int m, int n, cuComplex alpha, Pointer A, int lda, Pointer B, int ldb);





    /**
     * <pre>
     * double
     * cublasDasum (int n, const double *x, int incx)
     *
     * computes the sum of the absolute values of the elements of double
     * precision vector x; that is, the result is the sum from i = 0 to n - 1 of
     * abs(x[1 + i * incx]).
     *
     * Input
     * -----
     * n      number of elements in input vector
     * x      double-precision vector with n elements
     * incx   storage spacing between elements of x
     *
     * Output
     * ------
     * returns the double-precision sum of absolute values
     * (0 if n <= 0 or incx <= 0, or if an error occurs)
     *
     * Reference: http://www.netlib.org/blas/dasum.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static double cublasDasum(int n, Pointer x, int incx)
    {
        double result = cublasDasumNative(n, x, incx);
        checkResultBLAS();
        return result;
    }
    private static native double cublasDasumNative(int n, Pointer x, int incx);





    /**
     * <pre>
     * void
     * cublasDaxpy (int n, double alpha, const double *x, int incx, double *y,
     *              int incy)
     *
     * multiplies double-precision vector x by double-precision scalar alpha
     * and adds the result to double-precision vector y; that is, it overwrites
     * double-precision y with double-precision alpha * x + y. For i = 0 to n-1,
     * it replaces y[ly + i * incy] with alpha * x[lx + i * incx] + y[ly + i*incy],
     * where lx = 1 if incx >= 0, else lx = 1 + (1 - n) * incx; ly is defined in a
     * similar way using incy.
     *
     * Input
     * -----
     * n      number of elements in input vectors
     * alpha  double-precision scalar multiplier
     * x      double-precision vector with n elements
     * incx   storage spacing between elements of x
     * y      double-precision vector with n elements
     * incy   storage spacing between elements of y
     *
     * Output
     * ------
     * y      double-precision result (unchanged if n <= 0)
     *
     * Reference: http://www.netlib.org/blas/daxpy.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library was not initialized
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasDaxpy(int n, double alpha, Pointer x, int incx, Pointer y, int incy)
    {
        cublasDaxpyNative(n, alpha, x, incx, y, incy);
        checkResultBLAS();
    }
    private static native void cublasDaxpyNative(int n, double alpha, Pointer x, int incx, Pointer y, int incy);





    /**
     * <pre>
     * void
     * cublasDcopy (int n, const double *x, int incx, double *y, int incy)
     *
     * copies the double-precision vector x to the double-precision vector y. For
     * i = 0 to n-1, copies x[lx + i * incx] to y[ly + i * incy], where lx = 1 if
     * incx >= 0, else lx = 1 + (1 - n) * incx, and ly is defined in a similar
     * way using incy.
     *
     * Input
     * -----
     * n      number of elements in input vectors
     * x      double-precision vector with n elements
     * incx   storage spacing between elements of x
     * y      double-precision vector with n elements
     * incy   storage spacing between elements of y
     *
     * Output
     * ------
     * y      contains double precision vector x
     *
     * Reference: http://www.netlib.org/blas/dcopy.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasDcopy(int n, Pointer x, int incx, Pointer y, int incy)
    {
        cublasDcopyNative(n, x, incx, y, incy);
        checkResultBLAS();
    }
    private static native void cublasDcopyNative(int n, Pointer x, int incx, Pointer y, int incy);





    /**
     * <pre>
     * double
     * cublasDdot (int n, const double *x, int incx, const double *y, int incy)
     *
     * computes the dot product of two double-precision vectors. It returns the
     * dot product of the double precision vectors x and y if successful, and
     * 0.0f otherwise. It computes the sum for i = 0 to n - 1 of x[lx + i *
     * incx] * y[ly + i * incy], where lx = 1 if incx >= 0, else lx = 1 + (1 - n)
     * *incx, and ly is defined in a similar way using incy.
     *
     * Input
     * -----
     * n      number of elements in input vectors
     * x      double-precision vector with n elements
     * incx   storage spacing between elements of x
     * y      double-precision vector with n elements
     * incy   storage spacing between elements of y
     *
     * Output
     * ------
     * returns double-precision dot product (zero if n <= 0)
     *
     * Reference: http://www.netlib.org/blas/ddot.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has nor been initialized
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to execute on GPU
     * </pre>
     */

    public static double cublasDdot(int n, Pointer x, int incx, Pointer y, int incy)
    {
        double result = cublasDdotNative(n, x, incx, y, incy);
        checkResultBLAS();
        return result;
    }
    private static native double cublasDdotNative(int n, Pointer x, int incx, Pointer y, int incy);





    /**
     * <pre>
     * double
     * dnrm2 (int n, const double *x, int incx)
     *
     * computes the Euclidean norm of the double-precision n-vector x (with
     * storage increment incx). This code uses a multiphase model of
     * accumulation to avoid intermediate underflow and overflow.
     *
     * Input
     * -----
     * n      number of elements in input vector
     * x      double-precision vector with n elements
     * incx   storage spacing between elements of x
     *
     * Output
     * ------
     * returns Euclidian norm (0 if n <= 0 or incx <= 0, or if an error occurs)
     *
     * Reference: http://www.netlib.org/blas/dnrm2.f
     * Reference: http://www.netlib.org/slatec/lin/dnrm2.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static double cublasDnrm2(int n, Pointer x, int incx)
    {
        double result = cublasDnrm2Native(n, x, incx);
        checkResultBLAS();
        return result;
    }
    private static native double cublasDnrm2Native(int n, Pointer x, int incx);





    /**
     * <pre>
     * void
     * cublasDrot (int n, double *x, int incx, double *y, int incy, double sc,
     *             double ss)
     *
     * multiplies a 2x2 matrix ( sc ss) with the 2xn matrix ( transpose(x) )
     *                         (-ss sc)                     ( transpose(y) )
     *
     * The elements of x are in x[lx + i * incx], i = 0 ... n - 1, where lx = 1 if
     * incx >= 0, else lx = 1 + (1 - n) * incx, and similarly for y using ly and
     * incy.
     *
     * Input
     * -----
     * n      number of elements in input vectors
     * x      double-precision vector with n elements
     * incx   storage spacing between elements of x
     * y      double-precision vector with n elements
     * incy   storage spacing between elements of y
     * sc     element of rotation matrix
     * ss     element of rotation matrix
     *
     * Output
     * ------
     * x      rotated vector x (unchanged if n <= 0)
     * y      rotated vector y (unchanged if n <= 0)
     *
     * Reference  http://www.netlib.org/blas/drot.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasDrot(int n, Pointer x, int incx, Pointer y, int incy, double sc, double ss)
    {
        cublasDrotNative(n, x, incx, y, incy, sc, ss);
        checkResultBLAS();
    }
    private static native void cublasDrotNative(int n, Pointer x, int incx, Pointer y, int incy, double sc, double ss);





    /**
     * <pre>
     * void
     * cublasDrotg (double *host_sa, double *host_sb, double *host_sc, double *host_ss)
     *
     * constructs the Givens tranformation
     *
     *        ( sc  ss )
     *    G = (        ) ,  sc^2 + ss^2 = 1,
     *        (-ss  sc )
     *
     * which zeros the second entry of the 2-vector transpose(sa, sb).
     *
     * The quantity r = (+/-) sqrt (sa^2 + sb^2) overwrites sa in storage. The
     * value of sb is overwritten by a value z which allows sc and ss to be
     * recovered by the following algorithm:
     *
     *    if z=1          set sc = 0.0 and ss = 1.0
     *    if abs(z) < 1   set sc = sqrt(1-z^2) and ss = z
     *    if abs(z) > 1   set sc = 1/z and ss = sqrt(1-sc^2)
     *
     * The function drot (n, x, incx, y, incy, sc, ss) normally is called next
     * to apply the transformation to a 2 x n matrix.
     * Note that is function is provided for completeness and run exclusively
     * on the Host.
     *
     * Input
     * -----
     * sa     double-precision scalar
     * sb     double-precision scalar
     *
     * Output
     * ------
     * sa     double-precision r
     * sb     double-precision z
     * sc     double-precision result
     * ss     double-precision result
     *
     * Reference: http://www.netlib.org/blas/drotg.f
     *
     * This function does not set any error status.
     * </pre>
     */

    public static void cublasDrotg(Pointer host_sa, Pointer host_sb, Pointer host_sc, Pointer host_ss)
    {
        cublasDrotgNative(host_sa, host_sb, host_sc, host_ss);
        checkResultBLAS();
    }
    private static native void cublasDrotgNative(Pointer host_sa, Pointer host_sb, Pointer host_sc, Pointer host_ss);





    /**
     * <pre>
     * void
     * cublasDscal (int n, double alpha, double *x, int incx)
     *
     * replaces double-precision vector x with double-precision alpha * x. For
     * i = 0 to n-1, it replaces x[lx + i * incx] with alpha * x[lx + i * incx],
     * where lx = 1 if incx >= 0, else lx = 1 + (1 - n) * incx.
     *
     * Input
     * -----
     * n      number of elements in input vector
     * alpha  double-precision scalar multiplier
     * x      double-precision vector with n elements
     * incx   storage spacing between elements of x
     *
     * Output
     * ------
     * x      double-precision result (unchanged if n <= 0 or incx <= 0)
     *
     * Reference: http://www.netlib.org/blas/dscal.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library was not initialized
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasDscal(int n, double alpha, Pointer x, int incx)
    {
        cublasDscalNative(n, alpha, x, incx);
        checkResultBLAS();
    }
    private static native void cublasDscalNative(int n, double alpha, Pointer x, int incx);





    /**
     * <pre>
     * void
     * cublasDswap (int n, double *x, int incx, double *y, int incy)
     *
     * interchanges the double-precision vector x with the double-precision vector y.
     * For i = 0 to n-1, interchanges x[lx + i * incx] with y[ly + i * incy], where
     * lx = 1 if incx >= 0, else lx = 1 + (1 - n) * incx, and ly is defined in a
     * similar way using incy.
     *
     * Input
     * -----
     * n      number of elements in input vectors
     * x      double precision vector with n elements
     * incx   storage spacing between elements of x
     * y      double precision vector with n elements
     * incy   storage spacing between elements of y
     *
     * Output
     * ------
     * x      contains double precision vector y
     * y      contains double precision vector x
     *
     * Reference: http://www.netlib.org/blas/dswap.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasDswap(int n, Pointer x, int incx, Pointer y, int incy)
    {
        cublasDswapNative(n, x, incx, y, incy);
        checkResultBLAS();
    }
    private static native void cublasDswapNative(int n, Pointer x, int incx, Pointer y, int incy);





    /**
     * <pre>
     * int
     * idamax (int n, const double *x, int incx)
     *
     * finds the smallest index of the maximum magnitude element of double-
     * precision vector x; that is, the result is the first i, i = 0 to n - 1,
     * that maximizes abs(x[1 + i * incx])).
     *
     * Input
     * -----
     * n      number of elements in input vector
     * x      double-precision vector with n elements
     * incx   storage spacing between elements of x
     *
     * Output
     * ------
     * returns the smallest index (0 if n <= 0 or incx <= 0)
     *
     * Reference: http://www.netlib.org/blas/idamax.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static int cublasIdamax(int n, Pointer x, int incx)
    {
        int result = cublasIdamaxNative(n, x, incx);
        checkResultBLAS();
        return result;
    }
    private static native int cublasIdamaxNative(int n, Pointer x, int incx);





    /**
     * <pre>
     * int
     * idamin (int n, const double *x, int incx)
     *
     * finds the smallest index of the minimum magnitude element of double-
     * precision vector x; that is, the result is the first i, i = 0 to n - 1,
     * that minimizes abs(x[1 + i * incx])).
     *
     * Input
     * -----
     * n      number of elements in input vector
     * x      double-precision vector with n elements
     * incx   storage spacing between elements of x
     *
     * Output
     * ------
     * returns the smallest index (0 if n <= 0 or incx <= 0)
     *
     * Reference: http://www.netlib.org/scilib/blass.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static int cublasIdamin(int n, Pointer x, int incx)
    {
        int result = cublasIdaminNative(n, x, incx);
        checkResultBLAS();
        return result;
    }
    private static native int cublasIdaminNative(int n, Pointer x, int incx);





    /**
     * <pre>
     * cublasDgemv (char trans, int m, int n, double alpha, const double *A,
     *              int lda, const double *x, int incx, double beta, double *y,
     *              int incy)
     *
     * performs one of the matrix-vector operations
     *
     *    y = alpha * op(A) * x + beta * y,
     *
     * where op(A) is one of
     *
     *    op(A) = A   or   op(A) = transpose(A)
     *
     * where alpha and beta are double precision scalars, x and y are double
     * precision vectors, and A is an m x n matrix consisting of double precision
     * elements. Matrix A is stored in column major format, and lda is the leading
     * dimension of the two-dimensional array in which A is stored.
     *
     * Input
     * -----
     * trans  specifies op(A). If transa = 'n' or 'N', op(A) = A. If trans =
     *        trans = 't', 'T', 'c', or 'C', op(A) = transpose(A)
     * m      specifies the number of rows of the matrix A. m must be at least
     *        zero.
     * n      specifies the number of columns of the matrix A. n must be at least
     *        zero.
     * alpha  double precision scalar multiplier applied to op(A).
     * A      double precision array of dimensions (lda, n) if trans = 'n' or
     *        'N'), and of dimensions (lda, m) otherwise. lda must be at least
     *        max(1, m) and at least max(1, n) otherwise.
     * lda    leading dimension of two-dimensional array used to store matrix A
     * x      double precision array of length at least (1 + (n - 1) * abs(incx))
     *        when trans = 'N' or 'n' and at least (1 + (m - 1) * abs(incx))
     *        otherwise.
     * incx   specifies the storage spacing between elements of x. incx must not
     *        be zero.
     * beta   double precision scalar multiplier applied to vector y. If beta
     *        is zero, y is not read.
     * y      double precision array of length at least (1 + (m - 1) * abs(incy))
     *        when trans = 'N' or 'n' and at least (1 + (n - 1) * abs(incy))
     *        otherwise.
     * incy   specifies the storage spacing between elements of x. incx must not
     *        be zero.
     *
     * Output
     * ------
     * y      updated according to alpha * op(A) * x + beta * y
     *
     * Reference: http://www.netlib.org/blas/dgemv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if m or n are < 0, or if incx or incy == 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasDgemv(char trans, int m, int n, double alpha, Pointer A, int lda, Pointer x, int incx, double beta, Pointer y, int incy)
    {
        cublasDgemvNative(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
        checkResultBLAS();
    }
    private static native void cublasDgemvNative(char trans, int m, int n, double alpha, Pointer A, int lda, Pointer x, int incx, double beta, Pointer y, int incy);





    /**
     * <pre>
     * cublasDger (int m, int n, double alpha, const double *x, int incx,
     *             const double *y, int incy, double *A, int lda)
     *
     * performs the symmetric rank 1 operation
     *
     *    A = alpha * x * transpose(y) + A,
     *
     * where alpha is a double precision scalar, x is an m element double
     * precision vector, y is an n element double precision vector, and A
     * is an m by n matrix consisting of double precision elements. Matrix A
     * is stored in column major format, and lda is the leading dimension of
     * the two-dimensional array used to store A.
     *
     * Input
     * -----
     * m      specifies the number of rows of the matrix A. It must be at least
     *        zero.
     * n      specifies the number of columns of the matrix A. It must be at
     *        least zero.
     * alpha  double precision scalar multiplier applied to x * transpose(y)
     * x      double precision array of length at least (1 + (m - 1) * abs(incx))
     * incx   specifies the storage spacing between elements of x. incx must not
     *        be zero.
     * y      double precision array of length at least (1 + (n - 1) * abs(incy))
     * incy   specifies the storage spacing between elements of y. incy must not
     *        be zero.
     * A      double precision array of dimensions (lda, n).
     * lda    leading dimension of two-dimensional array used to store matrix A
     *
     * Output
     * ------
     * A      updated according to A = alpha * x * transpose(y) + A
     *
     * Reference: http://www.netlib.org/blas/dger.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0, incx == 0, incy == 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasDger(int m, int n, double alpha, Pointer x, int incx, Pointer y, int incy, Pointer A, int lda)
    {
        cublasDgerNative(m, n, alpha, x, incx, y, incy, A, lda);
        checkResultBLAS();
    }
    private static native void cublasDgerNative(int m, int n, double alpha, Pointer x, int incx, Pointer y, int incy, Pointer A, int lda);





    /**
     * <pre>
     * void
     * cublasDsyr (char uplo, int n, double alpha, const double *x, int incx,
     *             double *A, int lda)
     *
     * performs the symmetric rank 1 operation
     *
     *    A = alpha * x * transpose(x) + A,
     *
     * where alpha is a double precision scalar, x is an n element double
     * precision vector and A is an n x n symmetric matrix consisting of
     * double precision elements. Matrix A is stored in column major format,
     * and lda is the leading dimension of the two-dimensional array
     * containing A.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix data is stored in the upper or
     *        the lower triangular part of array A. If uplo = 'U' or 'u',
     *        then only the upper triangular part of A may be referenced.
     *        If uplo = 'L' or 'l', then only the lower triangular part of
     *        A may be referenced.
     * n      specifies the number of rows and columns of the matrix A. It
     *        must be at least 0.
     * alpha  double precision scalar multiplier applied to x * transpose(x)
     * x      double precision array of length at least (1 + (n - 1) * abs(incx))
     * incx   specifies the storage spacing between elements of x. incx must
     *        not be zero.
     * A      double precision array of dimensions (lda, n). If uplo = 'U' or
     *        'u', then A must contain the upper triangular part of a symmetric
     *        matrix, and the strictly lower triangular part is not referenced.
     *        If uplo = 'L' or 'l', then A contains the lower triangular part
     *        of a symmetric matrix, and the strictly upper triangular part is
     *        not referenced.
     * lda    leading dimension of the two-dimensional array containing A. lda
     *        must be at least max(1, n).
     *
     * Output
     * ------
     * A      updated according to A = alpha * x * transpose(x) + A
     *
     * Reference: http://www.netlib.org/blas/dsyr.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or incx == 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasDsyr(char uplo, int n, double alpha, Pointer x, int incx, Pointer A, int lda)
    {
        cublasDsyrNative(uplo, n, alpha, x, incx, A, lda);
        checkResultBLAS();
    }
    private static native void cublasDsyrNative(char uplo, int n, double alpha, Pointer x, int incx, Pointer A, int lda);





    /**
     * <pre>
     * void cublasDsyr2 (char uplo, int n, double alpha, const double *x, int incx,
     *                   const double *y, int incy, double *A, int lda)
     *
     * performs the symmetric rank 2 operation
     *
     *    A = alpha*x*transpose(y) + alpha*y*transpose(x) + A,
     *
     * where alpha is a double precision scalar, x and y are n element double
     * precision vector and A is an n by n symmetric matrix consisting of double
     * precision elements.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix data is stored in the upper or the lower
     *        triangular part of array A. If uplo == 'U' or 'u', then only the
     *        upper triangular part of A may be referenced and the lower triangular
     *        part of A is inferred. If uplo == 'L' or 'l', then only the lower
     *        triangular part of A may be referenced and the upper triangular part
     *        of A is inferred.
     * n      specifies the number of rows and columns of the matrix A. It must be
     *        at least zero.
     * alpha  double precision scalar multiplier applied to x * transpose(y) +
     *        y * transpose(x).
     * x      double precision array of length at least (1 + (n - 1) * abs (incx)).
     * incx   storage spacing between elements of x. incx must not be zero.
     * y      double precision array of length at least (1 + (n - 1) * abs (incy)).
     * incy   storage spacing between elements of y. incy must not be zero.
     * A      double precision array of dimensions (lda, n). If uplo == 'U' or 'u',
     *        then A must contains the upper triangular part of a symmetric matrix,
     *        and the strictly lower triangular parts is not referenced. If uplo ==
     *        'L' or 'l', then A contains the lower triangular part of a symmetric
     *        matrix, and the strictly upper triangular part is not referenced.
     * lda    leading dimension of A. It must be at least max(1, n).
     *
     * Output
     * ------
     * A      updated according to A = alpha*x*transpose(y)+alpha*y*transpose(x)+A
     *
     * Reference: http://www.netlib.org/blas/dsyr2.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0, incx == 0, incy == 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasDsyr2(char uplo, int n, double alpha, Pointer x, int incx, Pointer y, int incy, Pointer A, int lda)
    {
        cublasDsyr2Native(uplo, n, alpha, x, incx, y, incy, A, lda);
        checkResultBLAS();
    }
    private static native void cublasDsyr2Native(char uplo, int n, double alpha, Pointer x, int incx, Pointer y, int incy, Pointer A, int lda);





    /**
     * <pre>
     * void
     * cublasDspr (char uplo, int n, double alpha, const double *x, int incx,
     *             double *AP)
     *
     * performs the symmetric rank 1 operation
     *
     *    A = alpha * x * transpose(x) + A,
     *
     * where alpha is a double precision scalar and x is an n element double
     * precision vector. A is a symmetric n x n matrix consisting of double
     * precision elements that is supplied in packed form.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix data is stored in the upper or the lower
     *        triangular part of array AP. If uplo == 'U' or 'u', then the upper
     *        triangular part of A is supplied in AP. If uplo == 'L' or 'l', then
     *        the lower triangular part of A is supplied in AP.
     * n      specifies the number of rows and columns of the matrix A. It must be
     *        at least zero.
     * alpha  double precision scalar multiplier applied to x * transpose(x).
     * x      double precision array of length at least (1 + (n - 1) * abs(incx)).
     * incx   storage spacing between elements of x. incx must not be zero.
     * AP     double precision array with at least ((n * (n + 1)) / 2) elements. If
     *        uplo == 'U' or 'u', the array AP contains the upper triangular part
     *        of the symmetric matrix A, packed sequentially, column by column;
     *        that is, if i <= j, then A[i,j] is stored is AP[i+(j*(j+1)/2)]. If
     *        uplo == 'L' or 'L', the array AP contains the lower triangular part
     *        of the symmetric matrix A, packed sequentially, column by column;
     *        that is, if i >= j, then A[i,j] is stored in AP[i+((2*n-j+1)*j)/2].
     *
     * Output
     * ------
     * A      updated according to A = alpha * x * transpose(x) + A
     *
     * Reference: http://www.netlib.org/blas/dspr.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or incx == 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasDspr(char uplo, int n, double alpha, Pointer x, int incx, Pointer AP)
    {
        cublasDsprNative(uplo, n, alpha, x, incx, AP);
        checkResultBLAS();
    }
    private static native void cublasDsprNative(char uplo, int n, double alpha, Pointer x, int incx, Pointer AP);





    /**
     * <pre>
     * void
     * cublasDspr2 (char uplo, int n, double alpha, const double *x, int incx,
     *              const double *y, int incy, double *AP)
     *
     * performs the symmetric rank 2 operation
     *
     *    A = alpha*x*transpose(y) + alpha*y*transpose(x) + A,
     *
     * where alpha is a double precision scalar, and x and y are n element double
     * precision vectors. A is a symmetric n x n matrix consisting of double
     * precision elements that is supplied in packed form.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix data is stored in the upper or the lower
     *        triangular part of array A. If uplo == 'U' or 'u', then only the
     *        upper triangular part of A may be referenced and the lower triangular
     *        part of A is inferred. If uplo == 'L' or 'l', then only the lower
     *        triangular part of A may be referenced and the upper triangular part
     *        of A is inferred.
     * n      specifies the number of rows and columns of the matrix A. It must be
     *        at least zero.
     * alpha  double precision scalar multiplier applied to x * transpose(y) +
     *        y * transpose(x).
     * x      double precision array of length at least (1 + (n - 1) * abs (incx)).
     * incx   storage spacing between elements of x. incx must not be zero.
     * y      double precision array of length at least (1 + (n - 1) * abs (incy)).
     * incy   storage spacing between elements of y. incy must not be zero.
     * AP     double precision array with at least ((n * (n + 1)) / 2) elements. If
     *        uplo == 'U' or 'u', the array AP contains the upper triangular part
     *        of the symmetric matrix A, packed sequentially, column by column;
     *        that is, if i <= j, then A[i,j] is stored is AP[i+(j*(j+1)/2)]. If
     *        uplo == 'L' or 'L', the array AP contains the lower triangular part
     *        of the symmetric matrix A, packed sequentially, column by column;
     *        that is, if i >= j, then A[i,j] is stored in AP[i+((2*n-j+1)*j)/2].
     *
     * Output
     * ------
     * A      updated according to A = alpha*x*transpose(y)+alpha*y*transpose(x)+A
     *
     * Reference: http://www.netlib.org/blas/dspr2.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0, incx == 0, incy == 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasDspr2(char uplo, int n, double alpha, Pointer x, int incx, Pointer y, int incy, Pointer AP)
    {
        cublasDspr2Native(uplo, n, alpha, x, incx, y, incy, AP);
        checkResultBLAS();
    }
    private static native void cublasDspr2Native(char uplo, int n, double alpha, Pointer x, int incx, Pointer y, int incy, Pointer AP);





    /**
     * <pre>
     * void
     * cublasDtrsv (char uplo, char trans, char diag, int n, const double *A,
     *              int lda, double *x, int incx)
     *
     * solves a system of equations op(A) * x = b, where op(A) is either A or
     * transpose(A). b and x are double precision vectors consisting of n
     * elements, and A is an n x n matrix composed of a unit or non-unit, upper
     * or lower triangular matrix. Matrix A is stored in column major format,
     * and lda is the leading dimension of the two-dimensional array containing
     * A.
     *
     * No test for singularity or near-singularity is included in this function.
     * Such tests must be performed before calling this function.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix data is stored in the upper or the
     *        lower triangular part of array A. If uplo = 'U' or 'u', then only
     *        the upper triangular part of A may be referenced. If uplo = 'L' or
     *        'l', then only the lower triangular part of A may be referenced.
     * trans  specifies op(A). If transa = 'n' or 'N', op(A) = A. If transa = 't',
     *        'T', 'c', or 'C', op(A) = transpose(A)
     * diag   specifies whether or not A is a unit triangular matrix like so:
     *        if diag = 'U' or 'u', A is assumed to be unit triangular. If
     *        diag = 'N' or 'n', then A is not assumed to be unit triangular.
     * n      specifies the number of rows and columns of the matrix A. It
     *        must be at least 0.
     * A      is a double precision array of dimensions (lda, n). If uplo = 'U'
     *        or 'u', then A must contains the upper triangular part of a symmetric
     *        matrix, and the strictly lower triangular parts is not referenced.
     *        If uplo = 'L' or 'l', then A contains the lower triangular part of
     *        a symmetric matrix, and the strictly upper triangular part is not
     *        referenced.
     * lda    is the leading dimension of the two-dimensional array containing A.
     *        lda must be at least max(1, n).
     * x      double precision array of length at least (1 + (n - 1) * abs(incx)).
     *        On entry, x contains the n element right-hand side vector b. On exit,
     *        it is overwritten with the solution vector x.
     * incx   specifies the storage spacing between elements of x. incx must not
     *        be zero.
     *
     * Output
     * ------
     * x      updated to contain the solution vector x that solves op(A) * x = b.
     *
     * Reference: http://www.netlib.org/blas/dtrsv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if incx == 0 or if n < 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasDtrsv(char uplo, char trans, char diag, int n, Pointer A, int lda, Pointer x, int incx)
    {
        cublasDtrsvNative(uplo, trans, diag, n, A, lda, x, incx);
        checkResultBLAS();
    }
    private static native void cublasDtrsvNative(char uplo, char trans, char diag, int n, Pointer A, int lda, Pointer x, int incx);





    /**
     * <pre>
     * void
     * cublasDtrmv (char uplo, char trans, char diag, int n, const double *A,
     *              int lda, double *x, int incx);
     *
     * performs one of the matrix-vector operations x = op(A) * x, where op(A) =
     = A, or op(A) = transpose(A). x is an n-element single precision vector, and
     * A is an n x n, unit or non-unit, upper or lower, triangular matrix composed
     * of single precision elements.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix A is an upper or lower triangular
     *        matrix. If uplo = 'U' or 'u', then A is an upper triangular matrix.
     *        If uplo = 'L' or 'l', then A is a lower triangular matrix.
     * trans  specifies op(A). If transa = 'N' or 'n', op(A) = A. If trans = 'T',
     *        't', 'C', or 'c', op(A) = transpose(A)
     * diag   specifies whether or not matrix A is unit triangular. If diag = 'U'
     *        or 'u', A is assumed to be unit triangular. If diag = 'N' or 'n', A
     *        is not assumed to be unit triangular.
     * n      specifies the number of rows and columns of the matrix A. n must be
     *        at least zero.
     * A      single precision array of dimension (lda, n). If uplo = 'U' or 'u',
     *        the leading n x n upper triangular part of the array A must contain
     *        the upper triangular matrix and the strictly lower triangular part
     *        of A is not referenced. If uplo = 'L' or 'l', the leading n x n lower
     *        triangular part of the array A must contain the lower triangular
     *        matrix and the strictly upper triangular part of A is not referenced.
     *        When diag = 'U' or 'u', the diagonal elements of A are not referenced
     *        either, but are are assumed to be unity.
     * lda    is the leading dimension of A. It must be at least max (1, n).
     * x      single precision array of length at least (1 + (n - 1) * abs(incx) ).
     *        On entry, x contains the source vector. On exit, x is overwritten
     *        with the result vector.
     * incx   specifies the storage spacing for elements of x. incx must not be
     *        zero.
     *
     * Output
     * ------
     * x      updated according to x = op(A) * x,
     *
     * Reference: http://www.netlib.org/blas/dtrmv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if incx == 0 or if n < 0
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasDtrmv(char uplo, char trans, char diag, int n, Pointer A, int lda, Pointer x, int incx)
    {
        cublasDtrmvNative(uplo, trans, diag, n, A, lda, x, incx);
        checkResultBLAS();
    }
    private static native void cublasDtrmvNative(char uplo, char trans, char diag, int n, Pointer A, int lda, Pointer x, int incx);





    /**
     * <pre>
     * void
     * cublasDgbmv (char trans, int m, int n, int kl, int ku, double alpha,
     *              const double *A, int lda, const double *x, int incx, double beta,
     *              double *y, int incy);
     *
     * performs one of the matrix-vector operations
     *
     *    y = alpha*op(A)*x + beta*y,  op(A)=A or op(A) = transpose(A)
     *
     * alpha and beta are double precision scalars. x and y are double precision
     * vectors. A is an m by n band matrix consisting of double precision elements
     * with kl sub-diagonals and ku super-diagonals.
     *
     * Input
     * -----
     * trans  specifies op(A). If trans == 'N' or 'n', op(A) = A. If trans == 'T',
     *        't', 'C', or 'c', op(A) = transpose(A)
     * m      specifies the number of rows of the matrix A. m must be at least
     *        zero.
     * n      specifies the number of columns of the matrix A. n must be at least
     *        zero.
     * kl     specifies the number of sub-diagonals of matrix A. It must be at
     *        least zero.
     * ku     specifies the number of super-diagonals of matrix A. It must be at
     *        least zero.
     * alpha  double precision scalar multiplier applied to op(A).
     * A      double precision array of dimensions (lda, n). The leading
     *        (kl + ku + 1) x n part of the array A must contain the band matrix A,
     *        supplied column by column, with the leading diagonal of the matrix
     *        in row (ku + 1) of the array, the first super-diagonal starting at
     *        position 2 in row ku, the first sub-diagonal starting at position 1
     *        in row (ku + 2), and so on. Elements in the array A that do not
     *        correspond to elements in the band matrix (such as the top left
     *        ku x ku triangle) are not referenced.
     * lda    leading dimension of A. lda must be at least (kl + ku + 1).
     * x      double precision array of length at least (1+(n-1)*abs(incx)) when
     *        trans == 'N' or 'n' and at least (1+(m-1)*abs(incx)) otherwise.
     * incx   specifies the increment for the elements of x. incx must not be zero.
     * beta   double precision scalar multiplier applied to vector y. If beta is
     *        zero, y is not read.
     * y      double precision array of length at least (1+(m-1)*abs(incy)) when
     *        trans == 'N' or 'n' and at least (1+(n-1)*abs(incy)) otherwise. If
     *        beta is zero, y is not read.
     * incy   On entry, incy specifies the increment for the elements of y. incy
     *        must not be zero.
     *
     * Output
     * ------
     * y      updated according to y = alpha*op(A)*x + beta*y
     *
     * Reference: http://www.netlib.org/blas/dgbmv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or if incx or incy == 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasDgbmv(char trans, int m, int n, int kl, int ku, double alpha, Pointer A, int lda, Pointer x, int incx, double beta, Pointer y, int incy)
    {
        cublasDgbmvNative(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
        checkResultBLAS();
    }
    private static native void cublasDgbmvNative(char trans, int m, int n, int kl, int ku, double alpha, Pointer A, int lda, Pointer x, int incx, double beta, Pointer y, int incy);





    /**
     * <pre>
     * void
     * cublasDtbmv (char uplo, char trans, char diag, int n, int k, const double *A,
     *              int lda, double *x, int incx)
     *
     * performs one of the matrix-vector operations x = op(A) * x, where op(A) = A,
     * or op(A) = transpose(A). x is an n-element double precision vector, and A is
     * an n x n, unit or non-unit, upper or lower triangular band matrix composed
     * of double precision elements.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix A is an upper or lower triangular band
     *        matrix. If uplo == 'U' or 'u', A is an upper triangular band matrix.
     *        If uplo == 'L' or 'l', A is a lower triangular band matrix.
     * trans  specifies op(A). If transa == 'N' or 'n', op(A) = A. If trans == 'T',
     *        't', 'C', or 'c', op(A) = transpose(A)
     * diag   specifies whether or not matrix A is unit triangular. If diag == 'U'
     *        or 'u', A is assumed to be unit triangular. If diag == 'N' or 'n', A
     *        is not assumed to be unit triangular.
     * n      specifies the number of rows and columns of the matrix A. n must be
     *        at least zero.
     * k      specifies the number of super- or sub-diagonals. If uplo == 'U' or
     *        'u', k specifies the number of super-diagonals. If uplo == 'L' or
     *        'l', k specifies the number of sub-diagonals. k must at least be
     *        zero.
     * A      double precision array of dimension (lda, n). If uplo == 'U' or 'u',
     *        the leading (k + 1) x n part of the array A must contain the upper
     *        triangular band matrix, supplied column by column, with the leading
     *        diagonal of the matrix in row (k + 1) of the array, the first
     *        super-diagonal starting at position 2 in row k, and so on. The top
     *        left k x k triangle of the array A is not referenced. If uplo == 'L'
     *        or 'l', the leading (k + 1) x n part of the array A must constain the
     *        lower triangular band matrix, supplied column by column, with the
     *        leading diagonal of the matrix in row 1 of the array, the first
     *        sub-diagonal startingat position 1 in row 2, and so on. The bottom
     *        right k x k triangle of the array is not referenced.
     * lda    is the leading dimension of A. It must be at least (k + 1).
     * x      double precision array of length at least (1 + (n - 1) * abs(incx)).
     *        On entry, x contains the source vector. On exit, x is overwritten
     *        with the result vector.
     * incx   specifies the storage spacing for elements of x. incx must not be
     *        zero.
     *
     * Output
     * ------
     * x      updated according to x = op(A) * x
     *
     * Reference: http://www.netlib.org/blas/dtbmv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n or k < 0, or if incx == 0
     * CUBLAS_STATUS_ALLOC_FAILED     if function cannot allocate enough internal scratch vector memory
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasDtbmv(char uplo, char trans, char diag, int n, int k, Pointer A, int lda, Pointer x, int incx)
    {
        cublasDtbmvNative(uplo, trans, diag, n, k, A, lda, x, incx);
        checkResultBLAS();
    }
    private static native void cublasDtbmvNative(char uplo, char trans, char diag, int n, int k, Pointer A, int lda, Pointer x, int incx);





    /**
     * <pre>
     * void
     * cublasDtpmv (char uplo, char trans, char diag, int n, const double *AP,
     *              double *x, int incx);
     *
     * performs one of the matrix-vector operations x = op(A) * x, where op(A) = A,
     * or op(A) = transpose(A). x is an n element double precision vector, and A
     * is an n x n, unit or non-unit, upper or lower triangular matrix composed
     * of double precision elements.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix A is an upper or lower triangular
     *        matrix. If uplo == 'U' or 'u', then A is an upper triangular matrix.
     *        If uplo == 'L' or 'l', then A is a lower triangular matrix.
     * trans  specifies op(A). If transa == 'N' or 'n', op(A) = A. If trans == 'T',
     *        't', 'C', or 'c', op(A) = transpose(A)
     * diag   specifies whether or not matrix A is unit triangular. If diag == 'U'
     *        or 'u', A is assumed to be unit triangular. If diag == 'N' or 'n', A
     *        is not assumed to be unit triangular.
     * n      specifies the number of rows and columns of the matrix A. n must be
     *        at least zero. In the current implementation n must not exceed 4070.
     * AP     double precision array with at least ((n * (n + 1)) / 2) elements. If
     *        uplo == 'U' or 'u', the array AP contains the upper triangular part
     *        of the symmetric matrix A, packed sequentially, column by column;
     *        that is, if i <= j, then A[i,j] is stored in AP[i+(j*(j+1)/2)]. If
     *        uplo == 'L' or 'L', the array AP contains the lower triangular part
     *        of the symmetric matrix A, packed sequentially, column by column;
     *        that is, if i >= j, then A[i,j] is stored in AP[i+((2*n-j+1)*j)/2].
     * x      double precision array of length at least (1 + (n - 1) * abs(incx)).
     *        On entry, x contains the source vector. On exit, x is overwritten
     *        with the result vector.
     * incx   specifies the storage spacing for elements of x. incx must not be
     *        zero.
     *
     * Output
     * ------
     * x      updated according to x = op(A) * x,
     *
     * Reference: http://www.netlib.org/blas/dtpmv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if incx == 0 or n < 0
     * CUBLAS_STATUS_ALLOC_FAILED     if function cannot allocate enough internal scratch vector memory
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasDtpmv(char uplo, char trans, char diag, int n, Pointer AP, Pointer x, int incx)
    {
        cublasDtpmvNative(uplo, trans, diag, n, AP, x, incx);
        checkResultBLAS();
    }
    private static native void cublasDtpmvNative(char uplo, char trans, char diag, int n, Pointer AP, Pointer x, int incx);





    /**
     * <pre>
     * void
     * cublasDtpsv (char uplo, char trans, char diag, int n, const double *AP,
     *              double *X, int incx)
     *
     * solves one of the systems of equations op(A)*x = b, where op(A) is either
     * op(A) = A or op(A) = transpose(A). b and x are n element vectors, and A is
     * an n x n unit or non-unit, upper or lower triangular matrix. No test for
     * singularity or near-singularity is included in this routine. Such tests
     * must be performed before calling this routine.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix is an upper or lower triangular matrix
     *        as follows: If uplo == 'U' or 'u', A is an upper triangluar matrix.
     *        If uplo == 'L' or 'l', A is a lower triangular matrix.
     * trans  specifies op(A). If trans == 'N' or 'n', op(A) = A. If trans == 'T',
     *        't', 'C', or 'c', op(A) = transpose(A).
     * diag   specifies whether A is unit triangular. If diag == 'U' or 'u', A is
     *        assumed to be unit triangular; thas is, diagonal elements are not
     *        read and are assumed to be unity. If diag == 'N' or 'n', A is not
     *        assumed to be unit triangular.
     * n      specifies the number of rows and columns of the matrix A. n must be
     *        at least zero.
     * AP     double precision array with at least ((n*(n+1))/2) elements. If uplo
     *        == 'U' or 'u', the array AP contains the upper triangular matrix A,
     *        packed sequentially, column by column; that is, if i <= j, then
     *        A[i,j] is stored is AP[i+(j*(j+1)/2)]. If uplo == 'L' or 'L', the
     *        array AP contains the lower triangular matrix A, packed sequentially,
     *        column by column; that is, if i >= j, then A[i,j] is stored in
     *        AP[i+((2*n-j+1)*j)/2]. When diag = 'U' or 'u', the diagonal elements
     *        of A are not referenced and are assumed to be unity.
     * x      double precision array of length at least (1+(n-1)*abs(incx)).
     * incx   storage spacing between elements of x. It must not be zero.
     *
     * Output
     * ------
     * x      updated to contain the solution vector x that solves op(A) * x = b.
     *
     * Reference: http://www.netlib.org/blas/dtpsv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if incx == 0 or if n < 0 or n > 2035
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasDtpsv(char uplo, char trans, char diag, int n, Pointer AP, Pointer x, int incx)
    {
        cublasDtpsvNative(uplo, trans, diag, n, AP, x, incx);
        checkResultBLAS();
    }
    private static native void cublasDtpsvNative(char uplo, char trans, char diag, int n, Pointer AP, Pointer x, int incx);





    /**
     * <pre>
     * void cublasDtbsv (char uplo, char trans, char diag, int n, int k,
     *                   const double *A, int lda, double *X, int incx)
     *
     * solves one of the systems of equations op(A)*x = b, where op(A) is either
     * op(A) = A or op(A) = transpose(A). b and x are n element vectors, and A is
     * an n x n unit or non-unit, upper or lower triangular band matrix with k + 1
     * diagonals. No test for singularity or near-singularity is included in this
     * function. Such tests must be performed before calling this function.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix is an upper or lower triangular band
     *        matrix as follows: If uplo == 'U' or 'u', A is an upper triangular
     *        band matrix. If uplo == 'L' or 'l', A is a lower triangular band
     *        matrix.
     * trans  specifies op(A). If trans == 'N' or 'n', op(A) = A. If trans == 'T',
     *        't', 'C', or 'c', op(A) = transpose(A).
     * diag   specifies whether A is unit triangular. If diag == 'U' or 'u', A is
     *        assumed to be unit triangular; thas is, diagonal elements are not
     *        read and are assumed to be unity. If diag == 'N' or 'n', A is not
     *        assumed to be unit triangular.
     * n      specifies the number of rows and columns of the matrix A. n must be
     *        at least zero.
     * k      specifies the number of super- or sub-diagonals. If uplo == 'U' or
     *        'u', k specifies the number of super-diagonals. If uplo == 'L' or
     *        'l', k specifies the number of sub-diagonals. k must at least be
     *        zero.
     * A      double precision array of dimension (lda, n). If uplo == 'U' or 'u',
     *        the leading (k + 1) x n part of the array A must contain the upper
     *        triangular band matrix, supplied column by column, with the leading
     *        diagonal of the matrix in row (k + 1) of the array, the first super-
     *        diagonal starting at position 2 in row k, and so on. The top left
     *        k x k triangle of the array A is not referenced. If uplo == 'L' or
     *        'l', the leading (k + 1) x n part of the array A must constain the
     *        lower triangular band matrix, supplied column by column, with the
     *        leading diagonal of the matrix in row 1 of the array, the first
     *        sub-diagonal starting at position 1 in row 2, and so on. The bottom
     *        right k x k triangle of the array is not referenced.
     * x      double precision array of length at least (1+(n-1)*abs(incx)).
     * incx   storage spacing between elements of x. It must not be zero.
     *
     * Output
     * ------
     * x      updated to contain the solution vector x that solves op(A) * x = b.
     *
     * Reference: http://www.netlib.org/blas/dtbsv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if incx == 0, n < 0 or n > 2035
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasDtbsv(char uplo, char trans, char diag, int n, int k, Pointer A, int lda, Pointer x, int incx)
    {
        cublasDtbsvNative(uplo, trans, diag, n, k, A, lda, x, incx);
        checkResultBLAS();
    }
    private static native void cublasDtbsvNative(char uplo, char trans, char diag, int n, int k, Pointer A, int lda, Pointer x, int incx);





    /**
     * <pre>
     * void
     * cublasDsymv (char uplo, int n, double alpha, const double *A, int lda,
     *              const double *x, int incx, double beta, double *y, int incy)
     *
     * performs the matrix-vector operation
     *
     *     y = alpha*A*x + beta*y
     *
     * Alpha and beta are double precision scalars, and x and y are double
     * precision vectors, each with n elements. A is a symmetric n x n matrix
     * consisting of double precision elements that is stored in either upper or
     * lower storage mode.
     *
     * Input
     * -----
     * uplo   specifies whether the upper or lower triangular part of the array A
     *        is to be referenced. If uplo == 'U' or 'u', the symmetric matrix A
     *        is stored in upper storage mode, i.e. only the upper triangular part
     *        of A is to be referenced while the lower triangular part of A is to
     *        be inferred. If uplo == 'L' or 'l', the symmetric matrix A is stored
     *        in lower storage mode, i.e. only the lower triangular part of A is
     *        to be referenced while the upper triangular part of A is to be
     *        inferred.
     * n      specifies the number of rows and the number of columns of the
     *        symmetric matrix A. n must be at least zero.
     * alpha  double precision scalar multiplier applied to A*x.
     * A      double precision array of dimensions (lda, n). If uplo == 'U' or 'u',
     *        the leading n x n upper triangular part of the array A must contain
     *        the upper triangular part of the symmetric matrix and the strictly
     *        lower triangular part of A is not referenced. If uplo == 'L' or 'l',
     *        the leading n x n lower triangular part of the array A must contain
     *        the lower triangular part of the symmetric matrix and the strictly
     *        upper triangular part of A is not referenced.
     * lda    leading dimension of A. It must be at least max (1, n).
     * x      double precision array of length at least (1 + (n - 1) * abs(incx)).
     * incx   storage spacing between elements of x. incx must not be zero.
     * beta   double precision scalar multiplier applied to vector y.
     * y      double precision array of length at least (1 + (n - 1) * abs(incy)).
     *        If beta is zero, y is not read.
     * incy   storage spacing between elements of y. incy must not be zero.
     *
     * Output
     * ------
     * y      updated according to y = alpha*A*x + beta*y
     *
     * Reference: http://www.netlib.org/blas/dsymv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or if incx or incy == 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasDsymv(char uplo, int n, double alpha, Pointer A, int lda, Pointer x, int incx, double beta, Pointer y, int incy)
    {
        cublasDsymvNative(uplo, n, alpha, A, lda, x, incx, beta, y, incy);
        checkResultBLAS();
    }
    private static native void cublasDsymvNative(char uplo, int n, double alpha, Pointer A, int lda, Pointer x, int incx, double beta, Pointer y, int incy);





    /**
     * <pre>
     * void
     * cublasDsbmv (char uplo, int n, int k, double alpha, const double *A, int lda,
     *              const double *x, int incx, double beta, double *y, int incy)
     *
     * performs the matrix-vector operation
     *
     *     y := alpha*A*x + beta*y
     *
     * alpha and beta are double precision scalars. x and y are double precision
     * vectors with n elements. A is an n by n symmetric band matrix consisting
     * of double precision elements, with k super-diagonals and the same number
     * of subdiagonals.
     *
     * Input
     * -----
     * uplo   specifies whether the upper or lower triangular part of the symmetric
     *        band matrix A is being supplied. If uplo == 'U' or 'u', the upper
     *        triangular part is being supplied. If uplo == 'L' or 'l', the lower
     *        triangular part is being supplied.
     * n      specifies the number of rows and the number of columns of the
     *        symmetric matrix A. n must be at least zero.
     * k      specifies the number of super-diagonals of matrix A. Since the matrix
     *        is symmetric, this is also the number of sub-diagonals. k must be at
     *        least zero.
     * alpha  double precision scalar multiplier applied to A*x.
     * A      double precision array of dimensions (lda, n). When uplo == 'U' or
     *        'u', the leading (k + 1) x n part of array A must contain the upper
     *        triangular band of the symmetric matrix, supplied column by column,
     *        with the leading diagonal of the matrix in row (k+1) of the array,
     *        the first super-diagonal starting at position 2 in row k, and so on.
     *        The top left k x k triangle of the array A is not referenced. When
     *        uplo == 'L' or 'l', the leading (k + 1) x n part of the array A must
     *        contain the lower triangular band part of the symmetric matrix,
     *        supplied column by column, with the leading diagonal of the matrix in
     *        row 1 of the array, the first sub-diagonal starting at position 1 in
     *        row 2, and so on. The bottom right k x k triangle of the array A is
     *        not referenced.
     * lda    leading dimension of A. lda must be at least (k + 1).
     * x      double precision array of length at least (1 + (n - 1) * abs(incx)).
     * incx   storage spacing between elements of x. incx must not be zero.
     * beta   double precision scalar multiplier applied to vector y. If beta is
     *        zero, y is not read.
     * y      double precision array of length at least (1 + (n - 1) * abs(incy)).
     *        If beta is zero, y is not read.
     * incy   storage spacing between elements of y. incy must not be zero.
     *
     * Output
     * ------
     * y      updated according to alpha*A*x + beta*y
     *
     * Reference: http://www.netlib.org/blas/dsbmv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if k or n < 0, or if incx or incy == 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasDsbmv(char uplo, int n, int k, double alpha, Pointer A, int lda, Pointer x, int incx, double beta, Pointer y, int incy)
    {
        cublasDsbmvNative(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
        checkResultBLAS();
    }
    private static native void cublasDsbmvNative(char uplo, int n, int k, double alpha, Pointer A, int lda, Pointer x, int incx, double beta, Pointer y, int incy);





    /**
     * <pre>
     * void
     * cublasDspmv (char uplo, int n, double alpha, const double *AP, const double *x,
     *              int incx, double beta, double *y, int incy)
     *
     * performs the matrix-vector operation
     *
     *    y = alpha * A * x + beta * y
     *
     * Alpha and beta are double precision scalars, and x and y are double
     * precision vectors with n elements. A is a symmetric n x n matrix
     * consisting of double precision elements that is supplied in packed form.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix data is stored in the upper or the lower
     *        triangular part of array AP. If uplo == 'U' or 'u', then the upper
     *        triangular part of A is supplied in AP. If uplo == 'L' or 'l', then
     *        the lower triangular part of A is supplied in AP.
     * n      specifies the number of rows and columns of the matrix A. It must be
     *        at least zero.
     * alpha  double precision scalar multiplier applied to A*x.
     * AP     double precision array with at least ((n * (n + 1)) / 2) elements. If
     *        uplo == 'U' or 'u', the array AP contains the upper triangular part
     *        of the symmetric matrix A, packed sequentially, column by column;
     *        that is, if i <= j, then A[i,j] is stored is AP[i+(j*(j+1)/2)]. If
     *        uplo == 'L' or 'L', the array AP contains the lower triangular part
     *        of the symmetric matrix A, packed sequentially, column by column;
     *        that is, if i >= j, then A[i,j] is stored in AP[i+((2*n-j+1)*j)/2].
     * x      double precision array of length at least (1 + (n - 1) * abs(incx)).
     * incx   storage spacing between elements of x. incx must not be zero.
     * beta   double precision scalar multiplier applied to vector y;
     * y      double precision array of length at least (1 + (n - 1) * abs(incy)).
     *        If beta is zero, y is not read.
     * incy   storage spacing between elements of y. incy must not be zero.
     *
     * Output
     * ------
     * y      updated according to y = alpha*A*x + beta*y
     *
     * Reference: http://www.netlib.org/blas/dspmv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or if incx or incy == 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasDspmv(char uplo, int n, double alpha, Pointer AP, Pointer x, int incx, double beta, Pointer y, int incy)
    {
        cublasDspmvNative(uplo, n, alpha, AP, x, incx, beta, y, incy);
        checkResultBLAS();
    }
    private static native void cublasDspmvNative(char uplo, int n, double alpha, Pointer AP, Pointer x, int incx, double beta, Pointer y, int incy);





    /**
     * <pre>
     * void
     * cublasDgemm (char transa, char transb, int m, int n, int k, double alpha,
     *              const double *A, int lda, const double *B, int ldb,
     *              double beta, double *C, int ldc)
     *
     * computes the product of matrix A and matrix B, multiplies the result
     * by scalar alpha, and adds the sum to the product of matrix C and
     * scalar beta. It performs one of the matrix-matrix operations:
     *
     * C = alpha * op(A) * op(B) + beta * C,
     * where op(X) = X or op(X) = transpose(X),
     *
     * and alpha and beta are double-precision scalars. A, B and C are matrices
     * consisting of double-precision elements, with op(A) an m x k matrix,
     * op(B) a k x n matrix, and C an m x n matrix. Matrices A, B, and C are
     * stored in column-major format, and lda, ldb, and ldc are the leading
     * dimensions of the two-dimensional arrays containing A, B, and C.
     *
     * Input
     * -----
     * transa specifies op(A). If transa == 'N' or 'n', op(A) = A.
     *        If transa == 'T', 't', 'C', or 'c', op(A) = transpose(A).
     * transb specifies op(B). If transb == 'N' or 'n', op(B) = B.
     *        If transb == 'T', 't', 'C', or 'c', op(B) = transpose(B).
     * m      number of rows of matrix op(A) and rows of matrix C; m must be at
     *        least zero.
     * n      number of columns of matrix op(B) and number of columns of C;
     *        n must be at least zero.
     * k      number of columns of matrix op(A) and number of rows of op(B);
     *        k must be at least zero.
     * alpha  double-precision scalar multiplier applied to op(A) * op(B).
     * A      double-precision array of dimensions (lda, k) if transa == 'N' or
     *        'n', and of dimensions (lda, m) otherwise. If transa == 'N' or
     *        'n' lda must be at least max(1, m), otherwise lda must be at
     *        least max(1, k).
     * lda    leading dimension of two-dimensional array used to store matrix A.
     * B      double-precision array of dimensions (ldb, n) if transb == 'N' or
     *        'n', and of dimensions (ldb, k) otherwise. If transb == 'N' or
     *        'n' ldb must be at least max (1, k), otherwise ldb must be at
     *        least max(1, n).
     * ldb    leading dimension of two-dimensional array used to store matrix B.
     * beta   double-precision scalar multiplier applied to C. If zero, C does not
     *        have to be a valid input
     * C      double-precision array of dimensions (ldc, n); ldc must be at least
     *        max(1, m).
     * ldc    leading dimension of two-dimensional array used to store matrix C.
     *
     * Output
     * ------
     * C      updated based on C = alpha * op(A)*op(B) + beta * C.
     *
     * Reference: http://www.netlib.org/blas/sgemm.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS was not initialized
     * CUBLAS_STATUS_INVALID_VALUE    if m < 0, n < 0, or k < 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasDgemm(char transa, char transb, int m, int n, int k, double alpha, Pointer A, int lda, Pointer B, int ldb, double beta, Pointer C, int ldc)
    {
        cublasDgemmNative(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        checkResultBLAS();
    }
    private static native void cublasDgemmNative(char transa, char transb, int m, int n, int k, double alpha, Pointer A, int lda, Pointer B, int ldb, double beta, Pointer C, int ldc);





    /**
     * <pre>
     * void
     * cublasDtrsm (char side, char uplo, char transa, char diag, int m, int n,
     *              double alpha, const double *A, int lda, double *B, int ldb)
     *
     * solves one of the matrix equations
     *
     *    op(A) * X = alpha * B,   or   X * op(A) = alpha * B,
     *
     * where alpha is a double precision scalar, and X and B are m x n matrices
     * that are composed of double precision elements. A is a unit or non-unit,
     * upper or lower triangular matrix, and op(A) is one of
     *
     *    op(A) = A  or  op(A) = transpose(A)
     *
     * The result matrix X overwrites input matrix B; that is, on exit the result
     * is stored in B. Matrices A and B are stored in column major format, and
     * lda and ldb are the leading dimensions of the two-dimensonials arrays that
     * contain A and B, respectively.
     *
     * Input
     * -----
     * side   specifies whether op(A) appears on the left or right of X as
     *        follows: side = 'L' or 'l' indicates solve op(A) * X = alpha * B.
     *        side = 'R' or 'r' indicates solve X * op(A) = alpha * B.
     * uplo   specifies whether the matrix A is an upper or lower triangular
     *        matrix as follows: uplo = 'U' or 'u' indicates A is an upper
     *        triangular matrix. uplo = 'L' or 'l' indicates A is a lower
     *        triangular matrix.
     * transa specifies the form of op(A) to be used in matrix multiplication
     *        as follows: If transa = 'N' or 'N', then op(A) = A. If transa =
     *        'T', 't', 'C', or 'c', then op(A) = transpose(A).
     * diag   specifies whether or not A is a unit triangular matrix like so:
     *        if diag = 'U' or 'u', A is assumed to be unit triangular. If
     *        diag = 'N' or 'n', then A is not assumed to be unit triangular.
     * m      specifies the number of rows of B. m must be at least zero.
     * n      specifies the number of columns of B. n must be at least zero.
     * alpha  is a double precision scalar to be multiplied with B. When alpha is
     *        zero, then A is not referenced and B need not be set before entry.
     * A      is a double precision array of dimensions (lda, k), where k is
     *        m when side = 'L' or 'l', and is n when side = 'R' or 'r'. If
     *        uplo = 'U' or 'u', the leading k x k upper triangular part of
     *        the array A must contain the upper triangular matrix and the
     *        strictly lower triangular matrix of A is not referenced. When
     *        uplo = 'L' or 'l', the leading k x k lower triangular part of
     *        the array A must contain the lower triangular matrix and the
     *        strictly upper triangular part of A is not referenced. Note that
     *        when diag = 'U' or 'u', the diagonal elements of A are not
     *        referenced, and are assumed to be unity.
     * lda    is the leading dimension of the two dimensional array containing A.
     *        When side = 'L' or 'l' then lda must be at least max(1, m), when
     *        side = 'R' or 'r' then lda must be at least max(1, n).
     * B      is a double precision array of dimensions (ldb, n). ldb must be
     *        at least max (1,m). The leading m x n part of the array B must
     *        contain the right-hand side matrix B. On exit B is overwritten
     *        by the solution matrix X.
     * ldb    is the leading dimension of the two dimensional array containing B.
     *        ldb must be at least max(1, m).
     *
     * Output
     * ------
     * B      contains the solution matrix X satisfying op(A) * X = alpha * B,
     *        or X * op(A) = alpha * B
     *
     * Reference: http://www.netlib.org/blas/dtrsm.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if m or n < 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasDtrsm(char side, char uplo, char transa, char diag, int m, int n, double alpha, Pointer A, int lda, Pointer B, int ldb)
    {
        cublasDtrsmNative(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
        checkResultBLAS();
    }
    private static native void cublasDtrsmNative(char side, char uplo, char transa, char diag, int m, int n, double alpha, Pointer A, int lda, Pointer B, int ldb);





    /**
     * <pre>
     * void
     * cublasZtrsm (char side, char uplo, char transa, char diag, int m, int n,
     *              cuDoubleComplex alpha, const cuDoubleComplex *A, int lda,
     *              cuDoubleComplex *B, int ldb)
     *
     * solves one of the matrix equations
     *
     *    op(A) * X = alpha * B,   or   X * op(A) = alpha * B,
     *
     * where alpha is a double precision complex scalar, and X and B are m x n matrices
     * that are composed of double precision complex elements. A is a unit or non-unit,
     * upper or lower triangular matrix, and op(A) is one of
     *
     *    op(A) = A  or  op(A) = transpose(A)  or  op( A ) = conj( A' ).
     *
     * The result matrix X overwrites input matrix B; that is, on exit the result
     * is stored in B. Matrices A and B are stored in column major format, and
     * lda and ldb are the leading dimensions of the two-dimensonials arrays that
     * contain A and B, respectively.
     *
     * Input
     * -----
     * side   specifies whether op(A) appears on the left or right of X as
     *        follows: side = 'L' or 'l' indicates solve op(A) * X = alpha * B.
     *        side = 'R' or 'r' indicates solve X * op(A) = alpha * B.
     * uplo   specifies whether the matrix A is an upper or lower triangular
     *        matrix as follows: uplo = 'U' or 'u' indicates A is an upper
     *        triangular matrix. uplo = 'L' or 'l' indicates A is a lower
     *        triangular matrix.
     * transa specifies the form of op(A) to be used in matrix multiplication
     *        as follows: If transa = 'N' or 'N', then op(A) = A. If transa =
     *        'T', 't', 'C', or 'c', then op(A) = transpose(A).
     * diag   specifies whether or not A is a unit triangular matrix like so:
     *        if diag = 'U' or 'u', A is assumed to be unit triangular. If
     *        diag = 'N' or 'n', then A is not assumed to be unit triangular.
     * m      specifies the number of rows of B. m must be at least zero.
     * n      specifies the number of columns of B. n must be at least zero.
     * alpha  is a double precision complex scalar to be multiplied with B. When alpha is
     *        zero, then A is not referenced and B need not be set before entry.
     * A      is a double precision complex array of dimensions (lda, k), where k is
     *        m when side = 'L' or 'l', and is n when side = 'R' or 'r'. If
     *        uplo = 'U' or 'u', the leading k x k upper triangular part of
     *        the array A must contain the upper triangular matrix and the
     *        strictly lower triangular matrix of A is not referenced. When
     *        uplo = 'L' or 'l', the leading k x k lower triangular part of
     *        the array A must contain the lower triangular matrix and the
     *        strictly upper triangular part of A is not referenced. Note that
     *        when diag = 'U' or 'u', the diagonal elements of A are not
     *        referenced, and are assumed to be unity.
     * lda    is the leading dimension of the two dimensional array containing A.
     *        When side = 'L' or 'l' then lda must be at least max(1, m), when
     *        side = 'R' or 'r' then lda must be at least max(1, n).
     * B      is a double precision complex array of dimensions (ldb, n). ldb must be
     *        at least max (1,m). The leading m x n part of the array B must
     *        contain the right-hand side matrix B. On exit B is overwritten
     *        by the solution matrix X.
     * ldb    is the leading dimension of the two dimensional array containing B.
     *        ldb must be at least max(1, m).
     *
     * Output
     * ------
     * B      contains the solution matrix X satisfying op(A) * X = alpha * B,
     *        or X * op(A) = alpha * B
     *
     * Reference: http://www.netlib.org/blas/ztrsm.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if m or n < 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasZtrsm(char side, char uplo, char transa, char diag, int m, int n, cuDoubleComplex alpha, Pointer A, int lda, Pointer B, int ldb)
    {
        cublasZtrsmNative(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
        checkResultBLAS();
    }
    private static native void cublasZtrsmNative(char side, char uplo, char transa, char diag, int m, int n, cuDoubleComplex alpha, Pointer A, int lda, Pointer B, int ldb);





    /**
     * <pre>
     * void
     * cublasDtrmm (char side, char uplo, char transa, char diag, int m, int n,
     *              double alpha, const double *A, int lda, const double *B, int ldb)
     *
     * performs one of the matrix-matrix operations
     *
     *   B = alpha * op(A) * B,  or  B = alpha * B * op(A)
     *
     * where alpha is a double-precision scalar, B is an m x n matrix composed
     * of double precision elements, and A is a unit or non-unit, upper or lower,
     * triangular matrix composed of double precision elements. op(A) is one of
     *
     *   op(A) = A  or  op(A) = transpose(A)
     *
     * Matrices A and B are stored in column major format, and lda and ldb are
     * the leading dimensions of the two-dimensonials arrays that contain A and
     * B, respectively.
     *
     * Input
     * -----
     * side   specifies whether op(A) multiplies B from the left or right.
     *        If side = 'L' or 'l', then B = alpha * op(A) * B. If side =
     *        'R' or 'r', then B = alpha * B * op(A).
     * uplo   specifies whether the matrix A is an upper or lower triangular
     *        matrix. If uplo = 'U' or 'u', A is an upper triangular matrix.
     *        If uplo = 'L' or 'l', A is a lower triangular matrix.
     * transa specifies the form of op(A) to be used in the matrix
     *        multiplication. If transa = 'N' or 'n', then op(A) = A. If
     *        transa = 'T', 't', 'C', or 'c', then op(A) = transpose(A).
     * diag   specifies whether or not A is unit triangular. If diag = 'U'
     *        or 'u', A is assumed to be unit triangular. If diag = 'N' or
     *        'n', A is not assumed to be unit triangular.
     * m      the number of rows of matrix B. m must be at least zero.
     * n      the number of columns of matrix B. n must be at least zero.
     * alpha  double precision scalar multiplier applied to op(A)*B, or
     *        B*op(A), respectively. If alpha is zero no accesses are made
     *        to matrix A, and no read accesses are made to matrix B.
     * A      double precision array of dimensions (lda, k). k = m if side =
     *        'L' or 'l', k = n if side = 'R' or 'r'. If uplo = 'U' or 'u'
     *        the leading k x k upper triangular part of the array A must
     *        contain the upper triangular matrix, and the strictly lower
     *        triangular part of A is not referenced. If uplo = 'L' or 'l'
     *        the leading k x k lower triangular part of the array A must
     *        contain the lower triangular matrix, and the strictly upper
     *        triangular part of A is not referenced. When diag = 'U' or 'u'
     *        the diagonal elements of A are no referenced and are assumed
     *        to be unity.
     * lda    leading dimension of A. When side = 'L' or 'l', it must be at
     *        least max(1,m) and at least max(1,n) otherwise
     * B      double precision array of dimensions (ldb, n). On entry, the
     *        leading m x n part of the array contains the matrix B. It is
     *        overwritten with the transformed matrix on exit.
     * ldb    leading dimension of B. It must be at least max (1, m).
     *
     * Output
     * ------
     * B      updated according to B = alpha * op(A) * B  or B = alpha * B * op(A)
     *
     * Reference: http://www.netlib.org/blas/dtrmm.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if m or n < 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasDtrmm(char side, char uplo, char transa, char diag, int m, int n, double alpha, Pointer A, int lda, Pointer B, int ldb)
    {
        cublasDtrmmNative(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
        checkResultBLAS();
    }
    private static native void cublasDtrmmNative(char side, char uplo, char transa, char diag, int m, int n, double alpha, Pointer A, int lda, Pointer B, int ldb);





    /**
     * <pre>
     * void
     * cublasDsymm (char side, char uplo, int m, int n, double alpha,
     *              const double *A, int lda, const double *B, int ldb,
     *              double beta, double *C, int ldc);
     *
     * performs one of the matrix-matrix operations
     *
     *   C = alpha * A * B + beta * C, or
     *   C = alpha * B * A + beta * C,
     *
     * where alpha and beta are double precision scalars, A is a symmetric matrix
     * consisting of double precision elements and stored in either lower or upper
     * storage mode, and B and C are m x n matrices consisting of double precision
     * elements.
     *
     * Input
     * -----
     * side   specifies whether the symmetric matrix A appears on the left side
     *        hand side or right hand side of matrix B, as follows. If side == 'L'
     *        or 'l', then C = alpha * A * B + beta * C. If side = 'R' or 'r',
     *        then C = alpha * B * A + beta * C.
     * uplo   specifies whether the symmetric matrix A is stored in upper or lower
     *        storage mode, as follows. If uplo == 'U' or 'u', only the upper
     *        triangular part of the symmetric matrix is to be referenced, and the
     *        elements of the strictly lower triangular part are to be infered from
     *        those in the upper triangular part. If uplo == 'L' or 'l', only the
     *        lower triangular part of the symmetric matrix is to be referenced,
     *        and the elements of the strictly upper triangular part are to be
     *        infered from those in the lower triangular part.
     * m      specifies the number of rows of the matrix C, and the number of rows
     *        of matrix B. It also specifies the dimensions of symmetric matrix A
     *        when side == 'L' or 'l'. m must be at least zero.
     * n      specifies the number of columns of the matrix C, and the number of
     *        columns of matrix B. It also specifies the dimensions of symmetric
     *        matrix A when side == 'R' or 'r'. n must be at least zero.
     * alpha  double precision scalar multiplier applied to A * B, or B * A
     * A      double precision array of dimensions (lda, ka), where ka is m when
     *        side == 'L' or 'l' and is n otherwise. If side == 'L' or 'l' the
     *        leading m x m part of array A must contain the symmetric matrix,
     *        such that when uplo == 'U' or 'u', the leading m x m part stores the
     *        upper triangular part of the symmetric matrix, and the strictly lower
     *        triangular part of A is not referenced, and when uplo == 'U' or 'u',
     *        the leading m x m part stores the lower triangular part of the
     *        symmetric matrix and the strictly upper triangular part is not
     *        referenced. If side == 'R' or 'r' the leading n x n part of array A
     *        must contain the symmetric matrix, such that when uplo == 'U' or 'u',
     *        the leading n x n part stores the upper triangular part of the
     *        symmetric matrix and the strictly lower triangular part of A is not
     *        referenced, and when uplo == 'U' or 'u', the leading n x n part
     *        stores the lower triangular part of the symmetric matrix and the
     *        strictly upper triangular part is not referenced.
     * lda    leading dimension of A. When side == 'L' or 'l', it must be at least
     *        max(1, m) and at least max(1, n) otherwise.
     * B      double precision array of dimensions (ldb, n). On entry, the leading
     *        m x n part of the array contains the matrix B.
     * ldb    leading dimension of B. It must be at least max (1, m).
     * beta   double precision scalar multiplier applied to C. If beta is zero, C
     *        does not have to be a valid input
     * C      double precision array of dimensions (ldc, n)
     * ldc    leading dimension of C. Must be at least max(1, m)
     *
     * Output
     * ------
     * C      updated according to C = alpha * A * B + beta * C, or C = alpha *
     *        B * A + beta * C
     *
     * Reference: http://www.netlib.org/blas/dsymm.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if m or n are < 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasDsymm(char side, char uplo, int m, int n, double alpha, Pointer A, int lda, Pointer B, int ldb, double beta, Pointer C, int ldc)
    {
        cublasDsymmNative(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
        checkResultBLAS();
    }
    private static native void cublasDsymmNative(char side, char uplo, int m, int n, double alpha, Pointer A, int lda, Pointer B, int ldb, double beta, Pointer C, int ldc);





    /**
     * <pre>
     * void
     * cublasZsymm (char side, char uplo, int m, int n, cuDoubleComplex alpha,
     *              const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb,
     *              cuDoubleComplex beta, cuDoubleComplex *C, int ldc);
     *
     * performs one of the matrix-matrix operations
     *
     *   C = alpha * A * B + beta * C, or
     *   C = alpha * B * A + beta * C,
     *
     * where alpha and beta are double precision complex scalars, A is a symmetric matrix
     * consisting of double precision complex elements and stored in either lower or upper
     * storage mode, and B and C are m x n matrices consisting of double precision
     * complex elements.
     *
     * Input
     * -----
     * side   specifies whether the symmetric matrix A appears on the left side
     *        hand side or right hand side of matrix B, as follows. If side == 'L'
     *        or 'l', then C = alpha * A * B + beta * C. If side = 'R' or 'r',
     *        then C = alpha * B * A + beta * C.
     * uplo   specifies whether the symmetric matrix A is stored in upper or lower
     *        storage mode, as follows. If uplo == 'U' or 'u', only the upper
     *        triangular part of the symmetric matrix is to be referenced, and the
     *        elements of the strictly lower triangular part are to be infered from
     *        those in the upper triangular part. If uplo == 'L' or 'l', only the
     *        lower triangular part of the symmetric matrix is to be referenced,
     *        and the elements of the strictly upper triangular part are to be
     *        infered from those in the lower triangular part.
     * m      specifies the number of rows of the matrix C, and the number of rows
     *        of matrix B. It also specifies the dimensions of symmetric matrix A
     *        when side == 'L' or 'l'. m must be at least zero.
     * n      specifies the number of columns of the matrix C, and the number of
     *        columns of matrix B. It also specifies the dimensions of symmetric
     *        matrix A when side == 'R' or 'r'. n must be at least zero.
     * alpha  double precision scalar multiplier applied to A * B, or B * A
     * A      double precision array of dimensions (lda, ka), where ka is m when
     *        side == 'L' or 'l' and is n otherwise. If side == 'L' or 'l' the
     *        leading m x m part of array A must contain the symmetric matrix,
     *        such that when uplo == 'U' or 'u', the leading m x m part stores the
     *        upper triangular part of the symmetric matrix, and the strictly lower
     *        triangular part of A is not referenced, and when uplo == 'U' or 'u',
     *        the leading m x m part stores the lower triangular part of the
     *        symmetric matrix and the strictly upper triangular part is not
     *        referenced. If side == 'R' or 'r' the leading n x n part of array A
     *        must contain the symmetric matrix, such that when uplo == 'U' or 'u',
     *        the leading n x n part stores the upper triangular part of the
     *        symmetric matrix and the strictly lower triangular part of A is not
     *        referenced, and when uplo == 'U' or 'u', the leading n x n part
     *        stores the lower triangular part of the symmetric matrix and the
     *        strictly upper triangular part is not referenced.
     * lda    leading dimension of A. When side == 'L' or 'l', it must be at least
     *        max(1, m) and at least max(1, n) otherwise.
     * B      double precision array of dimensions (ldb, n). On entry, the leading
     *        m x n part of the array contains the matrix B.
     * ldb    leading dimension of B. It must be at least max (1, m).
     * beta   double precision scalar multiplier applied to C. If beta is zero, C
     *        does not have to be a valid input
     * C      double precision array of dimensions (ldc, n)
     * ldc    leading dimension of C. Must be at least max(1, m)
     *
     * Output
     * ------
     * C      updated according to C = alpha * A * B + beta * C, or C = alpha *
     *        B * A + beta * C
     *
     * Reference: http://www.netlib.org/blas/zsymm.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if m or n are < 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasZsymm(char side, char uplo, int m, int n, cuDoubleComplex alpha, Pointer A, int lda, Pointer B, int ldb, cuDoubleComplex beta, Pointer C, int ldc)
    {
        cublasZsymmNative(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
        checkResultBLAS();
    }
    private static native void cublasZsymmNative(char side, char uplo, int m, int n, cuDoubleComplex alpha, Pointer A, int lda, Pointer B, int ldb, cuDoubleComplex beta, Pointer C, int ldc);





    /**
     * <pre>
     * void
     * cublasDsyrk (char uplo, char trans, int n, int k, double alpha,
     *              const double *A, int lda, double beta, double *C, int ldc)
     *
     * performs one of the symmetric rank k operations
     *
     *   C = alpha * A * transpose(A) + beta * C, or
     *   C = alpha * transpose(A) * A + beta * C.
     *
     * Alpha and beta are double precision scalars. C is an n x n symmetric matrix
     * consisting of double precision elements and stored in either lower or
     * upper storage mode. A is a matrix consisting of double precision elements
     * with dimension of n x k in the first case, and k x n in the second case.
     *
     * Input
     * -----
     * uplo   specifies whether the symmetric matrix C is stored in upper or lower
     *        storage mode as follows. If uplo == 'U' or 'u', only the upper
     *        triangular part of the symmetric matrix is to be referenced, and the
     *        elements of the strictly lower triangular part are to be infered from
     *        those in the upper triangular part. If uplo == 'L' or 'l', only the
     *        lower triangular part of the symmetric matrix is to be referenced,
     *        and the elements of the strictly upper triangular part are to be
     *        infered from those in the lower triangular part.
     * trans  specifies the operation to be performed. If trans == 'N' or 'n', C =
     *        alpha * transpose(A) + beta * C. If trans == 'T', 't', 'C', or 'c',
     *        C = transpose(A) * A + beta * C.
     * n      specifies the number of rows and the number columns of matrix C. If
     *        trans == 'N' or 'n', n specifies the number of rows of matrix A. If
     *        trans == 'T', 't', 'C', or 'c', n specifies the columns of matrix A.
     *        n must be at least zero.
     * k      If trans == 'N' or 'n', k specifies the number of rows of matrix A.
     *        If trans == 'T', 't', 'C', or 'c', k specifies the number of rows of
     *        matrix A. k must be at least zero.
     * alpha  double precision scalar multiplier applied to A * transpose(A) or
     *        transpose(A) * A.
     * A      double precision array of dimensions (lda, ka), where ka is k when
     *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n',
     *        the leading n x k part of array A must contain the matrix A,
     *        otherwise the leading k x n part of the array must contains the
     *        matrix A.
     * lda    leading dimension of A. When trans == 'N' or 'n' then lda must be at
     *        least max(1, n). Otherwise lda must be at least max(1, k).
     * beta   double precision scalar multiplier applied to C. If beta izs zero, C
     *        does not have to be a valid input
     * C      double precision array of dimensions (ldc, n). If uplo = 'U' or 'u',
     *        the leading n x n triangular part of the array C must contain the
     *        upper triangular part of the symmetric matrix C and the strictly
     *        lower triangular part of C is not referenced. On exit, the upper
     *        triangular part of C is overwritten by the upper triangular part of
     *        the updated matrix. If uplo = 'L' or 'l', the leading n x n
     *        triangular part of the array C must contain the lower triangular part
     *        of the symmetric matrix C and the strictly upper triangular part of C
     *        is not referenced. On exit, the lower triangular part of C is
     *        overwritten by the lower triangular part of the updated matrix.
     * ldc    leading dimension of C. It must be at least max(1, n).
     *
     * Output
     * ------
     * C      updated according to C = alpha * A * transpose(A) + beta * C, or C =
     *        alpha * transpose(A) * A + beta * C
     *
     * Reference: http://www.netlib.org/blas/dsyrk.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0 or k < 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasDsyrk(char uplo, char trans, int n, int k, double alpha, Pointer A, int lda, double beta, Pointer C, int ldc)
    {
        cublasDsyrkNative(uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
        checkResultBLAS();
    }
    private static native void cublasDsyrkNative(char uplo, char trans, int n, int k, double alpha, Pointer A, int lda, double beta, Pointer C, int ldc);





    /**
     * <pre>
     * void
     * cublasZsyrk (char uplo, char trans, int n, int k, cuDoubleComplex alpha,
     *              const cuDoubleComplex *A, int lda, cuDoubleComplex beta, cuDoubleComplex *C, int ldc)
     *
     * performs one of the symmetric rank k operations
     *
     *   C = alpha * A * transpose(A) + beta * C, or
     *   C = alpha * transpose(A) * A + beta * C.
     *
     * Alpha and beta are double precision complex scalars. C is an n x n symmetric matrix
     * consisting of double precision complex elements and stored in either lower or
     * upper storage mode. A is a matrix consisting of double precision complex elements
     * with dimension of n x k in the first case, and k x n in the second case.
     *
     * Input
     * -----
     * uplo   specifies whether the symmetric matrix C is stored in upper or lower
     *        storage mode as follows. If uplo == 'U' or 'u', only the upper
     *        triangular part of the symmetric matrix is to be referenced, and the
     *        elements of the strictly lower triangular part are to be infered from
     *        those in the upper triangular part. If uplo == 'L' or 'l', only the
     *        lower triangular part of the symmetric matrix is to be referenced,
     *        and the elements of the strictly upper triangular part are to be
     *        infered from those in the lower triangular part.
     * trans  specifies the operation to be performed. If trans == 'N' or 'n', C =
     *        alpha * transpose(A) + beta * C. If trans == 'T', 't', 'C', or 'c',
     *        C = transpose(A) * A + beta * C.
     * n      specifies the number of rows and the number columns of matrix C. If
     *        trans == 'N' or 'n', n specifies the number of rows of matrix A. If
     *        trans == 'T', 't', 'C', or 'c', n specifies the columns of matrix A.
     *        n must be at least zero.
     * k      If trans == 'N' or 'n', k specifies the number of rows of matrix A.
     *        If trans == 'T', 't', 'C', or 'c', k specifies the number of rows of
     *        matrix A. k must be at least zero.
     * alpha  double precision complex scalar multiplier applied to A * transpose(A) or
     *        transpose(A) * A.
     * A      double precision complex array of dimensions (lda, ka), where ka is k when
     *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n',
     *        the leading n x k part of array A must contain the matrix A,
     *        otherwise the leading k x n part of the array must contains the
     *        matrix A.
     * lda    leading dimension of A. When trans == 'N' or 'n' then lda must be at
     *        least max(1, n). Otherwise lda must be at least max(1, k).
     * beta   double precision complex scalar multiplier applied to C. If beta izs zero, C
     *        does not have to be a valid input
     * C      double precision complex array of dimensions (ldc, n). If uplo = 'U' or 'u',
     *        the leading n x n triangular part of the array C must contain the
     *        upper triangular part of the symmetric matrix C and the strictly
     *        lower triangular part of C is not referenced. On exit, the upper
     *        triangular part of C is overwritten by the upper triangular part of
     *        the updated matrix. If uplo = 'L' or 'l', the leading n x n
     *        triangular part of the array C must contain the lower triangular part
     *        of the symmetric matrix C and the strictly upper triangular part of C
     *        is not referenced. On exit, the lower triangular part of C is
     *        overwritten by the lower triangular part of the updated matrix.
     * ldc    leading dimension of C. It must be at least max(1, n).
     *
     * Output
     * ------
     * C      updated according to C = alpha * A * transpose(A) + beta * C, or C =
     *        alpha * transpose(A) * A + beta * C
     *
     * Reference: http://www.netlib.org/blas/zsyrk.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0 or k < 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasZsyrk(char uplo, char trans, int n, int k, cuDoubleComplex alpha, Pointer A, int lda, cuDoubleComplex beta, Pointer C, int ldc)
    {
        cublasZsyrkNative(uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
        checkResultBLAS();
    }
    private static native void cublasZsyrkNative(char uplo, char trans, int n, int k, cuDoubleComplex alpha, Pointer A, int lda, cuDoubleComplex beta, Pointer C, int ldc);





    /**
     * <pre>
     * void
     * cublasZsyr2k (char uplo, char trans, int n, int k, cuDoubleComplex alpha,
     *               const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb,
     *               cuDoubleComplex beta, cuDoubleComplex *C, int ldc)
     *
     * performs one of the symmetric rank 2k operations
     *
     *    C = alpha * A * transpose(B) + alpha * B * transpose(A) + beta * C, or
     *    C = alpha * transpose(A) * B + alpha * transpose(B) * A + beta * C.
     *
     * Alpha and beta are double precision complex scalars. C is an n x n symmetric matrix
     * consisting of double precision complex elements and stored in either lower or upper
     * storage mode. A and B are matrices consisting of double precision complex elements
     * with dimension of n x k in the first case, and k x n in the second case.
     *
     * Input
     * -----
     * uplo   specifies whether the symmetric matrix C is stored in upper or lower
     *        storage mode, as follows. If uplo == 'U' or 'u', only the upper
     *        triangular part of the symmetric matrix is to be referenced, and the
     *        elements of the strictly lower triangular part are to be infered from
     *        those in the upper triangular part. If uplo == 'L' or 'l', only the
     *        lower triangular part of the symmetric matrix is to be references,
     *        and the elements of the strictly upper triangular part are to be
     *        infered from those in the lower triangular part.
     * trans  specifies the operation to be performed. If trans == 'N' or 'n',
     *        C = alpha * A * transpose(B) + alpha * B * transpose(A) + beta * C,
     *        If trans == 'T', 't', 'C', or 'c', C = alpha * transpose(A) * B +
     *        alpha * transpose(B) * A + beta * C.
     * n      specifies the number of rows and the number columns of matrix C. If
     *        trans == 'N' or 'n', n specifies the number of rows of matrix A. If
     *        trans == 'T', 't', 'C', or 'c', n specifies the columns of matrix A.
     *        n must be at least zero.
     * k      If trans == 'N' or 'n', k specifies the number of rows of matrix A.
     *        If trans == 'T', 't', 'C', or 'c', k specifies the number of rows of
     *        matrix A. k must be at least zero.
     * alpha  double precision scalar multiplier.
     * A      double precision array of dimensions (lda, ka), where ka is k when
     *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n',
     *        the leading n x k part of array A must contain the matrix A,
     *        otherwise the leading k x n part of the array must contain the matrix
     *        A.
     * lda    leading dimension of A. When trans == 'N' or 'n' then lda must be at
     *        least max(1, n). Otherwise lda must be at least max(1,k).
     * B      double precision array of dimensions (lda, kb), where kb is k when
     *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n',
     *        the leading n x k part of array B must contain the matrix B,
     *        otherwise the leading k x n part of the array must contain the matrix
     *        B.
     * ldb    leading dimension of N. When trans == 'N' or 'n' then ldb must be at
     *        least max(1, n). Otherwise ldb must be at least max(1, k).
     * beta   double precision scalar multiplier applied to C. If beta is zero, C
     *        does not have to be a valid input.
     * C      double precision array of dimensions (ldc, n). If uplo == 'U' or 'u',
     *        the leading n x n triangular part of the array C must contain the
     *        upper triangular part of the symmetric matrix C and the strictly
     *        lower triangular part of C is not referenced. On exit, the upper
     *        triangular part of C is overwritten by the upper triangular part of
     *        the updated matrix. If uplo == 'L' or 'l', the leading n x n
     *        triangular part of the array C must contain the lower triangular part
     *        of the symmetric matrix C and the strictly upper triangular part of C
     *        is not referenced. On exit, the lower triangular part of C is
     *        overwritten by the lower triangular part of the updated matrix.
     * ldc    leading dimension of C. Must be at least max(1, n).
     *
     * Output
     * ------
     * C      updated according to alpha*A*transpose(B) + alpha*B*transpose(A) +
     *        beta*C or alpha*transpose(A)*B + alpha*transpose(B)*A + beta*C
     *
     * Reference:   http://www.netlib.org/blas/zsyr2k.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0 or k < 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasZsyr2k(char uplo, char trans, int n, int k, cuDoubleComplex alpha, Pointer A, int lda, Pointer B, int ldb, cuDoubleComplex beta, Pointer C, int ldc)
    {
        cublasZsyr2kNative(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        checkResultBLAS();
    }
    private static native void cublasZsyr2kNative(char uplo, char trans, int n, int k, cuDoubleComplex alpha, Pointer A, int lda, Pointer B, int ldb, cuDoubleComplex beta, Pointer C, int ldc);





    /**
     * <pre>
     * void
     * cublasZher2k (char uplo, char trans, int n, int k, cuDoubleComplex alpha,
     *               const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb,
     *               double beta, cuDoubleComplex *C, int ldc)
     *
     * performs one of the hermitian rank 2k operations
     *
     *    C =   alpha * A * conjugate(transpose(B))
     *        + conjugate(alpha) * B * conjugate(transpose(A))
     *        + beta * C ,
     *    or
     *    C =  alpha * conjugate(transpose(A)) * B
     *       + conjugate(alpha) * conjugate(transpose(B)) * A
     *       + beta * C.
     *
     * Alpha is double precision complex scalar whereas Beta is a double precision real scalar.
     * C is an n x n hermitian matrix consisting of double precision complex elements and
     * stored in either lower or upper storage mode. A and B are matrices consisting of
     * double precision complex elements with dimension of n x k in the first case,
     * and k x n in the second case.
     *
     * Input
     * -----
     * uplo   specifies whether the hermitian matrix C is stored in upper or lower
     *        storage mode, as follows. If uplo == 'U' or 'u', only the upper
     *        triangular part of the hermitian matrix is to be referenced, and the
     *        elements of the strictly lower triangular part are to be infered from
     *        those in the upper triangular part. If uplo == 'L' or 'l', only the
     *        lower triangular part of the hermitian matrix is to be references,
     *        and the elements of the strictly upper triangular part are to be
     *        infered from those in the lower triangular part.
     * trans  specifies the operation to be performed. If trans == 'N' or 'n',
     *        C =   alpha * A * conjugate(transpose(B))
     *            + conjugate(alpha) * B * conjugate(transpose(A))
     *            + beta * C .
     *        If trans == 'T', 't', 'C', or 'c',
     *        C =  alpha * conjugate(transpose(A)) * B
     *          + conjugate(alpha) * conjugate(transpose(B)) * A
     *          + beta * C.
     * n      specifies the number of rows and the number columns of matrix C. If
     *        trans == 'N' or 'n', n specifies the number of rows of matrix A. If
     *        trans == 'T', 't', 'C', or 'c', n specifies the columns of matrix A.
     *        n must be at least zero.
     * k      If trans == 'N' or 'n', k specifies the number of rows of matrix A.
     *        If trans == 'T', 't', 'C', or 'c', k specifies the number of rows of
     *        matrix A. k must be at least zero.
     * alpha  double precision scalar multiplier.
     * A      double precision array of dimensions (lda, ka), where ka is k when
     *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n',
     *        the leading n x k part of array A must contain the matrix A,
     *        otherwise the leading k x n part of the array must contain the matrix
     *        A.
     * lda    leading dimension of A. When trans == 'N' or 'n' then lda must be at
     *        least max(1, n). Otherwise lda must be at least max(1,k).
     * B      double precision array of dimensions (lda, kb), where kb is k when
     *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n',
     *        the leading n x k part of array B must contain the matrix B,
     *        otherwise the leading k x n part of the array must contain the matrix
     *        B.
     * ldb    leading dimension of N. When trans == 'N' or 'n' then ldb must be at
     *        least max(1, n). Otherwise ldb must be at least max(1, k).
     * beta   double precision scalar multiplier applied to C. If beta is zero, C
     *        does not have to be a valid input.
     * C      double precision array of dimensions (ldc, n). If uplo == 'U' or 'u',
     *        the leading n x n triangular part of the array C must contain the
     *        upper triangular part of the hermitian matrix C and the strictly
     *        lower triangular part of C is not referenced. On exit, the upper
     *        triangular part of C is overwritten by the upper triangular part of
     *        the updated matrix. If uplo == 'L' or 'l', the leading n x n
     *        triangular part of the array C must contain the lower triangular part
     *        of the hermitian matrix C and the strictly upper triangular part of C
     *        is not referenced. On exit, the lower triangular part of C is
     *        overwritten by the lower triangular part of the updated matrix.
     *        The imaginary parts of the diagonal elements need
     *        not be set,  they are assumed to be zero,  and on exit they
     *        are set to zero.
     * ldc    leading dimension of C. Must be at least max(1, n).
     *
     * Output
     * ------
     * C      updated according to alpha*A*conjugate(transpose(B)) +
     *        + conjugate(alpha)*B*conjugate(transpose(A)) + beta*C or
     *        alpha*conjugate(transpose(A))*B + conjugate(alpha)*conjugate(transpose(B))*A
     *        + beta*C.
     *
     * Reference:   http://www.netlib.org/blas/zher2k.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0 or k < 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasZher2k(char uplo, char trans, int n, int k, cuDoubleComplex alpha, Pointer A, int lda, Pointer B, int ldb, double beta, Pointer C, int ldc)
    {
        cublasZher2kNative(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        checkResultBLAS();
    }
    private static native void cublasZher2kNative(char uplo, char trans, int n, int k, cuDoubleComplex alpha, Pointer A, int lda, Pointer B, int ldb, double beta, Pointer C, int ldc);





    /**
     * <pre>
     * void
     * cublasZher (char uplo, int n, double alpha, const cuDoubleComplex *x, int incx,
     *             cuDoubleComplex *A, int lda)
     *
     * performs the hermitian rank 1 operation
     *
     *    A = alpha * x * conjugate(transpose(x) + A,
     *
     * where alpha is a double precision real scalar, x is an n element double
     * precision complex vector and A is an n x n hermitian matrix consisting of
     * double precision complex elements. Matrix A is stored in column major format,
     * and lda is the leading dimension of the two-dimensional array
     * containing A.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix data is stored in the upper or
     *        the lower triangular part of array A. If uplo = 'U' or 'u',
     *        then only the upper triangular part of A may be referenced.
     *        If uplo = 'L' or 'l', then only the lower triangular part of
     *        A may be referenced.
     * n      specifies the number of rows and columns of the matrix A. It
     *        must be at least 0.
     * alpha  double precision real scalar multiplier applied to
     *        x * conjugate(transpose(x))
     * x      double precision complex array of length at least (1 + (n - 1) * abs(incx))
     * incx   specifies the storage spacing between elements of x. incx must
     *        not be zero.
     * A      double precision complex array of dimensions (lda, n). If uplo = 'U' or
     *        'u', then A must contain the upper triangular part of a hermitian
     *        matrix, and the strictly lower triangular part is not referenced.
     *        If uplo = 'L' or 'l', then A contains the lower triangular part
     *        of a hermitian matrix, and the strictly upper triangular part is
     *        not referenced. The imaginary parts of the diagonal elements need
     *        not be set, they are assumed to be zero, and on exit they
     *        are set to zero.
     * lda    leading dimension of the two-dimensional array containing A. lda
     *        must be at least max(1, n).
     *
     * Output
     * ------
     * A      updated according to A = alpha * x * conjugate(transpose(x)) + A
     *
     * Reference: http://www.netlib.org/blas/zher.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or incx == 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasZher(char uplo, int n, double alpha, Pointer x, int incx, Pointer A, int lda)
    {
        cublasZherNative(uplo, n, alpha, x, incx, A, lda);
        checkResultBLAS();
    }
    private static native void cublasZherNative(char uplo, int n, double alpha, Pointer x, int incx, Pointer A, int lda);





    /**
     * <pre>
     * void
     * cublasZhpr (char uplo, int n, double alpha, const cuDoubleComplex *x, int incx,
     *             cuDoubleComplex *AP)
     *
     * performs the hermitian rank 1 operation
     *
     *    A = alpha * x * conjugate(transpose(x)) + A,
     *
     * where alpha is a double precision real scalar and x is an n element double
     * precision complex vector. A is a hermitian n x n matrix consisting of double
     * precision complex elements that is supplied in packed form.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix data is stored in the upper or the lower
     *        triangular part of array AP. If uplo == 'U' or 'u', then the upper
     *        triangular part of A is supplied in AP. If uplo == 'L' or 'l', then
     *        the lower triangular part of A is supplied in AP.
     * n      specifies the number of rows and columns of the matrix A. It must be
     *        at least zero.
     * alpha  double precision real scalar multiplier applied to x * conjugate(transpose(x)).
     * x      double precision array of length at least (1 + (n - 1) * abs(incx)).
     * incx   storage spacing between elements of x. incx must not be zero.
     * AP     double precision complex array with at least ((n * (n + 1)) / 2) elements. If
     *        uplo == 'U' or 'u', the array AP contains the upper triangular part
     *        of the hermitian matrix A, packed sequentially, column by column;
     *        that is, if i <= j, then A[i,j] is stored is AP[i+(j*(j+1)/2)]. If
     *        uplo == 'L' or 'L', the array AP contains the lower triangular part
     *        of the hermitian matrix A, packed sequentially, column by column;
     *        that is, if i >= j, then A[i,j] is stored in AP[i+((2*n-j+1)*j)/2].
     *        The imaginary parts of the diagonal elements need not be set, they
     *        are assumed to be zero, and on exit they are set to zero.
     *
     * Output
     * ------
     * A      updated according to A = alpha * x * conjugate(transpose(x)) + A
     *
     * Reference: http://www.netlib.org/blas/zhpr.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or incx == 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasZhpr(char uplo, int n, double alpha, Pointer x, int incx, Pointer AP)
    {
        cublasZhprNative(uplo, n, alpha, x, incx, AP);
        checkResultBLAS();
    }
    private static native void cublasZhprNative(char uplo, int n, double alpha, Pointer x, int incx, Pointer AP);





    /**
     * <pre>
     * void
     * cublasZhpr2 (char uplo, int n, cuDoubleComplex alpha, const cuDoubleComplex *x, int incx,
     *              const cuDoubleComplex *y, int incy, cuDoubleComplex *AP)
     *
     * performs the hermitian rank 2 operation
     *
     *    A = alpha*x*conjugate(transpose(y)) + conjugate(alpha)*y*conjugate(transpose(x)) + A,
     *
     * where alpha is a double precision complex scalar, and x and y are n element double
     * precision complex vectors. A is a hermitian n x n matrix consisting of double
     * precision complex elements that is supplied in packed form.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix data is stored in the upper or the lower
     *        triangular part of array A. If uplo == 'U' or 'u', then only the
     *        upper triangular part of A may be referenced and the lower triangular
     *        part of A is inferred. If uplo == 'L' or 'l', then only the lower
     *        triangular part of A may be referenced and the upper triangular part
     *        of A is inferred.
     * n      specifies the number of rows and columns of the matrix A. It must be
     *        at least zero.
     * alpha  double precision complex scalar multiplier applied to x * conjugate(transpose(y)) +
     *        y * conjugate(transpose(x)).
     * x      double precision complex array of length at least (1 + (n - 1) * abs (incx)).
     * incx   storage spacing between elements of x. incx must not be zero.
     * y      double precision complex array of length at least (1 + (n - 1) * abs (incy)).
     * incy   storage spacing between elements of y. incy must not be zero.
     * AP     double precision complex array with at least ((n * (n + 1)) / 2) elements. If
     *        uplo == 'U' or 'u', the array AP contains the upper triangular part
     *        of the hermitian matrix A, packed sequentially, column by column;
     *        that is, if i <= j, then A[i,j] is stored is AP[i+(j*(j+1)/2)]. If
     *        uplo == 'L' or 'L', the array AP contains the lower triangular part
     *        of the hermitian matrix A, packed sequentially, column by column;
     *        that is, if i >= j, then A[i,j] is stored in AP[i+((2*n-j+1)*j)/2].
     *        The imaginary parts of the diagonal elements need not be set, they
     *        are assumed to be zero, and on exit they are set to zero.
     *
     * Output
     * ------
     * A      updated according to A = alpha*x*conjugate(transpose(y))
     *                               + conjugate(alpha)*y*conjugate(transpose(x))+A
     *
     * Reference: http://www.netlib.org/blas/zhpr2.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0, incx == 0, incy == 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasZhpr2(char uplo, int n, cuDoubleComplex alpha, Pointer x, int incx, Pointer y, int incy, Pointer AP)
    {
        cublasZhpr2Native(uplo, n, alpha, x, incx, y, incy, AP);
        checkResultBLAS();
    }
    private static native void cublasZhpr2Native(char uplo, int n, cuDoubleComplex alpha, Pointer x, int incx, Pointer y, int incy, Pointer AP);





    /**
     * <pre>
     * void cublasZher2 (char uplo, int n, cuDoubleComplex alpha, const cuDoubleComplex *x, int incx,
     *                   const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda)
     *
     * performs the hermitian rank 2 operation
     *
     *    A = alpha*x*conjugate(transpose(y)) + conjugate(alpha)*y*conjugate(transpose(x)) + A,
     *
     * where alpha is a double precision complex scalar, x and y are n element double
     * precision complex vector and A is an n by n hermitian matrix consisting of double
     * precision complex elements.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix data is stored in the upper or the lower
     *        triangular part of array A. If uplo == 'U' or 'u', then only the
     *        upper triangular part of A may be referenced and the lower triangular
     *        part of A is inferred. If uplo == 'L' or 'l', then only the lower
     *        triangular part of A may be referenced and the upper triangular part
     *        of A is inferred.
     * n      specifies the number of rows and columns of the matrix A. It must be
     *        at least zero.
     * alpha  double precision complex scalar multiplier applied to x * conjugate(transpose(y)) +
     *        y * conjugate(transpose(x)).
     * x      double precision array of length at least (1 + (n - 1) * abs (incx)).
     * incx   storage spacing between elements of x. incx must not be zero.
     * y      double precision array of length at least (1 + (n - 1) * abs (incy)).
     * incy   storage spacing between elements of y. incy must not be zero.
     * A      double precision complex array of dimensions (lda, n). If uplo == 'U' or 'u',
     *        then A must contains the upper triangular part of a hermitian matrix,
     *        and the strictly lower triangular parts is not referenced. If uplo ==
     *        'L' or 'l', then A contains the lower triangular part of a hermitian
     *        matrix, and the strictly upper triangular part is not referenced.
     *        The imaginary parts of the diagonal elements need not be set,
     *        they are assumed to be zero, and on exit they are set to zero.
     *
     * lda    leading dimension of A. It must be at least max(1, n).
     *
     * Output
     * ------
     * A      updated according to A = alpha*x*conjugate(transpose(y))
     *                               + conjugate(alpha)*y*conjugate(transpose(x))+A
     *
     * Reference: http://www.netlib.org/blas/zher2.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0, incx == 0, incy == 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasZher2(char uplo, int n, cuDoubleComplex alpha, Pointer x, int incx, Pointer y, int incy, Pointer A, int lda)
    {
        cublasZher2Native(uplo, n, alpha, x, incx, y, incy, A, lda);
        checkResultBLAS();
    }
    private static native void cublasZher2Native(char uplo, int n, cuDoubleComplex alpha, Pointer x, int incx, Pointer y, int incy, Pointer A, int lda);





    /**
     * <pre>
     * void
     * cublasDsyr2k (char uplo, char trans, int n, int k, double alpha,
     *               const double *A, int lda, const double *B, int ldb,
     *               double beta, double *C, int ldc)
     *
     * performs one of the symmetric rank 2k operations
     *
     *    C = alpha * A * transpose(B) + alpha * B * transpose(A) + beta * C, or
     *    C = alpha * transpose(A) * B + alpha * transpose(B) * A + beta * C.
     *
     * Alpha and beta are double precision scalars. C is an n x n symmetric matrix
     * consisting of double precision elements and stored in either lower or upper
     * storage mode. A and B are matrices consisting of double precision elements
     * with dimension of n x k in the first case, and k x n in the second case.
     *
     * Input
     * -----
     * uplo   specifies whether the symmetric matrix C is stored in upper or lower
     *        storage mode, as follows. If uplo == 'U' or 'u', only the upper
     *        triangular part of the symmetric matrix is to be referenced, and the
     *        elements of the strictly lower triangular part are to be infered from
     *        those in the upper triangular part. If uplo == 'L' or 'l', only the
     *        lower triangular part of the symmetric matrix is to be references,
     *        and the elements of the strictly upper triangular part are to be
     *        infered from those in the lower triangular part.
     * trans  specifies the operation to be performed. If trans == 'N' or 'n',
     *        C = alpha * A * transpose(B) + alpha * B * transpose(A) + beta * C,
     *        If trans == 'T', 't', 'C', or 'c', C = alpha * transpose(A) * B +
     *        alpha * transpose(B) * A + beta * C.
     * n      specifies the number of rows and the number columns of matrix C. If
     *        trans == 'N' or 'n', n specifies the number of rows of matrix A. If
     *        trans == 'T', 't', 'C', or 'c', n specifies the columns of matrix A.
     *        n must be at least zero.
     * k      If trans == 'N' or 'n', k specifies the number of rows of matrix A.
     *        If trans == 'T', 't', 'C', or 'c', k specifies the number of rows of
     *        matrix A. k must be at least zero.
     * alpha  double precision scalar multiplier.
     * A      double precision array of dimensions (lda, ka), where ka is k when
     *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n',
     *        the leading n x k part of array A must contain the matrix A,
     *        otherwise the leading k x n part of the array must contain the matrix
     *        A.
     * lda    leading dimension of A. When trans == 'N' or 'n' then lda must be at
     *        least max(1, n). Otherwise lda must be at least max(1,k).
     * B      double precision array of dimensions (lda, kb), where kb is k when
     *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n',
     *        the leading n x k part of array B must contain the matrix B,
     *        otherwise the leading k x n part of the array must contain the matrix
     *        B.
     * ldb    leading dimension of N. When trans == 'N' or 'n' then ldb must be at
     *        least max(1, n). Otherwise ldb must be at least max(1, k).
     * beta   double precision scalar multiplier applied to C. If beta is zero, C
     *        does not have to be a valid input.
     * C      double precision array of dimensions (ldc, n). If uplo == 'U' or 'u',
     *        the leading n x n triangular part of the array C must contain the
     *        upper triangular part of the symmetric matrix C and the strictly
     *        lower triangular part of C is not referenced. On exit, the upper
     *        triangular part of C is overwritten by the upper triangular part of
     *        the updated matrix. If uplo == 'L' or 'l', the leading n x n
     *        triangular part of the array C must contain the lower triangular part
     *        of the symmetric matrix C and the strictly upper triangular part of C
     *        is not referenced. On exit, the lower triangular part of C is
     *        overwritten by the lower triangular part of the updated matrix.
     * ldc    leading dimension of C. Must be at least max(1, n).
     *
     * Output
     * ------
     * C      updated according to alpha*A*transpose(B) + alpha*B*transpose(A) +
     *        beta*C or alpha*transpose(A)*B + alpha*transpose(B)*A + beta*C
     *
     * Reference:   http://www.netlib.org/blas/dsyr2k.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0 or k < 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasDsyr2k(char uplo, char trans, int n, int k, double alpha, Pointer A, int lda, Pointer B, int ldb, double beta, Pointer C, int ldc)
    {
        cublasDsyr2kNative(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        checkResultBLAS();
    }
    private static native void cublasDsyr2kNative(char uplo, char trans, int n, int k, double alpha, Pointer A, int lda, Pointer B, int ldb, double beta, Pointer C, int ldc);





    /**
     * <pre>
     * void cublasZgemm (char transa, char transb, int m, int n, int k,
     *                   cuDoubleComplex alpha, const cuDoubleComplex *A, int lda,
     *                   const cuDoubleComplex *B, int ldb, cuDoubleComplex beta,
     *                   cuDoubleComplex *C, int ldc)
     *
     * zgemm performs one of the matrix-matrix operations
     *
     *    C = alpha * op(A) * op(B) + beta*C,
     *
     * where op(X) is one of
     *
     *    op(X) = X   or   op(X) = transpose  or  op(X) = conjg(transpose(X))
     *
     * alpha and beta are double-complex scalars, and A, B and C are matrices
     * consisting of double-complex elements, with op(A) an m x k matrix, op(B)
     * a k x n matrix and C an m x n matrix.
     *
     * Input
     * -----
     * transa specifies op(A). If transa == 'N' or 'n', op(A) = A. If transa ==
     *        'T' or 't', op(A) = transpose(A). If transa == 'C' or 'c', op(A) =
     *        conjg(transpose(A)).
     * transb specifies op(B). If transa == 'N' or 'n', op(B) = B. If transb ==
     *        'T' or 't', op(B) = transpose(B). If transb == 'C' or 'c', op(B) =
     *        conjg(transpose(B)).
     * m      number of rows of matrix op(A) and rows of matrix C. It must be at
     *        least zero.
     * n      number of columns of matrix op(B) and number of columns of C. It
     *        must be at least zero.
     * k      number of columns of matrix op(A) and number of rows of op(B). It
     *        must be at least zero.
     * alpha  double-complex scalar multiplier applied to op(A)op(B)
     * A      double-complex array of dimensions (lda, k) if transa ==  'N' or
     *        'n'), and of dimensions (lda, m) otherwise.
     * lda    leading dimension of A. When transa == 'N' or 'n', it must be at
     *        least max(1, m) and at least max(1, k) otherwise.
     * B      double-complex array of dimensions (ldb, n) if transb == 'N' or 'n',
     *        and of dimensions (ldb, k) otherwise
     * ldb    leading dimension of B. When transb == 'N' or 'n', it must be at
     *        least max(1, k) and at least max(1, n) otherwise.
     * beta   double-complex scalar multiplier applied to C. If beta is zero, C
     *        does not have to be a valid input.
     * C      double precision array of dimensions (ldc, n)
     * ldc    leading dimension of C. Must be at least max(1, m).
     *
     * Output
     * ------
     * C      updated according to C = alpha*op(A)*op(B) + beta*C
     *
     * Reference: http://www.netlib.org/blas/zgemm.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if any of m, n, or k are < 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasZgemm(char transa, char transb, int m, int n, int k, cuDoubleComplex alpha, Pointer A, int lda, Pointer B, int ldb, cuDoubleComplex beta, Pointer C, int ldc)
    {
        cublasZgemmNative(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        checkResultBLAS();
    }
    private static native void cublasZgemmNative(char transa, char transb, int m, int n, int k, cuDoubleComplex alpha, Pointer A, int lda, Pointer B, int ldb, cuDoubleComplex beta, Pointer C, int ldc);





    /**
     * <pre>
     * void
     * cublasZtrmm (char side, char uplo, char transa, char diag, int m, int n,
     *              cuDoubleComplex alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B,
     *              int ldb)
     *
     * performs one of the matrix-matrix operations
     *
     *   B = alpha * op(A) * B,  or  B = alpha * B * op(A)
     *
     * where alpha is a double-precision complex scalar, B is an m x n matrix composed
     * of double precision complex elements, and A is a unit or non-unit, upper or lower,
     * triangular matrix composed of double precision complex elements. op(A) is one of
     *
     *   op(A) = A  , op(A) = transpose(A) or op(A) = conjugate(transpose(A))
     *
     * Matrices A and B are stored in column major format, and lda and ldb are
     * the leading dimensions of the two-dimensonials arrays that contain A and
     * B, respectively.
     *
     * Input
     * -----
     * side   specifies whether op(A) multiplies B from the left or right.
     *        If side = 'L' or 'l', then B = alpha * op(A) * B. If side =
     *        'R' or 'r', then B = alpha * B * op(A).
     * uplo   specifies whether the matrix A is an upper or lower triangular
     *        matrix. If uplo = 'U' or 'u', A is an upper triangular matrix.
     *        If uplo = 'L' or 'l', A is a lower triangular matrix.
     * transa specifies the form of op(A) to be used in the matrix
     *        multiplication. If transa = 'N' or 'n', then op(A) = A. If
     *        transa = 'T' or 't', then op(A) = transpose(A).
     *        If transa = 'C' or 'c', then op(A) = conjugate(transpose(A)).
     * diag   specifies whether or not A is unit triangular. If diag = 'U'
     *        or 'u', A is assumed to be unit triangular. If diag = 'N' or
     *        'n', A is not assumed to be unit triangular.
     * m      the number of rows of matrix B. m must be at least zero.
     * n      the number of columns of matrix B. n must be at least zero.
     * alpha  double precision complex scalar multiplier applied to op(A)*B, or
     *        B*op(A), respectively. If alpha is zero no accesses are made
     *        to matrix A, and no read accesses are made to matrix B.
     * A      double precision complex array of dimensions (lda, k). k = m if side =
     *        'L' or 'l', k = n if side = 'R' or 'r'. If uplo = 'U' or 'u'
     *        the leading k x k upper triangular part of the array A must
     *        contain the upper triangular matrix, and the strictly lower
     *        triangular part of A is not referenced. If uplo = 'L' or 'l'
     *        the leading k x k lower triangular part of the array A must
     *        contain the lower triangular matrix, and the strictly upper
     *        triangular part of A is not referenced. When diag = 'U' or 'u'
     *        the diagonal elements of A are no referenced and are assumed
     *        to be unity.
     * lda    leading dimension of A. When side = 'L' or 'l', it must be at
     *        least max(1,m) and at least max(1,n) otherwise
     * B      double precision complex array of dimensions (ldb, n). On entry, the
     *        leading m x n part of the array contains the matrix B. It is
     *        overwritten with the transformed matrix on exit.
     * ldb    leading dimension of B. It must be at least max (1, m).
     *
     * Output
     * ------
     * B      updated according to B = alpha * op(A) * B  or B = alpha * B * op(A)
     *
     * Reference: http://www.netlib.org/blas/ztrmm.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if m or n < 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasZtrmm(char side, char uplo, char transa, char diag, int m, int n, cuDoubleComplex alpha, Pointer A, int lda, Pointer B, int ldb)
    {
        cublasZtrmmNative(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
        checkResultBLAS();
    }
    private static native void cublasZtrmmNative(char side, char uplo, char transa, char diag, int m, int n, cuDoubleComplex alpha, Pointer A, int lda, Pointer B, int ldb);





    /**
     * <pre>
     * cublasZgeru (int m, int n, cuDoubleComplex alpha, const cuDoubleComplex *x, int incx,
     *             const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda)
     *
     * performs the symmetric rank 1 operation
     *
     *    A = alpha * x * transpose(y) + A,
     *
     * where alpha is a double precision complex scalar, x is an m element double
     * precision complex vector, y is an n element double precision complex vector, and A
     * is an m by n matrix consisting of double precision complex elements. Matrix A
     * is stored in column major format, and lda is the leading dimension of
     * the two-dimensional array used to store A.
     *
     * Input
     * -----
     * m      specifies the number of rows of the matrix A. It must be at least
     *        zero.
     * n      specifies the number of columns of the matrix A. It must be at
     *        least zero.
     * alpha  double precision complex scalar multiplier applied to x * transpose(y)
     * x      double precision complex array of length at least (1 + (m - 1) * abs(incx))
     * incx   specifies the storage spacing between elements of x. incx must not
     *        be zero.
     * y      double precision complex array of length at least (1 + (n - 1) * abs(incy))
     * incy   specifies the storage spacing between elements of y. incy must not
     *        be zero.
     * A      double precision complex array of dimensions (lda, n).
     * lda    leading dimension of two-dimensional array used to store matrix A
     *
     * Output
     * ------
     * A      updated according to A = alpha * x * transpose(y) + A
     *
     * Reference: http://www.netlib.org/blas/zgeru.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if m < 0, n < 0, incx == 0, incy == 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasZgeru(int m, int n, cuDoubleComplex alpha, Pointer x, int incx, Pointer y, int incy, Pointer A, int lda)
    {
        cublasZgeruNative(m, n, alpha, x, incx, y, incy, A, lda);
        checkResultBLAS();
    }
    private static native void cublasZgeruNative(int m, int n, cuDoubleComplex alpha, Pointer x, int incx, Pointer y, int incy, Pointer A, int lda);





    /**
     * <pre>
     * cublasZgerc (int m, int n, cuDoubleComplex alpha, const cuDoubleComplex *x, int incx,
     *             const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda)
     *
     * performs the symmetric rank 1 operation
     *
     *    A = alpha * x * conjugate(transpose(y)) + A,
     *
     * where alpha is a double precision complex scalar, x is an m element double
     * precision complex vector, y is an n element double precision complex vector, and A
     * is an m by n matrix consisting of double precision complex elements. Matrix A
     * is stored in column major format, and lda is the leading dimension of
     * the two-dimensional array used to store A.
     *
     * Input
     * -----
     * m      specifies the number of rows of the matrix A. It must be at least
     *        zero.
     * n      specifies the number of columns of the matrix A. It must be at
     *        least zero.
     * alpha  double precision complex scalar multiplier applied to x * conjugate(transpose(y))
     * x      double precision array of length at least (1 + (m - 1) * abs(incx))
     * incx   specifies the storage spacing between elements of x. incx must not
     *        be zero.
     * y      double precision complex array of length at least (1 + (n - 1) * abs(incy))
     * incy   specifies the storage spacing between elements of y. incy must not
     *        be zero.
     * A      double precision complex array of dimensions (lda, n).
     * lda    leading dimension of two-dimensional array used to store matrix A
     *
     * Output
     * ------
     * A      updated according to A = alpha * x * conjugate(transpose(y)) + A
     *
     * Reference: http://www.netlib.org/blas/zgerc.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if m < 0, n < 0, incx == 0, incy == 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasZgerc(int m, int n, cuDoubleComplex alpha, Pointer x, int incx, Pointer y, int incy, Pointer A, int lda)
    {
        cublasZgercNative(m, n, alpha, x, incx, y, incy, A, lda);
        checkResultBLAS();
    }
    private static native void cublasZgercNative(int m, int n, cuDoubleComplex alpha, Pointer x, int incx, Pointer y, int incy, Pointer A, int lda);





    /**
     * <pre>
     * void
     * cublasZherk (char uplo, char trans, int n, int k, double alpha,
     *              const cuDoubleComplex *A, int lda, double beta, cuDoubleComplex *C, int ldc)
     *
     * performs one of the hermitian rank k operations
     *
     *   C = alpha * A * conjugate(transpose(A)) + beta * C, or
     *   C = alpha * conjugate(transpose(A)) * A + beta * C.
     *
     * Alpha and beta are double precision scalars. C is an n x n hermitian matrix
     * consisting of double precision complex elements and stored in either lower or
     * upper storage mode. A is a matrix consisting of double precision complex elements
     * with dimension of n x k in the first case, and k x n in the second case.
     *
     * Input
     * -----
     * uplo   specifies whether the hermitian matrix C is stored in upper or lower
     *        storage mode as follows. If uplo == 'U' or 'u', only the upper
     *        triangular part of the hermitian matrix is to be referenced, and the
     *        elements of the strictly lower triangular part are to be infered from
     *        those in the upper triangular part. If uplo == 'L' or 'l', only the
     *        lower triangular part of the hermitian matrix is to be referenced,
     *        and the elements of the strictly upper triangular part are to be
     *        infered from those in the lower triangular part.
     * trans  specifies the operation to be performed. If trans == 'N' or 'n', C =
     *        alpha * A * conjugate(transpose(A)) + beta * C. If trans == 'T', 't', 'C', or 'c',
     *        C = alpha * conjugate(transpose(A)) * A + beta * C.
     * n      specifies the number of rows and the number columns of matrix C. If
     *        trans == 'N' or 'n', n specifies the number of rows of matrix A. If
     *        trans == 'T', 't', 'C', or 'c', n specifies the columns of matrix A.
     *        n must be at least zero.
     * k      If trans == 'N' or 'n', k specifies the number of columns of matrix A.
     *        If trans == 'T', 't', 'C', or 'c', k specifies the number of rows of
     *        matrix A. k must be at least zero.
     * alpha  double precision scalar multiplier applied to A * conjugate(transpose(A)) or
     *        conjugate(transpose(A)) * A.
     * A      double precision complex array of dimensions (lda, ka), where ka is k when
     *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n',
     *        the leading n x k part of array A must contain the matrix A,
     *        otherwise the leading k x n part of the array must contains the
     *        matrix A.
     * lda    leading dimension of A. When trans == 'N' or 'n' then lda must be at
     *        least max(1, n). Otherwise lda must be at least max(1, k).
     * beta   double precision scalar multiplier applied to C. If beta is zero, C
     *        does not have to be a valid input
     * C      double precision complex array of dimensions (ldc, n). If uplo = 'U' or 'u',
     *        the leading n x n triangular part of the array C must contain the
     *        upper triangular part of the hermitian matrix C and the strictly
     *        lower triangular part of C is not referenced. On exit, the upper
     *        triangular part of C is overwritten by the upper triangular part of
     *        the updated matrix. If uplo = 'L' or 'l', the leading n x n
     *        triangular part of the array C must contain the lower triangular part
     *        of the hermitian matrix C and the strictly upper triangular part of C
     *        is not referenced. On exit, the lower triangular part of C is
     *        overwritten by the lower triangular part of the updated matrix.
     *        The imaginary parts of the diagonal elements need
     *        not be set,  they are assumed to be zero,  and on exit they
     *        are set to zero.
     * ldc    leading dimension of C. It must be at least max(1, n).
     *
     * Output
     * ------
     * C      updated according to C = alpha * A * conjugate(transpose(A)) + beta * C, or C =
     *        alpha * conjugate(transpose(A)) * A + beta * C
     *
     * Reference: http://www.netlib.org/blas/zherk.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if n < 0 or k < 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasZherk(char uplo, char trans, int n, int k, double alpha, Pointer A, int lda, double beta, Pointer C, int ldc)
    {
        cublasZherkNative(uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
        checkResultBLAS();
    }
    private static native void cublasZherkNative(char uplo, char trans, int n, int k, double alpha, Pointer A, int lda, double beta, Pointer C, int ldc);





    /**
     * <pre>
     * void
     * cublasZhemm (char side, char uplo, int m, int n, cuDoubleComplex alpha,
     *              const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb,
     *              cuDoubleComplex beta, cuDoubleComplex *C, int ldc);
     *
     * performs one of the matrix-matrix operations
     *
     *   C = alpha * A * B + beta * C, or
     *   C = alpha * B * A + beta * C,
     *
     * where alpha and beta are double precision complex scalars, A is a hermitian matrix
     * consisting of double precision complex elements and stored in either lower or upper
     * storage mode, and B and C are m x n matrices consisting of double precision
     * complex elements.
     *
     * Input
     * -----
     * side   specifies whether the hermitian matrix A appears on the left side
     *        hand side or right hand side of matrix B, as follows. If side == 'L'
     *        or 'l', then C = alpha * A * B + beta * C. If side = 'R' or 'r',
     *        then C = alpha * B * A + beta * C.
     * uplo   specifies whether the hermitian matrix A is stored in upper or lower
     *        storage mode, as follows. If uplo == 'U' or 'u', only the upper
     *        triangular part of the hermitian matrix is to be referenced, and the
     *        elements of the strictly lower triangular part are to be infered from
     *        those in the upper triangular part. If uplo == 'L' or 'l', only the
     *        lower triangular part of the hermitian matrix is to be referenced,
     *        and the elements of the strictly upper triangular part are to be
     *        infered from those in the lower triangular part.
     * m      specifies the number of rows of the matrix C, and the number of rows
     *        of matrix B. It also specifies the dimensions of hermitian matrix A
     *        when side == 'L' or 'l'. m must be at least zero.
     * n      specifies the number of columns of the matrix C, and the number of
     *        columns of matrix B. It also specifies the dimensions of hermitian
     *        matrix A when side == 'R' or 'r'. n must be at least zero.
     * alpha  double precision scalar multiplier applied to A * B, or B * A
     * A      double precision complex array of dimensions (lda, ka), where ka is m when
     *        side == 'L' or 'l' and is n otherwise. If side == 'L' or 'l' the
     *        leading m x m part of array A must contain the hermitian matrix,
     *        such that when uplo == 'U' or 'u', the leading m x m part stores the
     *        upper triangular part of the hermitian matrix, and the strictly lower
     *        triangular part of A is not referenced, and when uplo == 'U' or 'u',
     *        the leading m x m part stores the lower triangular part of the
     *        hermitian matrix and the strictly upper triangular part is not
     *        referenced. If side == 'R' or 'r' the leading n x n part of array A
     *        must contain the hermitian matrix, such that when uplo == 'U' or 'u',
     *        the leading n x n part stores the upper triangular part of the
     *        hermitian matrix and the strictly lower triangular part of A is not
     *        referenced, and when uplo == 'U' or 'u', the leading n x n part
     *        stores the lower triangular part of the hermitian matrix and the
     *        strictly upper triangular part is not referenced. The imaginary parts
     *        of the diagonal elements need not be set, they are assumed to be zero.
     *
     * lda    leading dimension of A. When side == 'L' or 'l', it must be at least
     *        max(1, m) and at least max(1, n) otherwise.
     * B      double precision complex array of dimensions (ldb, n). On entry, the leading
     *        m x n part of the array contains the matrix B.
     * ldb    leading dimension of B. It must be at least max (1, m).
     * beta   double precision complex scalar multiplier applied to C. If beta is zero, C
     *        does not have to be a valid input
     * C      double precision complex array of dimensions (ldc, n)
     * ldc    leading dimension of C. Must be at least max(1, m)
     *
     * Output
     * ------
     * C      updated according to C = alpha * A * B + beta * C, or C = alpha *
     *        B * A + beta * C
     *
     * Reference: http://www.netlib.org/blas/zhemm.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if m or n are < 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasZhemm(char side, char uplo, int m, int n, cuDoubleComplex alpha, Pointer A, int lda, Pointer B, int ldb, cuDoubleComplex beta, Pointer C, int ldc)
    {
        cublasZhemmNative(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
        checkResultBLAS();
    }
    private static native void cublasZhemmNative(char side, char uplo, int m, int n, cuDoubleComplex alpha, Pointer A, int lda, Pointer B, int ldb, cuDoubleComplex beta, Pointer C, int ldc);





    /**
     * <pre>
     * void
     * cublasZtrsv (char uplo, char trans, char diag, int n, const cuDoubleComplex *A,
     *              int lda, cuDoubleComplex *x, int incx)
     *
     * solves a system of equations op(A) * x = b, where op(A) is either A,
     * transpose(A) or conjugate(transpose(A)). b and x are double precision
     * complex vectors consisting of n elements, and A is an n x n matrix
     * composed of a unit or non-unit, upper or lower triangular matrix.
     * Matrix A is stored in column major format, and lda is the leading
     * dimension of the two-dimensional array containing A.
     *
     * No test for singularity or near-singularity is included in this function.
     * Such tests must be performed before calling this function.
     *
     * Input
     * -----
     * uplo   specifies whether the matrix data is stored in the upper or the
     *        lower triangular part of array A. If uplo = 'U' or 'u', then only
     *        the upper triangular part of A may be referenced. If uplo = 'L' or
     *        'l', then only the lower triangular part of A may be referenced.
     * trans  specifies op(A). If transa = 'n' or 'N', op(A) = A. If transa = 't',
     *        'T', 'c', or 'C', op(A) = transpose(A)
     * diag   specifies whether or not A is a unit triangular matrix like so:
     *        if diag = 'U' or 'u', A is assumed to be unit triangular. If
     *        diag = 'N' or 'n', then A is not assumed to be unit triangular.
     * n      specifies the number of rows and columns of the matrix A. It
     *        must be at least 0.
     * A      is a double precision complex array of dimensions (lda, n). If uplo = 'U'
     *        or 'u', then A must contains the upper triangular part of a symmetric
     *        matrix, and the strictly lower triangular parts is not referenced.
     *        If uplo = 'L' or 'l', then A contains the lower triangular part of
     *        a symmetric matrix, and the strictly upper triangular part is not
     *        referenced.
     * lda    is the leading dimension of the two-dimensional array containing A.
     *        lda must be at least max(1, n).
     * x      double precision complex array of length at least (1 + (n - 1) * abs(incx)).
     *        On entry, x contains the n element right-hand side vector b. On exit,
     *        it is overwritten with the solution vector x.
     * incx   specifies the storage spacing between elements of x. incx must not
     *        be zero.
     *
     * Output
     * ------
     * x      updated to contain the solution vector x that solves op(A) * x = b.
     *
     * Reference: http://www.netlib.org/blas/ztrsv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if incx == 0 or if n < 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasZtrsv(char uplo, char trans, char diag, int n, Pointer A, int lda, Pointer x, int incx)
    {
        cublasZtrsvNative(uplo, trans, diag, n, A, lda, x, incx);
        checkResultBLAS();
    }
    private static native void cublasZtrsvNative(char uplo, char trans, char diag, int n, Pointer A, int lda, Pointer x, int incx);





    /**
     * <pre>
     * void
     * cublasZhbmv (char uplo, int n, int k, cuDoubleComplex alpha, const cuDoubleComplex *A, int lda,
     *              const cuDoubleComplex *x, int incx, cuDoubleComplex beta, cuDoubleComplex *y, int incy)
     *
     * performs the matrix-vector operation
     *
     *     y := alpha*A*x + beta*y
     *
     * alpha and beta are double precision complex scalars. x and y are double precision
     * complex vectors with n elements. A is an n by n hermitian band matrix consisting
     * of double precision complex elements, with k super-diagonals and the same number
     * of subdiagonals.
     *
     * Input
     * -----
     * uplo   specifies whether the upper or lower triangular part of the hermitian
     *        band matrix A is being supplied. If uplo == 'U' or 'u', the upper
     *        triangular part is being supplied. If uplo == 'L' or 'l', the lower
     *        triangular part is being supplied.
     * n      specifies the number of rows and the number of columns of the
     *        hermitian matrix A. n must be at least zero.
     * k      specifies the number of super-diagonals of matrix A. Since the matrix
     *        is hermitian, this is also the number of sub-diagonals. k must be at
     *        least zero.
     * alpha  double precision complex scalar multiplier applied to A*x.
     * A      double precision complex array of dimensions (lda, n). When uplo == 'U' or
     *        'u', the leading (k + 1) x n part of array A must contain the upper
     *        triangular band of the hermitian matrix, supplied column by column,
     *        with the leading diagonal of the matrix in row (k+1) of the array,
     *        the first super-diagonal starting at position 2 in row k, and so on.
     *        The top left k x k triangle of the array A is not referenced. When
     *        uplo == 'L' or 'l', the leading (k + 1) x n part of the array A must
     *        contain the lower triangular band part of the hermitian matrix,
     *        supplied column by column, with the leading diagonal of the matrix in
     *        row 1 of the array, the first sub-diagonal starting at position 1 in
     *        row 2, and so on. The bottom right k x k triangle of the array A is
     *        not referenced. The imaginary parts of the diagonal elements need
     *        not be set, they are assumed to be zero.
     * lda    leading dimension of A. lda must be at least (k + 1).
     * x      double precision complex array of length at least (1 + (n - 1) * abs(incx)).
     * incx   storage spacing between elements of x. incx must not be zero.
     * beta   double precision complex scalar multiplier applied to vector y. If beta is
     *        zero, y is not read.
     * y      double precision complex array of length at least (1 + (n - 1) * abs(incy)).
     *        If beta is zero, y is not read.
     * incy   storage spacing between elements of y. incy must not be zero.
     *
     * Output
     * ------
     * y      updated according to alpha*A*x + beta*y
     *
     * Reference: http://www.netlib.org/blas/zhbmv.f
     *
     * Error status for this function can be retrieved via cublasGetError().
     *
     * Error Status
     * ------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if k or n < 0, or if incx or incy == 0
     * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
     * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
     * </pre>
     */

    public static void cublasZhbmv(char uplo, int n, int k, cuDoubleComplex alpha, Pointer A, int lda, Pointer x, int incx, cuDoubleComplex beta, Pointer y, int incy)
    {
        cublasZhbmvNative(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
        checkResultBLAS();
    }
    private static native void cublasZhbmvNative(char uplo, int n, int k, cuDoubleComplex alpha, Pointer A, int lda, Pointer x, int incx, cuDoubleComplex beta, Pointer y, int incy);


















}


