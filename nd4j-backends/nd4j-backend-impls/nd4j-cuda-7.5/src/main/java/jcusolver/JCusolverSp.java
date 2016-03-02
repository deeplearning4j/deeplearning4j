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
import jcuda.jcusparse.cusparseMatDescr;
import jcuda.runtime.cudaStream_t;

/**
 * Java bindings for CUSOLVER, the NVIDIA CUDA solver library. <br />
 * <br />
 * The documentation is taken from the CUSOLVER header files.
 */
public class JCusolverSp
{
    /* Private constructor to prevent instantiation */
    private JCusolverSp()
    {
    }
    
    static
    {
        JCusolver.initialize();
    }
    
    /**
     * Delegates to {@link JCusolver#checkResult(int)}
     * 
     * @param result The result to check
     * @return The result that was given as the parameter
     * @throws CudaException As described in {@link JCusolver#checkResult(int)}
     */
    private static int checkResult(int result)
    {
        return JCusolver.checkResult(result);
    }
    
    //=== Auto-generated part: ===============================================
    
    public static int cusolverSpCreate(
        cusolverSpHandle handle)
    {
        return checkResult(cusolverSpCreateNative(handle));
    }
    private static native int cusolverSpCreateNative(
        cusolverSpHandle handle);


    public static int cusolverSpDestroy(
        cusolverSpHandle handle)
    {
        return checkResult(cusolverSpDestroyNative(handle));
    }
    private static native int cusolverSpDestroyNative(
        cusolverSpHandle handle);


    public static int cusolverSpSetStream(
        cusolverSpHandle handle, 
        cudaStream_t streamId)
    {
        return checkResult(cusolverSpSetStreamNative(handle, streamId));
    }
    private static native int cusolverSpSetStreamNative(
        cusolverSpHandle handle, 
        cudaStream_t streamId);


    public static int cusolverSpGetStream(
        cusolverSpHandle handle, 
        cudaStream_t streamId)
    {
        return checkResult(cusolverSpGetStreamNative(handle, streamId));
    }
    private static native int cusolverSpGetStreamNative(
        cusolverSpHandle handle, 
        cudaStream_t streamId);


    public static int cusolverSpXcsrissymHost(
        cusolverSpHandle handle, 
        int m, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrRowPtrA, 
        Pointer csrEndPtrA, 
        Pointer csrColIndA, 
        Pointer issym)
    {
        return checkResult(cusolverSpXcsrissymHostNative(handle, m, nnzA, descrA, csrRowPtrA, csrEndPtrA, csrColIndA, issym));
    }
    private static native int cusolverSpXcsrissymHostNative(
        cusolverSpHandle handle, 
        int m, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrRowPtrA, 
        Pointer csrEndPtrA, 
        Pointer csrColIndA, 
        Pointer issym);


    /**
     * <pre>
     * -------- GPU linear solver based on LU factorization
     *       solve A*x = b, A can be singular 
     * [ls] stands for linear solve
     * [v] stands for vector
     * [lu] stands for LU factorization
     * </pre>
     */
    public static int cusolverSpScsrlsvluHost(
        cusolverSpHandle handle, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer b, 
        float tol, 
        int reorder, 
        Pointer x, 
        Pointer singularity)
    {
        return checkResult(cusolverSpScsrlsvluHostNative(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity));
    }
    private static native int cusolverSpScsrlsvluHostNative(
        cusolverSpHandle handle, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer b, 
        float tol, 
        int reorder, 
        Pointer x, 
        Pointer singularity);


    public static int cusolverSpDcsrlsvluHost(
        cusolverSpHandle handle, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer b, 
        double tol, 
        int reorder, 
        Pointer x, 
        Pointer singularity)
    {
        return checkResult(cusolverSpDcsrlsvluHostNative(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity));
    }
    private static native int cusolverSpDcsrlsvluHostNative(
        cusolverSpHandle handle, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer b, 
        double tol, 
        int reorder, 
        Pointer x, 
        Pointer singularity);


    public static int cusolverSpCcsrlsvluHost(
        cusolverSpHandle handle, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer b, 
        float tol, 
        int reorder, 
        Pointer x, 
        Pointer singularity)
    {
        return checkResult(cusolverSpCcsrlsvluHostNative(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity));
    }
    private static native int cusolverSpCcsrlsvluHostNative(
        cusolverSpHandle handle, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer b, 
        float tol, 
        int reorder, 
        Pointer x, 
        Pointer singularity);


    public static int cusolverSpZcsrlsvluHost(
        cusolverSpHandle handle, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer b, 
        double tol, 
        int reorder, 
        Pointer x, 
        Pointer singularity)
    {
        return checkResult(cusolverSpZcsrlsvluHostNative(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity));
    }
    private static native int cusolverSpZcsrlsvluHostNative(
        cusolverSpHandle handle, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer b, 
        double tol, 
        int reorder, 
        Pointer x, 
        Pointer singularity);


    /**
     * <pre>
     * -------- GPU linear solver based on QR factorization
     *       solve A*x = b, A can be singular 
     * [ls] stands for linear solve
     * [v] stands for vector
     * [qr] stands for QR factorization
     * </pre>
     */
    public static int cusolverSpScsrlsvqr(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        Pointer b, 
        float tol, 
        int reorder, 
        Pointer x, 
        Pointer singularity)
    {
        return checkResult(cusolverSpScsrlsvqrNative(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity));
    }
    private static native int cusolverSpScsrlsvqrNative(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        Pointer b, 
        float tol, 
        int reorder, 
        Pointer x, 
        Pointer singularity);


    public static int cusolverSpDcsrlsvqr(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        Pointer b, 
        double tol, 
        int reorder, 
        Pointer x, 
        Pointer singularity)
    {
        return checkResult(cusolverSpDcsrlsvqrNative(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity));
    }
    private static native int cusolverSpDcsrlsvqrNative(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        Pointer b, 
        double tol, 
        int reorder, 
        Pointer x, 
        Pointer singularity);


    public static int cusolverSpCcsrlsvqr(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        Pointer b, 
        float tol, 
        int reorder, 
        Pointer x, 
        Pointer singularity)
    {
        return checkResult(cusolverSpCcsrlsvqrNative(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity));
    }
    private static native int cusolverSpCcsrlsvqrNative(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        Pointer b, 
        float tol, 
        int reorder, 
        Pointer x, 
        Pointer singularity);


    public static int cusolverSpZcsrlsvqr(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        Pointer b, 
        double tol, 
        int reorder, 
        Pointer x, 
        Pointer singularity)
    {
        return checkResult(cusolverSpZcsrlsvqrNative(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity));
    }
    private static native int cusolverSpZcsrlsvqrNative(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        Pointer b, 
        double tol, 
        int reorder, 
        Pointer x, 
        Pointer singularity);


    /**
     * <pre>
     * -------- CPU linear solver based on QR factorization
     *       solve A*x = b, A can be singular 
     * [ls] stands for linear solve
     * [v] stands for vector
     * [qr] stands for QR factorization
     * </pre>
     */
    public static int cusolverSpScsrlsvqrHost(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer b, 
        float tol, 
        int reorder, 
        Pointer x, 
        Pointer singularity)
    {
        return checkResult(cusolverSpScsrlsvqrHostNative(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity));
    }
    private static native int cusolverSpScsrlsvqrHostNative(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer b, 
        float tol, 
        int reorder, 
        Pointer x, 
        Pointer singularity);


    public static int cusolverSpDcsrlsvqrHost(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer b, 
        double tol, 
        int reorder, 
        Pointer x, 
        Pointer singularity)
    {
        return checkResult(cusolverSpDcsrlsvqrHostNative(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity));
    }
    private static native int cusolverSpDcsrlsvqrHostNative(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer b, 
        double tol, 
        int reorder, 
        Pointer x, 
        Pointer singularity);


    public static int cusolverSpCcsrlsvqrHost(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer b, 
        float tol, 
        int reorder, 
        Pointer x, 
        Pointer singularity)
    {
        return checkResult(cusolverSpCcsrlsvqrHostNative(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity));
    }
    private static native int cusolverSpCcsrlsvqrHostNative(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer b, 
        float tol, 
        int reorder, 
        Pointer x, 
        Pointer singularity);


    public static int cusolverSpZcsrlsvqrHost(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer b, 
        double tol, 
        int reorder, 
        Pointer x, 
        Pointer singularity)
    {
        return checkResult(cusolverSpZcsrlsvqrHostNative(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity));
    }
    private static native int cusolverSpZcsrlsvqrHostNative(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer b, 
        double tol, 
        int reorder, 
        Pointer x, 
        Pointer singularity);


    /**
     * <pre>
     * -------- CPU linear solver based on Cholesky factorization
     *       solve A*x = b, A can be singular 
     * [ls] stands for linear solve
     * [v] stands for vector
     * [chol] stands for Cholesky factorization
     *
     * Only works for symmetric positive definite matrix.
     * The upper part of A is ignored.
     * </pre>
     */
    public static int cusolverSpScsrlsvcholHost(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        Pointer b, 
        float tol, 
        int reorder, 
        Pointer x, 
        Pointer singularity)
    {
        return checkResult(cusolverSpScsrlsvcholHostNative(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity));
    }
    private static native int cusolverSpScsrlsvcholHostNative(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        Pointer b, 
        float tol, 
        int reorder, 
        Pointer x, 
        Pointer singularity);


    public static int cusolverSpDcsrlsvcholHost(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        Pointer b, 
        double tol, 
        int reorder, 
        Pointer x, 
        Pointer singularity)
    {
        return checkResult(cusolverSpDcsrlsvcholHostNative(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity));
    }
    private static native int cusolverSpDcsrlsvcholHostNative(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        Pointer b, 
        double tol, 
        int reorder, 
        Pointer x, 
        Pointer singularity);


    public static int cusolverSpCcsrlsvcholHost(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        Pointer b, 
        float tol, 
        int reorder, 
        Pointer x, 
        Pointer singularity)
    {
        return checkResult(cusolverSpCcsrlsvcholHostNative(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity));
    }
    private static native int cusolverSpCcsrlsvcholHostNative(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        Pointer b, 
        float tol, 
        int reorder, 
        Pointer x, 
        Pointer singularity);


    public static int cusolverSpZcsrlsvcholHost(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        Pointer b, 
        double tol, 
        int reorder, 
        Pointer x, 
        Pointer singularity)
    {
        return checkResult(cusolverSpZcsrlsvcholHostNative(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity));
    }
    private static native int cusolverSpZcsrlsvcholHostNative(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        Pointer b, 
        double tol, 
        int reorder, 
        Pointer x, 
        Pointer singularity);


    /**
     * <pre>
     * -------- GPU linear solver based on Cholesky factorization
     *       solve A*x = b, A can be singular 
     * [ls] stands for linear solve
     * [v] stands for vector
     * [chol] stands for Cholesky factorization
     *
     * Only works for symmetric positive definite matrix.
     * The upper part of A is ignored.
     * </pre>
     */
    public static int cusolverSpScsrlsvchol(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        Pointer b, 
        float tol, 
        int reorder, 
        // output
        Pointer x, 
        Pointer singularity)
    {
        return checkResult(cusolverSpScsrlsvcholNative(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity));
    }
    private static native int cusolverSpScsrlsvcholNative(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        Pointer b, 
        float tol, 
        int reorder, 
        // output
        Pointer x, 
        Pointer singularity);


    public static int cusolverSpDcsrlsvchol(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        Pointer b, 
        double tol, 
        int reorder, 
        // output
        Pointer x, 
        Pointer singularity)
    {
        return checkResult(cusolverSpDcsrlsvcholNative(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity));
    }
    private static native int cusolverSpDcsrlsvcholNative(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        Pointer b, 
        double tol, 
        int reorder, 
        // output
        Pointer x, 
        Pointer singularity);


    public static int cusolverSpCcsrlsvchol(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        Pointer b, 
        float tol, 
        int reorder, 
        // output
        Pointer x, 
        Pointer singularity)
    {
        return checkResult(cusolverSpCcsrlsvcholNative(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity));
    }
    private static native int cusolverSpCcsrlsvcholNative(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        Pointer b, 
        float tol, 
        int reorder, 
        // output
        Pointer x, 
        Pointer singularity);


    public static int cusolverSpZcsrlsvchol(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        Pointer b, 
        double tol, 
        int reorder, 
        // output
        Pointer x, 
        Pointer singularity)
    {
        return checkResult(cusolverSpZcsrlsvcholNative(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity));
    }
    private static native int cusolverSpZcsrlsvcholNative(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        Pointer b, 
        double tol, 
        int reorder, 
        // output
        Pointer x, 
        Pointer singularity);


    /**
     * <pre>
     * ----------- CPU least square solver based on QR factorization
     *       solve min|b - A*x| 
     * [lsq] stands for least square
     * [v] stands for vector
     * [qr] stands for QR factorization
     * </pre>
     */
    public static int cusolverSpScsrlsqvqrHost(
        cusolverSpHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer b, 
        float tol, 
        Pointer rankA, 
        Pointer x, 
        Pointer p, 
        Pointer min_norm)
    {
        return checkResult(cusolverSpScsrlsqvqrHostNative(handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, rankA, x, p, min_norm));
    }
    private static native int cusolverSpScsrlsqvqrHostNative(
        cusolverSpHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer b, 
        float tol, 
        Pointer rankA, 
        Pointer x, 
        Pointer p, 
        Pointer min_norm);


    public static int cusolverSpDcsrlsqvqrHost(
        cusolverSpHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer b, 
        double tol, 
        Pointer rankA, 
        Pointer x, 
        Pointer p, 
        Pointer min_norm)
    {
        return checkResult(cusolverSpDcsrlsqvqrHostNative(handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, rankA, x, p, min_norm));
    }
    private static native int cusolverSpDcsrlsqvqrHostNative(
        cusolverSpHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer b, 
        double tol, 
        Pointer rankA, 
        Pointer x, 
        Pointer p, 
        Pointer min_norm);


    public static int cusolverSpCcsrlsqvqrHost(
        cusolverSpHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer b, 
        float tol, 
        Pointer rankA, 
        Pointer x, 
        Pointer p, 
        Pointer min_norm)
    {
        return checkResult(cusolverSpCcsrlsqvqrHostNative(handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, rankA, x, p, min_norm));
    }
    private static native int cusolverSpCcsrlsqvqrHostNative(
        cusolverSpHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer b, 
        float tol, 
        Pointer rankA, 
        Pointer x, 
        Pointer p, 
        Pointer min_norm);


    public static int cusolverSpZcsrlsqvqrHost(
        cusolverSpHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer b, 
        double tol, 
        Pointer rankA, 
        Pointer x, 
        Pointer p, 
        Pointer min_norm)
    {
        return checkResult(cusolverSpZcsrlsqvqrHostNative(handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, rankA, x, p, min_norm));
    }
    private static native int cusolverSpZcsrlsqvqrHostNative(
        cusolverSpHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer b, 
        double tol, 
        Pointer rankA, 
        Pointer x, 
        Pointer p, 
        Pointer min_norm);


    /**
     * <pre>
     * --------- CPU eigenvalue solver based on shift inverse
     *      solve A*x = lambda * x 
     *   where lambda is the eigenvalue nearest mu0.
     * [eig] stands for eigenvalue solver
     * [si] stands for shift-inverse
     * </pre>
     */
    public static int cusolverSpScsreigvsiHost(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        float mu0, 
        Pointer x0, 
        int maxite, 
        float tol, 
        Pointer mu, 
        Pointer x)
    {
        return checkResult(cusolverSpScsreigvsiHostNative(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, tol, mu, x));
    }
    private static native int cusolverSpScsreigvsiHostNative(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        float mu0, 
        Pointer x0, 
        int maxite, 
        float tol, 
        Pointer mu, 
        Pointer x);


    public static int cusolverSpDcsreigvsiHost(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        double mu0, 
        Pointer x0, 
        int maxite, 
        double tol, 
        Pointer mu, 
        Pointer x)
    {
        return checkResult(cusolverSpDcsreigvsiHostNative(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, tol, mu, x));
    }
    private static native int cusolverSpDcsreigvsiHostNative(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        double mu0, 
        Pointer x0, 
        int maxite, 
        double tol, 
        Pointer mu, 
        Pointer x);


    public static int cusolverSpCcsreigvsiHost(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cuComplex mu0, 
        Pointer x0, 
        int maxite, 
        float tol, 
        Pointer mu, 
        Pointer x)
    {
        return checkResult(cusolverSpCcsreigvsiHostNative(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, tol, mu, x));
    }
    private static native int cusolverSpCcsreigvsiHostNative(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cuComplex mu0, 
        Pointer x0, 
        int maxite, 
        float tol, 
        Pointer mu, 
        Pointer x);


    public static int cusolverSpZcsreigvsiHost(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cuDoubleComplex mu0, 
        Pointer x0, 
        int maxite, 
        double tol, 
        Pointer mu, 
        Pointer x)
    {
        return checkResult(cusolverSpZcsreigvsiHostNative(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, tol, mu, x));
    }
    private static native int cusolverSpZcsreigvsiHostNative(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cuDoubleComplex mu0, 
        Pointer x0, 
        int maxite, 
        double tol, 
        Pointer mu, 
        Pointer x);


    /**
     * <pre>
     * --------- GPU eigenvalue solver based on shift inverse
     *      solve A*x = lambda * x 
     *   where lambda is the eigenvalue nearest mu0.
     * [eig] stands for eigenvalue solver
     * [si] stands for shift-inverse
     * </pre>
     */
    public static int cusolverSpScsreigvsi(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        float mu0, 
        Pointer x0, 
        int maxite, 
        float eps, 
        Pointer mu, 
        Pointer x)
    {
        return checkResult(cusolverSpScsreigvsiNative(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, eps, mu, x));
    }
    private static native int cusolverSpScsreigvsiNative(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        float mu0, 
        Pointer x0, 
        int maxite, 
        float eps, 
        Pointer mu, 
        Pointer x);


    public static int cusolverSpDcsreigvsi(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        double mu0, 
        Pointer x0, 
        int maxite, 
        double eps, 
        Pointer mu, 
        Pointer x)
    {
        return checkResult(cusolverSpDcsreigvsiNative(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, eps, mu, x));
    }
    private static native int cusolverSpDcsreigvsiNative(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        double mu0, 
        Pointer x0, 
        int maxite, 
        double eps, 
        Pointer mu, 
        Pointer x);


    public static int cusolverSpCcsreigvsi(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cuComplex mu0, 
        Pointer x0, 
        int maxite, 
        float eps, 
        Pointer mu, 
        Pointer x)
    {
        return checkResult(cusolverSpCcsreigvsiNative(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, eps, mu, x));
    }
    private static native int cusolverSpCcsreigvsiNative(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cuComplex mu0, 
        Pointer x0, 
        int maxite, 
        float eps, 
        Pointer mu, 
        Pointer x);


    public static int cusolverSpZcsreigvsi(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cuDoubleComplex mu0, 
        Pointer x0, 
        int maxite, 
        double eps, 
        Pointer mu, 
        Pointer x)
    {
        return checkResult(cusolverSpZcsreigvsiNative(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, eps, mu, x));
    }
    private static native int cusolverSpZcsreigvsiNative(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cuDoubleComplex mu0, 
        Pointer x0, 
        int maxite, 
        double eps, 
        Pointer mu, 
        Pointer x);


    // ----------- enclosed eigenvalues
    public static int cusolverSpScsreigsHost(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cuComplex left_bottom_corner, 
        cuComplex right_upper_corner, 
        Pointer num_eigs)
    {
        return checkResult(cusolverSpScsreigsHostNative(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, left_bottom_corner, right_upper_corner, num_eigs));
    }
    private static native int cusolverSpScsreigsHostNative(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cuComplex left_bottom_corner, 
        cuComplex right_upper_corner, 
        Pointer num_eigs);


    public static int cusolverSpDcsreigsHost(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cuDoubleComplex left_bottom_corner, 
        cuDoubleComplex right_upper_corner, 
        Pointer num_eigs)
    {
        return checkResult(cusolverSpDcsreigsHostNative(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, left_bottom_corner, right_upper_corner, num_eigs));
    }
    private static native int cusolverSpDcsreigsHostNative(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cuDoubleComplex left_bottom_corner, 
        cuDoubleComplex right_upper_corner, 
        Pointer num_eigs);


    public static int cusolverSpCcsreigsHost(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cuComplex left_bottom_corner, 
        cuComplex right_upper_corner, 
        Pointer num_eigs)
    {
        return checkResult(cusolverSpCcsreigsHostNative(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, left_bottom_corner, right_upper_corner, num_eigs));
    }
    private static native int cusolverSpCcsreigsHostNative(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cuComplex left_bottom_corner, 
        cuComplex right_upper_corner, 
        Pointer num_eigs);


    public static int cusolverSpZcsreigsHost(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cuDoubleComplex left_bottom_corner, 
        cuDoubleComplex right_upper_corner, 
        Pointer num_eigs)
    {
        return checkResult(cusolverSpZcsreigsHostNative(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, left_bottom_corner, right_upper_corner, num_eigs));
    }
    private static native int cusolverSpZcsreigsHostNative(
        cusolverSpHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cuDoubleComplex left_bottom_corner, 
        cuDoubleComplex right_upper_corner, 
        Pointer num_eigs);


    /**
     * <pre>
     * --------- CPU symrcm
     *   Symmetric reverse Cuthill McKee permutation         
     *
     * </pre>
     */
    public static int cusolverSpXcsrsymrcmHost(
        cusolverSpHandle handle, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer p)
    {
        return checkResult(cusolverSpXcsrsymrcmHostNative(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, p));
    }
    private static native int cusolverSpXcsrsymrcmHostNative(
        cusolverSpHandle handle, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer p);

    /**
     * <pre>
     * --------- CPU symmdq
     *   Symmetric minimum degree algorithm based on quotient graph
     *
     * </pre>
     */
    public static int cusolverSpXcsrsymmdqHost(
        cusolverSpHandle handle, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer p)
    {
        return checkResult(cusolverSpXcsrsymmdqHostNative(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, p));
    }
    private static native int cusolverSpXcsrsymmdqHostNative(
        cusolverSpHandle handle, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer p);


    /**
     * <pre>
     * --------- CPU symmdq
     *   Symmetric Approximate minimum degree algorithm based on quotient graph
     *
     * </pre>
     */
    public static int cusolverSpXcsrsymamdHost(
        cusolverSpHandle handle, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer p)
    {
        return checkResult(cusolverSpXcsrsymamdHostNative(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, p));
    }
    private static native int cusolverSpXcsrsymamdHostNative(
        cusolverSpHandle handle, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer p);

    /**
     * <pre>
     * --------- CPU permuation
     *   P*A*Q^T        
     *
     * </pre>
     */
    public static int cusolverSpXcsrperm_bufferSizeHost(
        cusolverSpHandle handle, 
        int m, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer p, 
        Pointer q, 
        long[] bufferSizeInBytes)
    {
        return checkResult(cusolverSpXcsrperm_bufferSizeHostNative(handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, p, q, bufferSizeInBytes));
    }
    private static native int cusolverSpXcsrperm_bufferSizeHostNative(
        cusolverSpHandle handle, 
        int m, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer p, 
        Pointer q, 
        long[] bufferSizeInBytes);


    public static int cusolverSpXcsrpermHost(
        cusolverSpHandle handle, 
        int m, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer p, 
        Pointer q, 
        Pointer map, 
        Pointer pBuffer)
    {
        return checkResult(cusolverSpXcsrpermHostNative(handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, p, q, map, pBuffer));
    }
    private static native int cusolverSpXcsrpermHostNative(
        cusolverSpHandle handle, 
        int m, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer p, 
        Pointer q, 
        Pointer map, 
        Pointer pBuffer);


    /**
     * <pre>
     *  Low-level API: Batched QR
     *
     * </pre>
     */
    public static int cusolverSpCreateCsrqrInfo(
        csrqrInfo info)
    {
        return checkResult(cusolverSpCreateCsrqrInfoNative(info));
    }
    private static native int cusolverSpCreateCsrqrInfoNative(
        csrqrInfo info);


    public static int cusolverSpDestroyCsrqrInfo(
        csrqrInfo info)
    {
        return checkResult(cusolverSpDestroyCsrqrInfoNative(info));
    }
    private static native int cusolverSpDestroyCsrqrInfoNative(
        csrqrInfo info);


    public static int cusolverSpXcsrqrAnalysisBatched(
        cusolverSpHandle handle, 
        int m, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrqrInfo info)
    {
        return checkResult(cusolverSpXcsrqrAnalysisBatchedNative(handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, info));
    }
    private static native int cusolverSpXcsrqrAnalysisBatchedNative(
        cusolverSpHandle handle, 
        int m, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrqrInfo info);


    public static int cusolverSpScsrqrBufferInfoBatched(
        cusolverSpHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        int batchSize, 
        csrqrInfo info, 
        long[] internalDataInBytes, 
        long[] workspaceInBytes)
    {
        return checkResult(cusolverSpScsrqrBufferInfoBatchedNative(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, batchSize, info, internalDataInBytes, workspaceInBytes));
    }
    private static native int cusolverSpScsrqrBufferInfoBatchedNative(
        cusolverSpHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        int batchSize, 
        csrqrInfo info, 
        long[] internalDataInBytes, 
        long[] workspaceInBytes);


    public static int cusolverSpDcsrqrBufferInfoBatched(
        cusolverSpHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        int batchSize, 
        csrqrInfo info, 
        long[] internalDataInBytes, 
        long[] workspaceInBytes)
    {
        return checkResult(cusolverSpDcsrqrBufferInfoBatchedNative(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, batchSize, info, internalDataInBytes, workspaceInBytes));
    }
    private static native int cusolverSpDcsrqrBufferInfoBatchedNative(
        cusolverSpHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        int batchSize, 
        csrqrInfo info, 
        long[] internalDataInBytes, 
        long[] workspaceInBytes);


    public static int cusolverSpCcsrqrBufferInfoBatched(
        cusolverSpHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        int batchSize, 
        csrqrInfo info, 
        long[] internalDataInBytes, 
        long[] workspaceInBytes)
    {
        return checkResult(cusolverSpCcsrqrBufferInfoBatchedNative(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, batchSize, info, internalDataInBytes, workspaceInBytes));
    }
    private static native int cusolverSpCcsrqrBufferInfoBatchedNative(
        cusolverSpHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        int batchSize, 
        csrqrInfo info, 
        long[] internalDataInBytes, 
        long[] workspaceInBytes);


    public static int cusolverSpZcsrqrBufferInfoBatched(
        cusolverSpHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        int batchSize, 
        csrqrInfo info, 
        long[] internalDataInBytes, 
        long[] workspaceInBytes)
    {
        return checkResult(cusolverSpZcsrqrBufferInfoBatchedNative(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, batchSize, info, internalDataInBytes, workspaceInBytes));
    }
    private static native int cusolverSpZcsrqrBufferInfoBatchedNative(
        cusolverSpHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        int batchSize, 
        csrqrInfo info, 
        long[] internalDataInBytes, 
        long[] workspaceInBytes);


    public static int cusolverSpScsrqrsvBatched(
        cusolverSpHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer b, 
        Pointer x, 
        int batchSize, 
        csrqrInfo info, 
        Pointer pBuffer)
    {
        return checkResult(cusolverSpScsrqrsvBatchedNative(handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, x, batchSize, info, pBuffer));
    }
    private static native int cusolverSpScsrqrsvBatchedNative(
        cusolverSpHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer b, 
        Pointer x, 
        int batchSize, 
        csrqrInfo info, 
        Pointer pBuffer);


    public static int cusolverSpDcsrqrsvBatched(
        cusolverSpHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer b, 
        Pointer x, 
        int batchSize, 
        csrqrInfo info, 
        Pointer pBuffer)
    {
        return checkResult(cusolverSpDcsrqrsvBatchedNative(handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, x, batchSize, info, pBuffer));
    }
    private static native int cusolverSpDcsrqrsvBatchedNative(
        cusolverSpHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer b, 
        Pointer x, 
        int batchSize, 
        csrqrInfo info, 
        Pointer pBuffer);


    public static int cusolverSpCcsrqrsvBatched(
        cusolverSpHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer b, 
        Pointer x, 
        int batchSize, 
        csrqrInfo info, 
        Pointer pBuffer)
    {
        return checkResult(cusolverSpCcsrqrsvBatchedNative(handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, x, batchSize, info, pBuffer));
    }
    private static native int cusolverSpCcsrqrsvBatchedNative(
        cusolverSpHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer b, 
        Pointer x, 
        int batchSize, 
        csrqrInfo info, 
        Pointer pBuffer);


    public static int cusolverSpZcsrqrsvBatched(
        cusolverSpHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer b, 
        Pointer x, 
        int batchSize, 
        csrqrInfo info, 
        Pointer pBuffer)
    {
        return checkResult(cusolverSpZcsrqrsvBatchedNative(handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, x, batchSize, info, pBuffer));
    }
    private static native int cusolverSpZcsrqrsvBatchedNative(
        cusolverSpHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer b, 
        Pointer x, 
        int batchSize, 
        csrqrInfo info, 
        Pointer pBuffer);

}
