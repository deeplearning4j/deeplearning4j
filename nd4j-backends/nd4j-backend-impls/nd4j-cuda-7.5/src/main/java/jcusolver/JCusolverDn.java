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
import jcuda.runtime.cudaStream_t;

/**
 * Java bindings for CUSOLVER, the NVIDIA CUDA solver library. <br />
 * <br />
 * The documentation is taken from the CUSOLVER header files.
 */
public class JCusolverDn
{
    /* Private constructor to prevent instantiation */
    private JCusolverDn()
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
    
    public static int cusolverDnCreate(
        cusolverDnHandle handle)
    {
        return checkResult(cusolverDnCreateNative(handle));
    }
    private static native int cusolverDnCreateNative(
        cusolverDnHandle handle);


    public static int cusolverDnDestroy(
        cusolverDnHandle handle)
    {
        return checkResult(cusolverDnDestroyNative(handle));
    }
    private static native int cusolverDnDestroyNative(
        cusolverDnHandle handle);


    public static int cusolverDnSetStream(
        cusolverDnHandle handle, 
        cudaStream_t streamId)
    {
        return checkResult(cusolverDnSetStreamNative(handle, streamId));
    }
    private static native int cusolverDnSetStreamNative(
        cusolverDnHandle handle, 
        cudaStream_t streamId);


    public static int cusolverDnGetStream(
        cusolverDnHandle handle, 
        cudaStream_t streamId)
    {
        return checkResult(cusolverDnGetStreamNative(handle, streamId));
    }
    private static native int cusolverDnGetStreamNative(
        cusolverDnHandle handle, 
        cudaStream_t streamId);


    /** Cholesky factorization and its solver */
    public static int cusolverDnSpotrf_bufferSize(
        cusolverDnHandle handle, 
        int uplo, 
        int n, 
        Pointer A, 
        int lda, 
        int[] Lwork)
    {
        return checkResult(cusolverDnSpotrf_bufferSizeNative(handle, uplo, n, A, lda, Lwork));
    }
    private static native int cusolverDnSpotrf_bufferSizeNative(
        cusolverDnHandle handle, 
        int uplo, 
        int n, 
        Pointer A, 
        int lda, 
        int[] Lwork);


    public static int cusolverDnDpotrf_bufferSize(
        cusolverDnHandle handle, 
        int uplo, 
        int n, 
        Pointer A, 
        int lda, 
        int[] Lwork)
    {
        return checkResult(cusolverDnDpotrf_bufferSizeNative(handle, uplo, n, A, lda, Lwork));
    }
    private static native int cusolverDnDpotrf_bufferSizeNative(
        cusolverDnHandle handle, 
        int uplo, 
        int n, 
        Pointer A, 
        int lda, 
        int[] Lwork);


    public static int cusolverDnCpotrf_bufferSize(
        cusolverDnHandle handle, 
        int uplo, 
        int n, 
        Pointer A, 
        int lda, 
        int[] Lwork)
    {
        return checkResult(cusolverDnCpotrf_bufferSizeNative(handle, uplo, n, A, lda, Lwork));
    }
    private static native int cusolverDnCpotrf_bufferSizeNative(
        cusolverDnHandle handle, 
        int uplo, 
        int n, 
        Pointer A, 
        int lda, 
        int[] Lwork);


    public static int cusolverDnZpotrf_bufferSize(
        cusolverDnHandle handle, 
        int uplo, 
        int n, 
        Pointer A, 
        int lda, 
        int[] Lwork)
    {
        return checkResult(cusolverDnZpotrf_bufferSizeNative(handle, uplo, n, A, lda, Lwork));
    }
    private static native int cusolverDnZpotrf_bufferSizeNative(
        cusolverDnHandle handle, 
        int uplo, 
        int n, 
        Pointer A, 
        int lda, 
        int[] Lwork);


    public static int cusolverDnSpotrf(
        cusolverDnHandle handle, 
        int uplo, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer Workspace, 
        int Lwork, 
        Pointer devInfo)
    {
        return checkResult(cusolverDnSpotrfNative(handle, uplo, n, A, lda, Workspace, Lwork, devInfo));
    }
    private static native int cusolverDnSpotrfNative(
        cusolverDnHandle handle, 
        int uplo, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer Workspace, 
        int Lwork, 
        Pointer devInfo);


    public static int cusolverDnDpotrf(
        cusolverDnHandle handle, 
        int uplo, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer Workspace, 
        int Lwork, 
        Pointer devInfo)
    {
        return checkResult(cusolverDnDpotrfNative(handle, uplo, n, A, lda, Workspace, Lwork, devInfo));
    }
    private static native int cusolverDnDpotrfNative(
        cusolverDnHandle handle, 
        int uplo, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer Workspace, 
        int Lwork, 
        Pointer devInfo);


    public static int cusolverDnCpotrf(
        cusolverDnHandle handle, 
        int uplo, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer Workspace, 
        int Lwork, 
        Pointer devInfo)
    {
        return checkResult(cusolverDnCpotrfNative(handle, uplo, n, A, lda, Workspace, Lwork, devInfo));
    }
    private static native int cusolverDnCpotrfNative(
        cusolverDnHandle handle, 
        int uplo, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer Workspace, 
        int Lwork, 
        Pointer devInfo);


    public static int cusolverDnZpotrf(
        cusolverDnHandle handle, 
        int uplo, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer Workspace, 
        int Lwork, 
        Pointer devInfo)
    {
        return checkResult(cusolverDnZpotrfNative(handle, uplo, n, A, lda, Workspace, Lwork, devInfo));
    }
    private static native int cusolverDnZpotrfNative(
        cusolverDnHandle handle, 
        int uplo, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer Workspace, 
        int Lwork, 
        Pointer devInfo);


    public static int cusolverDnSpotrs(
        cusolverDnHandle handle, 
        int uplo, 
        int n, 
        int nrhs, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer devInfo)
    {
        return checkResult(cusolverDnSpotrsNative(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo));
    }
    private static native int cusolverDnSpotrsNative(
        cusolverDnHandle handle, 
        int uplo, 
        int n, 
        int nrhs, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer devInfo);


    public static int cusolverDnDpotrs(
        cusolverDnHandle handle, 
        int uplo, 
        int n, 
        int nrhs, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer devInfo)
    {
        return checkResult(cusolverDnDpotrsNative(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo));
    }
    private static native int cusolverDnDpotrsNative(
        cusolverDnHandle handle, 
        int uplo, 
        int n, 
        int nrhs, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer devInfo);


    public static int cusolverDnCpotrs(
        cusolverDnHandle handle, 
        int uplo, 
        int n, 
        int nrhs, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer devInfo)
    {
        return checkResult(cusolverDnCpotrsNative(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo));
    }
    private static native int cusolverDnCpotrsNative(
        cusolverDnHandle handle, 
        int uplo, 
        int n, 
        int nrhs, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer devInfo);


    public static int cusolverDnZpotrs(
        cusolverDnHandle handle, 
        int uplo, 
        int n, 
        int nrhs, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer devInfo)
    {
        return checkResult(cusolverDnZpotrsNative(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo));
    }
    private static native int cusolverDnZpotrsNative(
        cusolverDnHandle handle, 
        int uplo, 
        int n, 
        int nrhs, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer devInfo);


    /** LU Factorization */
    public static int cusolverDnSgetrf_bufferSize(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        int[] Lwork)
    {
        return checkResult(cusolverDnSgetrf_bufferSizeNative(handle, m, n, A, lda, Lwork));
    }
    private static native int cusolverDnSgetrf_bufferSizeNative(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        int[] Lwork);


    public static int cusolverDnDgetrf_bufferSize(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        int[] Lwork)
    {
        return checkResult(cusolverDnDgetrf_bufferSizeNative(handle, m, n, A, lda, Lwork));
    }
    private static native int cusolverDnDgetrf_bufferSizeNative(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        int[] Lwork);


    public static int cusolverDnCgetrf_bufferSize(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        int[] Lwork)
    {
        return checkResult(cusolverDnCgetrf_bufferSizeNative(handle, m, n, A, lda, Lwork));
    }
    private static native int cusolverDnCgetrf_bufferSizeNative(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        int[] Lwork);


    public static int cusolverDnZgetrf_bufferSize(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        int[] Lwork)
    {
        return checkResult(cusolverDnZgetrf_bufferSizeNative(handle, m, n, A, lda, Lwork));
    }
    private static native int cusolverDnZgetrf_bufferSizeNative(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        int[] Lwork);


    public static int cusolverDnSgetrf(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer Workspace, 
        Pointer devIpiv, 
        Pointer devInfo)
    {
        return checkResult(cusolverDnSgetrfNative(handle, m, n, A, lda, Workspace, devIpiv, devInfo));
    }
    private static native int cusolverDnSgetrfNative(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer Workspace, 
        Pointer devIpiv, 
        Pointer devInfo);


    public static int cusolverDnDgetrf(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer Workspace, 
        Pointer devIpiv, 
        Pointer devInfo)
    {
        return checkResult(cusolverDnDgetrfNative(handle, m, n, A, lda, Workspace, devIpiv, devInfo));
    }
    private static native int cusolverDnDgetrfNative(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer Workspace, 
        Pointer devIpiv, 
        Pointer devInfo);


    public static int cusolverDnCgetrf(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer Workspace, 
        Pointer devIpiv, 
        Pointer devInfo)
    {
        return checkResult(cusolverDnCgetrfNative(handle, m, n, A, lda, Workspace, devIpiv, devInfo));
    }
    private static native int cusolverDnCgetrfNative(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer Workspace, 
        Pointer devIpiv, 
        Pointer devInfo);


    public static int cusolverDnZgetrf(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer Workspace, 
        Pointer devIpiv, 
        Pointer devInfo)
    {
        return checkResult(cusolverDnZgetrfNative(handle, m, n, A, lda, Workspace, devIpiv, devInfo));
    }
    private static native int cusolverDnZgetrfNative(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer Workspace, 
        Pointer devIpiv, 
        Pointer devInfo);


    /** Row pivoting */
    public static int cusolverDnSlaswp(
        cusolverDnHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        int k1, 
        int k2, 
        Pointer devIpiv, 
        int incx)
    {
        return checkResult(cusolverDnSlaswpNative(handle, n, A, lda, k1, k2, devIpiv, incx));
    }
    private static native int cusolverDnSlaswpNative(
        cusolverDnHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        int k1, 
        int k2, 
        Pointer devIpiv, 
        int incx);


    public static int cusolverDnDlaswp(
        cusolverDnHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        int k1, 
        int k2, 
        Pointer devIpiv, 
        int incx)
    {
        return checkResult(cusolverDnDlaswpNative(handle, n, A, lda, k1, k2, devIpiv, incx));
    }
    private static native int cusolverDnDlaswpNative(
        cusolverDnHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        int k1, 
        int k2, 
        Pointer devIpiv, 
        int incx);


    public static int cusolverDnClaswp(
        cusolverDnHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        int k1, 
        int k2, 
        Pointer devIpiv, 
        int incx)
    {
        return checkResult(cusolverDnClaswpNative(handle, n, A, lda, k1, k2, devIpiv, incx));
    }
    private static native int cusolverDnClaswpNative(
        cusolverDnHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        int k1, 
        int k2, 
        Pointer devIpiv, 
        int incx);


    public static int cusolverDnZlaswp(
        cusolverDnHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        int k1, 
        int k2, 
        Pointer devIpiv, 
        int incx)
    {
        return checkResult(cusolverDnZlaswpNative(handle, n, A, lda, k1, k2, devIpiv, incx));
    }
    private static native int cusolverDnZlaswpNative(
        cusolverDnHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        int k1, 
        int k2, 
        Pointer devIpiv, 
        int incx);


    /** LU solve */
    public static int cusolverDnSgetrs(
        cusolverDnHandle handle, 
        int trans, 
        int n, 
        int nrhs, 
        Pointer A, 
        int lda, 
        Pointer devIpiv, 
        Pointer B, 
        int ldb, 
        Pointer devInfo)
    {
        return checkResult(cusolverDnSgetrsNative(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo));
    }
    private static native int cusolverDnSgetrsNative(
        cusolverDnHandle handle, 
        int trans, 
        int n, 
        int nrhs, 
        Pointer A, 
        int lda, 
        Pointer devIpiv, 
        Pointer B, 
        int ldb, 
        Pointer devInfo);


    public static int cusolverDnDgetrs(
        cusolverDnHandle handle, 
        int trans, 
        int n, 
        int nrhs, 
        Pointer A, 
        int lda, 
        Pointer devIpiv, 
        Pointer B, 
        int ldb, 
        Pointer devInfo)
    {
        return checkResult(cusolverDnDgetrsNative(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo));
    }
    private static native int cusolverDnDgetrsNative(
        cusolverDnHandle handle, 
        int trans, 
        int n, 
        int nrhs, 
        Pointer A, 
        int lda, 
        Pointer devIpiv, 
        Pointer B, 
        int ldb, 
        Pointer devInfo);


    public static int cusolverDnCgetrs(
        cusolverDnHandle handle, 
        int trans, 
        int n, 
        int nrhs, 
        Pointer A, 
        int lda, 
        Pointer devIpiv, 
        Pointer B, 
        int ldb, 
        Pointer devInfo)
    {
        return checkResult(cusolverDnCgetrsNative(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo));
    }
    private static native int cusolverDnCgetrsNative(
        cusolverDnHandle handle, 
        int trans, 
        int n, 
        int nrhs, 
        Pointer A, 
        int lda, 
        Pointer devIpiv, 
        Pointer B, 
        int ldb, 
        Pointer devInfo);


    public static int cusolverDnZgetrs(
        cusolverDnHandle handle, 
        int trans, 
        int n, 
        int nrhs, 
        Pointer A, 
        int lda, 
        Pointer devIpiv, 
        Pointer B, 
        int ldb, 
        Pointer devInfo)
    {
        return checkResult(cusolverDnZgetrsNative(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo));
    }
    private static native int cusolverDnZgetrsNative(
        cusolverDnHandle handle, 
        int trans, 
        int n, 
        int nrhs, 
        Pointer A, 
        int lda, 
        Pointer devIpiv, 
        Pointer B, 
        int ldb, 
        Pointer devInfo);


    /** QR factorization */
    public static int cusolverDnSgeqrf(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer TAU, 
        Pointer Workspace, 
        int Lwork, 
        Pointer devInfo)
    {
        return checkResult(cusolverDnSgeqrfNative(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo));
    }
    private static native int cusolverDnSgeqrfNative(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer TAU, 
        Pointer Workspace, 
        int Lwork, 
        Pointer devInfo);


    public static int cusolverDnDgeqrf(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer TAU, 
        Pointer Workspace, 
        int Lwork, 
        Pointer devInfo)
    {
        return checkResult(cusolverDnDgeqrfNative(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo));
    }
    private static native int cusolverDnDgeqrfNative(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer TAU, 
        Pointer Workspace, 
        int Lwork, 
        Pointer devInfo);


    public static int cusolverDnCgeqrf(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer TAU, 
        Pointer Workspace, 
        int Lwork, 
        Pointer devInfo)
    {
        return checkResult(cusolverDnCgeqrfNative(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo));
    }
    private static native int cusolverDnCgeqrfNative(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer TAU, 
        Pointer Workspace, 
        int Lwork, 
        Pointer devInfo);


    public static int cusolverDnZgeqrf(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer TAU, 
        Pointer Workspace, 
        int Lwork, 
        Pointer devInfo)
    {
        return checkResult(cusolverDnZgeqrfNative(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo));
    }
    private static native int cusolverDnZgeqrfNative(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer TAU, 
        Pointer Workspace, 
        int Lwork, 
        Pointer devInfo);


    public static int cusolverDnSormqr(
        cusolverDnHandle handle, 
        int side, 
        int trans, 
        int m, 
        int n, 
        int k, 
        Pointer A, 
        int lda, 
        Pointer tau, 
        Pointer C, 
        int ldc, 
        Pointer work, 
        int lwork, 
        Pointer devInfo)
    {
        return checkResult(cusolverDnSormqrNative(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo));
    }
    private static native int cusolverDnSormqrNative(
        cusolverDnHandle handle, 
        int side, 
        int trans, 
        int m, 
        int n, 
        int k, 
        Pointer A, 
        int lda, 
        Pointer tau, 
        Pointer C, 
        int ldc, 
        Pointer work, 
        int lwork, 
        Pointer devInfo);


    public static int cusolverDnDormqr(
        cusolverDnHandle handle, 
        int side, 
        int trans, 
        int m, 
        int n, 
        int k, 
        Pointer A, 
        int lda, 
        Pointer tau, 
        Pointer C, 
        int ldc, 
        Pointer work, 
        int lwork, 
        Pointer devInfo)
    {
        return checkResult(cusolverDnDormqrNative(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo));
    }
    private static native int cusolverDnDormqrNative(
        cusolverDnHandle handle, 
        int side, 
        int trans, 
        int m, 
        int n, 
        int k, 
        Pointer A, 
        int lda, 
        Pointer tau, 
        Pointer C, 
        int ldc, 
        Pointer work, 
        int lwork, 
        Pointer devInfo);


    public static int cusolverDnCunmqr(
        cusolverDnHandle handle, 
        int side, 
        int trans, 
        int m, 
        int n, 
        int k, 
        Pointer A, 
        int lda, 
        Pointer tau, 
        Pointer C, 
        int ldc, 
        Pointer work, 
        int lwork, 
        Pointer devInfo)
    {
        return checkResult(cusolverDnCunmqrNative(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo));
    }
    private static native int cusolverDnCunmqrNative(
        cusolverDnHandle handle, 
        int side, 
        int trans, 
        int m, 
        int n, 
        int k, 
        Pointer A, 
        int lda, 
        Pointer tau, 
        Pointer C, 
        int ldc, 
        Pointer work, 
        int lwork, 
        Pointer devInfo);


    public static int cusolverDnZunmqr(
        cusolverDnHandle handle, 
        int side, 
        int trans, 
        int m, 
        int n, 
        int k, 
        Pointer A, 
        int lda, 
        Pointer tau, 
        Pointer C, 
        int ldc, 
        Pointer work, 
        int lwork, 
        Pointer devInfo)
    {
        return checkResult(cusolverDnZunmqrNative(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo));
    }
    private static native int cusolverDnZunmqrNative(
        cusolverDnHandle handle, 
        int side, 
        int trans, 
        int m, 
        int n, 
        int k, 
        Pointer A, 
        int lda, 
        Pointer tau, 
        Pointer C, 
        int ldc, 
        Pointer work, 
        int lwork, 
        Pointer devInfo);


    /** QR factorization workspace query */
    public static int cusolverDnSgeqrf_bufferSize(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        int[] Lwork)
    {
        return checkResult(cusolverDnSgeqrf_bufferSizeNative(handle, m, n, A, lda, Lwork));
    }
    private static native int cusolverDnSgeqrf_bufferSizeNative(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        int[] Lwork);


    public static int cusolverDnDgeqrf_bufferSize(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        int[] Lwork)
    {
        return checkResult(cusolverDnDgeqrf_bufferSizeNative(handle, m, n, A, lda, Lwork));
    }
    private static native int cusolverDnDgeqrf_bufferSizeNative(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        int[] Lwork);


    public static int cusolverDnCgeqrf_bufferSize(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        int[] Lwork)
    {
        return checkResult(cusolverDnCgeqrf_bufferSizeNative(handle, m, n, A, lda, Lwork));
    }
    private static native int cusolverDnCgeqrf_bufferSizeNative(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        int[] Lwork);


    public static int cusolverDnZgeqrf_bufferSize(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        int[] Lwork)
    {
        return checkResult(cusolverDnZgeqrf_bufferSizeNative(handle, m, n, A, lda, Lwork));
    }
    private static native int cusolverDnZgeqrf_bufferSizeNative(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        int[] Lwork);


    /** bidiagonal */
    public static int cusolverDnSgebrd(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer D, 
        Pointer E, 
        Pointer TAUQ, 
        Pointer TAUP, 
        Pointer Work, 
        int Lwork, 
        Pointer devInfo)
    {
        return checkResult(cusolverDnSgebrdNative(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo));
    }
    private static native int cusolverDnSgebrdNative(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer D, 
        Pointer E, 
        Pointer TAUQ, 
        Pointer TAUP, 
        Pointer Work, 
        int Lwork, 
        Pointer devInfo);


    public static int cusolverDnDgebrd(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer D, 
        Pointer E, 
        Pointer TAUQ, 
        Pointer TAUP, 
        Pointer Work, 
        int Lwork, 
        Pointer devInfo)
    {
        return checkResult(cusolverDnDgebrdNative(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo));
    }
    private static native int cusolverDnDgebrdNative(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer D, 
        Pointer E, 
        Pointer TAUQ, 
        Pointer TAUP, 
        Pointer Work, 
        int Lwork, 
        Pointer devInfo);


    public static int cusolverDnCgebrd(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer D, 
        Pointer E, 
        Pointer TAUQ, 
        Pointer TAUP, 
        Pointer Work, 
        int Lwork, 
        Pointer devInfo)
    {
        return checkResult(cusolverDnCgebrdNative(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo));
    }
    private static native int cusolverDnCgebrdNative(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer D, 
        Pointer E, 
        Pointer TAUQ, 
        Pointer TAUP, 
        Pointer Work, 
        int Lwork, 
        Pointer devInfo);


    public static int cusolverDnZgebrd(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer D, 
        Pointer E, 
        Pointer TAUQ, 
        Pointer TAUP, 
        Pointer Work, 
        int Lwork, 
        Pointer devInfo)
    {
        return checkResult(cusolverDnZgebrdNative(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo));
    }
    private static native int cusolverDnZgebrdNative(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer D, 
        Pointer E, 
        Pointer TAUQ, 
        Pointer TAUP, 
        Pointer Work, 
        int Lwork, 
        Pointer devInfo);


    public static int cusolverDnSsytrd(
        cusolverDnHandle handle, 
        char uplo, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer D, 
        Pointer E, 
        Pointer tau, 
        Pointer Work, 
        int Lwork, 
        Pointer info)
    {
        return checkResult(cusolverDnSsytrdNative(handle, uplo, n, A, lda, D, E, tau, Work, Lwork, info));
    }
    private static native int cusolverDnSsytrdNative(
        cusolverDnHandle handle, 
        char uplo, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer D, 
        Pointer E, 
        Pointer tau, 
        Pointer Work, 
        int Lwork, 
        Pointer info);


    public static int cusolverDnDsytrd(
        cusolverDnHandle handle, 
        char uplo, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer D, 
        Pointer E, 
        Pointer tau, 
        Pointer Work, 
        int Lwork, 
        Pointer info)
    {
        return checkResult(cusolverDnDsytrdNative(handle, uplo, n, A, lda, D, E, tau, Work, Lwork, info));
    }
    private static native int cusolverDnDsytrdNative(
        cusolverDnHandle handle, 
        char uplo, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer D, 
        Pointer E, 
        Pointer tau, 
        Pointer Work, 
        int Lwork, 
        Pointer info);


    /** bidiagonal factorization workspace query */
    public static int cusolverDnSgebrd_bufferSize(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        int[] Lwork)
    {
        return checkResult(cusolverDnSgebrd_bufferSizeNative(handle, m, n, Lwork));
    }
    private static native int cusolverDnSgebrd_bufferSizeNative(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        int[] Lwork);


    public static int cusolverDnDgebrd_bufferSize(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        int[] Lwork)
    {
        return checkResult(cusolverDnDgebrd_bufferSizeNative(handle, m, n, Lwork));
    }
    private static native int cusolverDnDgebrd_bufferSizeNative(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        int[] Lwork);


    public static int cusolverDnCgebrd_bufferSize(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        int[] Lwork)
    {
        return checkResult(cusolverDnCgebrd_bufferSizeNative(handle, m, n, Lwork));
    }
    private static native int cusolverDnCgebrd_bufferSizeNative(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        int[] Lwork);


    public static int cusolverDnZgebrd_bufferSize(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        int[] Lwork)
    {
        return checkResult(cusolverDnZgebrd_bufferSizeNative(handle, m, n, Lwork));
    }
    private static native int cusolverDnZgebrd_bufferSizeNative(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        int[] Lwork);


    /** singular value decomposition, A = U * Sigma * V^H */
    public static int cusolverDnSgesvd_bufferSize(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        int[] Lwork)
    {
        return checkResult(cusolverDnSgesvd_bufferSizeNative(handle, m, n, Lwork));
    }
    private static native int cusolverDnSgesvd_bufferSizeNative(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        int[] Lwork);


    public static int cusolverDnDgesvd_bufferSize(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        int[] Lwork)
    {
        return checkResult(cusolverDnDgesvd_bufferSizeNative(handle, m, n, Lwork));
    }
    private static native int cusolverDnDgesvd_bufferSizeNative(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        int[] Lwork);


    public static int cusolverDnCgesvd_bufferSize(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        int[] Lwork)
    {
        return checkResult(cusolverDnCgesvd_bufferSizeNative(handle, m, n, Lwork));
    }
    private static native int cusolverDnCgesvd_bufferSizeNative(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        int[] Lwork);


    public static int cusolverDnZgesvd_bufferSize(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        int[] Lwork)
    {
        return checkResult(cusolverDnZgesvd_bufferSizeNative(handle, m, n, Lwork));
    }
    private static native int cusolverDnZgesvd_bufferSizeNative(
        cusolverDnHandle handle, 
        int m, 
        int n, 
        int[] Lwork);


    public static int cusolverDnSgesvd(
        cusolverDnHandle handle, 
        char jobu, 
        char jobvt, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer S, 
        Pointer U, 
        int ldu, 
        Pointer VT, 
        int ldvt, 
        Pointer Work, 
        int Lwork, 
        Pointer rwork, 
        Pointer devInfo)
    {
        return checkResult(cusolverDnSgesvdNative(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, Work, Lwork, rwork, devInfo));
    }
    private static native int cusolverDnSgesvdNative(
        cusolverDnHandle handle, 
        char jobu, 
        char jobvt, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer S, 
        Pointer U, 
        int ldu, 
        Pointer VT, 
        int ldvt, 
        Pointer Work, 
        int Lwork, 
        Pointer rwork, 
        Pointer devInfo);


    public static int cusolverDnDgesvd(
        cusolverDnHandle handle, 
        char jobu, 
        char jobvt, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer S, 
        Pointer U, 
        int ldu, 
        Pointer VT, 
        int ldvt, 
        Pointer Work, 
        int Lwork, 
        Pointer rwork, 
        Pointer devInfo)
    {
        return checkResult(cusolverDnDgesvdNative(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, Work, Lwork, rwork, devInfo));
    }
    private static native int cusolverDnDgesvdNative(
        cusolverDnHandle handle, 
        char jobu, 
        char jobvt, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer S, 
        Pointer U, 
        int ldu, 
        Pointer VT, 
        int ldvt, 
        Pointer Work, 
        int Lwork, 
        Pointer rwork, 
        Pointer devInfo);


    public static int cusolverDnCgesvd(
        cusolverDnHandle handle, 
        char jobu, 
        char jobvt, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer S, 
        Pointer U, 
        int ldu, 
        Pointer VT, 
        int ldvt, 
        Pointer Work, 
        int Lwork, 
        Pointer rwork, 
        Pointer devInfo)
    {
        return checkResult(cusolverDnCgesvdNative(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, Work, Lwork, rwork, devInfo));
    }
    private static native int cusolverDnCgesvdNative(
        cusolverDnHandle handle, 
        char jobu, 
        char jobvt, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer S, 
        Pointer U, 
        int ldu, 
        Pointer VT, 
        int ldvt, 
        Pointer Work, 
        int Lwork, 
        Pointer rwork, 
        Pointer devInfo);


    public static int cusolverDnZgesvd(
        cusolverDnHandle handle, 
        char jobu, 
        char jobvt, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer S, 
        Pointer U, 
        int ldu, 
        Pointer VT, 
        int ldvt, 
        Pointer Work, 
        int Lwork, 
        Pointer rwork, 
        Pointer devInfo)
    {
        return checkResult(cusolverDnZgesvdNative(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, Work, Lwork, rwork, devInfo));
    }
    private static native int cusolverDnZgesvdNative(
        cusolverDnHandle handle, 
        char jobu, 
        char jobvt, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer S, 
        Pointer U, 
        int ldu, 
        Pointer VT, 
        int ldvt, 
        Pointer Work, 
        int Lwork, 
        Pointer rwork, 
        Pointer devInfo);


    /** LDLT,UDUT factorization */
    public static int cusolverDnSsytrf(
        cusolverDnHandle handle, 
        int uplo, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer ipiv, 
        Pointer work, 
        int lwork, 
        Pointer devInfo)
    {
        return checkResult(cusolverDnSsytrfNative(handle, uplo, n, A, lda, ipiv, work, lwork, devInfo));
    }
    private static native int cusolverDnSsytrfNative(
        cusolverDnHandle handle, 
        int uplo, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer ipiv, 
        Pointer work, 
        int lwork, 
        Pointer devInfo);


    public static int cusolverDnDsytrf(
        cusolverDnHandle handle, 
        int uplo, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer ipiv, 
        Pointer work, 
        int lwork, 
        Pointer devInfo)
    {
        return checkResult(cusolverDnDsytrfNative(handle, uplo, n, A, lda, ipiv, work, lwork, devInfo));
    }
    private static native int cusolverDnDsytrfNative(
        cusolverDnHandle handle, 
        int uplo, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer ipiv, 
        Pointer work, 
        int lwork, 
        Pointer devInfo);


    public static int cusolverDnCsytrf(
        cusolverDnHandle handle, 
        int uplo, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer ipiv, 
        Pointer work, 
        int lwork, 
        Pointer devInfo)
    {
        return checkResult(cusolverDnCsytrfNative(handle, uplo, n, A, lda, ipiv, work, lwork, devInfo));
    }
    private static native int cusolverDnCsytrfNative(
        cusolverDnHandle handle, 
        int uplo, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer ipiv, 
        Pointer work, 
        int lwork, 
        Pointer devInfo);


    public static int cusolverDnZsytrf(
        cusolverDnHandle handle, 
        int uplo, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer ipiv, 
        Pointer work, 
        int lwork, 
        Pointer devInfo)
    {
        return checkResult(cusolverDnZsytrfNative(handle, uplo, n, A, lda, ipiv, work, lwork, devInfo));
    }
    private static native int cusolverDnZsytrfNative(
        cusolverDnHandle handle, 
        int uplo, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer ipiv, 
        Pointer work, 
        int lwork, 
        Pointer devInfo);


    /** SYTRF factorization workspace query */
    public static int cusolverDnSsytrf_bufferSize(
        cusolverDnHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        int[] Lwork)
    {
        return checkResult(cusolverDnSsytrf_bufferSizeNative(handle, n, A, lda, Lwork));
    }
    private static native int cusolverDnSsytrf_bufferSizeNative(
        cusolverDnHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        int[] Lwork);


    public static int cusolverDnDsytrf_bufferSize(
        cusolverDnHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        int[] Lwork)
    {
        return checkResult(cusolverDnDsytrf_bufferSizeNative(handle, n, A, lda, Lwork));
    }
    private static native int cusolverDnDsytrf_bufferSizeNative(
        cusolverDnHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        int[] Lwork);


    public static int cusolverDnCsytrf_bufferSize(
        cusolverDnHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        int[] Lwork)
    {
        return checkResult(cusolverDnCsytrf_bufferSizeNative(handle, n, A, lda, Lwork));
    }
    private static native int cusolverDnCsytrf_bufferSizeNative(
        cusolverDnHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        int[] Lwork);


    public static int cusolverDnZsytrf_bufferSize(
        cusolverDnHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        int[] Lwork)
    {
        return checkResult(cusolverDnZsytrf_bufferSizeNative(handle, n, A, lda, Lwork));
    }
    private static native int cusolverDnZsytrf_bufferSizeNative(
        cusolverDnHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        int[] Lwork);
    

}
