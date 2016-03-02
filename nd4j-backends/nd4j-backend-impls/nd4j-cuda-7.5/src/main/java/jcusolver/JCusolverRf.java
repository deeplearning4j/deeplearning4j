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

/**
 * Java bindings for CUSOLVER, the NVIDIA CUDA solver library. <br />
 * <br />
 * The documentation is taken from the CUSOLVER header files.
 */
public class JCusolverRf
{
    /* Private constructor to prevent instantiation */
    private JCusolverRf()
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
    

    /** CUSOLVERRF create (allocate memory) and destroy (free memory) in the handle */
    public static int cusolverRfCreate(
        cusolverRfHandle handle)
    {
        return checkResult(cusolverRfCreateNative(handle));
    }
    private static native int cusolverRfCreateNative(
        cusolverRfHandle handle);


    public static int cusolverRfDestroy(
        cusolverRfHandle handle)
    {
        return checkResult(cusolverRfDestroyNative(handle));
    }
    private static native int cusolverRfDestroyNative(
        cusolverRfHandle handle);


    /** CUSOLVERRF set and get input format */
    public static int cusolverRfGetMatrixFormat(
        cusolverRfHandle handle, 
        int[] format, 
        int[] diag)
    {
        return checkResult(cusolverRfGetMatrixFormatNative(handle, format, diag));
    }
    private static native int cusolverRfGetMatrixFormatNative(
        cusolverRfHandle handle, 
        int[] format, 
        int[] diag);


    public static int cusolverRfSetMatrixFormat(
        cusolverRfHandle handle, 
        int format, 
        int diag)
    {
        return checkResult(cusolverRfSetMatrixFormatNative(handle, format, diag));
    }
    private static native int cusolverRfSetMatrixFormatNative(
        cusolverRfHandle handle, 
        int format, 
        int diag);


    /** CUSOLVERRF set and get numeric properties */
    public static int cusolverRfSetNumericProperties(
        cusolverRfHandle handle, 
        double zero, 
        double boost)
    {
        return checkResult(cusolverRfSetNumericPropertiesNative(handle, zero, boost));
    }
    private static native int cusolverRfSetNumericPropertiesNative(
        cusolverRfHandle handle, 
        double zero, 
        double boost);


    public static int cusolverRfGetNumericProperties(
        cusolverRfHandle handle, 
        double[] zero, 
        double[] boost)
    {
        return checkResult(cusolverRfGetNumericPropertiesNative(handle, zero, boost));
    }
    private static native int cusolverRfGetNumericPropertiesNative(
        cusolverRfHandle handle, 
        double[] zero, 
        double[] boost);


    public static int cusolverRfGetNumericBoostReport(
        cusolverRfHandle handle, 
        int[] report)
    {
        return checkResult(cusolverRfGetNumericBoostReportNative(handle, report));
    }
    private static native int cusolverRfGetNumericBoostReportNative(
        cusolverRfHandle handle, 
        int[] report);


    /** CUSOLVERRF choose the triangular solve algorithm */
    public static int cusolverRfSetAlgs(
        cusolverRfHandle handle, 
        int factAlg, 
        int solveAlg)
    {
        return checkResult(cusolverRfSetAlgsNative(handle, factAlg, solveAlg));
    }
    private static native int cusolverRfSetAlgsNative(
        cusolverRfHandle handle, 
        int factAlg, 
        int solveAlg);


    public static int cusolverRfGetAlgs(
        cusolverRfHandle handle, 
        int[] factAlg, 
        int[] solveAlg)
    {
        return checkResult(cusolverRfGetAlgsNative(handle, factAlg, solveAlg));
    }
    private static native int cusolverRfGetAlgsNative(
        cusolverRfHandle handle, 
        int[] factAlg, 
        int[] solveAlg);


    /** CUSOLVERRF set and get fast mode */
    public static int cusolverRfGetResetValuesFastMode(
        cusolverRfHandle handle, 
        int[] fastMode)
    {
        return checkResult(cusolverRfGetResetValuesFastModeNative(handle, fastMode));
    }
    private static native int cusolverRfGetResetValuesFastModeNative(
        cusolverRfHandle handle, 
        int[] fastMode);


    public static int cusolverRfSetResetValuesFastMode(
        cusolverRfHandle handle, 
        int fastMode)
    {
        return checkResult(cusolverRfSetResetValuesFastModeNative(handle, fastMode));
    }
    private static native int cusolverRfSetResetValuesFastModeNative(
        cusolverRfHandle handle, 
        int fastMode);


    /*** Non-Batched Routines ***/
    /** CUSOLVERRF setup of internal structures from host or device memory */
    public static int cusolverRfSetupHost(
        int n, 
        int nnzA, 
        Pointer h_csrRowPtrA, 
        Pointer h_csrColIndA, 
        Pointer h_csrValA, 
        int nnzL, 
        Pointer h_csrRowPtrL, 
        Pointer h_csrColIndL, 
        Pointer h_csrValL, 
        int nnzU, 
        Pointer h_csrRowPtrU, 
        Pointer h_csrColIndU, 
        Pointer h_csrValU, 
        Pointer h_P, 
        Pointer h_Q, 
        /** Output */
        cusolverRfHandle handle)
    {
        return checkResult(cusolverRfSetupHostNative(n, nnzA, h_csrRowPtrA, h_csrColIndA, h_csrValA, nnzL, h_csrRowPtrL, h_csrColIndL, h_csrValL, nnzU, h_csrRowPtrU, h_csrColIndU, h_csrValU, h_P, h_Q, handle));
    }
    private static native int cusolverRfSetupHostNative(
        int n, 
        int nnzA, 
        Pointer h_csrRowPtrA, 
        Pointer h_csrColIndA, 
        Pointer h_csrValA, 
        int nnzL, 
        Pointer h_csrRowPtrL, 
        Pointer h_csrColIndL, 
        Pointer h_csrValL, 
        int nnzU, 
        Pointer h_csrRowPtrU, 
        Pointer h_csrColIndU, 
        Pointer h_csrValU, 
        Pointer h_P, 
        Pointer h_Q, 
        /** Output */
        cusolverRfHandle handle);
/** Input (in the host memory) */

    /**
     * Input (in the device memory) 
     */
    public static int cusolverRfSetupDevice(
        int n, 
        int nnzA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer csrValA, 
        int nnzL, 
        Pointer csrRowPtrL, 
        Pointer csrColIndL, 
        Pointer csrValL, 
        int nnzU, 
        Pointer csrRowPtrU, 
        Pointer csrColIndU, 
        Pointer csrValU, 
        Pointer P, 
        Pointer Q, 
        /** Output */
        cusolverRfHandle handle)
    {
        return checkResult(cusolverRfSetupDeviceNative(n, nnzA, csrRowPtrA, csrColIndA, csrValA, nnzL, csrRowPtrL, csrColIndL, csrValL, nnzU, csrRowPtrU, csrColIndU, csrValU, P, Q, handle));
    }
    private static native int cusolverRfSetupDeviceNative(
        int n, 
        int nnzA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer csrValA, 
        int nnzL, 
        Pointer csrRowPtrL, 
        Pointer csrColIndL, 
        Pointer csrValL, 
        int nnzU, 
        Pointer csrRowPtrU, 
        Pointer csrColIndU, 
        Pointer csrValU, 
        Pointer P, 
        Pointer Q, 
        /** Output */
        cusolverRfHandle handle);


    /** CUSOLVERRF update the matrix values (assuming the reordering, pivoting 
       and consequently the sparsity pattern of L and U did not change),
       and zero out the remaining values. */
    public static int cusolverRfResetValues(
        int n, 
        int nnzA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer csrValA, 
        Pointer P, 
        Pointer Q, 
        /** Output */
        cusolverRfHandle handle)
    {
        return checkResult(cusolverRfResetValuesNative(n, nnzA, csrRowPtrA, csrColIndA, csrValA, P, Q, handle));
    }
    private static native int cusolverRfResetValuesNative(
        int n, 
        int nnzA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer csrValA, 
        Pointer P, 
        Pointer Q, 
        /** Output */
        cusolverRfHandle handle);
/** Input (in the device memory) */

    /** CUSOLVERRF analysis (for parallelism) */
    public static int cusolverRfAnalyze(
        cusolverRfHandle handle)
    {
        return checkResult(cusolverRfAnalyzeNative(handle));
    }
    private static native int cusolverRfAnalyzeNative(
        cusolverRfHandle handle);


    /** CUSOLVERRF re-factorization (for parallelism) */
    public static int cusolverRfRefactor(
        cusolverRfHandle handle)
    {
        return checkResult(cusolverRfRefactorNative(handle));
    }
    private static native int cusolverRfRefactorNative(
        cusolverRfHandle handle);


    /** CUSOLVERRF extraction: Get L & U packed into a single matrix M */
    public static int cusolverRfAccessBundledFactorsDevice(
        cusolverRfHandle handle, 
        /** Output (in the host memory) */
        Pointer nnzM, 
        /** Output (in the device memory) */
        Pointer Mp, 
        Pointer Mi, 
        Pointer Mx)
    {
        return checkResult(cusolverRfAccessBundledFactorsDeviceNative(handle, nnzM, Mp, Mi, Mx));
    }
    private static native int cusolverRfAccessBundledFactorsDeviceNative(
        cusolverRfHandle handle, 
        /** Output (in the host memory) */
        Pointer nnzM, 
        /** Output (in the device memory) */
        Pointer Mp, 
        Pointer Mi, 
        Pointer Mx);
/** Input */

    /**
     * Input 
     */
    public static int cusolverRfExtractBundledFactorsHost(
        cusolverRfHandle handle, 
        /** Output (in the host memory) */
        Pointer h_nnzM, 
        Pointer h_Mp, 
        Pointer h_Mi, 
        Pointer h_Mx)
    {
        return checkResult(cusolverRfExtractBundledFactorsHostNative(handle, h_nnzM, h_Mp, h_Mi, h_Mx));
    }
    private static native int cusolverRfExtractBundledFactorsHostNative(
        cusolverRfHandle handle, 
        /** Output (in the host memory) */
        Pointer h_nnzM, 
        Pointer h_Mp, 
        Pointer h_Mi, 
        Pointer h_Mx);


    /** CUSOLVERRF extraction: Get L & U individually */
    public static int cusolverRfExtractSplitFactorsHost(
        cusolverRfHandle handle, 
        /** Output (in the host memory) */
        Pointer h_nnzL, 
        Pointer h_csrRowPtrL, 
        Pointer h_csrColIndL, 
        Pointer h_csrValL, 
        Pointer h_nnzU, 
        Pointer h_csrRowPtrU, 
        Pointer h_csrColIndU, 
        Pointer h_csrValU)
    {
        return checkResult(cusolverRfExtractSplitFactorsHostNative(handle, h_nnzL, h_csrRowPtrL, h_csrColIndL, h_csrValL, h_nnzU, h_csrRowPtrU, h_csrColIndU, h_csrValU));
    }
    private static native int cusolverRfExtractSplitFactorsHostNative(
        cusolverRfHandle handle, 
        /** Output (in the host memory) */
        Pointer h_nnzL, 
        Pointer h_csrRowPtrL, 
        Pointer h_csrColIndL, 
        Pointer h_csrValL, 
        Pointer h_nnzU, 
        Pointer h_csrRowPtrU, 
        Pointer h_csrColIndU, 
        Pointer h_csrValU);
/** Input */

    /** CUSOLVERRF (forward and backward triangular) solves */
    public static int cusolverRfSolve(
        cusolverRfHandle handle, 
        Pointer P, 
        Pointer Q, 
        int nrhs, //only nrhs=1 is supported
        Pointer Temp, //of size ldt*nrhs (ldt>=n)
        int ldt, 
        /** Input/Output (in the device memory) */
        Pointer XF, 
        /** Input */
        int ldxf)
    {
        return checkResult(cusolverRfSolveNative(handle, P, Q, nrhs, Temp, ldt, XF, ldxf));
    }
    private static native int cusolverRfSolveNative(
        cusolverRfHandle handle, 
        Pointer P, 
        Pointer Q, 
        int nrhs, //only nrhs=1 is supported
        Pointer Temp, //of size ldt*nrhs (ldt>=n)
        int ldt, 
        /** Input/Output (in the device memory) */
        Pointer XF, 
        /** Input */
        int ldxf);
/** Input (in the device memory) */

    /*** Batched Routines ***/
    /** CUSOLVERRF-batch setup of internal structures from host */
    public static int cusolverRfBatchSetupHost(
        int batchSize, 
        int n, 
        int nnzA, 
        Pointer h_csrRowPtrA, 
        Pointer h_csrColIndA, 
        Pointer h_csrValA_array, 
        int nnzL, 
        Pointer h_csrRowPtrL, 
        Pointer h_csrColIndL, 
        Pointer h_csrValL, 
        int nnzU, 
        Pointer h_csrRowPtrU, 
        Pointer h_csrColIndU, 
        Pointer h_csrValU, 
        Pointer h_P, 
        Pointer h_Q, 
        /** Output (in the device memory) */
        cusolverRfHandle handle)
    {
        return checkResult(cusolverRfBatchSetupHostNative(batchSize, n, nnzA, h_csrRowPtrA, h_csrColIndA, h_csrValA_array, nnzL, h_csrRowPtrL, h_csrColIndL, h_csrValL, nnzU, h_csrRowPtrU, h_csrColIndU, h_csrValU, h_P, h_Q, handle));
    }
    private static native int cusolverRfBatchSetupHostNative(
        int batchSize, 
        int n, 
        int nnzA, 
        Pointer h_csrRowPtrA, 
        Pointer h_csrColIndA, 
        Pointer h_csrValA_array, 
        int nnzL, 
        Pointer h_csrRowPtrL, 
        Pointer h_csrColIndL, 
        Pointer h_csrValL, 
        int nnzU, 
        Pointer h_csrRowPtrU, 
        Pointer h_csrColIndU, 
        Pointer h_csrValU, 
        Pointer h_P, 
        Pointer h_Q, 
        /** Output (in the device memory) */
        cusolverRfHandle handle);
/** Input (in the host memory)*/

    /** CUSOLVERRF-batch update the matrix values (assuming the reordering, pivoting 
       and consequently the sparsity pattern of L and U did not change),
       and zero out the remaining values. */
    public static int cusolverRfBatchResetValues(
        int batchSize, 
        int n, 
        int nnzA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer csrValA_array, 
        Pointer P, 
        Pointer Q, 
        /** Output */
        cusolverRfHandle handle)
    {
        return checkResult(cusolverRfBatchResetValuesNative(batchSize, n, nnzA, csrRowPtrA, csrColIndA, csrValA_array, P, Q, handle));
    }
    private static native int cusolverRfBatchResetValuesNative(
        int batchSize, 
        int n, 
        int nnzA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer csrValA_array, 
        Pointer P, 
        Pointer Q, 
        /** Output */
        cusolverRfHandle handle);
/** Input (in the device memory) */

    /** CUSOLVERRF-batch analysis (for parallelism) */
    public static int cusolverRfBatchAnalyze(
        cusolverRfHandle handle)
    {
        return checkResult(cusolverRfBatchAnalyzeNative(handle));
    }
    private static native int cusolverRfBatchAnalyzeNative(
        cusolverRfHandle handle);


    /** CUSOLVERRF-batch re-factorization (for parallelism) */
    public static int cusolverRfBatchRefactor(
        cusolverRfHandle handle)
    {
        return checkResult(cusolverRfBatchRefactorNative(handle));
    }
    private static native int cusolverRfBatchRefactorNative(
        cusolverRfHandle handle);


    /** CUSOLVERRF-batch (forward and backward triangular) solves */
    public static int cusolverRfBatchSolve(
        cusolverRfHandle handle, 
        Pointer P, 
        Pointer Q, 
        int nrhs, //only nrhs=1 is supported
        Pointer Temp, //of size 2*batchSize*(n*nrhs)
        int ldt, //only ldt=n is supported
        /** Input/Output (in the device memory) */
        Pointer XF_array, 
        /** Input */
        int ldxf)
    {
        return checkResult(cusolverRfBatchSolveNative(handle, P, Q, nrhs, Temp, ldt, XF_array, ldxf));
    }
    private static native int cusolverRfBatchSolveNative(
        cusolverRfHandle handle, 
        Pointer P, 
        Pointer Q, 
        int nrhs, //only nrhs=1 is supported
        Pointer Temp, //of size 2*batchSize*(n*nrhs)
        int ldt, //only ldt=n is supported
        /** Input/Output (in the device memory) */
        Pointer XF_array, 
        /** Input */
        int ldxf);
/** Input (in the device memory) */

    /** CUSOLVERRF-batch obtain the position of zero pivot */
    public static int cusolverRfBatchZeroPivot(
        cusolverRfHandle handle, 
        /** Output (in the host memory) */
        Pointer position)
    {
        return checkResult(cusolverRfBatchZeroPivotNative(handle, position));
    }
    private static native int cusolverRfBatchZeroPivotNative(
        cusolverRfHandle handle, 
        /** Output (in the host memory) */
        Pointer position);
/** Input */
    
}
