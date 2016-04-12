/*
 * JCufft - Java bindings for CUFFT, the NVIDIA CUDA FFT library,
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

package jcuda.jcufft;

import jcuda.*;
import jcuda.runtime.*;
import org.nd4j.linalg.api.buffer.util.LibUtils;

/**
 * Java bindings for CUFFT, the NVIDIA CUDA FFT library.<br />
 * <br />
 * Most comments are taken from the CUFFT library documentation<br />
 * <br />
 */
public class JCufft
{
    /**
     * CUFFT transform direction
     */
    public static final int CUFFT_FORWARD = -1;

    /**
     * CUFFT transform direction
     */
    public static final int CUFFT_INVERSE = 1;


    /**
     * The flag that indicates whether the native library has been
     * loaded
     */
    private static boolean initialized = false;

    /**
     * Whether a CudaException should be thrown if a method is about
     * to return a result code that is not cufftResult.CUFFT_SUCCESS
     */
    private static boolean exceptionsEnabled = false;

    /* Private constructor to prevent instantiation */
    private JCufft()
    {
    }

    // Initialize the native library.
    static
    {
        initialize();
    }

    /**
     * Initializes the native library. Note that this method
     * does not have to be called explicitly by the user of
     * the library: The library will automatically be
     * loaded when this class is loaded.
     */
    public static void initialize()
    {
        if (!initialized)
        {
            LibUtils.loadLibrary("JCufft");
            initialized = true;
        }
    }


    /**
     * Set the specified log level for the JCufft library.<br />
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
        setLogLevel(logLevel.ordinal());
    }

    private static native void setLogLevel(int logLevel);


    /**
     * Enables or disables exceptions. By default, the methods of this class
     * only return the cufftResult error code from the underlying CUDA function.
     * If exceptions are enabled, a CudaException with a detailed error
     * message will be thrown if a method is about to return a result code
     * that is not cufftResult.CUFFT_SUCCESS
     *
     * @param enabled Whether exceptions are enabled
     */
    public static void setExceptionsEnabled(boolean enabled)
    {
        exceptionsEnabled = enabled;
    }

    /**
     * If the given result is different to cufftResult.CUFFT_SUCCESS and
     * exceptions have been enabled, this method will throw a
     * CudaException with an error message that corresponds to the
     * given result code. Otherwise, the given result is simply
     * returned.
     *
     * @param result The result to check
     * @return The result that was given as the parameter
     * @throws CudaException If exceptions have been enabled and
     * the given result code is not cufftResult.CUFFT_SUCCESS
     */
    private static int checkResult(int result)
    {
        if (exceptionsEnabled && result != cufftResult.CUFFT_SUCCESS)
        {
            throw new CudaException(cufftResult.stringFor(result));
        }
        return result;
    }


    /**
     * Writes the CUFFT version into the given argument.
     *
     * @param version The version
     * @return The cufftResult code.
     */
    public static int cufftGetVersion(int version[])
    {
        return cufftGetVersionNative(version);
    }

    private static native int cufftGetVersionNative(int version[]);

    /**
     * <pre>
     * Creates a 1D FFT plan configuration for a specified signal size and data
     * type.
     *
     * cufftResult cufftPlan1d( cufftHandle *plan, int nx, cufftType type, int batch );
     *
     * The batch input parameter tells CUFFT how many 1D transforms to configure.
     *
     * Input
     * ----
     * plan  Pointer to a cufftHandle object
     * nx    The transform size (e.g., 256 for a 256-point FFT)
     * type  The transform data type (e.g., CUFFT_C2C for complex to complex)
     * batch Number of transforms of size nx
     *
     * Output
     * ----
     * plan     Contains a CUFFT 1D plan handle value
     *
     * Return Values
     * ----
     * CUFFT_SETUP_FAILED CUFFT library failed to initialize.
     * CUFFT_INVALID_SIZE The nx parameter is not a supported size.
     * CUFFT_INVALID_TYPE The type parameter is not supported.
     * CUFFT_ALLOC_FAILED Allocation of GPU resources for the plan failed.
     * CUFFT_SUCCESS      CUFFT successfully created the FFT plan.
     *
     * JCUFFT_INTERNAL_ERROR If an internal JCufft error occurred
     * <pre>
     * NOTE: Batch sizes other than 1 for cufftPlan1d() have been
     * deprecated as of CUDA 6.0RC. Use cufftPlanMany() for
     * multiple batch execution.
     */
    public static int cufftPlan1d(cufftHandle plan, int nx, int type, int batch)
    {
        plan.setDimension(1);
        plan.setType(type);
        plan.setSize(nx, 0, 0);
        plan.setBatchSize(batch);
        return checkResult(cufftPlan1dNative(plan, nx, type, batch));
    }
    private static native int cufftPlan1dNative(cufftHandle plan, int nx, int type, int batch);


    /**
     * <pre>
     * Creates a 2D FFT plan configuration according to specified signal sizes
     * and data type.
     *
     * cufftResult cufftPlan2d( cufftHandle *plan, int nx, int ny, cufftType type );
     *
     * This function is the same as cufftPlan1d() except that
     * it takes a second size parameter, ny, and does not support batching.
     *
     * Input
     * ----
     * plan Pointer to a cufftHandle object
     * nx   The transform size in the X dimension (number of rows)
     * ny   The transform size in the Y dimension (number of columns)
     * type The transform data type (e.g., CUFFT_C2R for complex to real)
     *
     * Output
     * ----
     * plan Contains a CUFFT 2D plan handle value
     *
     * Return Values
     * ----
     * CUFFT_SETUP_FAILED CUFFT library failed to initialize.
     * CUFFT_INVALID_SIZE The nx or ny parameter is not a supported size.
     * CUFFT_INVALID_TYPE The type parameter is not supported.
     * CUFFT_ALLOC_FAILED Allocation of GPU resources for the plan failed.
     * CUFFT_SUCCESS      CUFFT successfully created the FFT plan.
     *
     * JCUFFT_INTERNAL_ERROR If an internal JCufft error occurred
     * <pre>
     */
    public static int cufftPlan2d(cufftHandle plan, int nx, int ny, int type)
    {
        plan.setDimension(2);
        plan.setType(type);
        plan.setSize(nx, ny, 0);
        return checkResult(cufftPlan2dNative(plan, nx, ny, type));
    }
    private static native int cufftPlan2dNative(cufftHandle plan, int nx, int ny, int type);

    /**
     * <pre>
     * Creates a 3D FFT plan configuration according to specified signal sizes
     * and data type.
     *
     * cufftResult cufftPlan3d( cufftHandle *plan, int nx, int ny, int nz, cufftType type );
     *
     * This function is the same as cufftPlan2d() except that
     * it takes a third size parameter nz.
     *
     * Input
     * ----
     * plan Pointer to a cufftHandle object
     * nx The transform size in the X dimension
     * ny The transform size in the Y dimension
     * nz The transform size in the Z dimension
     * type The transform data type (e.g., CUFFT_R2C for real to complex)
     *
     * Output
     * ----
     * plan Contains a CUFFT 3D plan handle value
     *
     * Return Values
     * ----
     * CUFFT_SETUP_FAILED CUFFT library failed to initialize.
     * CUFFT_INVALID_SIZE Parameter nx, ny, or nz is not a supported size.
     * CUFFT_INVALID_TYPE The type parameter is not supported.
     * CUFFT_ALLOC_FAILED Allocation of GPU resources for the plan failed.
     * CUFFT_SUCCESS      CUFFT successfully created the FFT plan.
     *
     * JCUFFT_INTERNAL_ERROR If an internal JCufft error occurred
     * <pre>
     */
    public static int cufftPlan3d(cufftHandle plan, int nx, int ny, int nz, int type)
    {
        plan.setDimension(3);
        plan.setType(type);
        plan.setSize(nx, ny, nz);
        return checkResult(cufftPlan3dNative(plan, nx, ny, nz, type));
    }

    private static native int cufftPlan3dNative(cufftHandle plan, int nx, int ny, int nz, int type);


    /**
     * <pre>
     * Creates a FFT plan configuration of dimension rank, with sizes
     * specified in the array n.
     *
     * cufftResult cufftPlanMany(cufftHandle *plan, int rank, int *n,
     *     int *inembed, int istride, int idist,
     *     int *onembed, int ostride, int odist,
     *     cufftType type, int batch );
     *
     * The batch input parameter tells CUFFT how many transforms to
     * configure in parallel. With this function, batched plans of
     * any dimension may be created. Input parameters inembed, istride,
     * and idist and output parameters onembed, ostride, and odist
     * will allow setup of noncontiguous input data in a future version.
     * Note that for CUFFT 3.0, these parameters are ignored and the
     * layout of batched data must be side-by-side and not interleaved.
     *
     * Input
     * ----
     * plan Pointer to a cufftHandle object
     * rank Dimensionality of the transform (1, 2, or 3)
     * n An array of size rank, describing the size of each dimension
     * inembed Unused: pass NULL
     * istride Unused: pass 1
     * idist Unused: pass 0
     * onembed Unused: pass NULL
     * ostride Unused: pass 1
     * odist Unused: pass 0
     * type Transform data type (e.g., CUFFT_C2C, as per other CUFFT calls)
     * batch Batch size for this transform
     *
     * Output
     * ----
     * plan Contains a CUFFT plan handle
     *
     * Return Values
     * ----
     * CUFFT_SETUP_FAILED CUFFT library failed to initialize.
     * CUFFT_INVALID_SIZE Parameter is not a supported size.
     * CUFFT_INVALID_TYPE The type parameter is not supported
     * </pre>
     */
    public static int cufftPlanMany(cufftHandle plan, int rank, int n[],
        int inembed[], int istride, int idist,
        int onembed[], int ostride, int odist,
        int type, int batch)
    {
        return checkResult(cufftPlanManyNative(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch));
    }

    private static native int cufftPlanManyNative(cufftHandle plan, int rank, int n[],
        int inembed[], int istride, int idist,
        int onembed[], int ostride, int odist,
        int type, int batch);





    public static int cufftMakePlan1d(
        cufftHandle plan, int nx, int type,
        int batch, /* deprecated - use cufftPlanMany */
        long workSize[])
    {
        return checkResult(cufftMakePlan1dNative(plan, nx, type, batch, workSize));
    }
    private static native int cufftMakePlan1dNative(
        cufftHandle plan, int nx, int type,
        int batch, /* deprecated - use cufftPlanMany */
        long workSize[]);



    public static int cufftMakePlan2d(
        cufftHandle plan, int nx, int ny, int type,
        long workSize[])
    {
        return checkResult(cufftMakePlan2dNative(plan, nx, ny, type, workSize));
    }
    private static native int cufftMakePlan2dNative(
        cufftHandle plan, int nx, int ny, int type,
        long workSize[]);



    public static int cufftMakePlan3d(
        cufftHandle plan, int nx, int ny, int nz, int type,
        long workSize[])
    {
        return checkResult(cufftMakePlan3dNative(plan, nx, ny, nz, type, workSize));
    }
    private static native int cufftMakePlan3dNative(
        cufftHandle plan, int nx, int ny, int nz, int type,
        long workSize[]);



    public static int cufftMakePlanMany(
        cufftHandle plan, int rank, int n[],
        int inembed[], int istride, int idist,
        int onembed[], int ostride, int odist,
        int type, int batch, long workSize[])
    {
        return checkResult(cufftMakePlanManyNative(
            plan, rank, n,
            inembed, istride, idist,
            onembed, ostride, odist,
            type, batch, workSize));
    }
    private static native int cufftMakePlanManyNative(
        cufftHandle plan, int rank, int n[],
        int inembed[], int istride, int idist,
        int onembed[], int ostride, int odist,
        int type, int batch, long workSize[]);


    public static int cufftMakePlanMany64(
        cufftHandle plan, 
        int rank, 
        long n[],
        long inembed[], 
        long istride, 
        long idist,
        long onembed[], 
        long ostride, 
        long odist,
        int type, 
        long batch, 
        long workSize[])
    {
        return checkResult(cufftMakePlanManyNative64(
            plan, rank, n,
            inembed, istride, idist,
            onembed, ostride, odist,
            type, batch, workSize));
    }
    private static native int cufftMakePlanManyNative64(
        cufftHandle plan, 
        int rank, 
        long n[],
        long inembed[], 
        long istride, 
        long idist,
        long onembed[], 
        long ostride, 
        long odist,
        int type, 
        long batch, 
        long workSize[]);

    
    public static int cufftGetSizeMany64(
        cufftHandle plan, 
        int rank, 
        long n[],
        long inembed[], 
        long istride, 
        long idist,
        long onembed[], 
        long ostride, 
        long odist,
        int type, 
        long batch, 
        long workSize[])
    {
        return checkResult(cufftGetSizeMany64Native(
            plan, rank, n,
            inembed, istride, idist,
            onembed, ostride, odist,
            type, batch, workSize));
    }
    private static native int cufftGetSizeMany64Native(
        cufftHandle plan, 
        int rank, 
        long n[],
        long inembed[], 
        long istride, 
        long idist,
        long onembed[], 
        long ostride, 
        long odist,
        int type, 
        long batch, 
        long workSize[]);
    
    

    public static int cufftEstimate1d(int nx,
        int type,
            int batch, /* deprecated - use cufftPlanMany */
            long workSize[])
    {
        return checkResult(cufftEstimate1dNative(nx, type, batch, workSize));
    }
    private static native int cufftEstimate1dNative(
        int nx, int type,
        int batch, /* deprecated - use cufftPlanMany */
        long workSize[]);



    public static int cufftEstimate2d(
        int nx, int ny, int type,
        long workSize[])
    {
        return checkResult(cufftEstimate2dNative(nx, ny, type, workSize));
    }
    private static native int cufftEstimate2dNative(
        int nx, int ny, int type,
        long workSize[]);



    public static int cufftEstimate3d(
        int nx, int ny, int nz, int type,
        long workSize[])
    {
        return checkResult(cufftEstimate3dNative(nx, ny, nz, type, workSize));
    }
    private static native int cufftEstimate3dNative(
        int nx, int ny, int nz, int type,
        long workSize[]);



    public static int cufftEstimateMany(
        int rank, int n[],
        int inembed[], int istride, int idist,
        int onembed[], int ostride, int odist,
        int type,
        int batch,
        long workSize[])
    {
        return checkResult(cufftEstimateManyNative(rank, n,
            inembed, istride, idist,
            onembed, ostride, odist, type, batch, workSize));
    }
    private static native int cufftEstimateManyNative(
        int rank, int n[],
        int inembed[], int istride, int idist,
        int onembed[], int ostride, int odist,
        int type,
        int batch,
        long workSize[]);



    public static int cufftCreate(cufftHandle cufftHandle)
    {
        return checkResult(cufftCreateNative(cufftHandle));
    }
    private static native int cufftCreateNative(cufftHandle cufftHandle);


    public static int cufftGetSize1d(cufftHandle handle,
        int nx,
        int type,
        int batch,
        long workSize[])
    {
        return checkResult(cufftGetSize1dNative(handle, nx, type, batch, workSize));
    }
    private static native int cufftGetSize1dNative(cufftHandle handle,
        int nx,
        int type,
        int batch,
        long workSize[]);


    public static int cufftGetSize2d(cufftHandle handle,
        int nx, int ny,
        int type,
        long workSize[])
    {
        return checkResult(cufftGetSize2dNative(handle, nx, ny, type, workSize));
    }
    private static native int cufftGetSize2dNative(cufftHandle handle,
        int nx, int ny,
        int type,
        long workSize[]);



    public static int cufftGetSize3d(cufftHandle handle,
        int nx, int ny, int nz,
        int type,
        long workSize[])
    {
        return checkResult(cufftGetSize3dNative(handle, nx, ny, nz, type, workSize));
    }
    private static native int cufftGetSize3dNative(cufftHandle handle,
        int nx, int ny, int nz,
        int type,
        long workSize[]);



    public static int cufftGetSizeMany(cufftHandle handle,
        int rank, int n[],
        int inembed[], int istride, int idist,
        int onembed[], int ostride, int odist,
        int type, int batch, long workArea[])
    {
        return checkResult(cufftGetSizeManyNative(handle, rank, n,
            inembed, istride, idist,
            onembed, ostride, odist, type, batch, workArea));
    }
    private static native int cufftGetSizeManyNative(cufftHandle handle,
        int rank, int n[],
        int inembed[], int istride, int idist,
        int onembed[], int ostride, int odist,
        int type, int batch, long workArea[]);


    public static int cufftGetSize(cufftHandle handle, long workSize[])
    {
        return checkResult(cufftGetSizeNative(handle, workSize));
    }
    private static native int cufftGetSizeNative(cufftHandle handle, long workSize[]);

    public static int cufftSetWorkArea(cufftHandle plan, Pointer workArea)
    {
        return checkResult(cufftSetWorkAreaNative(plan, workArea));
    }
    private static native int cufftSetWorkAreaNative(cufftHandle plan, Pointer workArea);

    public static int cufftSetAutoAllocation(cufftHandle plan, int autoAllocate)
    {
        return checkResult(cufftSetAutoAllocationNative(plan, autoAllocate));
    }
    private static native int cufftSetAutoAllocationNative(cufftHandle plan, int autoAllocate);




    /**
     * <pre>
     * Frees all GPU resources associated with a CUFFT plan and destroys the
     * internal plan data structure.
     *
     * cufftResult cufftDestroy( cufftHandle plan );
     *
     * This function should be called once a plan
     * is no longer needed to avoid wasting GPU memory.
     *
     * Input
     * ----
     * plan The cufftHandle object of the plan to be destroyed.
     *
     * Return Values
     * ----
     * CUFFT_SETUP_FAILED    CUFFT library failed to initialize.
     * CUFFT_SHUTDOWN_FAILED CUFFT library failed to shut down.
     * CUFFT_INVALID_PLAN    The plan parameter is not a valid handle.
     * CUFFT_SUCCESS         CUFFT successfully destroyed the FFT plan.
     *
     * JCUFFT_INTERNAL_ERROR If an internal JCufft error occurred
     * <pre>
     */
    public static int cufftDestroy(cufftHandle plan)
    {
        return checkResult(cufftDestroyNative(plan));
    }

    private static native int cufftDestroyNative(cufftHandle plan);

    /**
     * <pre>
     * Associates a CUDA stream with a CUFFT plan.
     *
     * cufftResult cufftSetStream( cufftHandle plan, cudaStream_t stream );
     *
     * All kernel launches made during plan execution are now done through
     * the associated stream, enabling overlap with activity in other
     * streams (for example, data copying). The association remains until
     * the plan is destroyed or the stream is changed with another call
     * to cufftSetStream().
     *
     * Input
     * plan The cufftHandle object to associate with the stream
     * stream A valid CUDA stream created with cudaStreamCreate() (or 0
     * for the default stream)
     *
     * Return Values
     * CUFFT_INVALID_PLAN The plan parameter is not a valid handle.
     * CUFFT_SUCCESS The stream was successfully associated with the plan.
     * </pre>
     */
    public static int cufftSetStream(cufftHandle plan, cudaStream_t stream)
    {
        return checkResult(cufftSetStreamNative(plan, stream));
    }

    private static native int cufftSetStreamNative(cufftHandle plan, cudaStream_t stream);


    /**
     * <pre>
     * Configures the layout of CUFFT output in FFTW-compatible modes.
     *
     * When FFTW compatibility is desired, it can be configured for padding
     * only, for asymmetric complex inputs only, or to be fully compatible.
     *
     * Input
     * plan The cufftHandle object to associate with the stream
     * mode The cufftCompatibility option to be used:
     *     CUFFT_COMPATIBILITY_NATIVE:
     *     Disable any FFTW compatibility mode.
     *     CUFFT_COMPATIBILITY_FFTW_PADDING:
     *     Support FFTW data padding. (Default)
     *     CUFFT_COMPATIBILITY_FFTW_ASYMMETRIC:
     *     Waive the C2R symmetry requirement.
     *     Should be used with asymmetric input.
     *     CUFFT_COMPATIBILITY_FFTW_ALL:
     *     Enable full FFTW compatibility.
     *
     * Return Values
     * CUFFT_SETUP_FAILED CUFFT library failed to initialize.
     * CUFFT_INVALID_PLAN The plan parameter is not a valid handle.
     * CUFFT_SUCCESS CUFFT successfully executed the FFT plan.
     * </pre>
     */
    public static int cufftSetCompatibilityMode(cufftHandle plan, int mode)
    {
        return checkResult(cufftSetCompatibilityModeNative(plan, mode));
    }
    private static native int cufftSetCompatibilityModeNative(cufftHandle plan, int mode);



    //=== Single precision ===================================================

    /**
     * <pre>
     * Executes a CUFFT complex-to-complex transform plan.
     *
     * cufftResult cufftExecC2C( cufftHandle plan, cufftComplex *idata, cufftComplex *odata, int direction );
     *
     * CUFFT uses as input data the GPU memory pointed to by the idata parameter. This
     * function stores the Fourier coefficients in the odata array. If idata and
     * odata are the same, this method does an in-place transform.
     *
     * Input
     * ----
     * plan      The cufftHandle object for the plan to update
     * idata     Pointer to the input data (in GPU memory) to transform
     * odata     Pointer to the output data (in GPU memory)
     * direction The transform direction: CUFFT_FORWARD or CUFFT_INVERSE
     *
     * Output
     * ----
     * odata Contains the complex Fourier coefficients
     *
     * Return Values
     * ----
     * CUFFT_SETUP_FAILED  CUFFT library failed to initialize.
     * CUFFT_INVALID_PLAN  The plan parameter is not a valid handle.
     * CUFFT_INVALID_VALUE The idata, odata, and/or direction parameter is not valid.
     * CUFFT_EXEC_FAILED   CUFFT failed to execute the transform on GPU.
     * CUFFT_SUCCESS       CUFFT successfully executed the FFT plan
     *
     * JCUFFT_INTERNAL_ERROR If an internal JCufft error occurred
     * <pre>
     */
    public static int cufftExecC2C(cufftHandle plan, Pointer cIdata, Pointer cOdata, int direction)
    {
        return checkResult(cufftExecC2CNative(plan, cIdata, cOdata, direction));
    }
    private static native int cufftExecC2CNative(cufftHandle plan, Pointer cIdata, Pointer cOdata, int direction);


    /**
     * Convenience method for {@link JCufft#cufftExecC2C(cufftHandle, Pointer, Pointer, int)}.
     * Accepts arrays for input and output data and automatically performs the host-device
     * and device-host copies.
     *
     * @see JCufft#cufftExecC2C(cufftHandle, Pointer, Pointer, int)
     */
    public static int cufftExecC2C(cufftHandle plan, float cIdata[], float cOdata[], int direction)
    {
        int cudaResult = 0;

        boolean inPlace = (cIdata == cOdata);

        // Allocate space for the input data on the device
        Pointer hostCIdata = Pointer.to(cIdata);
        Pointer deviceCIdata = new Pointer();
        cudaResult = JCuda.cudaMalloc(deviceCIdata, cIdata.length * Sizeof.FLOAT);
        if (cudaResult != cudaError.cudaSuccess)
        {
            if (exceptionsEnabled)
            {
                throw new CudaException("JCuda error: "+cudaError.stringFor(cudaResult));
            }
            return cufftResult.JCUFFT_INTERNAL_ERROR;
        }

        // Set the output device data to be equal to the input
        // device data for in-place transforms, or allocate
        // the output device data if the transform is not
        // in-place
        Pointer hostCOdata = null;
        Pointer deviceCOdata = null;
        if (inPlace)
        {
            hostCOdata = hostCIdata;
            deviceCOdata = deviceCIdata;
        }
        else
        {
            hostCOdata = Pointer.to(cOdata);
            deviceCOdata = new Pointer();
            cudaResult = JCuda.cudaMalloc(deviceCOdata, cOdata.length * Sizeof.FLOAT);
            if (cudaResult != cudaError.cudaSuccess)
            {
                JCuda.cudaFree(deviceCIdata);
                if (exceptionsEnabled)
                {
                    throw new CudaException("JCuda error: "+cudaError.stringFor(cudaResult));
                }
                return cufftResult.JCUFFT_INTERNAL_ERROR;
            }
        }

        // Copy the host input data to the device
        cudaResult = JCuda.cudaMemcpy(deviceCIdata, hostCIdata, cIdata.length * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);
        if (cudaResult != cudaError.cudaSuccess)
        {
            JCuda.cudaFree(deviceCIdata);
            if (!inPlace)
            {
                JCuda.cudaFree(deviceCOdata);
            }

            if (exceptionsEnabled)
            {
                throw new CudaException("JCuda error: "+cudaError.stringFor(cudaResult));
            }
            return cufftResult.JCUFFT_INTERNAL_ERROR;
        }

        // Execute the transform
        int result = cufftResult.CUFFT_SUCCESS;
        try
        {
            result = JCufft.cufftExecC2C(plan, deviceCIdata, deviceCOdata, direction);
        }
        catch (CudaException e)
        {
            JCuda.cudaFree(deviceCIdata);
            if (!inPlace)
            {
                JCuda.cudaFree(deviceCOdata);
            }
            result = cufftResult.JCUFFT_INTERNAL_ERROR;
        }
        if (result != cufftResult.CUFFT_SUCCESS)
        {
            if (exceptionsEnabled)
            {
                throw new CudaException(cufftResult.stringFor(result));
            }
            return result;
        }

        // Copy the device output data to the host
        cudaResult = JCuda.cudaMemcpy(hostCOdata, deviceCOdata, cOdata.length * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost);
        if (cudaResult != cudaError.cudaSuccess)
        {
            JCuda.cudaFree(deviceCIdata);
            if (!inPlace)
            {
                JCuda.cudaFree(deviceCOdata);
            }
            if (exceptionsEnabled)
            {
                throw new CudaException("JCuda error: "+cudaError.stringFor(cudaResult));
            }
            return cufftResult.JCUFFT_INTERNAL_ERROR;
        }

        // Free the device data
        cudaResult = JCuda.cudaFree(deviceCIdata);
        if (cudaResult != cudaError.cudaSuccess)
        {
            if (exceptionsEnabled)
            {
                throw new CudaException("JCuda error: "+cudaError.stringFor(cudaResult));
            }
            return cufftResult.JCUFFT_INTERNAL_ERROR;
        }
        if (!inPlace)
        {
            cudaResult = JCuda.cudaFree(deviceCOdata);
            if (cudaResult != cudaError.cudaSuccess)
            {
                if (exceptionsEnabled)
                {
                    throw new CudaException("JCuda error: "+cudaError.stringFor(cudaResult));
                }
                return cufftResult.JCUFFT_INTERNAL_ERROR;
            }
        }
        return result;
    }



    /**
     * <pre>
     * Executes a CUFFT real-to-complex (implicitly forward) transform plan.
     *
     * cufftResult cufftExecR2C( cufftHandle plan, cufftReal *idata, cufftComplex *odata );
     *
     * CUFFT uses as input data the GPU memory pointed to by the idata
     * parameter. This function stores the non-redundant Fourier coefficients
     * in the odata array. If idata and odata are the same, this method does
     * an in-place transform (See CUFFT documentation for details on real
     * data FFTs.)
     *
     * Input
     * ----
     * plan      The cufftHandle object for the plan to update
     * idata     Pointer to the input data (in GPU memory) to transform
     * odata     Pointer to the output data (in GPU memory)
     * direction The transform direction: CUFFT_FORWARD or CUFFT_INVERSE
     *
     * Output
     * ----
     * odata Contains the complex Fourier coefficients
     *
     * Return Values
     * ----
     * CUFFT_SETUP_FAILED CUFFT library failed to initialize.
     * CUFFT_INVALID_PLAN The plan parameter is not a valid handle.
     * CUFFT_INVALID_VALUE The idata, odata, and/or direction parameter is not valid.
     * CUFFT_EXEC_FAILED CUFFT failed to execute the transform on GPU.
     * CUFFT_SUCCESS CUFFT successfully executed the FFT plan.
     *
     * JCUFFT_INTERNAL_ERROR If an internal JCufft error occurred
     * <pre>
     */
    public static int cufftExecR2C(cufftHandle plan, Pointer rIdata, Pointer cOdata)
    {
        return checkResult(cufftExecR2CNative(plan, rIdata, cOdata));
    }
    private static native int cufftExecR2CNative(cufftHandle plan, Pointer rIdata, Pointer cOdata);



    /**
     * Convenience method for {@link JCufft#cufftExecR2C(cufftHandle, Pointer, Pointer)}.
     * Accepts arrays for input and output data and automatically performs the host-device
     * and device-host copies.
     *
     * @see JCufft#cufftExecR2C(cufftHandle, Pointer, Pointer)
     */
    public static int cufftExecR2C(cufftHandle plan, float rIdata[], float cOdata[])
    {
        int cudaResult = 0;

        boolean inPlace = (rIdata == cOdata);

        // Allocate space for the input data on the device
        Pointer hostRIdata = Pointer.to(rIdata);
        Pointer deviceRIdata = new Pointer();
        cudaResult = JCuda.cudaMalloc(deviceRIdata, rIdata.length * Sizeof.FLOAT);
        if (cudaResult != cudaError.cudaSuccess)
        {
            if (exceptionsEnabled)
            {
                throw new CudaException("JCuda error: "+cudaError.stringFor(cudaResult));
            }
            return cufftResult.JCUFFT_INTERNAL_ERROR;
        }

        // Allocate the output device data
        Pointer hostCOdata = null;
        Pointer deviceCOdata = null;
        if (inPlace)
        {
            hostCOdata = hostRIdata;
            deviceCOdata = deviceRIdata;
        }
        else
        {
            hostCOdata = Pointer.to(cOdata);
            deviceCOdata = new Pointer();
            cudaResult = JCuda.cudaMalloc(deviceCOdata, cOdata.length * Sizeof.FLOAT);
            if (cudaResult != cudaError.cudaSuccess)
            {
                JCuda.cudaFree(deviceCOdata);
                if (exceptionsEnabled)
                {
                    throw new CudaException("JCuda error: "+cudaError.stringFor(cudaResult));
                }
                return cufftResult.JCUFFT_INTERNAL_ERROR;
            }
        }

        // Copy the host input data to the device
        cudaResult = JCuda.cudaMemcpy(deviceRIdata, hostRIdata, rIdata.length * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);
        if (cudaResult != cudaError.cudaSuccess)
        {
            JCuda.cudaFree(deviceRIdata);
            if (!inPlace)
            {
                JCuda.cudaFree(deviceCOdata);
            }
            if (exceptionsEnabled)
            {
                throw new CudaException("JCuda error: "+cudaError.stringFor(cudaResult));
            }
            return cufftResult.JCUFFT_INTERNAL_ERROR;
        }

        // Execute the transform
        int result = cufftResult.CUFFT_SUCCESS;
        try
        {
            result = JCufft.cufftExecR2C(plan, deviceRIdata, deviceCOdata);
        }
        catch (CudaException e)
        {
            JCuda.cudaFree(deviceRIdata);
            if (!inPlace)
            {
                JCuda.cudaFree(deviceCOdata);
            }
            result = cufftResult.JCUFFT_INTERNAL_ERROR;
        }
        if (result != cufftResult.CUFFT_SUCCESS)
        {
            if (exceptionsEnabled)
            {
                throw new CudaException(cufftResult.stringFor(cudaResult));
            }
            return result;
        }

        // Copy the device output data to the host
        cudaResult = JCuda.cudaMemcpy(hostCOdata, deviceCOdata, cOdata.length * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost);
        if (cudaResult != cudaError.cudaSuccess)
        {
            JCuda.cudaFree(deviceRIdata);
            if (!inPlace)
            {
                JCuda.cudaFree(deviceCOdata);
            }
            if (exceptionsEnabled)
            {
                throw new CudaException("JCuda error: "+cudaError.stringFor(cudaResult));
            }
            return cufftResult.JCUFFT_INTERNAL_ERROR;
        }

        // Free the device data
        cudaResult = JCuda.cudaFree(deviceRIdata);
        if (cudaResult != cudaError.cudaSuccess)
        {
            if (exceptionsEnabled)
            {
                throw new CudaException("JCuda error: "+cudaError.stringFor(cudaResult));
            }
            return cufftResult.JCUFFT_INTERNAL_ERROR;
        }
        if (!inPlace)
        {
            cudaResult = JCuda.cudaFree(deviceCOdata);
            if (cudaResult != cudaError.cudaSuccess)
            {
                if (exceptionsEnabled)
                {
                    throw new CudaException("JCuda error: "+cudaError.stringFor(cudaResult));
                }
                return cufftResult.JCUFFT_INTERNAL_ERROR;
            }
        }
        return result;
    }








    /**
     * <pre>
     * Executes a CUFFT complex-to-real (implicitly inverse) transform plan.
     *
     * cufftResult cufftExecC2R( cufftHandle plan, cufftComplex *idata, cufftReal *odata );
     *
     * CUFFT uses as input data the GPU memory pointed to by the idata
     * parameter. The input array holds only the non-redundant complex
     * Fourier coefficients. This function stores the real output values in the
     * odata array. If idata and odata are the same, this method does an inplace
     * transform. (See CUFFT documentation for details on real data FFTs.)
     *
     * Input
     * ----
     * plan The cufftHandle object for the plan to update
     * idata Pointer to the complex input data (in GPU memory) to transform
     * odata Pointer to the real output data (in GPU memory)
     *
     * Output
     * ----
     * odata Contains the real-valued output data
     *
     * Return Values
     * ----
     * CUFFT_SETUP_FAILED  CUFFT library failed to initialize.
     * CUFFT_INVALID_PLAN  The plan parameter is not a valid handle.
     * CUFFT_INVALID_VALUE The idata and/or odata parameter is not valid.
     * CUFFT_EXEC_FAILED   CUFFT failed to execute the transform on GPU.
     * CUFFT_SUCCESS       CUFFT successfully executed the FFT plan.
     *
     * JCUFFT_INTERNAL_ERROR If an internal JCufft error occurred
     * <pre>
     */
    public static int cufftExecC2R(cufftHandle plan, Pointer cIdata, Pointer rOdata)
    {
        return checkResult(cufftExecC2RNative(plan, cIdata, rOdata));
    }
    private static native int cufftExecC2RNative(cufftHandle plan, Pointer cIdata, Pointer rOdata);



    /**
     * Convenience method for {@link JCufft#cufftExecC2R(cufftHandle, Pointer, Pointer)}.
     * Accepts arrays for input and output data and automatically performs the host-device
     * and device-host copies.
     *
     * @see JCufft#cufftExecC2R(cufftHandle, Pointer, Pointer)
     */
    public static int cufftExecC2R(cufftHandle plan, float cIdata[], float rOdata[])
    {
        int cudaResult = 0;

        boolean inPlace = (cIdata == rOdata);

        // Allocate space for the input data on the device
        Pointer hostCIdata = Pointer.to(cIdata);
        Pointer deviceCIdata = new Pointer();
        cudaResult = JCuda.cudaMalloc(deviceCIdata, cIdata.length * Sizeof.FLOAT);
        if (cudaResult != cudaError.cudaSuccess)
        {
            if (exceptionsEnabled)
            {
                throw new CudaException("JCuda error: "+cudaError.stringFor(cudaResult));
            }
            return cufftResult.JCUFFT_INTERNAL_ERROR;
        }

        // Allocate the output device data
        Pointer hostROdata = null;
        Pointer deviceROdata = null;
        if (inPlace)
        {
            hostROdata = hostCIdata;
            deviceROdata = deviceCIdata;
        }
        else
        {
            hostROdata = Pointer.to(rOdata);
            deviceROdata = new Pointer();
            cudaResult = JCuda.cudaMalloc(deviceROdata, rOdata.length * Sizeof.FLOAT);
            if (cudaResult != cudaError.cudaSuccess)
            {
                JCuda.cudaFree(deviceCIdata);
                if (exceptionsEnabled)
                {
                    throw new CudaException("JCuda error: "+cudaError.stringFor(cudaResult));
                }
                return cufftResult.JCUFFT_INTERNAL_ERROR;
            }
        }

        // Copy the host input data to the device
        cudaResult = JCuda.cudaMemcpy(deviceCIdata, hostCIdata, cIdata.length * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);
        if (cudaResult != cudaError.cudaSuccess)
        {
            JCuda.cudaFree(deviceCIdata);
            if (!inPlace)
            {
                JCuda.cudaFree(deviceROdata);
            }
            if (exceptionsEnabled)
            {
                throw new CudaException("JCuda error: "+cudaError.stringFor(cudaResult));
            }
            return cufftResult.JCUFFT_INTERNAL_ERROR;
        }

        // Execute the transform
        int result = cufftResult.CUFFT_SUCCESS;
        try
        {
            result = JCufft.cufftExecC2R(plan, deviceCIdata, deviceROdata);
        }
        catch (CudaException e)
        {
            JCuda.cudaFree(deviceCIdata);
            if (!inPlace)
            {
                JCuda.cudaFree(deviceROdata);
            }
            result = cufftResult.JCUFFT_INTERNAL_ERROR;
        }
        if (result != cufftResult.CUFFT_SUCCESS)
        {
            if (exceptionsEnabled)
            {
                throw new CudaException(cufftResult.stringFor(cudaResult));
            }
            return result;
        }

        // Copy the device output data to the host
        cudaResult = JCuda.cudaMemcpy(hostROdata, deviceROdata, rOdata.length * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost);
        if (cudaResult != cudaError.cudaSuccess)
        {
            JCuda.cudaFree(deviceCIdata);
            if (!inPlace)
            {
                JCuda.cudaFree(deviceROdata);
            }
            if (exceptionsEnabled)
            {
                throw new CudaException("JCuda error: "+cudaError.stringFor(cudaResult));
            }
            return cufftResult.JCUFFT_INTERNAL_ERROR;
        }

        // Free the device data
        cudaResult = JCuda.cudaFree(deviceCIdata);
        if (cudaResult != cudaError.cudaSuccess)
        {
            if (exceptionsEnabled)
            {
                throw new CudaException("JCuda error: "+cudaError.stringFor(cudaResult));
            }
            return cufftResult.JCUFFT_INTERNAL_ERROR;
        }
        if (!inPlace)
        {
            cudaResult = JCuda.cudaFree(deviceROdata);
            if (cudaResult != cudaError.cudaSuccess)
            {
                if (exceptionsEnabled)
                {
                    throw new CudaException("JCuda error: "+cudaError.stringFor(cudaResult));
                }
                return cufftResult.JCUFFT_INTERNAL_ERROR;
            }
        }
        return result;
    }





    //=== Double precision ===================================================

    /**
     * <pre>
     * Executes a CUFFT complex-to-complex transform plan for double precision
     * values.
     *
     * cufftResult cufftExecZ2Z( cufftHandle plan, cufftDoubleComplex *idata, cufftDoubleComplex *odata, int direction );
     *
     * CUFFT uses as input data the GPU memory pointed to by the idata parameter. This
     * function stores the Fourier coefficients in the odata array. If idata and
     * odata are the same, this method does an in-place transform.
     *
     * Input
     * ----
     * plan      The cufftHandle object for the plan to update
     * idata     Pointer to the input data (in GPU memory) to transform
     * odata     Pointer to the output data (in GPU memory)
     * direction The transform direction: CUFFT_FORWARD or CUFFT_INVERSE
     *
     * Output
     * ----
     * odata Contains the complex Fourier coefficients
     *
     * Return Values
     * ----
     * CUFFT_SETUP_FAILED  CUFFT library failed to initialize.
     * CUFFT_INVALID_PLAN  The plan parameter is not a valid handle.
     * CUFFT_INVALID_VALUE The idata, odata, and/or direction parameter is not valid.
     * CUFFT_EXEC_FAILED   CUFFT failed to execute the transform on GPU.
     * CUFFT_SUCCESS       CUFFT successfully executed the FFT plan
     *
     * JCUFFT_INTERNAL_ERROR If an internal JCufft error occurred
     * <pre>
     */
    public static int cufftExecZ2Z(cufftHandle plan, Pointer cIdata, Pointer cOdata, int direction)
    {
        return checkResult(cufftExecZ2ZNative(plan, cIdata, cOdata, direction));
    }
    private static native int cufftExecZ2ZNative(cufftHandle plan, Pointer cIdata, Pointer cOdata, int direction);


    /**
     * Convenience method for {@link JCufft#cufftExecZ2Z(cufftHandle, Pointer, Pointer, int)}.
     * Accepts arrays for input and output data and automatically performs the host-device
     * and device-host copies.
     *
     * @see JCufft#cufftExecZ2Z(cufftHandle, Pointer, Pointer, int)
     */
    public static int cufftExecZ2Z(cufftHandle plan, double cIdata[], double cOdata[], int direction)
    {
        int cudaResult = 0;

        boolean inPlace = (cIdata == cOdata);

        // Allocate space for the input data on the device
        Pointer hostCIdata = Pointer.to(cIdata);
        Pointer deviceCIdata = new Pointer();
        cudaResult = JCuda.cudaMalloc(deviceCIdata, cIdata.length * Sizeof.DOUBLE);
        if (cudaResult != cudaError.cudaSuccess)
        {
            if (exceptionsEnabled)
            {
                throw new CudaException("JCuda error: "+cudaError.stringFor(cudaResult));
            }
            return cufftResult.JCUFFT_INTERNAL_ERROR;
        }

        // Set the output device data to be equal to the input
        // device data for in-place transforms, or allocate
        // the output device data if the transform is not
        // in-place
        Pointer hostCOdata = null;
        Pointer deviceCOdata = null;
        if (inPlace)
        {
            hostCOdata = hostCIdata;
            deviceCOdata = deviceCIdata;
        }
        else
        {
            hostCOdata = Pointer.to(cOdata);
            deviceCOdata = new Pointer();
            cudaResult = JCuda.cudaMalloc(deviceCOdata, cOdata.length * Sizeof.DOUBLE);
            if (cudaResult != cudaError.cudaSuccess)
            {
                JCuda.cudaFree(deviceCIdata);
                if (exceptionsEnabled)
                {
                    throw new CudaException("JCuda error: "+cudaError.stringFor(cudaResult));
                }
                return cufftResult.JCUFFT_INTERNAL_ERROR;
            }
        }

        // Copy the host input data to the device
        cudaResult = JCuda.cudaMemcpy(deviceCIdata, hostCIdata, cIdata.length * Sizeof.DOUBLE, cudaMemcpyKind.cudaMemcpyHostToDevice);
        if (cudaResult != cudaError.cudaSuccess)
        {
            JCuda.cudaFree(deviceCIdata);
            if (!inPlace)
            {
                JCuda.cudaFree(deviceCOdata);
            }

            if (exceptionsEnabled)
            {
                throw new CudaException("JCuda error: "+cudaError.stringFor(cudaResult));
            }
            return cufftResult.JCUFFT_INTERNAL_ERROR;
        }

        // Execute the transform
        int result = cufftResult.CUFFT_SUCCESS;
        try
        {
            result = JCufft.cufftExecZ2Z(plan, deviceCIdata, deviceCOdata, direction);
        }
        catch (CudaException e)
        {
            JCuda.cudaFree(deviceCIdata);
            if (!inPlace)
            {
                JCuda.cudaFree(deviceCOdata);
            }
            result = cufftResult.JCUFFT_INTERNAL_ERROR;
        }
        if (result != cufftResult.CUFFT_SUCCESS)
        {
            if (exceptionsEnabled)
            {
                throw new CudaException(cufftResult.stringFor(cudaResult));
            }
            return result;
        }

        // Copy the device output data to the host
        cudaResult = JCuda.cudaMemcpy(hostCOdata, deviceCOdata, cOdata.length * Sizeof.DOUBLE, cudaMemcpyKind.cudaMemcpyDeviceToHost);
        if (cudaResult != cudaError.cudaSuccess)
        {
            JCuda.cudaFree(deviceCIdata);
            if (!inPlace)
            {
                JCuda.cudaFree(deviceCOdata);
            }
            if (exceptionsEnabled)
            {
                throw new CudaException("JCuda error: "+cudaError.stringFor(cudaResult));
            }
            return cufftResult.JCUFFT_INTERNAL_ERROR;
        }

        // Free the device data
        cudaResult = JCuda.cudaFree(deviceCIdata);
        if (cudaResult != cudaError.cudaSuccess)
        {
            if (exceptionsEnabled)
            {
                throw new CudaException("JCuda error: "+cudaError.stringFor(cudaResult));
            }
            return cufftResult.JCUFFT_INTERNAL_ERROR;
        }
        if (!inPlace)
        {
            cudaResult = JCuda.cudaFree(deviceCOdata);
            if (cudaResult != cudaError.cudaSuccess)
            {
                if (exceptionsEnabled)
                {
                    throw new CudaException("JCuda error: "+cudaError.stringFor(cudaResult));
                }
                return cufftResult.JCUFFT_INTERNAL_ERROR;
            }
        }
        return result;
    }



    /**
     * <pre>
     * Executes a CUFFT real-to-complex (implicitly forward) transform plan
     * for double precision values.
     *
     * cufftResult cufftExecD2Z( cufftHandle plan, cufftDoubleReal *idata, cufftDoubleComplex *odata );
     *
     * CUFFT uses as input data the GPU memory pointed to by the idata
     * parameter. This function stores the non-redundant Fourier coefficients
     * in the odata array. If idata and odata are the same, this method does
     * an in-place transform (See CUFFT documentation for details on real
     * data FFTs.)
     *
     * Input
     * ----
     * plan      The cufftHandle object for the plan to update
     * idata     Pointer to the input data (in GPU memory) to transform
     * odata     Pointer to the output data (in GPU memory)
     * direction The transform direction: CUFFT_FORWARD or CUFFT_INVERSE
     *
     * Output
     * ----
     * odata Contains the complex Fourier coefficients
     *
     * Return Values
     * ----
     * CUFFT_SETUP_FAILED CUFFT library failed to initialize.
     * CUFFT_INVALID_PLAN The plan parameter is not a valid handle.
     * CUFFT_INVALID_VALUE The idata, odata, and/or direction parameter is not valid.
     * CUFFT_EXEC_FAILED CUFFT failed to execute the transform on GPU.
     * CUFFT_SUCCESS CUFFT successfully executed the FFT plan.
     *
     * JCUFFT_INTERNAL_ERROR If an internal JCufft error occurred
     * <pre>
     */
    public static int cufftExecD2Z(cufftHandle plan, Pointer rIdata, Pointer cOdata)
    {
        return checkResult(cufftExecD2ZNative(plan, rIdata, cOdata));
    }
    private static native int cufftExecD2ZNative(cufftHandle plan, Pointer rIdata, Pointer cOdata);



    /**
     * Convenience method for {@link JCufft#cufftExecD2Z(cufftHandle, Pointer, Pointer)}.
     * Accepts arrays for input and output data and automatically performs the host-device
     * and device-host copies.
     *
     * @see JCufft#cufftExecD2Z(cufftHandle, Pointer, Pointer)
     */
    public static int cufftExecD2Z(cufftHandle plan, double rIdata[], double cOdata[])
    {
        int cudaResult = 0;

        boolean inPlace = (rIdata == cOdata);

        // Allocate space for the input data on the device
        Pointer hostRIdata = Pointer.to(rIdata);
        Pointer deviceRIdata = new Pointer();
        cudaResult = JCuda.cudaMalloc(deviceRIdata, rIdata.length * Sizeof.DOUBLE);
        if (cudaResult != cudaError.cudaSuccess)
        {
            if (exceptionsEnabled)
            {
                throw new CudaException("JCuda error: "+cudaError.stringFor(cudaResult));
            }
            return cufftResult.JCUFFT_INTERNAL_ERROR;
        }

        // Allocate the output device data
        Pointer hostCOdata = null;
        Pointer deviceCOdata = null;
        if (inPlace)
        {
            hostCOdata = hostRIdata;
            deviceCOdata = deviceRIdata;
        }
        else
        {
            hostCOdata = Pointer.to(cOdata);
            deviceCOdata = new Pointer();
            cudaResult = JCuda.cudaMalloc(deviceCOdata, cOdata.length * Sizeof.DOUBLE);
            if (cudaResult != cudaError.cudaSuccess)
            {
                JCuda.cudaFree(deviceCOdata);
                if (exceptionsEnabled)
                {
                    throw new CudaException("JCuda error: "+cudaError.stringFor(cudaResult));
                }
                return cufftResult.JCUFFT_INTERNAL_ERROR;
            }
        }

        // Copy the host input data to the device
        cudaResult = JCuda.cudaMemcpy(deviceRIdata, hostRIdata, rIdata.length * Sizeof.DOUBLE, cudaMemcpyKind.cudaMemcpyHostToDevice);
        if (cudaResult != cudaError.cudaSuccess)
        {
            JCuda.cudaFree(deviceRIdata);
            if (!inPlace)
            {
                JCuda.cudaFree(deviceCOdata);
            }
            if (exceptionsEnabled)
            {
                throw new CudaException("JCuda error: "+cudaError.stringFor(cudaResult));
            }
            return cufftResult.JCUFFT_INTERNAL_ERROR;
        }

        // Execute the transform
        int result = cufftResult.CUFFT_SUCCESS;
        try
        {
            result = JCufft.cufftExecD2Z(plan, deviceRIdata, deviceCOdata);
        }
        catch (CudaException e)
        {
            JCuda.cudaFree(deviceRIdata);
            if (!inPlace)
            {
                JCuda.cudaFree(deviceCOdata);
            }
            result = cufftResult.JCUFFT_INTERNAL_ERROR;
        }
        if (result != cufftResult.CUFFT_SUCCESS)
        {
            if (exceptionsEnabled)
            {
                throw new CudaException(cufftResult.stringFor(cudaResult));
            }
            return result;
        }

        // Copy the device output data to the host
        cudaResult = JCuda.cudaMemcpy(hostCOdata, deviceCOdata, cOdata.length * Sizeof.DOUBLE, cudaMemcpyKind.cudaMemcpyDeviceToHost);
        if (cudaResult != cudaError.cudaSuccess)
        {
            JCuda.cudaFree(deviceRIdata);
            if (!inPlace)
            {
                JCuda.cudaFree(deviceCOdata);
            }
            if (exceptionsEnabled)
            {
                throw new CudaException("JCuda error: "+cudaError.stringFor(cudaResult));
            }
            return cufftResult.JCUFFT_INTERNAL_ERROR;
        }

        // Free the device data
        cudaResult = JCuda.cudaFree(deviceRIdata);
        if (cudaResult != cudaError.cudaSuccess)
        {
            if (exceptionsEnabled)
            {
                throw new CudaException("JCuda error: "+cudaError.stringFor(cudaResult));
            }
            return cufftResult.JCUFFT_INTERNAL_ERROR;
        }
        if (!inPlace)
        {
            cudaResult = JCuda.cudaFree(deviceCOdata);
            if (cudaResult != cudaError.cudaSuccess)
            {
                if (exceptionsEnabled)
                {
                    throw new CudaException("JCuda error: "+cudaError.stringFor(cudaResult));
                }
                return cufftResult.JCUFFT_INTERNAL_ERROR;
            }
        }
        return result;
    }





    /**
     * <pre>
     * Executes a CUFFT complex-to-real (implicitly inverse) transform plan
     * for double precision values.
     *
     * cufftResult cufftExecZ2D( cufftHandle plan, cufftDoubleComplex *idata, cufftDoubleReal *odata );
     *
     * CUFFT uses as input data the GPU memory pointed to by the idata
     * parameter. The input array holds only the non-redundant complex
     * Fourier coefficients. This function stores the real output values in the
     * odata array. If idata and odata are the same, this method does an inplace
     * transform. (See CUFFT documentation for details on real data FFTs.)
     *
     * Input
     * ----
     * plan The cufftHandle object for the plan to update
     * idata Pointer to the complex input data (in GPU memory) to transform
     * odata Pointer to the real output data (in GPU memory)
     *
     * Output
     * ----
     * odata Contains the real-valued output data
     *
     * Return Values
     * ----
     * CUFFT_SETUP_FAILED  CUFFT library failed to initialize.
     * CUFFT_INVALID_PLAN  The plan parameter is not a valid handle.
     * CUFFT_INVALID_VALUE The idata and/or odata parameter is not valid.
     * CUFFT_EXEC_FAILED   CUFFT failed to execute the transform on GPU.
     * CUFFT_SUCCESS       CUFFT successfully executed the FFT plan.
     *
     * JCUFFT_INTERNAL_ERROR If an internal JCufft error occurred
     * <pre>
     */
    public static int cufftExecZ2D(cufftHandle plan, Pointer cIdata, Pointer rOdata)
    {
        return checkResult(cufftExecZ2DNative(plan, cIdata, rOdata));
    }
    private static native int cufftExecZ2DNative(cufftHandle plan, Pointer cIdata, Pointer rOdata);



    /**
     * Convenience method for {@link JCufft#cufftExecZ2D(cufftHandle, Pointer, Pointer)}.
     * Accepts arrays for input and output data and automatically performs the host-device
     * and device-host copies.
     *
     * @see JCufft#cufftExecZ2D(cufftHandle, Pointer, Pointer)
     */
    public static int cufftExecZ2D(cufftHandle plan, double cIdata[], double rOdata[])
    {
        int cudaResult = 0;

        boolean inPlace = (cIdata == rOdata);

        // Allocate space for the input data on the device
        Pointer hostCIdata = Pointer.to(cIdata);
        Pointer deviceCIdata = new Pointer();
        cudaResult = JCuda.cudaMalloc(deviceCIdata, cIdata.length * Sizeof.DOUBLE);
        if (cudaResult != cudaError.cudaSuccess)
        {
            if (exceptionsEnabled)
            {
                throw new CudaException("JCuda error: "+cudaError.stringFor(cudaResult));
            }
            return cufftResult.JCUFFT_INTERNAL_ERROR;
        }

        // Allocate the output device data
        Pointer hostROdata = null;
        Pointer deviceROdata = null;
        if (inPlace)
        {
            hostROdata = hostCIdata;
            deviceROdata = deviceCIdata;
        }
        else
        {
            hostROdata = Pointer.to(rOdata);
            deviceROdata = new Pointer();
            cudaResult = JCuda.cudaMalloc(deviceROdata, rOdata.length * Sizeof.DOUBLE);
            if (cudaResult != cudaError.cudaSuccess)
            {
                JCuda.cudaFree(deviceCIdata);
                if (exceptionsEnabled)
                {
                    throw new CudaException("JCuda error: "+cudaError.stringFor(cudaResult));
                }
                return cufftResult.JCUFFT_INTERNAL_ERROR;
            }
        }

        // Copy the host input data to the device
        cudaResult = JCuda.cudaMemcpy(deviceCIdata, hostCIdata, cIdata.length * Sizeof.DOUBLE, cudaMemcpyKind.cudaMemcpyHostToDevice);
        if (cudaResult != cudaError.cudaSuccess)
        {
            JCuda.cudaFree(deviceCIdata);
            if (!inPlace)
            {
                JCuda.cudaFree(deviceROdata);
            }
            if (exceptionsEnabled)
            {
                throw new CudaException("JCuda error: "+cudaError.stringFor(cudaResult));
            }
            return cufftResult.JCUFFT_INTERNAL_ERROR;
        }

        // Execute the transform
        int result = cufftResult.CUFFT_SUCCESS;
        try
        {
            result = JCufft.cufftExecZ2D(plan, deviceCIdata, deviceROdata);
        }
        catch (CudaException e)
        {
            JCuda.cudaFree(deviceCIdata);
            if (!inPlace)
            {
                JCuda.cudaFree(deviceROdata);
            }
            result = cufftResult.JCUFFT_INTERNAL_ERROR;
        }
        if (result != cufftResult.CUFFT_SUCCESS)
        {
            if (exceptionsEnabled)
            {
                throw new CudaException(cufftResult.stringFor(cudaResult));
            }
            return result;
        }

        // Copy the device output data to the host
        cudaResult = JCuda.cudaMemcpy(hostROdata, deviceROdata, rOdata.length * Sizeof.DOUBLE, cudaMemcpyKind.cudaMemcpyDeviceToHost);
        if (cudaResult != cudaError.cudaSuccess)
        {
            JCuda.cudaFree(deviceCIdata);
            if (!inPlace)
            {
                JCuda.cudaFree(deviceROdata);
            }
            if (exceptionsEnabled)
            {
                throw new CudaException("JCuda error: "+cudaError.stringFor(cudaResult));
            }
            return cufftResult.JCUFFT_INTERNAL_ERROR;
        }

        // Free the device data
        cudaResult = JCuda.cudaFree(deviceCIdata);
        if (cudaResult != cudaError.cudaSuccess)
        {
            if (exceptionsEnabled)
            {
                throw new CudaException("JCuda error: "+cudaError.stringFor(cudaResult));
            }
            return cufftResult.JCUFFT_INTERNAL_ERROR;
        }
        if (!inPlace)
        {
            cudaResult = JCuda.cudaFree(deviceROdata);
            if (cudaResult != cudaError.cudaSuccess)
            {
                if (exceptionsEnabled)
                {
                    throw new CudaException("JCuda error: "+cudaError.stringFor(cudaResult));
                }
                return cufftResult.JCUFFT_INTERNAL_ERROR;
            }
        }
        return result;
    }

}




