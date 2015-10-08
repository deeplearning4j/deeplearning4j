package org.nd4j.linalg.jcublas.context;

import jcuda.driver.CUstream;
import jcuda.driver.CUstream_flags;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaStream_t;
import lombok.Data;

/**
 * A higher level class for handling
 * the different primitives around the cuda apis
 * This being:
 * streams (both old and new) as well as
 * the cublas handles.
 *
 *
 */
@Data
public class CudaContext implements AutoCloseable {
    private CUstream stream;
    private cudaStream_t oldStream;
    private cublasHandle handle;


    public CudaContext() {
        ContextHolder.getInstance().setContext();
    }

    /**
     * Synchronizes on the new
     * stream
     */
    public void syncStream() {
        JCudaDriver.cuStreamSynchronize(stream);
    }

    /**
     * Synchronizes
     * on the old stream
     */
    public void syncOldStream() {
        JCuda.cudaStreamSynchronize(oldStream);
    }

    /**
     * Synchrnonizes on
     * the old stream
     * since the given handle
     * will be associated with the
     * stream for this context
     */
    public void syncHandle() {
        syncOldStream();
    }


    /**
     * Associates
     * the handle on this context
     * to the given stream
     */
    public void associateHandle() {
        JCublas2.cublasSetStream(handle, oldStream);
    }

    /**
     * Initializes the stream
     */
    public void initStream() {
        stream = new CUstream();
        JCudaDriver.cuStreamCreate(stream, CUstream_flags.CU_STREAM_NON_BLOCKING);

    }

    /**
     * Initializes the old stream
     */
    public void initOldStream() {
        oldStream = new cudaStream_t();
        JCuda.cudaStreamCreate(oldStream);

    }


    /**
     * Initializes a handle and
     * associates with the given stream.
     * initOldStream() should be called first
     *
     */
    public void initHandle() {
        handle = new cublasHandle();
        JCublas2.cublasCreate(handle);
        associateHandle();
    }

    /**
     * Destroys the context
     * and associated resources
     */
    public void destroy() {
        if(handle != null) {
            JCublas2.cublasDestroy(handle);
        }
        if(stream != null) {
            JCudaDriver.cuStreamDestroy(stream);
        }
        if(oldStream != null) {
            JCuda.cudaStreamDestroy(oldStream);
        }
    }


    /**
     * Finishes a blas operation
     * and destroys this context
     */
    public void finishBlasOperation() {
        syncOldStream();
        destroy();
    }

    /**
     * Sets up a context with an old stream
     * and a blas handle
     * @return the cuda context
     * as setup for cublas usage
     */
    public static CudaContext getBlasContext() {
        CudaContext ctx = new CudaContext();
        ctx.initOldStream();
        ctx.initHandle();
        return ctx;
    }

    @Override
    public void close() throws Exception {
        destroy();
    }
}
