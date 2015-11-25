package org.nd4j.linalg.jcublas.context;

import jcuda.driver.CUstream;
import jcuda.driver.CUstream_flags;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaStream_t;
import lombok.Data;
import org.nd4j.linalg.jcublas.CublasPointer;

import java.util.concurrent.atomic.AtomicBoolean;

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
    private CublasPointer resultPointer;
    private AtomicBoolean oldStreamReturned = new AtomicBoolean(false);
    private AtomicBoolean handleReturned = new AtomicBoolean(false);
    private AtomicBoolean streamReturned = new AtomicBoolean(false);
    private boolean streamFromPool = true;
    private boolean handleFromPool = true;
    private boolean oldStreamFromPool = true;
    private boolean free = true;


    public CudaContext(boolean free) {
        this();
        this.free = free;
    }


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

    public CublasPointer getResultPointer() {
        return resultPointer;
    }

    /**
     * Associates
     * the handle on this context
     * to the given stream
     */
    public void associateHandle() {
        ContextHolder.getInstance().setContext();
        JCublas2.cublasSetStream(handle, oldStream);
    }

    /**
     * Initializes the stream
     */
    public void initStream() {
        if(stream == null) {
            try {
                stream = ContextHolder.getInstance().getStreamPool().borrowObject();
            } catch (Exception e) {
                stream = new CUstream();
                JCudaDriver.cuStreamCreate(stream, CUstream_flags.CU_STREAM_NON_BLOCKING);
                streamFromPool = false;
            }
        }


    }

    /**
     * Initializes the old stream
     */
    public void initOldStream() {
        if(oldStream == null)  {
            try {
                oldStream = ContextHolder.getInstance().getOldStreamPool().borrowObject();
            } catch (Exception e) {
                oldStreamFromPool = false;
                oldStream = new cudaStream_t();
                JCuda.cudaStreamCreate(oldStream);

            }
        }


    }


    /**
     * Initializes a handle and
     * associates with the given stream.
     * initOldStream() should be called first
     *
     */
    public void initHandle() {
        if(handle == null) {
            try {
                handle = ContextHolder.getInstance().getHandlePool().borrowObject();
            } catch (Exception e) {
                handle = new cublasHandle();
                JCublas2.cublasCreate(handle);
                handleFromPool = false;
            }
            associateHandle();
        }

    }

    /**
     * Destroys the context
     * and associated resources
     */
    public void destroy(CublasPointer resultPointer,boolean freeIfNotEqual) {
        if(handle != null && !handleReturned.get()) {
            try {
                if(handleFromPool)
                    ContextHolder.getInstance().getHandlePool().returnObject(handle);
                else {
                    JCublas2.cublasDestroy(handle);

                }
                handleReturned.set(true);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        if(stream != null && !streamReturned.get()) {
            try {
                if(streamFromPool)
                    ContextHolder.getInstance().getStreamPool().returnObject(stream);
                else {
                    JCudaDriver.cuStreamDestroy(stream);

                }
                streamReturned.set(true);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        if(oldStream != null && !oldStreamReturned.get()) {
            try {
                if(oldStreamFromPool)
                    ContextHolder.getInstance().getOldStreamPool().returnObject(oldStream);
                else {
                    JCuda.cudaStreamDestroy(oldStream);

                }
                oldStreamReturned.set(true);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        if(resultPointer != null && freeIfNotEqual && freeIfNotEqual) {
            resultPointer.copyToHost();
            try {
                resultPointer.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }


    /**
     * Destroys the context
     * and associated resources
     */
    public void destroy() {
        if(handle != null && !handleReturned.get()) {
            try {
                if(handleFromPool)
                    ContextHolder.getInstance().getHandlePool().returnObject(handle);
                else {
                    JCublas2.cublasDestroy(handle);

                }
                handleReturned.set(true);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        if(stream != null && !streamReturned.get()) {
            try {
                if(streamFromPool)
                    ContextHolder.getInstance().getStreamPool().returnObject(stream);
                else {
                    JCudaDriver.cuStreamDestroy(stream);

                }
                streamReturned.set(true);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        if(oldStream != null && !oldStreamReturned.get()) {
            try {
                if(oldStreamFromPool)
                    ContextHolder.getInstance().getOldStreamPool().returnObject(oldStream);
                else {
                    JCuda.cudaStreamDestroy(oldStream);

                }
                oldStreamReturned.set(true);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        if(resultPointer != null) {
            resultPointer.copyToHost();
            try {
                resultPointer.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }


    /**
     * Finishes a blas operation
     * and destroys this context
     */
    public void finishBlasOperation() {
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

    /**
     * Calls cuda device synchronize
     */
    public void syncDevice() {
        JCuda.cudaDeviceSynchronize();
    }

    @Override
    public void close() throws Exception {
        destroy();
    }
}
