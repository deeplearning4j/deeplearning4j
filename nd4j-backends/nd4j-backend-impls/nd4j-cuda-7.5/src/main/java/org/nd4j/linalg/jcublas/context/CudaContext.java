package org.nd4j.linalg.jcublas.context;

import jcuda.driver.CUcontext;
import jcuda.driver.CUstream;
import jcuda.driver.CUstream_flags;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaStream_t;
import lombok.Data;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
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
public class CudaContext {
    private CUcontext context;
    private CUstream stream;
    //private CUevent cUevent;
    private cudaStream_t oldStream;
    private cudaStream_t cublasStream;
    //private cudaEvent_t oldEvent;
    private cublasHandle handle;
    private CublasPointer resultPointer;
    private AtomicBoolean oldStreamReturned = new AtomicBoolean(false);
    private AtomicBoolean handleReturned = new AtomicBoolean(false);
    private AtomicBoolean streamReturned = new AtomicBoolean(false);
    private boolean streamFromPool = true;
    private boolean handleFromPool = true;
    private boolean oldStreamFromPool = true;
    private boolean free = true;
    private boolean oldEventDestroyed = true;
    private boolean eventDestroyed = true;


    public CudaContext(boolean free) {
        this();
        this.free = free;
    }


    public CudaContext() {
        //ContextHolder.getInstance().setContext();
    }

    /**
     * Synchronizes on the new
     * stream
     */
    public void syncStream() {
        JCudaDriver.cuStreamSynchronize(stream);}

    /**
     * Synchronizes
     * on the old stream
     */
    public void syncOldStream() {
        //ContextHolder.getInstance().setContext();
        JCuda.cudaStreamSynchronize(oldStream);

        syncCublasStream();
    }

    public void syncCublasStream() {
        if (cublasStream != null) {
            JCuda.cudaStreamSynchronize(cublasStream);
        }
    }


    /**
     * Associates
     * the handle on this context
     * to the given stream
     */
    public synchronized  void associateHandle() {
        //JCublas2.cublasSetStream(handle,oldStream);
    }




    /**
     * Initializes the stream
     */
    public void initStream() {
        //ContextHolder.getInstance().setContext();
        if(stream == null) {
            stream = new CUstream();
            JCudaDriver.cuStreamCreate(stream, CUstream_flags.CU_STREAM_DEFAULT);
            streamFromPool = false;
            eventDestroyed = false;
        }
    }

    /**
     * Initializes the old stream
     */
    public void initOldStream() {
        //ContextHolder.getInstance().setContext();
        if(oldStream == null)  {
            oldStreamFromPool = false;
            oldStream = new cudaStream_t();
            JCuda.cudaStreamCreate(oldStream);
        }

    }




    /**
     * Initializes a handle and
     * associates with the given stream.
     * initOldStream() should be called first
     *
     */
    public void initHandle() {
        /*

        We don't create handles here anymore

        if(handle == null) {
            handle = new cublasHandle();
            JCublas2.cublasCreate(handle);
            handleFromPool = false;
        }
        */
    }

    /**
     * Destroys the context
     * and associated resources
     */
    @Deprecated
    public void destroy(CublasPointer resultPointer,boolean freeIfNotEqual) {
    }


    /**
     * Destroys the context
     * and associated resources
     */
    @Deprecated
    public void destroy() {

    }


    /**
     * Finishes a blas operation
     * and destroys this context
     */
    public void finishBlasOperation() {
        //destroy();
    }

    /**
     * Sets up a context with an old stream
     * and a blas handle
     * @return the cuda context
     * as setup for cublas usage
     */
    public static CudaContext getBlasContext() {
        CudaContext context = (CudaContext) AtomicAllocator.getInstance().getDeviceContext().getContext();
        return context;
    }

}
