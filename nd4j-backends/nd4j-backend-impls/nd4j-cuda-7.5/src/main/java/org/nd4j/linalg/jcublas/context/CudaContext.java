package org.nd4j.linalg.jcublas.context;

import jcuda.driver.CUevent;
import jcuda.driver.CUstream;
import jcuda.driver.CUstream_flags;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaEvent_t;
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
    private CUstream stream;
    //private CUevent cUevent;
    private cudaStream_t oldStream;
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
        ContextHolder.getInstance().setContext();
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
        ContextHolder.getInstance().setContext();
        JCuda.cudaStreamSynchronize(oldStream);
/*
        if(!oldEventDestroyed) {
           JCuda.cudaStreamSynchronize(oldStream);
            JCuda.cudaStreamWaitEvent(oldStream,oldEvent,0);
            JCuda.cudaEventDestroy(oldEvent);
            oldEventDestroyed = true;

        }*/
    }

    /**
     * Synchronizes on
     * the old stream
     * since the given handle
     * will be associated with the
     * stream for this context
     */
    public void syncHandle() {
        syncOldStream();
    }

    /**
     * Get the result pointer for the context
     * @return
     */
    public CublasPointer getResultPointer() {
        return resultPointer;
    }

    /**
     * Associates
     * the handle on this context
     * to the given stream
     */
    public synchronized  void associateHandle() {
        JCublas2.cublasSetStream(handle,oldStream);
    }

    /**
     * Record an event.
     * This is for marking when an operation
     * starts.
     */
    public void startOldEvent() {
        //   JCuda.cudaEventRecord(oldEvent, oldStream);
    }

    /**
     * Record an  event (new).
     * This is for marking when an operation
     * starts.
     */
    public void startNewEvent() {
       // JCudaDriver.cuEventRecord(cUevent,stream);
    }


    /**
     * Initializes the stream
     */
    public void initStream() {
        ContextHolder.getInstance().setContext();
        if(stream == null) {
      /*      try {
                stream = ContextHolder.getInstance().getStreamPool().borrowObject();
            } catch (Exception e) {
                stream = new CUstream();
                JCudaDriver.cuStreamCreate(stream, CUstream_flags.CU_STREAM_NON_BLOCKING);
                streamFromPool = false;
            }
*/
            stream = new CUstream();
            JCudaDriver.cuStreamCreate(stream, CUstream_flags.CU_STREAM_DEFAULT);
            streamFromPool = false;
            //cUevent = new CUevent();
            //JCudaDriver.cuEventCreate(cUevent,0);
            eventDestroyed = false;
        }
    }

    /**
     * Initializes the old stream
     */
    public void initOldStream() {
        ContextHolder.getInstance().setContext();
        if(oldStream == null)  {
          /*  try {
                oldStream = ContextHolder.getInstance().getOldStreamPool().borrowObject();
            } catch (Exception e) {
                oldStreamFromPool = false;
                oldStream = new cudaStream_t();
                JCuda.cudaStreamCreate(oldStream);

            }*/

            oldStreamFromPool = false;
            oldStream = new cudaStream_t();
            JCuda.cudaStreamCreate(oldStream);

            //      oldEvent = new cudaEvent_t();
            //     JCuda.cudaEventCreate(oldEvent);
            //       oldEventDestroyed = false;
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
           /* try {
                handle = ContextHolder.getInstance().getHandlePool().borrowObject();
            } catch (Exception e) {
                handle = new cublasHandle();
                JCublas2.cublasCreate(handle);
                handleFromPool = false;
            }*/
            handle = new cublasHandle();
            JCublas2.cublasCreate(handle);
            handleFromPool = false;
           // associateHandle();
        }

    }

    /**
     * Destroys the context
     * and associated resources
     */
    @Deprecated
    public void destroy(CublasPointer resultPointer,boolean freeIfNotEqual) {
        /*
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

        if(!oldEventDestroyed) {
            oldEventDestroyed = true;
        }

        if(!eventDestroyed) {
            eventDestroyed = true;
        }
        */
    }


    /**
     * Destroys the context
     * and associated resources
     */
    @Deprecated
    public void destroy() {
        /*
        ContextHolder.getInstance().setContext();

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

        if(resultPointer != null) {
            if(stream != null)
                syncStream();
            if(oldStream != null)
                syncOldStream();
            resultPointer.copyToHost();
            try {
                resultPointer.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        */
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
        Allocator allocator = AtomicAllocator.getInstance();

        return allocator.getCudaContext();
        /*
        CudaContext ctx = new CudaContext();
        //ctx.initOldStream();
        ctx.initHandle();
        //ctx.startOldEvent();
        return ctx;*/
    }

    /**
     * Calls cuda device synchronize
     */
    public void syncDevice() {
        JCuda.cudaDeviceSynchronize();
    }

    /*
    @Override
    public void close() throws Exception {
        destroy();
    }
    */
}
