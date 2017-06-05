package org.nd4j.linalg.jcublas.context;

import lombok.Data;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.garbage.GarbageResourceReference;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.allocator.pointers.cuda.cublasHandle_t;
import org.nd4j.jita.allocator.pointers.cuda.cudaStream_t;
import org.nd4j.jita.allocator.pointers.cuda.cusolverDnHandle_t;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.jcublas.CublasPointer;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

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
    //private CUcontext context;
    //private CUstream stream;
    //private CUevent cUevent;
    private cudaStream_t oldStream;

    private cudaStream_t cublasStream;
    private cudaStream_t solverStream;

    private cudaStream_t specialStream;

    //private cudaEvent_t oldEvent;
    private cublasHandle_t handle;
    private cusolverDnHandle_t solverHandle;
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

    private Pointer bufferReduction;
    private Pointer bufferAllocation;
    private Pointer bufferScalar;
    private Pointer bufferSpecial;
    private GarbageResourceReference reference;

    private int deviceId = -1;

    private int laneId = 0;

    private static NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();


    public CudaContext(boolean free) {
        this();
        this.free = free;
    }

    public void attachReference(GarbageResourceReference ref) {
        reference = ref;
    }


    public CudaContext() {
        //   ContextHolder.getInstance().setContext();
    }

    /**
     * Synchronizes on the new
     * stream
     */
    public void syncStream() {
        //JCudaDriver.cuStreamSynchronize(stream);
    }

    /**
     * Synchronizes
     * on the old stream
     */
    public void syncOldStream() {
        //        ContextHolder.getInstance().setContext();
        //        System.out.println("old sync called");
        syncOldStream(false);
    }

    public void syncSpecialStream() {
        if (nativeOps.streamSynchronize(specialStream) == 0)
            throw new ND4JIllegalStateException("CUDA special stream synchronization failed");
    }

    public void syncOldStream(boolean syncCuBlas) {
        //        ContextHolder.getInstance().setContext();
        if (nativeOps.streamSynchronize(oldStream) == 0)
            throw new ND4JIllegalStateException("CUDA stream synchronization failed");

        if (syncCuBlas)
            syncCublasStream();
    }

    public void syncCublasStream() {
        if (cublasStream != null) {
            if (nativeOps.streamSynchronize(cublasStream) == 0)
                throw new ND4JIllegalStateException("CUDA stream synchronization failed");
        } else
            throw new IllegalStateException("cuBLAS stream isnt set");
    }


    public void syncSolverStream() {
        if (solverStream != null) {
            if (nativeOps.streamSynchronize(solverStream) == 0)
                throw new ND4JIllegalStateException("CUDA stream synchronization failed");
        } else
            throw new IllegalStateException("cuBLAS stream isnt set");
    }

    /**
     * Associates
     * the handle on this context
     * to the given stream
     */
    public synchronized void associateHandle() {
        //JCublas2.cublasSetStream(handle,oldStream);
    }



    /**
     * Initializes the stream
     */
    public void initStream() {
        //        ContextHolder.getInstance().setContext();
        /*
        if(stream == null) {
            stream = new CUstream();
            JCudaDriver.cuStreamCreate(stream, CUstream_flags.CU_STREAM_DEFAULT);
            streamFromPool = false;
            eventDestroyed = false;
        }
        */
    }

    /**
     * Initializes the old stream
     */
    public void initOldStream() {
        //        ContextHolder.getInstance().setContext();
        if (oldStream == null) {
            oldStreamFromPool = false;
            oldStream = new cudaStream_t(nativeOps.createStream());
            //JCuda.cudaStreamCreate(oldStream);

            specialStream = new cudaStream_t(nativeOps.createStream());
            //JCuda.cudaStreamCreate(specialStream);
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
    public void destroy(CublasPointer resultPointer, boolean freeIfNotEqual) {}


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
        //context.syncOldStream(false);
        return context;
    }

}
