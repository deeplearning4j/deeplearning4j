package org.nd4j.jita.allocator.context;

import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaStream_t;
import org.apache.commons.lang3.RandomUtils;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Semaphore;

/**
 * @author raver119@gmail.com
 */
public class ContextPool {
    private static final int MAX_STREAMS_PER_DEVICE = 15;

    private volatile Map<Integer, cublasHandle> cublasPool = new ConcurrentHashMap<>();

    private volatile Map<Long, CudaContext> contextsPool = new ConcurrentHashMap<>();

    private volatile Map<Integer, Map<Integer, CudaContext>> contextsForDevices = new ConcurrentHashMap<>();

    private Semaphore lock = new Semaphore(1);

    private static Logger logger = LoggerFactory.getLogger(ContextPool.class);

    public boolean containsContextForThread(long threadId) {
        return contextsPool.containsKey(threadId);
    }

    public CudaContext getContextForDevice(Integer deviceId) {
        return acquireContextForDevice(deviceId);
    }

    public CudaContext acquireContextForDevice(Integer deviceId) {
        /*
            We should check, if we have context for this specific thread/device
            If we don't have context for this thread - we should stick to one of existent contexts available at pool
         */
        Long threadId = Thread.currentThread().getId();
        if (!contextsPool.containsKey(threadId)) {
            // we don't have attached context for this thread. we should pick up existing context for target device (if any).

            try {
                // this is lockable thing, but since it locks once per thread initialization, performance impact won't be big
                lock.acquire();
                if (!contextsForDevices.containsKey(deviceId)) {
                    contextsForDevices.put(deviceId, new ConcurrentHashMap<Integer, CudaContext>());
                }

                // if we hadn't hit MAX_STREAMS_PER_DEVICE limit - we add new stream. Otherwise we use random one.
                if (contextsForDevices.get(deviceId).size() < MAX_STREAMS_PER_DEVICE) {
                    logger.debug("Creating new context...");
                    CudaContext context = createNewStream(deviceId);

                    if (contextsForDevices.get(deviceId).size() == 0) {
                        // if we have no contexts created - it's just awesome time to attach cuBLAS handle here
                        cudaStream_t cublasStream = createNewStream(deviceId).getOldStream();

                        cublasHandle handle = createNewCublasHandle(cublasStream);
                        context.setHandle(handle);
                        context.setCublasStream(cublasStream);

                        cublasPool.put(deviceId, handle);
                    } else {
                        // just pick handle out there
                        cublasHandle handle = cublasPool.get(deviceId);
                        context.setHandle(handle);

                        cudaStream_t cublasStream = new cudaStream_t();
                        JCublas2.cublasGetStream(handle, cublasStream);
                        context.setCublasStream(cublasStream);
                    }

                    contextsPool.put(threadId, context);
                    contextsForDevices.get(deviceId).put(contextsForDevices.get(deviceId).size(), context);

                    return context;
                } else {
                    Integer rand = RandomUtils.nextInt(0, MAX_STREAMS_PER_DEVICE);
                    logger.debug("Reusing context: " + rand);

                    JCuda.cudaSetDevice(deviceId);

                    CudaContext context = contextsForDevices.get(deviceId).get(rand);

                    contextsPool.put(threadId, context);
                    return context;
                }

            } catch (Exception e) {
                throw new RuntimeException(e);
            } finally {
                lock.release();
            }
        }

        return contextsPool.get(threadId);
    }

    private CudaContext createNewStream(Integer deviceId) {
        JCuda.cudaSetDevice(deviceId);

        CudaContext context = new CudaContext();
        context.initOldStream();

        //context.initHandle();
        //context.associateHandle();
        //context.initStream();

        return context;
    }

    private cublasHandle createNewCublasHandle(cudaStream_t stream) {
        cublasHandle handle = new cublasHandle();
        JCublas2.cublasCreate(handle);
        JCublas2.cublasSetStream(handle,stream);

        return handle;
    }
}
