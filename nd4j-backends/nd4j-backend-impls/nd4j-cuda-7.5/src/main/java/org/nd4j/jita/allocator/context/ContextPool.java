package org.nd4j.jita.allocator.context;

import jcuda.runtime.JCuda;
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
    private static final int MAX_STREAMS_PER_DEVICE = 2;

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
        context.initHandle();
        context.initOldStream();
        context.associateHandle();
        //context.initStream();

        return context;
    }
}
