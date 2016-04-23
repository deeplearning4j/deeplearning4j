package org.nd4j.jita.allocator.context.impl;

import org.nd4j.jita.allocator.context.ContextPack;
import org.nd4j.jita.allocator.context.ContextPool;
import org.nd4j.jita.allocator.pointers.cuda.cublasHandle_t;
import org.nd4j.jita.allocator.pointers.cuda.cudaStream_t;
import org.nd4j.linalg.jcublas.context.CudaContext;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * @author raver119@gmail.com
 */
public class PackedContextPool extends BasicContextPool implements ContextPool {

    protected static final int LANES_PER_THREAD = 4;

    private volatile Map<Long, ContextPack> contextsPool = new ConcurrentHashMap<>();

    @Override
    public CudaContext acquireContextForDevice(Integer deviceId) {
        return acquireContextPackForDevice(deviceId).getContextForLane(0);
    }

    @Override
    public ContextPack acquireContextPackForDevice(Integer deviceId) {
        Long threadId = Thread.currentThread().getId();
        if (!contextsPool.containsKey(threadId)) {
            try {
                lock.acquire();

                ContextPack pack = new ContextPack(LANES_PER_THREAD);
                for (int c = 0; c < LANES_PER_THREAD; c++) {
                    CudaContext context = createNewStream(deviceId);

                    getDeviceBuffers(context, deviceId);

                    if (cublasPool.get(deviceId) == null) {
                        // if we have no contexts created - it's just awesome time to attach cuBLAS handle here
                        logger.debug("Creating new cuBLAS handle for device [{}]", deviceId);

                        cudaStream_t cublasStream = createNewStream(deviceId).getOldStream();

                        cublasHandle_t handle = createNewCublasHandle(cublasStream);
                        context.setHandle(handle);
                        context.setCublasStream(cublasStream);

                        cublasPool.put(deviceId, handle);
                    } else {
                        // just pick handle out there
                        logger.debug("Reusing cuBLAS handle for device [{}]", deviceId);
                        cublasHandle_t handle = cublasPool.get(deviceId);
                        context.setHandle(handle);
                    }

                    pack.addLane(c, context);
                }

                contextsPool.put(threadId, pack);


            } catch (Exception e) {
                throw new RuntimeException(e);
            } finally {
                lock.release();
            }
        }

        return contextsPool.get(threadId);
    }
}
