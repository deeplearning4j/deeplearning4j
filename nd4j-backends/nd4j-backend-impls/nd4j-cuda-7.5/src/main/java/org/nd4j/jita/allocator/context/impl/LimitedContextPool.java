package org.nd4j.jita.allocator.context.impl;

import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.context.ContextPack;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.jita.allocator.pointers.cuda.cublasHandle_t;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author raver119@gmail.com
 */
public class LimitedContextPool extends BasicContextPool {

    // pool of free contexts
    protected Map<Integer, LinkedBlockingQueue<CudaContext>> pool = new HashMap<>();

    // pool of used pools
    protected Map<Long, CudaContext> acquired = new ConcurrentHashMap<>();
    protected AtomicInteger currentPoolSize = new AtomicInteger(0);

    public LimitedContextPool() {

        int perDevicePool = CudaEnvironment.getInstance().getConfiguration().getPoolSize();
        addResourcesToPool(perDevicePool, false);
        currentPoolSize.set(perDevicePool);
    }

    protected void addResourcesToPool(int numResources, boolean restoreDevice) {
        List<Integer> devices = CudaEnvironment.getInstance().getConfiguration().getAvailableDevices();

        int cDevice = 0;
        if (restoreDevice) {
            cDevice = AtomicAllocator.getInstance().getDeviceId();
        }

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

        for (Integer device: devices) {
            nativeOps.setDevice(new CudaPointer(device));
            pool.put(device, new LinkedBlockingQueue<CudaContext>());

            cublasHandle_t handle = createNewCublasHandle();
            for (int cnt = 0; cnt < numResources; cnt++ ) {
                CudaContext context = createNewStream(device);
                context.initOldStream();
                getDeviceBuffers(context, device);
                context.setHandle(handle);

                context.syncOldStream();

                pool.get(device).add(context);
            }
        }

        if (restoreDevice) {
            nativeOps.setDevice(new CudaPointer(cDevice));
        }
    }

    @Override
    public CudaContext acquireContextForDevice(Integer deviceId) {
        CudaContext context = null;
        long threadIdx = Thread.currentThread().getId();
        if (acquired.containsKey(threadIdx)) {
            context = acquired.get(threadIdx);
            logger.info("Serving context from cache");
            return context;
        }

        nativeOps.setDevice(new CudaPointer(deviceId));

        context = pool.get(deviceId).poll();
        if (context != null) {
            logger.info("Grabbing one context from pool");
            acquired.put(threadIdx, context);
            return context;
        } else {
            do {
                try {
                    context = pool.get(deviceId).poll(5, TimeUnit.SECONDS);
                    if (context != null) {
                        acquired.put(threadIdx, context);
                        return context;
                    }
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            } while (context == null);
        }

        throw new RuntimeException("EPTS");
    }

    @Override
    public ContextPack acquireContextPackForDevice(Integer deviceId) {
        return new ContextPack(acquireContextForDevice(deviceId));
    }

    @Override
    public CudaContext getContextForDevice(Integer deviceId) {
        return acquireContextForDevice(deviceId);
    }


}
