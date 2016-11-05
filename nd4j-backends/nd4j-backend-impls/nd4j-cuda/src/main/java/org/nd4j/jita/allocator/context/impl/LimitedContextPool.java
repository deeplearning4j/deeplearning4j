package org.nd4j.jita.allocator.context.impl;

import lombok.NonNull;
import org.apache.commons.lang3.RandomUtils;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.context.ContextPack;
import org.nd4j.jita.allocator.garbage.GarbageBufferReference;
import org.nd4j.jita.allocator.garbage.GarbageResourceReference;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.jita.allocator.pointers.cuda.cublasHandle_t;
import org.nd4j.jita.allocator.pointers.cuda.cusolverDnHandle_t;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.BaseDataBuffer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.lang.ref.ReferenceQueue;
import java.util.*;
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
    protected Map<Integer, ResourceGarbageCollectorThread> collectors = new HashMap<>();
    protected Map<Integer, ReferenceQueue<Thread>> queueMap = new HashMap<>();

    public LimitedContextPool() {

        int perDevicePool = CudaEnvironment.getInstance().getConfiguration().getPoolSize();

        for (int i = 0; i < 4; i++) {
            ReferenceQueue<Thread> queue = new ReferenceQueue<>();
            ResourceGarbageCollectorThread collector = new ResourceGarbageCollectorThread(i, queue);
            collector.start();

            collectors.put(i, collector);
            queueMap.put(i, queue);
        }

        addResourcesToPool(perDevicePool, false);
        currentPoolSize.set(perDevicePool);
    }

    protected synchronized void addResourcesToPool(int numResources, boolean restoreDevice) {
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
	cusolverDnHandle_t solverHandle = createNewSolverHandle();
            for (int cnt = 0; cnt < numResources; cnt++ ) {
                CudaContext context = createNewStream(device);
                context.initOldStream();
                getDeviceBuffers(context, device);
                context.setHandle(handle);
                context.setSolverHandle(solverHandle);

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
        long threadIdx = Thread.currentThread().getId();
        CudaContext context = acquired.get(threadIdx);
        if (context != null && deviceId == context.getDeviceId()) {
            return context;
        }

        nativeOps.setDevice(new CudaPointer(deviceId));

        context = pool.get(deviceId).poll();
        context.setDeviceId(deviceId);
        if (context != null) {
            int col = RandomUtils.nextInt(0, collectors.size());
            collectors.get(col);

            GarbageResourceReference reference = new GarbageResourceReference(Thread.currentThread(), queueMap.get(col), context, deviceId.intValue());
            context.attachReference(reference);
            //Garba reference = new GarbageBufferReference((BaseDataBuffer) buffer, queueMap.get(bucketId), point);
            //point.attachReference(reference);

            acquired.put(threadIdx, context);
            return context;
        } else {
            do {
                try {
                    context = pool.get(deviceId).poll(5, TimeUnit.SECONDS);
                    if (context != null) {
                        acquired.put(threadIdx, context);
                        return context;
                    } else {
                        if (currentPoolSize.get() < CudaEnvironment.getInstance().getConfiguration().getPoolSize() * 3) {
                            addResourcesToPool(16, true);

                            // there's possible race condition, but we don't really care
                            currentPoolSize.addAndGet(16);
                        }
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

    private class ResourceGarbageCollectorThread extends Thread implements Runnable {
        private final ReferenceQueue<Thread> queue;

        public ResourceGarbageCollectorThread(int threadId, @NonNull ReferenceQueue<Thread> queue) {
            this.queue = queue;
            this.setDaemon(true);
            this.setName("ResourceGC thread " + threadId);
        }

        @Override
        public void run() {
            while (true) {
                GarbageResourceReference reference = (GarbageResourceReference) queue.poll();
                if (reference != null) {
                    CudaContext context = reference.getContext();
                    Long threadId = reference.getThreadId();
                    int deviceId = reference.getDeviceId();

                    pool.get(deviceId).add(context);
                    acquired.remove(threadId);
                } else {
                    try {
                        Thread.sleep(100);
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                }
            }
        }
    }
}
