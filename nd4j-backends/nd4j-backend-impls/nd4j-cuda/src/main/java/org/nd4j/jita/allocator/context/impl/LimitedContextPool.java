package org.nd4j.jita.allocator.context.impl;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.RandomUtils;
import org.nd4j.jita.allocator.context.ContextPack;
import org.nd4j.jita.allocator.garbage.GarbageResourceReference;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.jita.allocator.pointers.cuda.cublasHandle_t;
import org.nd4j.jita.allocator.pointers.cuda.cusolverDnHandle_t;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.lang.ref.ReferenceQueue;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;

/**
 * @author raver119@gmail.com
 */
@Slf4j
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

        fillPoolWithResources(perDevicePool, false);
        currentPoolSize.set(perDevicePool);
    }

    protected void addResourcesToPool(int numResources) {
        int device = AtomicAllocator.getInstance().getDeviceId();

        cublasHandle_t handle = createNewCublasHandle();
        for (int cnt = 0; cnt < numResources; cnt++) {
            CudaContext context = createNewStream(device);
            context.initOldStream();
            getDeviceBuffers(context, device);
            context.setHandle(handle);

            context.syncOldStream();

            pool.get(device).add(context);
        }
    }

    protected synchronized void fillPoolWithResources(int numResources, boolean restoreDevice) {
        List<Integer> devices = CudaEnvironment.getInstance().getConfiguration().getAvailableDevices();

        int cDevice = 0;
        if (restoreDevice) {
            cDevice = AtomicAllocator.getInstance().getDeviceId();
        }

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

        for (Integer device : devices) {
            nativeOps.setDevice(new CudaPointer(device));
            pool.put(device, new LinkedBlockingQueue<CudaContext>());

            cublasHandle_t handle = createNewCublasHandle();
            cusolverDnHandle_t solverHandle = createNewSolverHandle();
            for (int cnt = 0; cnt < numResources; cnt++) {
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

        //log.info("Setting device to {}", deviceId);
        nativeOps.setDevice(new CudaPointer(deviceId));
        context = pool.get(deviceId).poll();
        if (context != null) {
            int col = RandomUtils.nextInt(0, collectors.size());
            collectors.get(col);

            GarbageResourceReference reference = new GarbageResourceReference(Thread.currentThread(), queueMap.get(col),
                            context, deviceId.intValue());
            context.attachReference(reference);
            //Garba reference = new GarbageBufferReference((BaseDataBuffer) buffer, queueMap.get(bucketId), point);
            //point.attachReference(reference);

            acquired.put(threadIdx, context);
            context.setDeviceId(deviceId);
            return context;
        } else {

            do {
                try {
                    Nd4j.getMemoryManager().invokeGc();

                    context = pool.get(deviceId).poll(1, TimeUnit.SECONDS);
                    if (context != null) {
                        int col = RandomUtils.nextInt(0, collectors.size());
                        collectors.get(col);

                        GarbageResourceReference reference = new GarbageResourceReference(Thread.currentThread(),
                                        queueMap.get(col), context, deviceId.intValue());
                        context.attachReference(reference);

                        acquired.put(threadIdx, context);
                        context.setDeviceId(deviceId);
                    } else {
                        if (currentPoolSize.get() < CudaEnvironment.getInstance().getConfiguration().getPoolSize()
                                        * 3) {
                            addResourcesToPool(16);

                            // there's possible race condition, but we don't really care
                            currentPoolSize.addAndGet(16);
                        } else {
                            log.warn("Can't allocate new context, sleeping...");

                            Nd4j.getMemoryManager().invokeGc();
                            try {
                                Thread.sleep(500);
                            } catch (Exception e) {
                                //
                            }
                        }
                    }
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            } while (context == null);

            return context;
        }
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
                    LockSupport.parkNanos(500000L);
                }
            }
        }
    }
}
