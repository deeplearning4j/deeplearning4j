/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.jita.allocator.context.impl;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import lombok.var;
import org.apache.commons.lang3.RandomUtils;
import org.nd4j.jita.allocator.context.ContextPack;
import org.nd4j.jita.allocator.garbage.DeallocatableThread;
import org.nd4j.jita.allocator.garbage.GarbageResourceReference;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.jita.allocator.pointers.cuda.cublasHandle_t;
import org.nd4j.jita.allocator.pointers.cuda.cusolverDnHandle_t;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.memory.Deallocatable;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.lang.ref.ReferenceQueue;
import java.util.ArrayList;
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
    //protected AtomicInteger currentPoolSize = new AtomicInteger(0);
    protected List<AtomicInteger> devicePoolSizes = new ArrayList<>();
    protected Map<Integer, ReferenceQueue<Thread>> queueMap = new HashMap<>();

    protected ThreadLocal<Deallocatable> threadHooks = new ThreadLocal<>();

    public LimitedContextPool() {

        int perDevicePool = CudaEnvironment.getInstance().getConfiguration().getPoolSize();

/*
        for (int i = 0; i < 4; i++) {
            val queue = new ReferenceQueue<Thread>();
            val collector = new ResourceGarbageCollectorThread(i, queue);
            collector.start();

            collectors.put(i, collector);
            queueMap.put(i, queue);
        }
*/
        fillPoolWithResources(perDevicePool, false);
    }

    protected void addResourcesToPool(int numResources) {
        int device = AtomicAllocator.getInstance().getDeviceId();

        val handle = createNewCublasHandle();
        for (int cnt = 0; cnt < numResources; cnt++) {
            val context = createNewStream(device);
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
            devicePoolSizes.add(new AtomicInteger(numResources));

            val handle = createNewCublasHandle();
            val solverHandle = createNewSolverHandle();
            for (int cnt = 0; cnt < numResources; cnt++) {
                val context = createNewStream(device);
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

    public void removeAcquired() {
        val threadIdx = Thread.currentThread().getId();
        acquired.remove(threadIdx);
    }

    @Override
    public CudaContext acquireContextForDevice(Integer deviceId) {
        val threadIdx = Thread.currentThread().getId();
        var context = acquired.get(threadIdx);
        if (context != null && deviceId == context.getDeviceId()) {
            return context;
        }

        //log.info("Setting device to {}", deviceId);
        nativeOps.setDevice(new CudaPointer(deviceId));
        context = pool.get(deviceId).poll();
        if (context != null) {
            //val reference = new GarbageResourceReference(Thread.currentThread(), queueMap.get(col), context, deviceId.intValue());
            //context.attachReference(reference);
            context.setDeviceId(deviceId);
            context.setThreadId(threadIdx);
            val hook = new DeallocatableThread(Thread.currentThread(), context);
            threadHooks.set(hook);
            Nd4j.getDeallocatorService().pickObject(hook);


            acquired.put(threadIdx, context);
            return context;
        } else {

            do {
                try {
                    Nd4j.getMemoryManager().invokeGc();

                    context = pool.get(deviceId).poll(1, TimeUnit.SECONDS);
                    if (context != null) {
                        //val reference = new GarbageResourceReference(Thread.currentThread(), queueMap.get(col), context, deviceId.intValue());
                        //context.attachReference(reference);
                        context.setDeviceId(deviceId);
                        context.setThreadId(threadIdx);
                        val hook = new DeallocatableThread(Thread.currentThread(), context);
                        threadHooks.set(hook);
                        Nd4j.getDeallocatorService().pickObject(hook);

                        acquired.put(threadIdx, context);
                    } else {
                        val currentPoolSize = devicePoolSizes.get(deviceId);
                        synchronized (currentPoolSize) {
                            if (currentPoolSize.get() < CudaEnvironment.getInstance().getConfiguration().getPoolSize() * 3) {
                                addResourcesToPool(16);

                                // there's possible race condition, but we don't really care
                                currentPoolSize.addAndGet(16);
                                log.warn("Initial pool size: {}; Current pool size: {}", CudaEnvironment.getInstance().getConfiguration().getPoolSize(), currentPoolSize.get());
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
                    }
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            } while (context == null);

            return context;
        }
    }

    @Override
    @Deprecated
    public ContextPack acquireContextPackForDevice(Integer deviceId) {
        return new ContextPack(acquireContextForDevice(deviceId));
    }

    @Override
    public CudaContext getContextForDevice(Integer deviceId) {
        return acquireContextForDevice(deviceId);
    }

    @Override
    public void releaseContext(CudaContext context) {
        val threadIdx = context.getThreadId();
        val deviceId = context.getDeviceId();

        context.setThreadId(-1);

        acquired.remove(threadIdx);
        pool.get(deviceId).add(context);
    }

    /*
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
                    val threadId = reference.getThreadId();
                    val deviceId = reference.getDeviceId();

                    // there's a chance context was already released
                    if (context.getThreadId() != threadId)
                        continue;

                    pool.get(deviceId).add(context);
                    acquired.remove(threadId);
                } else {
                    LockSupport.parkNanos(500000L);
                }
            }
        }
    }
    */
}
