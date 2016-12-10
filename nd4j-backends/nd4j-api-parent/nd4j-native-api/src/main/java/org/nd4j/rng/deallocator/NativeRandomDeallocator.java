package org.nd4j.rng.deallocator;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.lang.ref.ReferenceQueue;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Since NativeRandom assumes some native resources, we have to track their use, and deallocate them as soon they are released by JVM GC
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class NativeRandomDeallocator {
    private static final NativeRandomDeallocator INSTANCE = new NativeRandomDeallocator();

    // we don't really need concurrency here, so 1 queue will be just fine
    private final ReferenceQueue<NativePack> queue;
    private final Map<Long, GarbageStateReference> referenceMap;
    private List<DeallocatorThread> deallocatorThreads = new ArrayList<>();

    private NativeRandomDeallocator() {
        this.queue = new ReferenceQueue<>();
        this.referenceMap = new ConcurrentHashMap<>();

        DeallocatorThread thread = new DeallocatorThread(0, queue, referenceMap);
        thread.start();

        deallocatorThreads.add(thread);
    }

    public static NativeRandomDeallocator getInstance() {
        return INSTANCE;
    }


    /**
     * This method is used internally from NativeRandom deallocators
     * This method doesn't accept Random interface implementations intentionally.
     *
     * @param random
     */
    public void trackStatePointer(NativePack random) {
        if (random.getStatePointer() != null) {
            GarbageStateReference reference = new GarbageStateReference(random, queue);
            referenceMap.put(random.getStatePointer().address(), reference);
        }
    }


    /**
     * This class provides garbage collection for NativeRandom state memory. It's not too big amount of memory used, but we don't want any leaks.
     *
     */
    protected class DeallocatorThread extends Thread implements Runnable {
        private final ReferenceQueue<NativePack> queue;
        private final Map<Long, GarbageStateReference> referenceMap;

        protected DeallocatorThread(int threadId, @NonNull ReferenceQueue<NativePack> queue, Map<Long, GarbageStateReference> referenceMap) {
            this.queue = queue;
            this.referenceMap = referenceMap;
            this.setName("NativeRandomDeallocator thread " + threadId);
            this.setDaemon(true);
        }

        @Override
        public void run() {
            while (true) {
                GarbageStateReference reference = (GarbageStateReference) queue.poll();
                if (reference != null) {
                    if (reference.getStatePointer() != null) {
                        referenceMap.remove(reference.getStatePointer().address());
                        NativeOpsHolder.getInstance().getDeviceNativeOps().destroyRandom(reference.getStatePointer());
                    }
                } else {
                    try {
                        // state buffer size is very small, so we don't really care if we'll sleep for 5 seconds
                        Thread.sleep(5000);
                    } catch (Exception e) {
                        //
                    }
                }
            }
        }
    }
}
