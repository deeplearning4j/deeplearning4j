package org.nd4j.jita.allocator.garbage;

import org.nd4j.linalg.jcublas.context.CudaContext;

import java.lang.ref.ReferenceQueue;
import java.lang.ref.WeakReference;

/**
 * @author raver119@gmail.com
 */
public class GarbageResourceReference extends WeakReference<Thread> {
    private final CudaContext context;
    private final long threadId;
    private final int deviceId;

    public GarbageResourceReference(Thread referent, ReferenceQueue<? super Thread> q, CudaContext context,
                    int deviceId) {
        super(referent, q);
        this.context = context;
        this.threadId = referent.getId();
        this.deviceId = deviceId;
    }

    public CudaContext getContext() {
        return context;
    }

    public long getThreadId() {
        return threadId;
    }

    public int getDeviceId() {
        return deviceId;
    }
}
