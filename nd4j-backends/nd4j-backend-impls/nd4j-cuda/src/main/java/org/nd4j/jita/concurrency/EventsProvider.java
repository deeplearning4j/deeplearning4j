package org.nd4j.jita.concurrency;

import org.nd4j.jita.allocator.pointers.cuda.cudaEvent_t;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
public class EventsProvider {
    //private static final EventsProvider INSTANCE = new EventsProvider();

    private List<ConcurrentLinkedQueue<cudaEvent_t>> queue = new ArrayList<>();
    private AtomicLong newCounter = new AtomicLong(0);
    private AtomicLong cacheCounter = new AtomicLong(0);

    public EventsProvider() {
        int numDev = Nd4j.getAffinityManager().getNumberOfDevices();

        for (int i = 0; i < numDev; i++) {
            queue.add(new ConcurrentLinkedQueue<cudaEvent_t>());
        }
    }

    public cudaEvent_t getEvent() {
        int deviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        cudaEvent_t e = queue.get(deviceId).poll();
        if (e == null) {
            e = new cudaEvent_t(NativeOpsHolder.getInstance().getDeviceNativeOps().createEvent());
            e.setDeviceId(deviceId);
            newCounter.incrementAndGet();
        } else
            cacheCounter.incrementAndGet();

        return e;
    }

    public void storeEvent(cudaEvent_t event) {
        if (event != null)
            //            NativeOpsHolder.getInstance().getDeviceNativeOps().destroyEvent(event);
            queue.get(event.getDeviceId()).add(event);
    }

    public long getEventsNumber() {
        return newCounter.get();
    }

    public long getCachedNumber() {
        return cacheCounter.get();
    }

}
