package org.nd4j.jita.concurrency;

import org.nd4j.jita.allocator.pointers.cuda.cudaEvent_t;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
public class EventsProvider {
    private static final EventsProvider INSTANCE = new EventsProvider();

    private ConcurrentLinkedQueue<cudaEvent_t> queue = new ConcurrentLinkedQueue<>();
    private AtomicLong counter = new AtomicLong(0);

    private EventsProvider() {

    }

    public static EventsProvider getInstance() {
        return INSTANCE;
    }

    public cudaEvent_t getEvent() {
        cudaEvent_t e = queue.poll();
        if (e == null) {
            e = new cudaEvent_t(NativeOpsHolder.getInstance().getDeviceNativeOps().createEvent());
            counter.incrementAndGet();
        }

        return e;
    }

    public void storeEvent(cudaEvent_t event) {
        if (event != null)
            NativeOpsHolder.getInstance().getDeviceNativeOps().destroyEvent(event);
//            queue.add(event);
    }

    public long getEventsNumber() {
        return counter.get();
    }

}
