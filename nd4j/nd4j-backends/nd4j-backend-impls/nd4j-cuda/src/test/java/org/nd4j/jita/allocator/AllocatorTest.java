package test.org.nd4j.jita.allocator;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.jita.allocator.impl.MemoryTracker;
import static org.junit.Assert.assertTrue;


public class AllocatorTest {


    @Test
    public void testCounters() {
        int deviceId = 0;
        MemoryTracker tracker = MemoryTracker.getInstance();

        assertTrue(0 == tracker.getAllocated(deviceId));
        assertTrue(0 == tracker.getCached(deviceId));
        assertTrue(0 == tracker.getTotal(deviceId));

        tracker.incrementAllocated(deviceId, 10);
        assertTrue(10 == tracker.getAllocated(deviceId));

        tracker.incrementCached(deviceId, 5);
        assertTrue(5 == tracker.getCached(deviceId));

        tracker.decrementAllocated(deviceId, 5);
        assertTrue(5 == tracker.getAllocated(deviceId));

        tracker.decrementCached(deviceId, 5);
        assertTrue(0 == tracker.getCached(deviceId));

        assertTrue(0 == tracker.getTotal(deviceId));
    }
}
