package test.org.nd4j.jita.allocator;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.jita.allocator.impl.MemoryTracker;
import static org.junit.Assert.assertTrue;


class AllocatorTest {


    @Test
    public void test1() {
        int deviceId = 0;
        MemoryTracker tracker = MemoryTracker.getInstance();
        tracker.incrementAllocated(deviceId);
        assertTrue(1 == tracker.getAllocated(deviceId));

        tracker.incrementCached(deviceId);
        assertTrue(1 == tracker.getCached(deviceId));
    }
}
