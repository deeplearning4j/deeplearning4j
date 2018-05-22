package org.nd4j.jita.allocator.concurrency;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.jita.conf.Configuration;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
@Ignore
public class DeviceAllocationsTrackerTest {

    private static Configuration configuration = new Configuration();


    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testGetAllocatedSize1() throws Exception {
        DeviceAllocationsTracker tracker = new DeviceAllocationsTracker(configuration);


        tracker.addToAllocation(1L, 0, 100L);

        assertEquals(100, tracker.getAllocatedSize(0));

        tracker.subFromAllocation(1L, 0, 100L);

        assertEquals(0, tracker.getAllocatedSize(0));
    }

    @Test
    public void testGetAllocatedSize2() throws Exception {
        DeviceAllocationsTracker tracker = new DeviceAllocationsTracker(configuration);


        tracker.addToAllocation(1L, 0, 100L);
        tracker.addToAllocation(2L, 0, 100L);

        assertEquals(200, tracker.getAllocatedSize(0));

        tracker.subFromAllocation(1L, 0, 100L);

        assertEquals(100, tracker.getAllocatedSize(0));
    }

    @Test
    public void testGetAllocatedSize3() throws Exception {
        DeviceAllocationsTracker tracker = new DeviceAllocationsTracker(configuration);

        tracker.addToAllocation(1L, 0, 100L);
        tracker.addToAllocation(2L, 1, 100L);

        assertEquals(100, tracker.getAllocatedSize(0));
        assertEquals(100, tracker.getAllocatedSize(1));

        tracker.subFromAllocation(1L, 0, 100L);

        assertEquals(0, tracker.getAllocatedSize(0));
        assertEquals(100, tracker.getAllocatedSize(1));
    }

    @Test
    public void testGetAllocatedSize4() throws Exception {
        DeviceAllocationsTracker tracker = new DeviceAllocationsTracker(configuration);

        tracker.addToAllocation(1L, 0, 100L);
        tracker.addToAllocation(2L, 0, 150L);

        assertEquals(250, tracker.getAllocatedSize(0));

        assertEquals(100, tracker.getAllocatedSize(1L, 0));
        assertEquals(150, tracker.getAllocatedSize(2L, 0));

        tracker.subFromAllocation(1L, 0, 100L);

        assertEquals(150, tracker.getAllocatedSize(0));
    }

    @Test
    public void testReservedSpace1() throws Exception {
        DeviceAllocationsTracker tracker = new DeviceAllocationsTracker(configuration);

        tracker.addToReservedSpace(0, 1000L);
        assertEquals(1000L, tracker.getReservedSpace(0));

        tracker.subFromReservedSpace(0, 1000L);
        assertEquals(0L, tracker.getReservedSpace(0));
    }
}