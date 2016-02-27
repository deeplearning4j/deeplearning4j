package org.nd4j.jita.allocator.concurrency;

import org.junit.Before;
import org.junit.Test;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.jita.conf.DeviceInformation;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class DeviceAllocationsTrackerTest {

    private static Configuration configuration = new Configuration();
    private static CudaEnvironment environment;

    @Before
    public void setUp() throws Exception {
        if (environment == null) {
            environment = new CudaEnvironment(configuration);

            DeviceInformation device1 = new DeviceInformation();
            device1.setDeviceId(0);
            device1.setCcMajor(5);
            device1.setCcMinor(2);
            device1.setTotalMemory(4 * 1024 * 1024 * 1024L);
            device1.setAvailableMemory(4 * 1024 * 1024 * 1024L);

            DeviceInformation device2 = new DeviceInformation();
            device2.setDeviceId(1);
            device2.setCcMajor(5);
            device2.setCcMinor(2);
            device2.setTotalMemory(4 * 1024 * 1024 * 1024L);
            device2.setAvailableMemory(4 * 1024 * 1024 * 1024L);

            environment.addDevice(device1);
            environment.addDevice(device2);
        }
    }

    @Test
    public void testGetAllocatedSize1() throws Exception {
        DeviceAllocationsTracker tracker = new DeviceAllocationsTracker(environment, configuration);


        tracker.addToAllocation(1L, 0, 100L);

        assertEquals(100, tracker.getAllocatedSize(0));

        tracker.subFromAllocation(1L, 0, 100L);

        assertEquals(0, tracker.getAllocatedSize(0));
    }

    @Test
    public void testGetAllocatedSize2() throws Exception {
        DeviceAllocationsTracker tracker = new DeviceAllocationsTracker(environment, configuration);


        tracker.addToAllocation(1L, 0, 100L);
        tracker.addToAllocation(2L, 0, 100L);

        assertEquals(200, tracker.getAllocatedSize(0));

        tracker.subFromAllocation(1L, 0, 100L);

        assertEquals(100, tracker.getAllocatedSize(0));
    }

    @Test
    public void testGetAllocatedSize3() throws Exception {
        DeviceAllocationsTracker tracker = new DeviceAllocationsTracker(environment, configuration);

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
        DeviceAllocationsTracker tracker = new DeviceAllocationsTracker(environment, configuration);

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
        DeviceAllocationsTracker tracker = new DeviceAllocationsTracker(environment, configuration);

        tracker.addToReservedSpace(0, 1000L);
        assertEquals(1000L, tracker.getReservedSpace(0));

        tracker.subFromReservedSpace(0, 1000L);
        assertEquals(0L, tracker.getReservedSpace(0));
    }
}