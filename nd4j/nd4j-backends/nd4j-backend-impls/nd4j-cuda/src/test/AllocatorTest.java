import test.org.nd4j.jita.allocator.impl;


class CudaTest {


    @Test
    public void test1() {
        int deviceId = 0;
        MemoryTracker tracker = MemoryTracker.getInstance();
        tracker.incrementAllocated(deviceId);
        assertEquals(1, tracker.getAllocated(deviceId);

        tracker.incrementCacheed(deviceId);
        assertEquals(1, tracker.getCached(deviceId);
    }
}