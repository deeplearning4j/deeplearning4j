package org.nd4j.jita.allocator;

import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.jita.allocator.impl.MemoryTracker;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import lombok.val;

import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.MirroringPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;


public class AllocatorTest {
    private static final long SAFETY_OFFSET = 1024L;	

    @Test
    public void testCounters() {
        int deviceId = 0;
        MemoryTracker tracker = MemoryTracker.getInstance();

        assertTrue(0 == tracker.getAllocatedAmount(deviceId));
        assertTrue(0 == tracker.getCachedAmount(deviceId));
        assertTrue(0 == tracker.getTotalMemory(deviceId));

        tracker.incrementAllocatedAmount(deviceId, 10);
        assertTrue(10 == tracker.getAllocatedAmount(deviceId));

        tracker.incrementCachedAmount(deviceId, 5);
        assertTrue(5 == tracker.getCachedAmount(deviceId));

        tracker.decrementAllocatedAmount(deviceId, 5);
        assertTrue(5 == tracker.getAllocatedAmount(deviceId));

        tracker.decrementCachedAmount(deviceId, 5);
        assertTrue(0 == tracker.getCachedAmount(deviceId));

        assertTrue(0 == tracker.getTotalMemory(deviceId));
    }

    @Test
    public void testWorkspaceInitSize() {

        long initSize = 1024;
	    MemoryTracker tracker = MemoryTracker.getInstance();

        WorkspaceConfiguration workspaceConfig = WorkspaceConfiguration.builder()
                .policyAllocation(AllocationPolicy.STRICT)
                .initialSize(initSize)
                .build();

	    try (val ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(workspaceConfig, "test121")) {
        	assertEquals(initSize + SAFETY_OFFSET, tracker.getWorkspaceAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread()));
        }

	    val ws = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("test121");
	    ws.destroyWorkspace();

	    assertEquals(0, tracker.getWorkspaceAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread()));
    }


    @Test
    public void testWorkspaceSpilledSize() {

        long initSize = 0;
        MemoryTracker tracker = MemoryTracker.getInstance();

        WorkspaceConfiguration workspaceConfig = WorkspaceConfiguration.builder()
                .policyAllocation(AllocationPolicy.STRICT)
                .initialSize(initSize)
                .build();

        try (val ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(workspaceConfig, "test99323")) {
            assertEquals(0L, tracker.getWorkspaceAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread()));

            val array = Nd4j.createFromArray(1.f, 2.f, 3.f, 4.f);

            assertEquals(array.length() * array.data().getElementSize(), tracker.getWorkspaceAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread()));
        }

        val ws = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("test99323");
        ws.destroyWorkspace();

        assertEquals(0, tracker.getWorkspaceAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread()));
    }

    @Test
    public void testWorkspaceSpilledSizeHost() {

        long initSize = 0;
        MemoryTracker tracker = MemoryTracker.getInstance();

        WorkspaceConfiguration workspaceConfig = WorkspaceConfiguration.builder()
                .policyAllocation(AllocationPolicy.STRICT)
                .policyMirroring(MirroringPolicy.HOST_ONLY)
                .initialSize(initSize)
                .build();

        try (val ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(workspaceConfig, "test99323222")) {
            assertEquals(0L, tracker.getWorkspaceAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread()));

            val array = Nd4j.createFromArray(1.f, 2.f, 3.f, 4.f);

            assertEquals(0, tracker.getWorkspaceAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread()));
        }

        val ws = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("test99323222");
        ws.destroyWorkspace();

        assertEquals(0, tracker.getWorkspaceAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread()));
    }


    @Ignore
    @Test
    public void testWorkspaceAlloc() {

        long initSize = 0;
        long allocSize = 48;

        val workspaceConfig = WorkspaceConfiguration.builder()
                .policyAllocation(AllocationPolicy.STRICT)
                .initialSize(initSize)
                .policyMirroring(MirroringPolicy.HOST_ONLY) // Commenting this out makes it so that assert is not triggered (for at least 40 secs or so...)
                .build();

	    try (val ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(workspaceConfig, "test")) {
        	final INDArray zeros = Nd4j.zeros(allocSize, 'c');
		System.out.println("Alloc1:" + MemoryTracker.getInstance().getWorkspaceAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread()));
        	assertTrue(allocSize ==
                    MemoryTracker.getInstance().getWorkspaceAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread()));
	}
        assertTrue(allocSize == 
                MemoryTracker.getInstance().getWorkspaceAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread()));
        /*Nd4j.getWorkspaceManager().destroyWorkspace(ws);
        assertTrue(0L ==
                MemoryTracker.getInstance().getWorkspaceAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread()));*/
    }

}
