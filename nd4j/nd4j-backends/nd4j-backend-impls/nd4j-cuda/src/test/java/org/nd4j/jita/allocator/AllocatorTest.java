package test.org.nd4j.jita.allocator;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.jita.allocator.impl.MemoryTracker;
import static org.junit.Assert.assertTrue;

import lombok.val;

import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.MirroringPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;


public class AllocatorTest {
    private static final long SAFETY_OFFSET = 1024L;	

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

    @Test
    public void testWorkspaceInitSize() {

        long initSize = 1024;
	MemoryTracker tracker = MemoryTracker.getInstance();

        WorkspaceConfiguration workspaceConfig = WorkspaceConfiguration.builder()
                .policyAllocation(AllocationPolicy.STRICT)
                .initialSize(initSize)
                .build();

	try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(workspaceConfig, "test")) {
        	assertTrue(initSize + SAFETY_OFFSET  ==
                   tracker.getWorkspace(Nd4j.getAffinityManager().getDeviceForCurrentThread()));
        }
	assertTrue(0L == tracker.getWorkspace(Nd4j.getAffinityManager().getDeviceForCurrentThread()));
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
		System.out.println("Alloc1:" + MemoryTracker.getInstance().getWorkspace(Nd4j.getAffinityManager().getDeviceForCurrentThread()));
        	assertTrue(allocSize ==
                    MemoryTracker.getInstance().getWorkspace(Nd4j.getAffinityManager().getDeviceForCurrentThread()));
	}
        assertTrue(allocSize == 
                MemoryTracker.getInstance().getWorkspace(Nd4j.getAffinityManager().getDeviceForCurrentThread()));
        /*Nd4j.getWorkspaceManager().destroyWorkspace(ws);
        assertTrue(0L ==
                MemoryTracker.getInstance().getWorkspace(Nd4j.getAffinityManager().getDeviceForCurrentThread()));*/
    }

}
