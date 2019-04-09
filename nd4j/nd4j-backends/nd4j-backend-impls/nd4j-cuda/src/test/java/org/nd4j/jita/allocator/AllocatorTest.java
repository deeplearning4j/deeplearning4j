package test.org.nd4j.jita.allocator;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.jita.allocator.impl.MemoryTracker;
import static org.junit.Assert.assertTrue;

import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.MirroringPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.jita.allocator.impl.MemoryTracker;


public class AllocatorTest {

    private DataType initialType;

    public CudaWorkspaceTests(Nd4jBackend backend) {
        super(backend);
        this.initialType = Nd4j.dataType();
    }

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
    public void testWorkspaceReuse() {
        if (Nd4j.getExecutioner().type() != OpExecutioner.ExecutionerType.CUDA)
            return;

        val workspaceConfig = WorkspaceConfiguration.builder()
                .policyMirroring(MirroringPolicy.HOST_ONLY) // Commenting this out makes it so that assert is not triggered (for at least 40 secs or so...)
                .build();

        for (int  e = 0; e < 10; e++) {
           try (val ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(workspaceConfig, "test")) {
              final INDArray zeros = Nd4j.zeros(10, 'f');
              System.out.println(MemoryTracker.getInstance().getWorkspace(Nd4j.getAffinityManager().getDeviceForCurrentThread());
           }
        }
    }

}
