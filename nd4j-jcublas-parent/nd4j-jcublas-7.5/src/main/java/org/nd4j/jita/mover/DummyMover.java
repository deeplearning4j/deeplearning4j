package org.nd4j.jita.mover;

import lombok.NonNull;
import org.nd4j.jita.allocator.enums.SyncState;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.locks.Lock;
import org.nd4j.jita.allocator.utils.AllocationUtils;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This is dummy Mover implementation, suitable for tests. It does not handles any allocations, but provides proper responses :)
 *
 * PLEASE NOTE: Do not use it in production environment :)
 *
 * @author raver119@gmail.com
 */
public class DummyMover implements Mover {
    private Configuration configuration;
    private CudaEnvironment environment;
    private Lock locker;

    private static Logger log = LoggerFactory.getLogger(DummyMover.class);

    @Override
    public void init(@NonNull Configuration configuration, @NonNull CudaEnvironment environment, @NonNull Lock locker) {
        this.configuration = configuration;
        this.environment = environment;
        this.locker = locker;
    }

    /**
     * Allocate specified memory chunk on specified device/host
     *
     * @param targetMode valid arguments are DEVICE, ZERO
     * @param shape
     * @param deviceId   Id of the device for allocation. Value is ignored if UMA is available and/or HOST allocation is called  @return
     */
    @Override
    public Object alloc(AllocationStatus targetMode, AllocationShape shape, Integer deviceId) {
        if (!targetMode.equals(AllocationStatus.DEVICE) && !targetMode.equals(AllocationStatus.ZERO) )
            throw new UnsupportedOperationException("Target allocation ["+ targetMode+"] is not supported");
        return new Object();
    }

    /**
     * Relocates specific chunk of memory from one storage to another
     *
     * @param currentStatus
     * @param targetStatus
     * @param point
     */
    @Override
    public void relocate(AllocationStatus currentStatus, AllocationStatus targetStatus, AllocationPoint point) {
        if (currentStatus.equals(targetStatus)) return;

        switch (currentStatus) {
            case HOST:
            case ZERO: {
                    if (targetStatus.equals(AllocationStatus.DEVICE)) {
                        long memorySize = AllocationUtils.getRequiredMemory(point.getShape());


                        try {
                            locker.globalWriteLock();

                            // TODO: real memory query should be considered here in real mover
                            if (memorySize + environment.getAllocatedMemoryForDevice(1) >= configuration.getMaximumAllocation())
                                return;

                      //      log.info("Adding memory to alloc table: [" +memorySize + "]");

                            environment.trackAllocatedMemory(1, AllocationUtils.getRequiredMemory(point.getShape()));

                        } finally {
                            locker.globalWriteUnlock();
                        }



                        point.setAllocationStatus(targetStatus);
                        point.setDevicePointer(new Object());
                    } else throw new UnsupportedOperationException("HostMemory relocation in this direction isn't supported: [" + currentStatus + "] -> [" + targetStatus +"]");
                }
                break;
            default:
                throw new UnsupportedOperationException("Relocation in this direction isn't supported: [" + currentStatus + "] -> [" + targetStatus +"]");
        }
    }

    /**
     * Copies memory from device to host, if needed.
     * Device copy is preserved as is.
     *
     * @param point
     */
    @Override
    public void copyback(AllocationPoint point) {
        if (point.getAllocationStatus().equals(AllocationStatus.DEVICE) || point.getAllocationStatus().equals(AllocationStatus.ZERO)) {
            point.setAccessHost(point.getAccessDevice());
            point.setHostMemoryState(SyncState.SYNC);
        } else {
            throw new UnsupportedOperationException("Copyback is impossible for direction: ["+point.getAllocationStatus()+"] -> [HOST]");
        }
    }

    /**
     * This method frees memory chunk specified by allocation point
     *
     * @param point
     */
    @Override
    public void free(AllocationPoint point) {
        if (point.getAllocationStatus().equals(AllocationStatus.DEVICE) || point.getAllocationStatus().equals(AllocationStatus.ZERO)) {
            point.setAccessHost(point.getAccessDevice());
            point.setHostMemoryState(SyncState.SYNC);
            point.setDevicePointer(null);
            point.setAllocationStatus(AllocationStatus.HOST);
        } else {
            throw new UnsupportedOperationException("free() is impossible for : ["+point.getAllocationStatus()+"] allocation");
        }
    }
}
