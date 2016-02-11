package org.nd4j.jita.mover;

import jcuda.Pointer;
import lombok.NonNull;
import org.nd4j.jita.allocator.enums.SyncState;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.allocator.enums.AllocationStatus;
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

    private static Logger log = LoggerFactory.getLogger(DummyMover.class);

    @Override
    public void init(@NonNull Configuration configuration, @NonNull CudaEnvironment environment) {
        this.configuration = configuration;
        this.environment = environment;
    }

    /**
     * Allocate specified memory chunk on specified device/host
     *
     * @param targetMode valid arguments are DEVICE, ZERO
     * @param shape
     */
    @Override
    public Pointer alloc(AllocationStatus targetMode, AllocationPoint point, AllocationShape shape) {
        if (!targetMode.equals(AllocationStatus.DEVICE) && !targetMode.equals(AllocationStatus.ZERO) )
            throw new UnsupportedOperationException("Target allocation ["+ targetMode+"] is not supported");
        return new Pointer();
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
            case DEVICE: {
                if (targetStatus.equals(AllocationStatus.ZERO)) {
                    long memorySize = AllocationUtils.getRequiredMemory(point.getShape());

                    point.setAllocationStatus(targetStatus);
                    point.setCudaPointer(new Object());

                    try {
//                        locker.globalWriteLock();

                        log.info("Relocating: "+ point.getObjectId()+" Substracting memory from alloc table: [" +memorySize + "]. Direction is: [" + currentStatus + "] -> [" + targetStatus +"]");

                        environment.trackAllocatedMemory(point.getDeviceId(), -1 * memorySize);

                    } finally {
//                        locker.globalWriteUnlock();
                    }

                } else throw new UnsupportedOperationException("HostMemory relocation in this direction isn't supported: [" + currentStatus + "] -> [" + targetStatus +"]");
            }
            break;
            case HOST:
            case ZERO: {
                    if (targetStatus.equals(AllocationStatus.DEVICE)) {
                        long memorySize = AllocationUtils.getRequiredMemory(point.getShape());


                        try {
//                            locker.globalWriteLock();

                            // TODO: real memory query should be considered here in real mover
                            if (memorySize + environment.getAllocatedMemoryForDevice(1) >= configuration.getMaximumDeviceAllocation())
                                return;

                      //      log.info("Adding memory to alloc table: [" +memorySize + "]");
                            log.info("Relocating: "+ point.getObjectId()+" Adding memory to alloc table: [" +memorySize + "]. Direction is: [" + currentStatus + "] -> [" + targetStatus +"]");

                            environment.trackAllocatedMemory(point.getDeviceId(), AllocationUtils.getRequiredMemory(point.getShape()));

                        } finally {
//                            locker.globalWriteUnlock();
                        }



                        point.setAllocationStatus(targetStatus);
                        point.setCudaPointer(new Object());
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
     * Copies memory from host buffer to device.
     * Host copy is preserved as is.
     *
     * @param point
     */
    @Override
    public void copyforward(AllocationPoint point) {

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
            point.setCudaPointer(null);
            point.setAllocationStatus(AllocationStatus.HOST);
        } else {
            throw new UnsupportedOperationException("free() is impossible for : ["+point.getAllocationStatus()+"] allocation");
        }
    }
}
