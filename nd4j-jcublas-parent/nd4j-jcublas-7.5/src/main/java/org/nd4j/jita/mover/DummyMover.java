package org.nd4j.jita.mover;

import org.nd4j.jita.allocator.enums.SyncState;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.allocator.enums.AllocationStatus;

/**
 * This is dummy Mover implementation, suitable for tests. It does not handles any allocations, but provides proper responses :)
 *
 * PLEASE NOTE: Do not use it in production environment :)
 *
 * @author raver119@gmail.com
 */
public class DummyMover implements Mover {
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
                    point.setAllocationStatus(targetStatus);
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
}
