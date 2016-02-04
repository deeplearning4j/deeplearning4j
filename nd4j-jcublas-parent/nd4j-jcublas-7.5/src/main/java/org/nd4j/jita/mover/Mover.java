package org.nd4j.jita.mover;

import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.allocator.enums.AllocationStatus;

/**
 * Mover interface describes methods for data transfers between host and devices
 *
 * @author raver119@gmail.com
 */
public interface Mover {


    /**
     * Allocate specified memory chunk on specified device/host
     *
     * @param targetMode valid arguments are DEVICE, ZERO
     * @param deviceId Id of the device for allocation. Value is ignored if UMA is available and/or HOST allocation is called
     * @return
     */
    Object alloc(AllocationStatus targetMode,AllocationShape shape, Integer deviceId);


    /**
     *  Relocates specific chunk of memory from one storage to another
     *
     * @param currentStatus
     * @param targetStatus
     * @param point
     */
    void relocate(AllocationStatus currentStatus, AllocationStatus targetStatus, AllocationPoint point);

    /**
     * Copies memory from device to host, if needed.
     * Device copy is preserved as is.
     *
     * @param point
     */
    void copyback(AllocationPoint point);
}
