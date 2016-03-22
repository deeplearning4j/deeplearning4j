package org.nd4j.jita.mover;

import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.pointers.PointersPair;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.jcublas.context.CudaContext;

import java.util.Map;

/**
 * Mover interface describes methods for data transfers between host and devices
 *
 * @author raver119@gmail.com
 */
public interface Mover {

    void init(Configuration configuration, CudaEnvironment environment, Allocator allocator);

    /**
     * Allocate specified memory chunk on specified device/host
     *
     * @param targetMode valid arguments are DEVICE, ZERO
     * @return
     */
    PointersPair alloc(AllocationStatus targetMode, AllocationPoint point, AllocationShape shape);

    /**
     * This method checks if specified device has free memory
     *
     * @return
     */
    boolean pingDeviceForFreeMemory(Integer deviceId, long requiredMemory);

    /**
     *  Relocates specific chunk of memory from one storage to another
     *
     * @param currentStatus
     * @param targetStatus
     * @param point
     */
    void relocate(AllocationStatus currentStatus, AllocationStatus targetStatus, AllocationPoint point, AllocationShape shape);

    /**
     * Copies memory from device to host, if needed.
     * Device copy is preserved as is.
     *
     * @param point
     */
    void copyback(AllocationPoint point, AllocationShape shape);


    /**
     * Copies memory from host buffer to device.
     * Host copy is preserved as is.
     *
     * @param point
     */
    void copyforward(AllocationPoint point, AllocationShape shape);

    /**
     * Copies memory from device to zero-copy memory
     *
     * @param point
     * @param shape
     */
    void fallback(AllocationPoint point, AllocationShape shape);

    /**
     * This method frees memory chunk specified by pointer
     *
     * @param point
     */
    void free(AllocationPoint point, AllocationStatus target);

    /**
     * This method returns initial allocation location. So, it can be HOST, or DEVICE if environment allows that.
     *
     * @return
     */
    AllocationStatus getInitialLocation();

    /**
     * This method initializes specific device for current thread
     */
    void initializeDevice(Long threadId, Integer deviceId, Map<Long, CudaContext> contextPool);

    void memcpyBlocking(DataBuffer dstBuffer, jcuda.Pointer srcPointer, long length, long dstOffset);

    void memcpyAsync(DataBuffer dstBuffer, jcuda.Pointer srcPointer, long length, long dstOffset);

    void memcpy(DataBuffer dstBuffer, DataBuffer srcBuffer);
}
