package org.nd4j.jita.allocator;

import jcuda.Pointer;
import org.nd4j.jita.allocator.enums.SyncState;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.linalg.api.buffer.BaseDataBuffer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * Allocator interface provides methods for transparent memory management
 *
 *
 * @author raver119@gmail.com
 */
public interface Allocator {

    /**
     * Consume and apply configuration passed in as argument
     *
     * @param configuration configuration bean to be applied
     */
    void applyConfiguration(Configuration configuration);

    /**
     * Returns current Allocator configuration
     *
     * @return current configuration
     */
    Configuration getConfiguration();

    /**
     * This method registers buffer within allocator instance
     */
    void pickupSpan(DataBuffer buffer);

    /**
     * This method registers array's buffer within allocator instance
     * @param array INDArray object to be picked
     */
    void pickupSpan(INDArray array);

    /**
     * This method hints allocator, that specific object was accessed on host side.
     * This includes putRow, putScalar;
     *
     * @param objectId unique object ID
     */
    void tickHost(Long objectId);

    /**
     * This methods hints allocator, that specific object was accessed on device side.
     *
     * @param objectId unique object ID
     * @param deviceId device ID
     */
    void tickDevice(Long objectId, Integer deviceId);

    /**
     * This method returns actual device pointer valid for current object
     *
     * @param objectId
     */
    Object getDevicePointer(Long objectId);

    /**
     * This method returns actual device pointer valid for specified shape of current object
     *
     * @param objectId
     * @param shape
     */
    Object getDevicePointer(Long objectId, AllocationShape shape);

    /**
     * This method should be called to make sure that data on host size is actualized
     *
     * @param objectId
     */
    void synchronizeHostData(Long objectId);

    /**
     * This method returns current host memory state
     *
     * @param objectId
     * @return
     */
    SyncState getHostMemoryState(Long objectId);

    /**
     * This method returns the number of top-level memory allocation.
     * No descendants are included in this result.
     *
     * @return number of allocated top-level memory chunks
     */
    int tableSize();
}
