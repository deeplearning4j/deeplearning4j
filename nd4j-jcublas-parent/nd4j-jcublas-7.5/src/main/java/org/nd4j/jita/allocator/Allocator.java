package org.nd4j.jita.allocator;

import jcuda.Pointer;
import org.nd4j.jita.allocator.enums.SyncState;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.linalg.api.buffer.BaseDataBuffer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.jcublas.buffer.BaseCudaDataBuffer;

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
    Long pickupSpan(BaseCudaDataBuffer buffer, AllocationShape shape);

    /**
     * This method registers array's buffer within allocator instance
     * @param array INDArray object to be picked
     */
    Long pickupSpan(INDArray array);

    /**
     * This method hints allocator, that specific object was accessed on host side.
     * This includes putRow, putScalar;
     *
     * @param objectId unique object ID
     */
    void tickHost(BaseCudaDataBuffer objectId);

    /**
     * This methods hints allocator, that specific object was accessed on device side.
     *
     * @param objectId unique object ID
     * @param shape shape
     */
    @Deprecated
    void tickDevice(BaseCudaDataBuffer objectId, AllocationShape shape);


    void tackDevice(BaseCudaDataBuffer objectId, AllocationShape shape);

    /**
     * This method returns actual device pointer valid for current object
     *
     * @param objectId
     */
    Object getDevicePointer(BaseCudaDataBuffer objectId);

    /**
     * This method returns actual device pointer valid for specified shape of current object
     *
     * @param objectId
     * @param shape
     */
    Object getDevicePointer(BaseCudaDataBuffer objectId, AllocationShape shape);

    /**
     * This method should be called to make sure that data on host size is actualized
     *
     * @param objectId
     */
    void synchronizeHostData(BaseCudaDataBuffer objectId);

    /**
     * This method returns current host memory state
     *
     * @param objectId
     * @return
     */
    SyncState getHostMemoryState(BaseCudaDataBuffer objectId);

    /**
     * This method returns the number of top-level memory allocation.
     * No descendants are included in this result.
     *
     * @return number of allocated top-level memory chunks
     */
    int tableSize();


    /**
     * This method returns CUDA deviceId for specified buffer
     *
     * @param objectId
     * @return
     */
    Integer getDeviceId(BaseCudaDataBuffer objectId);

    /**
     * This method returns CUDA deviceId for current thread
     *
     * @return
     */
    Integer getDeviceId();
}
