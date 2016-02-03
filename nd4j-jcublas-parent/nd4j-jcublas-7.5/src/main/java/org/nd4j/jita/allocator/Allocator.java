package org.nd4j.jita.allocator;

import jcuda.Pointer;
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
     * @param array
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
    void getDevicePointer(Long objectId);
}
