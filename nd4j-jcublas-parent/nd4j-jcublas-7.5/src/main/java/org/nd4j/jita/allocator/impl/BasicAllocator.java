package org.nd4j.jita.allocator.impl;

import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * This is going to be basic JITA implementation.
 *
 * PLEASE NOTE: WORK IN PROGRESS, DO NOT EVER USE IT!
 *
 * @author raver119@gmail.com
 */
public final class BasicAllocator implements Allocator {
    private static final BasicAllocator INSTANCE = new BasicAllocator();
    private Configuration configuration = new Configuration();

    protected BasicAllocator() {
        //
    }

    public static BasicAllocator getInstance() {
        return INSTANCE;
    }


    /**
     * Consume and apply configuration passed in as argument
     *
     * @param configuration configuration bean to be applied
     */
    @Override
    public void applyConfiguration(Configuration configuration) {
        // TODO: lock to be implemented
        this.configuration = configuration;
    }

    /**
     * Returns current Allocator configuration
     *
     * @return current configuration
     */
    @Override
    public Configuration getConfiguration() {
        // TODO: lock to be implemented
        return configuration;
    }

    /**
     * This method registers buffer within allocator instance
     *
     * @param buffer
     */
    @Override
    public void pickupSpan(DataBuffer buffer) {

    }

    /**
     * This method registers array's buffer within allocator instance
     *
     * @param array
     */
    @Override
    public void pickupSpan(INDArray array) {

    }

    /**
     * This method hints allocator, that specific object was accessed on host side.
     * This includes putRow, putScalar etc methods as well as initial object instantiation.
     *
     * @param objectId unique object ID
     */
    @Override
    public void tickHost(Long objectId) {

    }

    /**
     * This methods hints allocator, that specific object was accessed on device side.
     *
     * @param objectId unique object ID
     * @param deviceId device ID
     */
    @Override
    public void tickDevice(Long objectId, Integer deviceId) {

    }

    /**
     * This method returns actual device pointer valid for current object
     *
     * @param objectId
     */
    @Override
    public void getDevicePointer(Long objectId) {

    }
}
