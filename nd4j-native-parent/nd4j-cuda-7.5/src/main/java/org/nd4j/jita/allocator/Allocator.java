package org.nd4j.jita.allocator;

import jcuda.Pointer;
import jcuda.jcublas.cublasHandle;
import org.nd4j.jita.allocator.enums.SyncState;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.jita.mover.Mover;
import org.nd4j.linalg.api.buffer.BaseDataBuffer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.jcublas.buffer.BaseCudaDataBuffer;
import org.nd4j.linalg.jcublas.context.CudaContext;

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
     * This method allows you to exclude specific device from being used for calculations
     *
     * Please note: you can call this method multiple times, to ban multiple devices
     *
     * @param deviceId deviceId to be banned
     */
    void banDevice(Integer deviceId);

    /**
     * Set active CUDA environment
     *
     * @param environment
     */
    void setEnvironment(CudaEnvironment environment);

    /**
     * This method returns CudaContext for current thread
     *
     * @return
     */
    CudaContext getCudaContext();

    /**
     * This methods specifies Mover implementation to be used internally
     *
     * @param mover
     */
    void setMover(Mover mover);

    /**
     * Returns current Allocator configuration
     *
     * @return current configuration
     */
    Configuration getConfiguration();

    /**
     * This method registers buffer within allocator instance
     */
   // Long pickupSpan(BaseCudaDataBuffer buffer, AllocationShape shape);

    /**
     * This method registers array's buffer within allocator instance
     * @param array INDArray object to be picked
     */
    Long pickupSpan(INDArray array);

    /**
     * This method hints allocator, that specific object was accessed on host side.
     * This includes putRow, putScalar;
     *
     * @param array
     */
    void tickHost(INDArray array);


    /**
     * This methods hints allocator, that specific object was accessed on device side.
     *
     * @param array
     */
    @Deprecated
    void tickDevice(INDArray array);


    /**
     * This method hints allocator, that specific object was released on device side
     *
     * @param array
     */
    void tackDevice(INDArray array);

    /**
     * This method notifies allocator, that specific object was changed on device side
     *
     * @param array
     */
    void tickDeviceWrite(INDArray array);

    /**
     * This method notifies allocator, that specific object was changed on host side
     *
     * @param array
     */
    void tickHostWrite(INDArray array);

    /**
     * This method returns actual device pointer valid for current object
     *
     * @param buffer
     */
    @Deprecated
    Pointer getDevicePointer(DataBuffer buffer);

    /**
     * This method returns actual device pointer valid for specified shape of current object
     *
     * @param buffer
     * @param shape
     */
    @Deprecated
    Pointer getDevicePointer(DataBuffer buffer, AllocationShape shape, boolean isView);


    /**
     * This method returns actual device pointer valid for specified INDArray
     */
    Pointer getDevicePointer(INDArray array);


    /**
     * This method returns actual host pointer, valid for specified shape of current object
     *
     * @param array
     * @return
     */
    Pointer getHostPointer(INDArray array);

    /**
     * This method should be callsd to make sure that data on host side is actualized
     *
     * @param array
     */
    void synchronizeHostData(INDArray array);

    /**
     * This method should be calls to make sure that data on host side is actualized
     *
     * @param buffer
     */
    void synchronizeHostData(DataBuffer buffer);

    /**
     * This method should be callsd to make sure that data on host side is actualized.
     * However, this method only tries to lock data before synchronization.
     *
     * PLEASE NOTE: This methos is considered non-safe.
     *
     * @param buffer
     */
    void trySynchronizeHostData(DataBuffer buffer);

    /**
     * This method returns current host memory state
     *
     * @param array
     * @return
     */
    SyncState getHostMemoryState(INDArray array);

    /**
     * This method returns the number of top-level memory allocation.
     * No descendants are included in this result.
     *
     * @return number of allocated top-level memory chunks
     */
    int tableSize();


    /**
     * This method returns CUDA deviceId for specified array
     *
     * @param array
     * @return
     */
    Integer getDeviceId(INDArray array);

    /**
     * This method returns CUDA deviceId for current thread
     *
     * @return
     */
    Integer getDeviceId();
}
