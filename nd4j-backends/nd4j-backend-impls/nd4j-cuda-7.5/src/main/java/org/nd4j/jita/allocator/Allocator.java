package org.nd4j.jita.allocator;

import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.enums.SyncState;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.allocator.context.ExternalContext;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.jita.mover.MemoryHandler;
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
    ExternalContext getDeviceContext();

    /**
     * This methods specifies Mover implementation to be used internally
     *
     * @param memoryHandler
     */
    void setMemoryHandler(MemoryHandler memoryHandler);

    /**
     * Returns current Allocator configuration
     *
     * @return current configuration
     */
    Configuration getConfiguration();

    /**
     * This method returns actual device pointer valid for current object
     *
     * @param buffer
     */
    Pointer getPointer(DataBuffer buffer);

    /**
     * This method returns actual device pointer valid for specified shape of current object
     *
     * @param buffer
     * @param shape
     */
     Pointer getPointer(DataBuffer buffer, AllocationShape shape, boolean isView);


    /**
     * This method returns actual device pointer valid for specified INDArray
     */
    Pointer getPointer(INDArray array);


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
     * This method returns deviceId for current thread
     * All values >= 0 are considered valid device IDs, all values < 0 are considered stubs.
     *
     * @return
     */
     Integer getDeviceId();

    /**
     *  This method allocates required chunk of memory
     *
     * @param requiredMemory
     */
    AllocationPoint allocateMemory(AllocationShape requiredMemory);

    /**
     * This method allocates required chunk of memory in specific location
     *
     * PLEASE NOTE: Do not use this method, unless you're 100% sure what you're doing
     *
     * @param requiredMemory
     * @param location
     */
    AllocationPoint allocateMemory(AllocationShape requiredMemory, AllocationStatus location);


    void memcpyBlocking(DataBuffer dstBuffer, jcuda.Pointer srcPointer, long length, long dstOffset);

    void memcpyAsync(DataBuffer dstBuffer, jcuda.Pointer srcPointer, long length, long dstOffset);

    void memcpy(DataBuffer dstBuffer, DataBuffer srcBuffer);
}
