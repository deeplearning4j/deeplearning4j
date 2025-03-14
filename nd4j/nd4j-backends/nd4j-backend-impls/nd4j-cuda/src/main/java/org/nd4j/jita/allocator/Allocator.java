/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.jita.allocator;

import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.flow.FlowController;
import org.nd4j.jita.handler.MemoryHandler;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
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
     * This method returns CudaContext for current thread
     *
     * @return
     */
    CudaContext getDeviceContext();

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
    Pointer getPointer(DataBuffer buffer, CudaContext context);

    /**
     * This method returns actual host pointer valid for current object
     *
     * @param buffer
     */
    Pointer getHostPointer(DataBuffer buffer);

    /**
     * This method returns actual host pointer valid for current object
     *
     * @param array
     */
    Pointer getHostPointer(INDArray array);

    /**
     * This method returns actual device pointer valid for specified shape of current object
     *
     * @param buffer
     * @param shape
     */
    Pointer getPointer(DataBuffer buffer, AllocationShape shape, boolean isView, CudaContext context);


    /**
     * This method returns actual device pointer valid for specified INDArray
     */
    Pointer getPointer(INDArray array, CudaContext context);


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

    /** Returns {@link #getDeviceId()} wrapped as a {@link Pointer}. */
    Pointer getDeviceIdPointer();

    /**
     *  This method allocates required chunk of memory
     *
     * @param requiredMemory
     */
    AllocationPoint allocateMemory(DataBuffer buffer, AllocationShape requiredMemory, boolean initialize);

    /**
     * This method allocates required chunk of memory in specific location
     *
     * PLEASE NOTE: Do not use this method, unless you're 100% sure what you're doing
     *
     * @param requiredMemory
     * @param location
     */
    AllocationPoint allocateMemory(DataBuffer buffer, AllocationShape requiredMemory, AllocationStatus location,
                    boolean initialize);


    void memcpyBlocking(DataBuffer dstBuffer, Pointer srcPointer, long length, long dstOffset);

    void memcpyAsync(DataBuffer dstBuffer, Pointer srcPointer, long length, long dstOffset);

    void memcpySpecial(DataBuffer dstBuffer, Pointer srcPointer, long length, long dstOffset);

    void memcpyDevice(DataBuffer dstBuffer, Pointer srcPointer, long length, long dstOffset, CudaContext context);

    void memcpy(DataBuffer dstBuffer, DataBuffer srcBuffer);

    void tickHostWrite(DataBuffer buffer);

    void tickHostWrite(INDArray array);

    void tickDeviceWrite(INDArray array);

    AllocationPoint getAllocationPoint(INDArray array);

    AllocationPoint getAllocationPoint(DataBuffer buffer);

    void registerAction(CudaContext context, INDArray result, INDArray... operands);

    FlowController getFlowController();

    DataBuffer getConstantBuffer(int[] array);

    DataBuffer getConstantBuffer(long[] array);

    DataBuffer getConstantBuffer(float[] array);

    DataBuffer getConstantBuffer(double[] array);

    MemoryHandler getMemoryHandler();
}
