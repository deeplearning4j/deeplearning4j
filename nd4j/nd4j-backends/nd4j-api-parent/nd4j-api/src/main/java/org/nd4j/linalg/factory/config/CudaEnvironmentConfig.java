/*
 *  ******************************************************************************
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
package org.nd4j.linalg.factory.config;

/**
 * CUDA device configuration: device selection, memory, P2P, allocator,
 * stream/event limits, graph optimization, tensor cores, device limits.
 */
public interface CudaEnvironmentConfig {

    int cudaDeviceCount();
    int cudaCurrentDevice();
    void setCudaCurrentDevice(int device);
    boolean cudaMemoryPinned();
    void setCudaMemoryPinned(boolean pinned);
    boolean cudaUseManagedMemory();
    void setCudaUseManagedMemory(boolean managed);
    int cudaMemoryPoolSize();
    void setCudaMemoryPoolSize(int sizeInMB);
    boolean cudaForceP2P();
    void setCudaForceP2P(boolean forceP2P);
    boolean cudaAllocatorEnabled();
    void setCudaAllocatorEnabled(boolean enabled);
    int cudaMaxBlocks();
    void setCudaMaxBlocks(int blocks);
    int cudaMaxThreadsPerBlock();
    void setCudaMaxThreadsPerBlock(int threads);
    boolean cudaAsyncExecution();
    void setCudaAsyncExecution(boolean async);
    int cudaStreamLimit();
    void setCudaStreamLimit(int limit);
    boolean cudaUseDeviceHost();
    void setCudaUseDeviceHost(boolean useDeviceHost);
    int cudaEventLimit();
    void setCudaEventLimit(int limit);
    int cudaCachingAllocatorLimit();
    void setCudaCachingAllocatorLimit(int limitInMB);
    long cudaPinnedHostLimit();
    void setCudaPinnedHostLimit(long limitInMB);
    boolean cudaUseUnifiedMemory();
    void setCudaUseUnifiedMemory(boolean unified);
    int cudaPrefetchSize();
    void setCudaPrefetchSize(int sizeInMB);
    boolean cudaGraphOptimization();
    void setCudaGraphOptimization(boolean enabled);
    boolean cudaTensorCoreEnabled();
    void setCudaTensorCoreEnabled(boolean enabled);
    int cudaBlockingSync();
    void setCudaBlockingSync(int mode);
    int cudaDeviceSchedule();
    void setCudaDeviceSchedule(int schedule);

    // CUDA Device Limits
    default long cudaGetLimit(int limitType) { return 0; }
    default void cudaSetLimit(int limitType, long value) {}
}
