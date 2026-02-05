/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author Adam Gibson
//
// Barrier management for mega-kernel synchronization.
//

#ifndef LIBND4J_MEGAKERNEL_BARRIER_MANAGER_H
#define LIBND4J_MEGAKERNEL_BARRIER_MANAGER_H

#include <system/common.h>
#include <cstdint>
#include <atomic>
#include <vector>
#include <mutex>

namespace sd {
namespace execution {

/**
 * Barrier state for device-side synchronization.
 * Uses atomic operations for lock-free signaling.
 */
struct DeviceBarrier {
    volatile int32_t count;     // Number of tasks that have reached the barrier
    int32_t target;              // Number of tasks that must reach before release
    volatile int32_t generation; // Incremented each time barrier is reset
    int32_t _padding;
};

/**
 * Manages barriers for inter-task synchronization in mega-kernels.
 *
 * Barriers are used to enforce execution ordering between tasks that
 * have data dependencies. A task waits on barriers from its producers
 * and signals barriers to its consumers.
 *
 * The barrier system supports:
 * - Multiple producers signaling a single barrier
 * - Multiple consumers waiting on a single barrier
 * - Automatic reset after all consumers have passed
 */
class BarrierManager {
public:
    BarrierManager();
    ~BarrierManager();

    /**
     * Initialize barriers for a task graph.
     * @param numBarriers Number of barriers needed
     * @param context Launch context for device allocation
     */
    void initialize(int32_t numBarriers, LaunchContext* context);

    /**
     * Reset all barriers to their initial state.
     */
    void reset();

    /**
     * Get a barrier ID for a producer-consumer edge.
     * Creates a new barrier if needed.
     * @param producerTaskId The task producing data
     * @param consumerTaskId The task consuming data
     * @return Barrier ID
     */
    int32_t getOrCreateBarrier(int32_t producerTaskId, int32_t consumerTaskId);

    /**
     * Configure a barrier's target count.
     * @param barrierId The barrier to configure
     * @param producerCount Number of producers that will signal
     * @param consumerCount Number of consumers that will wait
     */
    void configureBarrier(int32_t barrierId, int32_t producerCount, int32_t consumerCount);

    /**
     * Get the device pointer to the barrier array.
     */
    volatile int32_t* getDeviceBarriers() const;

    /**
     * Get the number of barriers.
     */
    int32_t getBarrierCount() const;

    /**
     * Host-side barrier wait (for debugging/testing).
     */
    void hostWait(int32_t barrierId);

    /**
     * Host-side barrier signal (for debugging/testing).
     */
    void hostSignal(int32_t barrierId);

    /**
     * Check if a barrier has been reached by all producers.
     */
    bool isBarrierReached(int32_t barrierId) const;

private:
    struct BarrierInfo {
        int32_t producerCount;
        int32_t consumerCount;
        std::vector<int32_t> producers;
        std::vector<int32_t> consumers;
    };

    std::vector<BarrierInfo> barrierInfo_;
    std::vector<DeviceBarrier> hostBarriers_;
    DeviceBarrier* deviceBarriers_;
    int32_t barrierCount_;
    int32_t allocatedCount_;

    mutable std::mutex mutex_;
    LaunchContext* context_;

    // Edge to barrier mapping
    std::vector<std::pair<int32_t, int32_t>> edgeToBarrier_;
};

/**
 * Device-side barrier operations.
 * These are implemented as __device__ functions in the CUDA file.
 */
#ifdef __CUDA_ARCH__
__device__ void barrierWait(volatile int32_t* barriers, int32_t barrierId, int32_t generation);
__device__ void barrierSignal(volatile int32_t* barriers, int32_t barrierId);
__device__ bool barrierTryWait(volatile int32_t* barriers, int32_t barrierId, int32_t generation);
#endif

}  // namespace execution
}  // namespace sd

#endif  // LIBND4J_MEGAKERNEL_BARRIER_MANAGER_H
