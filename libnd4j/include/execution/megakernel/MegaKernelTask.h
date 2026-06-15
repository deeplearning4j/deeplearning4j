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
// Mega-kernel task data structures for Luminal-inspired persistent kernel execution.
//

#ifndef LIBND4J_MEGAKERNEL_TASK_H
#define LIBND4J_MEGAKERNEL_TASK_H

#include <system/common.h>
#include <cstdint>

namespace sd {
namespace execution {

/**
 * Task execution state for device-side tracking.
 */
enum class TaskState : int32_t {
    PENDING = 0,      // Task is waiting to be executed
    READY = 1,        // All dependencies satisfied, ready to execute
    RUNNING = 2,      // Task is currently executing
    COMPLETED = 3,    // Task has finished execution
    FAILED = 4        // Task execution failed
};

/**
 * Op types supported by the mega-kernel interpreter.
 * These correspond to commonly fused operations in LLM workloads.
 */
enum class MegaKernelOpType : int32_t {
    // Memory operations
    COPY = 0,
    FILL = 1,

    // Elementwise unary ops
    RELU = 10,
    SIGMOID = 11,
    TANH = 12,
    GELU = 13,
    SILU = 14,
    SOFTMAX = 15,

    // Elementwise binary ops
    ADD = 20,
    SUB = 21,
    MUL = 22,
    DIV = 23,

    // Reduction ops
    SUM = 30,
    MEAN = 31,
    MAX = 32,
    MIN = 33,

    // Normalization ops
    LAYER_NORM = 40,
    RMS_NORM = 41,
    BATCH_NORM = 42,

    // Fused LLM ops
    FUSED_GELU = 50,
    FUSED_LAYER_NORM = 51,
    FUSED_ROPE = 52,
    FUSED_BIAS_DROPOUT_RESIDUAL = 53,
    FUSED_RMS_NORM_SWIGLU = 54,

    // Matrix operations (not typically fused, but useful for small matmuls)
    MATMUL_SMALL = 60,  // For small matrices that fit in shared memory

    // Attention ops
    FLASH_ATTENTION = 70,
    DOT_PRODUCT_ATTENTION = 71,

    // Barrier/sync
    BARRIER = 100,

    // Unknown/custom
    CUSTOM = 255
};

/**
 * Descriptor for a single operation within a mega-kernel task.
 * This is a compact representation designed for device-side interpretation.
 */
struct alignas(16) OpDescriptor {
    MegaKernelOpType opType;       // Type of operation
    int32_t inputCount;             // Number of inputs
    int32_t outputCount;            // Number of outputs
    int32_t paramOffset;            // Offset into parameter buffer

    // Input/output buffer indices (relative to task's buffer list)
    int32_t inputIndices[4];        // Up to 4 inputs
    int32_t outputIndices[2];       // Up to 2 outputs

    // Shape information for this op
    int32_t rank;
    int64_t shape[4];               // Compact shape storage
    int64_t stride[4];              // Strides for non-contiguous access

    // Padding for alignment
    int32_t _padding[2];
};

/**
 * A mega-kernel task represents a fused subgraph that can be executed
 * as a single persistent kernel invocation.
 *
 * The task contains:
 * - A sequence of operations to execute
 * - Input/output buffer references
 * - Barrier dependencies for synchronization
 */
struct alignas(64) MegaKernelTask {
    // Task identification
    int32_t taskId;
    int32_t graphId;                // Which graph this task belongs to

    // Operation list
    int32_t opCount;                // Number of operations in this task
    int32_t opDescriptorOffset;     // Offset into global op descriptor array

    // Buffer management
    int32_t inputCount;
    int32_t outputCount;
    int32_t tempBufferCount;        // Number of temporary buffers needed
    int32_t bufferListOffset;       // Offset into global buffer pointer array

    // Synchronization
    int32_t waitBarrierCount;       // Number of barriers to wait on before execution
    int32_t signalBarrierCount;     // Number of barriers to signal after completion
    int32_t waitBarrierOffset;      // Offset into barrier wait list
    int32_t signalBarrierOffset;    // Offset into barrier signal list

    // Execution state (volatile for device-side updates)
    volatile TaskState state;

    // Grid/block configuration for CUDA
    int32_t gridDimX;
    int32_t gridDimY;
    int32_t gridDimZ;
    int32_t blockDimX;
    int32_t blockDimY;
    int32_t blockDimZ;
    int32_t sharedMemBytes;

    // Padding to cache line
    int32_t _padding[7];
};

/**
 * Device-side task queue for the mega-kernel interpreter.
 * Uses lock-free atomic operations for concurrent access.
 */
struct alignas(64) DeviceTaskQueue {
    // Circular queue indices
    volatile int32_t head;          // Next task to dequeue (consumer)
    volatile int32_t tail;          // Next slot for enqueue (producer)
    int32_t capacity;               // Maximum number of tasks
    int32_t _padding1;

    // Task array pointer (device memory)
    MegaKernelTask* tasks;

    // Op descriptor array (shared across all tasks)
    OpDescriptor* opDescriptors;

    // Buffer pointer array (shared across all tasks)
    void** buffers;

    // Parameter buffer for op-specific data
    void* paramBuffer;
    int64_t paramBufferSize;

    // Barrier state array
    volatile int32_t* barriers;
    int32_t barrierCount;
    int32_t _padding2;
};

/**
 * Host-side handle for managing a mega-kernel task graph.
 */
struct MegaKernelTaskGraph {
    int32_t graphId;
    int32_t taskCount;

    // Host-side task storage
    MegaKernelTask* tasks;
    OpDescriptor* opDescriptors;
    int32_t totalOpCount;

    // Device-side copies
    MegaKernelTask* deviceTasks;
    OpDescriptor* deviceOpDescriptors;
    void** deviceBuffers;
    void* deviceParamBuffer;
    int32_t* deviceBarriers;

    // Graph metadata
    int32_t inputCount;
    int32_t outputCount;
    int32_t intermediateCount;
    int32_t barrierCount;

    // Execution statistics
    int64_t totalKernelLaunches;
    int64_t totalExecutionTimeNs;
};

}  // namespace execution
}  // namespace sd

#endif  // LIBND4J_MEGAKERNEL_TASK_H
