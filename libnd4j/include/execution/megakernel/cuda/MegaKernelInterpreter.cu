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
// CUDA mega-kernel interpreter - persistent kernel for task execution.
//

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <execution/megakernel/MegaKernelTask.h>

namespace cg = cooperative_groups;

namespace sd {
namespace execution {

//////////////////////////////////////////////////////////////////////////////
// Device-side barrier operations
//////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ void barrierWait(volatile int32_t* barriers, int32_t barrierId, int32_t targetCount) {
    // Spin-wait until barrier count reaches target
    while (barriers[barrierId] < targetCount) {
        __threadfence();
    }
}

__device__ __forceinline__ void barrierSignal(volatile int32_t* barriers, int32_t barrierId) {
    atomicAdd((int32_t*)&barriers[barrierId], 1);
    __threadfence();
}

__device__ __forceinline__ bool barrierTryWait(volatile int32_t* barriers, int32_t barrierId, int32_t targetCount) {
    return barriers[barrierId] >= targetCount;
}

//////////////////////////////////////////////////////////////////////////////
// Op executor device functions
//////////////////////////////////////////////////////////////////////////////

template<typename T>
__device__ void executeRelu(const T* input, T* output, int64_t n, int tid, int stride) {
    for (int64_t i = tid; i < n; i += stride) {
        T val = input[i];
        output[i] = val > static_cast<T>(0) ? val : static_cast<T>(0);
    }
}

template<typename T>
__device__ void executeSigmoid(const T* input, T* output, int64_t n, int tid, int stride) {
    for (int64_t i = tid; i < n; i += stride) {
        float val = static_cast<float>(input[i]);
        output[i] = static_cast<T>(1.0f / (1.0f + expf(-val)));
    }
}

template<typename T>
__device__ void executeTanh(const T* input, T* output, int64_t n, int tid, int stride) {
    for (int64_t i = tid; i < n; i += stride) {
        output[i] = static_cast<T>(tanhf(static_cast<float>(input[i])));
    }
}

template<typename T>
__device__ void executeSilu(const T* input, T* output, int64_t n, int tid, int stride) {
    for (int64_t i = tid; i < n; i += stride) {
        float val = static_cast<float>(input[i]);
        float sig = 1.0f / (1.0f + expf(-val));
        output[i] = static_cast<T>(val * sig);
    }
}

template<typename T>
__device__ void executeFusedGelu(const T* input, T* output, int64_t n, int tid, int stride) {
    for (int64_t i = tid; i < n; i += stride) {
        float val = static_cast<float>(input[i]);
        float sig = 1.0f / (1.0f + expf(-1.702f * val));
        output[i] = static_cast<T>(val * sig);
    }
}

template<typename T>
__device__ void executeAdd(const T* a, const T* b, T* out, int64_t n, int tid, int stride) {
    for (int64_t i = tid; i < n; i += stride) {
        out[i] = a[i] + b[i];
    }
}

template<typename T>
__device__ void executeMul(const T* a, const T* b, T* out, int64_t n, int tid, int stride) {
    for (int64_t i = tid; i < n; i += stride) {
        out[i] = a[i] * b[i];
    }
}

template<typename T>
__device__ void executeCopy(const T* src, T* dst, int64_t n, int tid, int stride) {
    for (int64_t i = tid; i < n; i += stride) {
        dst[i] = src[i];
    }
}

//////////////////////////////////////////////////////////////////////////////
// Op dispatcher
//////////////////////////////////////////////////////////////////////////////

template<typename T>
__device__ void dispatchOp(const OpDescriptor& op, void** buffers, void* params, int tid, int stride) {
    int64_t n = op.shape[0];
    for (int r = 1; r < op.rank; r++) {
        n *= op.shape[r];
    }

    switch (op.opType) {
        case MegaKernelOpType::COPY:
            executeCopy<T>(
                (const T*)buffers[op.inputIndices[0]],
                (T*)buffers[op.outputIndices[0]],
                n, tid, stride);
            break;

        case MegaKernelOpType::RELU:
            executeRelu<T>(
                (const T*)buffers[op.inputIndices[0]],
                (T*)buffers[op.outputIndices[0]],
                n, tid, stride);
            break;

        case MegaKernelOpType::SIGMOID:
            executeSigmoid<T>(
                (const T*)buffers[op.inputIndices[0]],
                (T*)buffers[op.outputIndices[0]],
                n, tid, stride);
            break;

        case MegaKernelOpType::TANH:
            executeTanh<T>(
                (const T*)buffers[op.inputIndices[0]],
                (T*)buffers[op.outputIndices[0]],
                n, tid, stride);
            break;

        case MegaKernelOpType::SILU:
            executeSilu<T>(
                (const T*)buffers[op.inputIndices[0]],
                (T*)buffers[op.outputIndices[0]],
                n, tid, stride);
            break;

        case MegaKernelOpType::FUSED_GELU:
            executeFusedGelu<T>(
                (const T*)buffers[op.inputIndices[0]],
                (T*)buffers[op.outputIndices[0]],
                n, tid, stride);
            break;

        case MegaKernelOpType::ADD:
            executeAdd<T>(
                (const T*)buffers[op.inputIndices[0]],
                (const T*)buffers[op.inputIndices[1]],
                (T*)buffers[op.outputIndices[0]],
                n, tid, stride);
            break;

        case MegaKernelOpType::MUL:
            executeMul<T>(
                (const T*)buffers[op.inputIndices[0]],
                (const T*)buffers[op.inputIndices[1]],
                (T*)buffers[op.outputIndices[0]],
                n, tid, stride);
            break;

        default:
            // Unsupported op - should not happen if graph analysis is correct
            break;
    }
}

//////////////////////////////////////////////////////////////////////////////
// Task executor
//////////////////////////////////////////////////////////////////////////////

template<typename T>
__device__ void executeTask(MegaKernelTask* task, OpDescriptor* opDescriptors, void** buffers,
                             void* paramBuffer, volatile int32_t* barriers, cg::grid_group& grid) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // Wait on all input barriers
    if (tid == 0) {
        for (int i = 0; i < task->waitBarrierCount; i++) {
            int32_t barrierId = task->waitBarrierOffset + i;
            // Barriers encode producer count in upper bits
            barrierWait(barriers, barrierId, 1);
        }
    }
    grid.sync();

    // Execute all operations in the task
    for (int i = 0; i < task->opCount; i++) {
        OpDescriptor& op = opDescriptors[task->opDescriptorOffset + i];
        void* params = (char*)paramBuffer + op.paramOffset;
        dispatchOp<T>(op, buffers, params, tid, stride);

        // Sync between ops within a task
        grid.sync();
    }

    // Signal completion barriers
    if (tid == 0) {
        for (int i = 0; i < task->signalBarrierCount; i++) {
            int32_t barrierId = task->signalBarrierOffset + i;
            barrierSignal(barriers, barrierId);
        }
        task->state = TaskState::COMPLETED;
    }
}

//////////////////////////////////////////////////////////////////////////////
// Main persistent kernel
//////////////////////////////////////////////////////////////////////////////

template<typename T>
__global__ void megaKernelInterpreter(DeviceTaskQueue* queue, volatile bool* shouldStop) {
    cg::grid_group grid = cg::this_grid();
    int globalTid = threadIdx.x + blockIdx.x * blockDim.x;

    while (!(*shouldStop)) {
        // Try to dequeue a task
        int taskIdx = -1;
        if (globalTid == 0) {
            int head = queue->head;
            int tail = queue->tail;
            if (head < tail) {
                taskIdx = atomicAdd((int32_t*)&queue->head, 1);
                if (taskIdx >= tail) {
                    taskIdx = -1;  // No task available
                }
            }
        }

        // Broadcast task index to all threads
        taskIdx = __shfl_sync(0xffffffff, taskIdx, 0);

        if (taskIdx >= 0 && taskIdx < queue->capacity) {
            MegaKernelTask* task = &queue->tasks[taskIdx];

            if (task->state == TaskState::READY) {
                if (globalTid == 0) {
                    task->state = TaskState::RUNNING;
                }
                grid.sync();

                executeTask<T>(task, queue->opDescriptors, queue->buffers,
                               queue->paramBuffer, queue->barriers, grid);
            }
        }

        // Brief pause to reduce contention
        grid.sync();
    }
}

//////////////////////////////////////////////////////////////////////////////
// Single-task kernel (non-persistent mode)
//////////////////////////////////////////////////////////////////////////////

template<typename T>
__global__ void singleTaskKernel(MegaKernelTask* task, OpDescriptor* opDescriptors,
                                  void** buffers, void* paramBuffer, volatile int32_t* barriers) {
    cg::grid_group grid = cg::this_grid();
    executeTask<T>(task, opDescriptors, buffers, paramBuffer, barriers, grid);
}

//////////////////////////////////////////////////////////////////////////////
// Host-side launcher functions
//////////////////////////////////////////////////////////////////////////////

void launchPersistentKernel(DeviceTaskQueue* queue, volatile bool* shouldStop,
                             int gridSize, int blockSize, cudaStream_t stream) {
    void* args[] = { &queue, &shouldStop };

    // Launch as cooperative kernel for grid-wide sync
    cudaLaunchCooperativeKernel(
        (void*)megaKernelInterpreter<float>,
        dim3(gridSize), dim3(blockSize),
        args, 0, stream);
}

void launchSingleTask(MegaKernelTask* task, OpDescriptor* opDescriptors,
                      void** buffers, void* paramBuffer, volatile int32_t* barriers,
                      cudaStream_t stream) {
    void* args[] = { &task, &opDescriptors, &buffers, &paramBuffer, &barriers };

    int gridSize = task->gridDimX * task->gridDimY * task->gridDimZ;
    int blockSize = task->blockDimX * task->blockDimY * task->blockDimZ;

    cudaLaunchCooperativeKernel(
        (void*)singleTaskKernel<float>,
        dim3(task->gridDimX, task->gridDimY, task->gridDimZ),
        dim3(task->blockDimX, task->blockDimY, task->blockDimZ),
        args, task->sharedMemBytes, stream);
}

}  // namespace execution
}  // namespace sd
