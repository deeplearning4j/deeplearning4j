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

#include <array/InteropDataBuffer.h>
#include <array/DataBuffer.h>
#include <array/DataTypeUtils.h>
#include <helpers/logger.h>
#include <legacy/NativeOps.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

/**
 * DEPRECATED: Device synchronization is now handled directly in dbClose
 * (see NativeOpsHelpers_DataBuffers_close.cu).
 *
 * This function is kept as a no-op for backwards compatibility in case
 * anything still references it.
 */
void syncBeforeDeallocation() {
    // No-op - synchronization now handled in platform-specific dbClose
}

/**
 * Batched asynchronous synchronization of multiple data buffers from host to device.
 * Uses multiple CUDA streams to transfer data in parallel, providing better performance
 * than individual synchronous transfers for model loading.
 *
 * This implementation:
 * 1. Creates the specified number of CUDA streams
 * 2. Issues cudaMemcpyAsync calls round-robin across streams
 * 3. Synchronizes all streams at the end
 *
 * @param buffers Array of OpaqueDataBuffer pointers to sync
 * @param bufferCount Number of buffers in the array
 * @param streamCount Number of CUDA streams to use for parallel transfers (typically 2-8)
 */
void batchSyncToSpecialAsync(OpaqueDataBuffer **buffers, int bufferCount, int streamCount) {
    if (buffers == nullptr || bufferCount <= 0) {
        return;
    }

    // Clamp stream count to reasonable range
    streamCount = std::max(1, std::min(streamCount, 16));

    // Early exit if nothing to transfer
    if (bufferCount == 0) {
        return;
    }

    // Create CUDA streams
    std::vector<cudaStream_t> streams(streamCount);
    for (int i = 0; i < streamCount; i++) {
        auto status = cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
        if (status != cudaSuccess) {
            // If stream creation fails, fall back to sequential sync
            for (int j = 0; j < i; j++) {
                cudaStreamDestroy(streams[j]);
            }
            // Sequential fallback
            for (int b = 0; b < bufferCount; b++) {
                if (buffers[b] != nullptr && buffers[b]->dataBuffer() != nullptr) {
                    buffers[b]->dataBuffer()->syncToSpecial();
                }
            }
            return;
        }
    }

    // Issue async transfers round-robin across streams
    for (int i = 0; i < bufferCount; i++) {
        OpaqueDataBuffer *buffer = buffers[i];
        if (buffer == nullptr || buffer->dataBuffer() == nullptr) {
            continue;
        }

        sd::DataBuffer *db = buffer->dataBuffer();
        if (db->getNumElements() == 0) {
            continue;
        }

        // Get the appropriate stream
        cudaStream_t stream = streams[i % streamCount];

        // Get pointers
        void *primary = db->primary();
        void *special = db->special();

        if (primary == nullptr || special == nullptr) {
            // Need to allocate special buffer first
            db->allocateSpecial();
            special = db->special();
        }

        if (primary != nullptr && special != nullptr) {
            size_t bytesToCopy = db->getNumElements() * sd::DataTypeUtils::sizeOf(db->getDataType());

            // Issue async memcpy
            auto status = cudaMemcpyAsync(special, primary, bytesToCopy, cudaMemcpyHostToDevice, stream);
            if (status != cudaSuccess) {
                // Log error but continue with other buffers
                sd_printf("batchSyncToSpecialAsync: cudaMemcpyAsync failed for buffer %d: %s\n",
                         i, cudaGetErrorString(status));
            }

            // Mark the buffer as having been synced
            db->writeSpecial();
        }
    }

    // Synchronize all streams
    for (int i = 0; i < streamCount; i++) {
        auto status = cudaStreamSynchronize(streams[i]);
        if (status != cudaSuccess) {
            sd_printf("batchSyncToSpecialAsync: cudaStreamSynchronize failed for stream %d: %s\n",
                     i, cudaGetErrorString(status));
        }
        cudaStreamDestroy(streams[i]);
    }
}
