/* ******************************************************************************
 *
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
// CUDA implementation of checkpoint offload helpers.
//
// D2H (checkpointOffloadD2H):
//   Issues cudaMemcpyAsync device→host on the launch-context stream so the
//   copy overlaps with subsequent device compute in the forward pass.
//
// H2D (checkpointPrefetchH2D):
//   Issues cudaMemcpyAsync host→device on the launch-context stream so the
//   prefetch overlaps with backward compute in earlier layers.
//
// Both helpers use the launch context's CUDA stream to preserve ordering with
// surrounding ops — no explicit stream synchronisation is needed here.
//

#include <cuda_runtime.h>
#include <helpers/DebugHelper.h>
#include <array/NDArray.h>
#include <system/common.h>
#include <ops/declarable/helpers/checkpoint_offload.h>

namespace sd {
namespace ops {
namespace helpers {

// -----------------------------------------------------------------------
// D2H: device → host
// -----------------------------------------------------------------------

void checkpointOffloadD2H(LaunchContext* context,
                           NDArray* src,
                           NDArray* dst,
                           int /*usePinned*/,
                           int /*streamId*/) {
    // src is device-resident; dst is the host-side output allocated by the op.
    //
    // src->specialBuffer() = device pointer
    // dst->buffer()        = host pointer (allocated when dst was created)
    //
    // We do NOT call src->syncToHost() because that would block and defeat
    // the purpose of async offload.  Instead we issue an async memcpy on the
    // execution stream and mark the host buffer valid afterwards.

    const LongType nbytes = static_cast<LongType>(src->lengthOf()) *
                            static_cast<LongType>(src->sizeOfT());

    // Make sure the device buffer reflects the most recent writes
    // (this is a host-side flag update only, no CUDA call)
    if (!src->isActualOnDeviceSide()) {
        // src is authoritative on host — sync it to device first so we have
        // a valid specialBuffer to copy from.
        src->syncToDevice();
    }

    cudaStream_t stream = *(context->getCudaStream());

    auto cudaErr = cudaMemcpyAsync(
            dst->buffer(),            // host destination
            src->specialBuffer(),     // device source
            static_cast<size_t>(nbytes),
            cudaMemcpyDeviceToHost,
            stream);

    if (cudaErr != cudaSuccess) {
        THROW_EXCEPTION(("checkpointOffloadD2H: cudaMemcpyAsync D2H failed: "
                         + std::string(cudaGetErrorString(cudaErr))).c_str());
    }

    // Mark the host buffer of dst as the authoritative copy.
    // The copy is in-flight on the stream; the caller must ensure the stream
    // completes before reading dst->buffer().
    dst->tickWriteHost();
    dst->tickReadHost();
}

// -----------------------------------------------------------------------
// H2D: host → device
// -----------------------------------------------------------------------

void checkpointPrefetchH2D(LaunchContext* context,
                            NDArray* src,
                            NDArray* dst,
                            int /*targetDeviceId*/) {
    // src is host-resident (the offloaded checkpoint).
    // dst is the device-side output allocated by the op.
    //
    // src->buffer()         = host pointer
    // dst->specialBuffer()  = device pointer

    const LongType nbytes = static_cast<LongType>(src->lengthOf()) *
                            static_cast<LongType>(src->sizeOfT());

    // Ensure host buffer is readable (it was written by the D2H phase)
    if (!src->isActualOnHostSide()) {
        src->syncToHost();
    }

    cudaStream_t stream = *(context->getCudaStream());

    auto cudaErr = cudaMemcpyAsync(
            dst->specialBuffer(),     // device destination
            src->buffer(),            // host source
            static_cast<size_t>(nbytes),
            cudaMemcpyHostToDevice,
            stream);

    if (cudaErr != cudaSuccess) {
        THROW_EXCEPTION(("checkpointPrefetchH2D: cudaMemcpyAsync H2D failed: "
                         + std::string(cudaGetErrorString(cudaErr))).c_str());
    }

    // Mark the device buffer of dst as the authoritative copy.
    dst->tickWriteDevice();
    dst->tickReadDevice();
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
