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

#ifndef LIBND4J_CHECKPOINT_OFFLOAD_H
#define LIBND4J_CHECKPOINT_OFFLOAD_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * checkpoint_offload_d2h - Copy a device (GPU) tensor to host (CPU) memory.
 *
 * On CUDA backends the copy is issued asynchronously on the execution stream
 * so it overlaps with subsequent device compute. On CPU backends the output
 * is simply a contiguous duplicate of the input.
 *
 * @param context      launch context
 * @param src          source tensor (device or host)
 * @param dst          pre-allocated output tensor on host
 * @param usePinned    1 = prefer pinned (page-locked) host memory, 0 = pageable
 * @param streamId     CUDA stream index (0 = default execution stream)
 */
SD_LIB_HIDDEN void checkpointOffloadD2H(LaunchContext* context,
                                         NDArray* src,
                                         NDArray* dst,
                                         int usePinned,
                                         int streamId);

/**
 * checkpoint_prefetch_h2d - Copy a host (CPU) tensor back to device (GPU) memory.
 *
 * On CUDA backends the copy is issued asynchronously on the execution stream.
 * On CPU backends the output is simply a contiguous duplicate of the input.
 *
 * @param context        launch context
 * @param src            source tensor on host
 * @param dst            pre-allocated output tensor on device
 * @param targetDeviceId target device ID (0 = current device)
 */
SD_LIB_HIDDEN void checkpointPrefetchH2D(LaunchContext* context,
                                          NDArray* src,
                                          NDArray* dst,
                                          int targetDeviceId);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_CHECKPOINT_OFFLOAD_H
