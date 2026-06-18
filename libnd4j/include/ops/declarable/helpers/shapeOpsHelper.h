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
// CUDA-graph-compatible helpers for shape metadata ops (shape_of, size, rank, size_at).
//
// These ops read shape metadata and write it to output tensors. On CPU they write to
// host buffers directly. On CUDA they use device-to-device operations (cudaMemcpyAsync
// D2D and simple kernels) so that the operations are capturable by CUDA graphs.
//
// Without this, shape_of etc. write to HOST memory then syncToDevice — the host write
// is invisible to CUDA graph capture/replay, causing downstream ops to read stale zeros.
//

#ifndef LIBND4J_SHAPE_OPS_HELPER_H
#define LIBND4J_SHAPE_OPS_HELPER_H

#include <array/NDArray.h>
#include <execution/LaunchContext.h>

namespace sd {
namespace ops {
namespace helpers {

SD_LIB_HIDDEN void shapeOfHelper(LaunchContext* context, NDArray* x, NDArray* z, int rank);
SD_LIB_HIDDEN void sizeHelper(LaunchContext* context, NDArray* input, NDArray* output);
SD_LIB_HIDDEN void rankHelper(LaunchContext* context, NDArray* input, NDArray* output);
SD_LIB_HIDDEN void sizeAtHelper(LaunchContext* context, NDArray* input, NDArray* output, int dim);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_SHAPE_OPS_HELPER_H
