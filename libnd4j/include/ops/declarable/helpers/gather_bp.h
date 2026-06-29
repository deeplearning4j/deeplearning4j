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
// Backward pass helper for the gather op.
// Inputs: indices (ALL_INTS), gradOut (ALL_FLOATS).
// Output: dInput — shape/dtype from IArgs / output allocation; scatter-add gradOut along axis.
// The forward input array is NOT passed — its shape is encoded in the output allocation
// (dInput already has the correct shape when the helper is called).
//

#ifndef LIBND4J_GATHER_BP_H
#define LIBND4J_GATHER_BP_H

#include <array/NDArray.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Gradient of gather w.r.t. the input tensor.
 *
 * @param context  launch context
 * @param indices  gather indices (integer type)
 * @param gradOut  upstream gradient (same shape as gather's forward output)
 * @param dInput   output gradient — same shape/dtype as the original forward input;
 *                 the implementation zeroes it internally before accumulating
 * @param axis     gather axis (already normalised to [0, rank))
 */
SD_LIB_HIDDEN void gatherBp(LaunchContext* context, NDArray* indices,
                             NDArray* gradOut, NDArray* dInput, LongType axis);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_GATHER_BP_H
