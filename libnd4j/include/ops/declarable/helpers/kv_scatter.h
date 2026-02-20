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

#ifndef LIBND4J_HELPERS_KV_SCATTER_H
#define LIBND4J_HELPERS_KV_SCATTER_H

#include <system/op_boilerplate.h>
#include <array/NDArray.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Copy present[batch, heads, lastPos, dim] -> output[batch, heads, cachePos, dim]
 *
 * @param present   [batch, heads, seqLen, dim] source tensor
 * @param output    [batch, heads, maxKvLen, dim] destination tensor (modified in-place)
 * @param cachePos  position along dim 2 in output to write to
 * @param context   launch context
 */
SD_LIB_HIDDEN void kvScatter(NDArray* present, NDArray* output,
                              LongType cachePos, LaunchContext* context);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif
