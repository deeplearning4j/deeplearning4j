/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author Eclipse Deeplearning4j
//

#ifndef LIBND4J_RWKV_WKV_H
#define LIBND4J_RWKV_WKV_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * RWKV-6 WKV recurrence. All sequence tensors are [B, T, H, S]; tf (time-first)
 * is [H, S]; state is [B, H, S, S]; output is [B, T, H, S]. Per (b,h), with an
 * S×S recurrent state, for each token t:
 *   out[t,j] = sum_i r[i] * (tf[i] * k[i]*v[j] + state[i,j])
 *   state[i,j] = td[i] * state[i,j] + k[i]*v[j]
 */
SD_LIB_HIDDEN void rwkvWkv6(LaunchContext* context, NDArray* k, NDArray* v, NDArray* r,
                            NDArray* tf, NDArray* td, NDArray* state, NDArray* output);

/**
 * RWKV-7 WKV recurrence (generalized delta rule). Sequence tensors r,w,k,v,a,b
 * are [B, T, H, S]; state is [B, H, S, S]; output [B, T, H, S]. Per (b,h,t):
 *   sa[j]     = sum_i a[i] * state[i,j]
 *   state[i,j] = w[i]*state[i,j] + k[i]*v[j] + b[i]*sa[j]
 *   out[t,j]  = sum_i r[i] * state[i,j]
 */
SD_LIB_HIDDEN void rwkvWkv7(LaunchContext* context, NDArray* r, NDArray* w, NDArray* k,
                            NDArray* v, NDArray* a, NDArray* b, NDArray* state, NDArray* output);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_RWKV_WKV_H
