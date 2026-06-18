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

#ifndef LIBND4J_HELPERS_MAMBA2_SSM_H
#define LIBND4J_HELPERS_MAMBA2_SSM_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Mamba-2 SSD (State Space Duality) recurrence.
 *
 * Head-structured state with scalar per-head decay:
 *   A_bar_h = exp(A[h] * dt[b,t,h])
 *   state[b,h,n,p] = A_bar_h * state[b,h,n,p] + B[b,t,n] * dt[b,t,h] * x[b,t,h*P+p]
 *   y[b,t,h*P+p] = sum_n(C[b,t,n] * state[b,h,n,p]) + D[h] * x[b,t,h*P+p]
 *
 * @param context   launch context
 * @param x         input [B, L, D] where D = H * P
 * @param A         per-head decay in log-space [H]
 * @param B         input-dependent state expansion [B, L, N]
 * @param C         input-dependent state contraction [B, L, N]
 * @param dt        discretization timestep (post-softplus) [B, L, H]
 * @param D         skip connection [H] (may be nullptr)
 * @param stateIn   previous state [B, H, N, P] (may be nullptr)
 * @param y         output [B, L, D]
 * @param stateOut  final state [B, H, N, P]
 * @param numHeads  H
 * @param headDim   P
 * @param stateDim  N
 */
SD_LIB_HIDDEN void mamba2Ssm(LaunchContext* context,
                              NDArray* x, NDArray* A, NDArray* B, NDArray* C,
                              NDArray* dt, NDArray* D, NDArray* stateIn,
                              NDArray* y, NDArray* stateOut,
                              int numHeads, int headDim, int stateDim);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_HELPERS_MAMBA2_SSM_H
