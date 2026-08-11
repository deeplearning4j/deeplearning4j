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

#ifndef LIBND4J_HELPERS_CAUSAL_CONV1D_H
#define LIBND4J_HELPERS_CAUSAL_CONV1D_H

#include <ops/declarable/helpers/helpers.h>

#include <string>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Backend-neutral decoding of causal_conv1d's data-input contract.
 *
 * Backend tensor/node types deliberately do not appear here. Callers provide
 * only a rank accessor, then use the returned indices to construct native,
 * OpenVINO, NNAPI, or other backend-specific values.
 */
struct CausalConv1dInputRoles {
  int bias = -1;
  int stateIn = -1;
  int actualLen = -1;
};

template <typename RankAt>
inline bool resolveCausalConv1dInputRoles(int numInputs, RankAt rankAt,
                                         CausalConv1dInputRoles& roles,
                                         std::string* reason = nullptr) {
  auto fail = [&](const char* message) {
    if (reason != nullptr) *reason = message;
    return false;
  };

  roles = {};
  if (numInputs < 2 || numInputs > 5)
    return fail("expected x, weight, and at most bias, stateIn, actualLen");
  if (rankAt(0) != 3) return fail("x must have rank 3 [B,L,D]");
  if (rankAt(1) != 2) return fail("weight must have rank 2 [D,K] or [K,D]");

  for (int inputIndex = 2; inputIndex < numInputs; ++inputIndex) {
    const int rank = rankAt(inputIndex);
    int* role = nullptr;
    if (rank == 0) {
      role = &roles.actualLen;
    } else if (rank == 1) {
      role = &roles.bias;
    } else if (rank >= 2) {
      role = &roles.stateIn;
    } else {
      return fail("optional input rank is unresolved");
    }

    if (*role >= 0) return fail("multiple optional inputs resolve to the same semantic role");
    *role = inputIndex;
  }
  return true;
}

SD_LIB_HIDDEN void causalConv1d(LaunchContext* context,
                                 NDArray* x, NDArray* weight, NDArray* bias, NDArray* stateIn,
                                 NDArray* actualLen, NDArray* output, NDArray* stateOut,
                                 int activation, int wFormat = 0);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif
