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

#ifndef LIBND4J_HELPERS_GATED_DELTA_RULE_H
#define LIBND4J_HELPERS_GATED_DELTA_RULE_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

SD_LIB_HIDDEN void gatedDeltaRule(LaunchContext* context,
                                   NDArray* Q, NDArray* K, NDArray* V,
                                   NDArray* beta, NDArray* gate, NDArray* stateIn,
                                   NDArray* actualLen, NDArray* output, NDArray* stateOut);

/**
 * Same recurrence as gatedDeltaRule, additionally capturing the state AFTER
 * each consumed input row t into prefixOut. prefixOut layout is time-leading
 * C-order [W, B, H, D_k, D_v]; slot t must remain exactly the working state
 * snapshot at that boundary (no rounded re-feeding). W = prefixOut->sizeAt(0)
 * and only t < min(actualLen, L) slots are written. May be null.
 */
SD_LIB_HIDDEN void gatedDeltaRuleWithPrefix(LaunchContext* context,
                                            NDArray* Q, NDArray* K, NDArray* V,
                                            NDArray* beta, NDArray* gate, NDArray* stateIn,
                                            NDArray* actualLen, NDArray* output, NDArray* stateOut,
                                            NDArray* prefixOut);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif
