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

#include <ops/declarable/helpers/layer_norm.h>
#include <system/common.h>

namespace sd {
namespace ops {
namespace helpers {

// CPU stub — the fused layerNorm kernel is CUDA-only.
// Callers gate this behind dspIsCudaBuild(), so this should never be reached.
// If it is, throw to catch the logic error.
void layerNorm(
    NDArray* input,
    NDArray* gain,
    NDArray* bias,
    NDArray* output,
    const std::vector<LongType>& axis,
    float epsilon,
    LaunchContext* context) {
  THROW_EXCEPTION("layerNorm: fused kernel not available on CPU — caller should use decomposed path");
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
