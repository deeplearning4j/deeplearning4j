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

#include <ops/declarable/helpers/lightning_attention.h>
#include <cmath>
#include <vector>

namespace sd {
namespace ops {
namespace helpers {

void lightningAttentionCpu(LaunchContext* context,
                           NDArray* query, NDArray* key, NDArray* value,
                           NDArray* output, double scale) {
    // query/key/value: [batch, heads, seqLen, headDim]
    auto batch = query->sizeAt(0);
    auto heads = query->sizeAt(1);
    auto seqLen = query->sizeAt(2);
    auto headDim = query->sizeAt(3);

    float scaleF = static_cast<float>(scale);

    // For each batch and head, compute linear attention via cumulative state
    for (LongType b = 0; b < batch; b++) {
        for (LongType h = 0; h < heads; h++) {
            // Running state S: [headDim, headDim] outer-product accumulator
            std::vector<float> state(headDim * headDim, 0.0f);

            for (LongType s = 0; s < seqLen; s++) {
                // Read K[b,h,s,:] and V[b,h,s,:]
                std::vector<float> kVec(headDim);
                std::vector<float> vVec(headDim);
                for (LongType d = 0; d < headDim; d++) {
                    kVec[d] = key->e<float>(b, h, s, d);
                    vVec[d] = value->e<float>(b, h, s, d);
                }

                // Accumulate outer product: S += K^T * V  (i.e., S[i,j] += K[i] * V[j])
                for (LongType i = 0; i < headDim; i++) {
                    for (LongType j = 0; j < headDim; j++) {
                        state[i * headDim + j] += kVec[i] * vVec[j];
                    }
                }

                // Compute output: O[b,h,s,:] = scale * Q[b,h,s,:] @ S
                for (LongType d = 0; d < headDim; d++) {
                    float sum = 0.0f;
                    for (LongType k2 = 0; k2 < headDim; k2++) {
                        float q = query->e<float>(b, h, s, k2);
                        sum += q * state[k2 * headDim + d];
                    }
                    output->p(b, h, s, d, sum * scaleF);
                }
            }
        }
    }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
