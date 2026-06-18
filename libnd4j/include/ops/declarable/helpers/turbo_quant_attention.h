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

//
// @author Adam Gibson
//
// TurboQuant asymmetric attention kernel.
//
// Computes attention scores using compressed key representations:
//   score(q, k) ≈ <q, k_mse> + ||r|| * sqrt(π/2)/m * <S@q, signs>
//
// This avoids full key decompression during attention computation.
// Only values need standard MSE decompression (error averages out
// in the softmax-weighted sum).
//

#ifndef LIBND4J_TURBO_QUANT_ATTENTION_H
#define LIBND4J_TURBO_QUANT_ATTENTION_H

#include <array/NDArray.h>
#include <system/common.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Configuration for TurboQuant asymmetric attention.
 */
struct TurboQuantAttentionConfig {
    int numHeads;    // Number of attention heads
    int headDim;     // Dimension per head (also QJL projection dimension m)
    float scale;     // Attention scale (0 = auto: 1/sqrt(headDim))

    TurboQuantAttentionConfig()
        : numHeads(0), headDim(0), scale(0.0f) {}
};

/**
 * TurboQuant asymmetric attention forward pass.
 *
 * Computes attention using compressed key representations without full decompression:
 *
 * 1. term1 = Q @ K_mse^T                    (standard matmul with MSE-reconstructed keys)
 * 2. q_projected = Q @ S^T                   (project queries through QJL matrix)
 * 3. qjl_ip = q_projected @ signs^T          (inner product with sign bits)
 * 4. term2 = sqrt(π/2)/m * qjl_ip * norms   (QJL correction term)
 * 5. scores = (term1 + term2) * scale + mask
 * 6. weights = softmax(scores, dim=-1)
 * 7. output = weights @ V
 *
 * @param query           [B, H, Sq, D] query tensor
 * @param kMse            [B, H, Sk, D] pre-reconstructed MSE keys
 * @param qjlSigns        [B, H, Sk, D] QJL sign bits (INT8, +1/-1)
 * @param residualNorms   [B, H, Sk] residual L2 norms
 * @param qjlMatrix       [D, D] random Gaussian projection matrix
 * @param values          [B, H, Sk, D] dequantized values
 * @param attentionMask   [B, 1, 1, Sk] or compatible broadcastable shape
 * @param output          [B, H, Sq, D] attention output
 * @param config          attention configuration
 * @param context         launch context (stream, etc.)
 */
SD_LIB_HIDDEN void turboQuantAttentionForward(
    NDArray* query,
    NDArray* kMse,
    NDArray* qjlSigns,
    NDArray* residualNorms,
    NDArray* qjlMatrix,
    NDArray* values,
    NDArray* attentionMask,
    NDArray* output,
    const TurboQuantAttentionConfig& config,
    LaunchContext* context = nullptr);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_TURBO_QUANT_ATTENTION_H
