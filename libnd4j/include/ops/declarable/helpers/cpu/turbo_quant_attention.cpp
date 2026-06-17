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

#include <ops/declarable/helpers/turbo_quant_attention.h>
#include <array/NDArrayFactory.h>
#include <math/templatemath.h>
#include <system/op_boilerplate.h>
#include <cmath>

#if NOT_EXCLUDED(OP_turbo_quant_attention)

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void turboQuantAttentionForward_(
    NDArray* query,
    NDArray* kMse,
    NDArray* qjlSigns,
    NDArray* residualNorms,
    NDArray* qjlMatrix,
    NDArray* values,
    NDArray* attentionMask,
    NDArray* output,
    const TurboQuantAttentionConfig& config,
    LaunchContext* context) {

  // Extract dimensions
  auto batch = query->sizeAt(0);
  auto numHeads = query->sizeAt(1);
  auto seqQ = query->sizeAt(2);
  auto headDim = query->sizeAt(3);
  auto seqK = kMse->sizeAt(2);

  T scale = static_cast<T>(config.scale);
  if (config.scale <= 0.0f) {
    scale = static_cast<T>(1.0f / std::sqrt(static_cast<float>(headDim)));
  }

  // QJL correction factor: sqrt(π/2) / m
  T qjlCorrectionScale = static_cast<T>(std::sqrt(M_PI / 2.0) / static_cast<float>(headDim));

  // Process each batch and head independently
  for (sd::LongType b = 0; b < batch; b++) {
    for (sd::LongType h = 0; h < numHeads; h++) {
      for (sd::LongType sq = 0; sq < seqQ; sq++) {

        // ---- Step 1: Compute asymmetric attention scores for all key positions ----
        // For each key position sk:
        //   term1 = dot(q, k_mse)
        //   q_proj = q @ S^T   (project query through QJL matrix)
        //   qjl_ip = dot(q_proj, signs_sk)
        //   term2 = ||r_sk|| * sqrt(π/2)/m * qjl_ip
        //   score = (term1 + term2) * scale + mask

        // Get query vector for this position
        // q: [headDim]

        // First pass: compute scores and find max for numerical stability
        T maxScore = -std::numeric_limits<T>::infinity();
        std::vector<T> scores(seqK);

        for (sd::LongType sk = 0; sk < seqK; sk++) {
          // Term 1: dot(q, k_mse)
          T term1 = static_cast<T>(0);
          for (sd::LongType d = 0; d < headDim; d++) {
            T qVal = query->e<T>(b, h, sq, d);
            T kVal = kMse->e<T>(b, h, sk, d);
            term1 += qVal * kVal;
          }

          // Term 2: QJL correction
          // Project query: q_proj = q @ S^T
          // Then dot with signs
          T qjlIp = static_cast<T>(0);
          for (sd::LongType d = 0; d < headDim; d++) {
            // q_proj[d] = sum_j q[j] * S[d][j]  (S is [D, D])
            T qProjD = static_cast<T>(0);
            for (sd::LongType j = 0; j < headDim; j++) {
              qProjD += query->e<T>(b, h, sq, j) * qjlMatrix->e<T>(d, j);
            }
            T signVal = static_cast<T>(qjlSigns->e<int8_t>(b, h, sk, d));
            qjlIp += qProjD * signVal;
          }

          T residualNorm = residualNorms->e<T>(b, h, sk);
          T term2 = residualNorm * qjlCorrectionScale * qjlIp;

          T score = (term1 + term2) * scale;

          // Apply attention mask
          if (attentionMask != nullptr && attentionMask->lengthOf() > 0) {
            // Broadcast mask: [B, 1, 1, Sk] or similar
            T maskVal = static_cast<T>(0);
            if (attentionMask->rankOf() == 4) {
              auto mb = std::min(b, attentionMask->sizeAt(0) - 1);
              auto mh = std::min((sd::LongType)0, attentionMask->sizeAt(1) - 1);
              auto msq = std::min((sd::LongType)0, attentionMask->sizeAt(2) - 1);
              auto msk = std::min(sk, attentionMask->sizeAt(3) - 1);
              maskVal = attentionMask->e<T>(mb, mh, msq, msk);
            }
            score += maskVal;
          }

          scores[sk] = score;
          if (score > maxScore) maxScore = score;
        }

        // ---- Step 2: Softmax ----
        T sumExp = static_cast<T>(0);
        std::vector<T> weights(seqK);
        for (sd::LongType sk = 0; sk < seqK; sk++) {
          weights[sk] = static_cast<T>(sd::math::sd_exp<float>(static_cast<float>(scores[sk] - maxScore)));
          sumExp += weights[sk];
        }
        if (sumExp > static_cast<T>(0)) {
          for (sd::LongType sk = 0; sk < seqK; sk++) {
            weights[sk] /= sumExp;
          }
        }

        // ---- Step 3: Weighted sum of values ----
        for (sd::LongType d = 0; d < headDim; d++) {
          T acc = static_cast<T>(0);
          for (sd::LongType sk = 0; sk < seqK; sk++) {
            acc += weights[sk] * values->e<T>(b, h, sk, d);
          }
          output->p<T>(b, h, sq, d, acc);
        }
      }
    }
  }
}

void turboQuantAttentionForward(
    NDArray* query,
    NDArray* kMse,
    NDArray* qjlSigns,
    NDArray* residualNorms,
    NDArray* qjlMatrix,
    NDArray* values,
    NDArray* attentionMask,
    NDArray* output,
    const TurboQuantAttentionConfig& config,
    LaunchContext* context) {

  BUILD_SINGLE_SELECTOR(query->dataType(), turboQuantAttentionForward_,
                        (query, kMse, qjlSigns, residualNorms, qjlMatrix, values,
                         attentionMask, output, config, context),
                        SD_FLOAT_TYPES);
}

BUILD_SINGLE_TEMPLATE(template void turboQuantAttentionForward_,
                      (NDArray*, NDArray*, NDArray*, NDArray*, NDArray*, NDArray*,
                       NDArray*, NDArray*, const TurboQuantAttentionConfig&, LaunchContext*),
                      SD_FLOAT_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_turbo_quant_attention)
