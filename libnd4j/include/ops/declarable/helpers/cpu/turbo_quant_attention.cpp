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
#include <cmath>

#if NOT_EXCLUDED(OP_turbo_quant_attention)

namespace sd {
namespace ops {
namespace helpers {

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

  // Extract dimensions
  auto batch = query->sizeAt(0);
  auto numHeads = query->sizeAt(1);
  auto seqQ = query->sizeAt(2);
  auto headDim = query->sizeAt(3);
  auto seqK = kMse->sizeAt(2);

  float scale = config.scale;
  if (scale <= 0.0f) {
    scale = 1.0f / std::sqrt(static_cast<float>(headDim));
  }

  // QJL correction factor: sqrt(π/2) / m
  float qjlCorrectionScale = std::sqrt(M_PI / 2.0) / static_cast<float>(headDim);

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
        float maxScore = -std::numeric_limits<float>::infinity();
        std::vector<float> scores(seqK);

        for (sd::LongType sk = 0; sk < seqK; sk++) {
          // Term 1: dot(q, k_mse)
          float term1 = 0.0f;
          for (sd::LongType d = 0; d < headDim; d++) {
            float qVal = query->e<float>(b, h, sq, d);
            float kVal = kMse->e<float>(b, h, sk, d);
            term1 += qVal * kVal;
          }

          // Term 2: QJL correction
          // Project query: q_proj = q @ S^T
          // Then dot with signs
          float qjlIp = 0.0f;
          for (sd::LongType d = 0; d < headDim; d++) {
            // q_proj[d] = sum_j q[j] * S[d][j]  (S is [D, D])
            float qProjD = 0.0f;
            for (sd::LongType j = 0; j < headDim; j++) {
              qProjD += query->e<float>(b, h, sq, j) * qjlMatrix->e<float>(d, j);
            }
            float signVal = static_cast<float>(qjlSigns->e<int8_t>(b, h, sk, d));
            qjlIp += qProjD * signVal;
          }

          float residualNorm = residualNorms->e<float>(b, h, sk);
          float term2 = residualNorm * qjlCorrectionScale * qjlIp;

          float score = (term1 + term2) * scale;

          // Apply attention mask
          if (attentionMask != nullptr && attentionMask->lengthOf() > 0) {
            // Broadcast mask: [B, 1, 1, Sk] or similar
            float maskVal = 0.0f;
            if (attentionMask->rankOf() == 4) {
              auto mb = std::min(b, attentionMask->sizeAt(0) - 1);
              auto mh = std::min((sd::LongType)0, attentionMask->sizeAt(1) - 1);
              auto msq = std::min((sd::LongType)0, attentionMask->sizeAt(2) - 1);
              auto msk = std::min(sk, attentionMask->sizeAt(3) - 1);
              maskVal = attentionMask->e<float>(mb, mh, msq, msk);
            }
            score += maskVal;
          }

          scores[sk] = score;
          if (score > maxScore) maxScore = score;
        }

        // ---- Step 2: Softmax ----
        float sumExp = 0.0f;
        std::vector<float> weights(seqK);
        for (sd::LongType sk = 0; sk < seqK; sk++) {
          weights[sk] = std::exp(scores[sk] - maxScore);
          sumExp += weights[sk];
        }
        if (sumExp > 0.0f) {
          for (sd::LongType sk = 0; sk < seqK; sk++) {
            weights[sk] /= sumExp;
          }
        }

        // ---- Step 3: Weighted sum of values ----
        for (sd::LongType d = 0; d < headDim; d++) {
          float acc = 0.0f;
          for (sd::LongType sk = 0; sk < seqK; sk++) {
            acc += weights[sk] * values->e<float>(b, h, sk, d);
          }
          output->p(b, h, sq, d, acc);
        }
      }
    }
  }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_turbo_quant_attention)
