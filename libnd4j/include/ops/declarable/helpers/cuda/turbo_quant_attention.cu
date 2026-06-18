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
#include <math/templatemath.h>
#include <helpers/PointersManager.h>

#if NOT_EXCLUDED(OP_turbo_quant_attention)

namespace sd {
namespace ops {
namespace helpers {

// ──────────────────────────────────────────────────────────────────────────
// CUDA kernel: TurboQuant asymmetric attention
//
// Grid layout:  (batch * numHeads, seqQ, 1)
// Block layout: (blockDim.x threads) — cooperate on one (b, h, sq) triple
//
// Each block computes attention output for one query position across
// all key positions. Uses shared memory for the reduction.
// ──────────────────────────────────────────────────────────────────────────

template <typename T>
static SD_KERNEL __launch_bounds__(256, 2) void turboQuantAttentionKernel(
    const T* __restrict__ query,         // [B, H, Sq, D]
    const T* __restrict__ kMse,          // [B, H, Sk, D]
    const int8_t* __restrict__ qjlSigns, // [B, H, Sk, D]
    const T* __restrict__ residualNorms, // [B, H, Sk]
    const T* __restrict__ qjlMatrix,     // [D, D]
    const T* __restrict__ values,        // [B, H, Sk, D]
    const T* __restrict__ attnMask,      // [B, 1, 1, Sk] or null
    T* __restrict__ output,              // [B, H, Sq, D]
    int batch,
    int numHeads,
    int seqQ,
    int seqK,
    int headDim,
    float scale,
    float qjlCorrectionScale,
    int maskRank,
    int maskDim1,
    int maskDim2,
    int maskDim3) {

  // Map grid position to (b, h, sq)
  int bhIdx = blockIdx.x;
  int b = bhIdx / numHeads;
  int h = bhIdx % numHeads;
  int sq = blockIdx.y;
  int tid = threadIdx.x;

  if (b >= batch || h >= numHeads || sq >= seqQ) return;

  // Strides for [B, H, S, D] layout
  int strideB = numHeads * seqQ * headDim;      // for query
  int strideH = seqQ * headDim;
  int strideBk = numHeads * seqK * headDim;     // for key/value
  int strideHk = seqK * headDim;
  int strideNB = numHeads * seqK;               // for norms [B, H, Sk]
  int strideNH = seqK;

  // Query base pointer for this (b, h, sq)
  int qOffset = b * strideB + h * strideH + sq * headDim;

  // ---- Phase 1: Compute attention scores ----

  extern __shared__ char smem[];
  float* sharedScores = reinterpret_cast<float*>(smem);
  float* sharedMax = sharedScores + seqK;  // single element for max

  // Each thread handles a subset of key positions
  float localMax = -1e30f;

  for (int sk = tid; sk < seqK; sk += blockDim.x) {
    int kOffset = b * strideBk + h * strideHk + sk * headDim;

    // Term 1: dot(q, k_mse)
    float term1 = 0.0f;
    for (int d = 0; d < headDim; d++) {
      term1 += static_cast<float>(query[qOffset + d]) *
               static_cast<float>(kMse[kOffset + d]);
    }

    // Term 2: QJL correction
    // For efficiency, compute q_proj on the fly and dot with signs
    float qjlIp = 0.0f;
    for (int d = 0; d < headDim; d++) {
      // q_proj[d] = sum_j q[j] * S[d][j]
      float qProjD = 0.0f;
      for (int j = 0; j < headDim; j++) {
        qProjD += static_cast<float>(query[qOffset + j]) *
                  static_cast<float>(qjlMatrix[d * headDim + j]);
      }
      float signVal = static_cast<float>(qjlSigns[kOffset + d]);
      qjlIp += qProjD * signVal;
    }

    int normOffset = b * strideNB + h * strideNH + sk;
    float residNorm = static_cast<float>(residualNorms[normOffset]);
    float term2 = residNorm * qjlCorrectionScale * qjlIp;

    float score = (term1 + term2) * scale;

    // Apply attention mask
    if (attnMask != nullptr && maskRank == 4) {
      int mb = min(b, maskDim1 > 0 ? maskDim1 - 1 : 0);
      int msk = min(sk, maskDim3 > 0 ? maskDim3 - 1 : 0);
      // Mask is [B, 1, 1, Sk] — index as [mb, 0, 0, msk]
      float maskVal = static_cast<float>(attnMask[mb * maskDim3 + msk]);
      score += maskVal;
    }

    sharedScores[sk] = score;
    if (score > localMax) localMax = score;
  }
  __syncthreads();

  // Reduce to find global max across threads
  // Use simple sequential reduction in shared memory
  if (tid == 0) {
    float globalMax = -1e30f;
    for (int sk = 0; sk < seqK; sk++) {
      if (sharedScores[sk] > globalMax) globalMax = sharedScores[sk];
    }
    sharedMax[0] = globalMax;
  }
  __syncthreads();

  float maxVal = sharedMax[0];

  // ---- Phase 2: Softmax (exp and sum) ----
  float localSum = 0.0f;
  for (int sk = tid; sk < seqK; sk += blockDim.x) {
    float w = __expf(sharedScores[sk] - maxVal);
    sharedScores[sk] = w;
    localSum += w;
  }
  __syncthreads();

  // Reduce sum
  if (tid == 0) {
    float totalSum = 0.0f;
    for (int sk = 0; sk < seqK; sk++) {
      totalSum += sharedScores[sk];
    }
    if (totalSum > 0.0f) {
      for (int sk = 0; sk < seqK; sk++) {
        sharedScores[sk] /= totalSum;
      }
    }
  }
  __syncthreads();

  // ---- Phase 3: Weighted sum of values ----
  int outOffset = b * strideB + h * strideH + sq * headDim;

  for (int d = tid; d < headDim; d += blockDim.x) {
    float acc = 0.0f;
    for (int sk = 0; sk < seqK; sk++) {
      int vOffset = b * strideBk + h * strideHk + sk * headDim + d;
      acc += sharedScores[sk] * static_cast<float>(values[vOffset]);
    }
    output[outOffset + d] = static_cast<T>(acc);
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

  auto stream = context->getCudaStream();

  int batch = static_cast<int>(query->sizeAt(0));
  int numHeads = static_cast<int>(query->sizeAt(1));
  int seqQ = static_cast<int>(query->sizeAt(2));
  int headDim = static_cast<int>(query->sizeAt(3));
  int seqK = static_cast<int>(kMse->sizeAt(2));

  float scale = config.scale;
  if (scale <= 0.0f) {
    scale = 1.0f / sd::math::sd_sqrt<float, float>(static_cast<float>(headDim));
  }

  float qjlCorrectionScale = sd::math::sd_sqrt<float, float>(static_cast<float>(M_PI / 2.0)) / static_cast<float>(headDim);

  // Ensure GPU buffers are current
  query->syncToDevice();
  kMse->syncToDevice();
  qjlSigns->syncToDevice();
  residualNorms->syncToDevice();
  qjlMatrix->syncToDevice();
  values->syncToDevice();
  if (attentionMask != nullptr) attentionMask->syncToDevice();

  // Grid/block configuration
  dim3 grid(batch * numHeads, seqQ, 1);
  int blockSize = std::min(256, seqK);
  if (blockSize < 32) blockSize = 32;
  dim3 block(blockSize, 1, 1);

  // Shared memory: scores array + 1 float for max
  int smemSize = (seqK + 1) * sizeof(float);

  // Mask info
  int maskRank = 0;
  int maskDim1 = 0, maskDim2 = 0, maskDim3 = 0;
  const void* maskPtr = nullptr;
  if (attentionMask != nullptr && attentionMask->lengthOf() > 0) {
    maskRank = static_cast<int>(attentionMask->rankOf());
    if (maskRank == 4) {
      maskDim1 = static_cast<int>(attentionMask->sizeAt(0));
      maskDim2 = static_cast<int>(attentionMask->sizeAt(1));
      maskDim3 = static_cast<int>(attentionMask->sizeAt(3));
      maskPtr = attentionMask->specialBuffer();
    }
  }

  // Dispatch based on query data type
  auto qType = query->dataType();
  if (qType == DataType::FLOAT32) {
    turboQuantAttentionKernel<float><<<grid, block, smemSize, *stream>>>(
        reinterpret_cast<const float*>(query->specialBuffer()),
        reinterpret_cast<const float*>(kMse->specialBuffer()),
        reinterpret_cast<const int8_t*>(qjlSigns->specialBuffer()),
        reinterpret_cast<const float*>(residualNorms->specialBuffer()),
        reinterpret_cast<const float*>(qjlMatrix->specialBuffer()),
        reinterpret_cast<const float*>(values->specialBuffer()),
        reinterpret_cast<const float*>(maskPtr),
        reinterpret_cast<float*>(output->specialBuffer()),
        batch, numHeads, seqQ, seqK, headDim,
        scale, qjlCorrectionScale,
        maskRank, maskDim1, maskDim2, maskDim3);
  } else if (qType == DataType::HALF) {
    turboQuantAttentionKernel<half><<<grid, block, smemSize, *stream>>>(
        reinterpret_cast<const half*>(query->specialBuffer()),
        reinterpret_cast<const half*>(kMse->specialBuffer()),
        reinterpret_cast<const int8_t*>(qjlSigns->specialBuffer()),
        reinterpret_cast<const half*>(residualNorms->specialBuffer()),
        reinterpret_cast<const half*>(qjlMatrix->specialBuffer()),
        reinterpret_cast<const half*>(values->specialBuffer()),
        reinterpret_cast<const half*>(maskPtr),
        reinterpret_cast<half*>(output->specialBuffer()),
        batch, numHeads, seqQ, seqK, headDim,
        scale, qjlCorrectionScale,
        maskRank, maskDim1, maskDim2, maskDim3);
  } else if (qType == DataType::DOUBLE) {
    turboQuantAttentionKernel<double><<<grid, block, smemSize, *stream>>>(
        reinterpret_cast<const double*>(query->specialBuffer()),
        reinterpret_cast<const double*>(kMse->specialBuffer()),
        reinterpret_cast<const int8_t*>(qjlSigns->specialBuffer()),
        reinterpret_cast<const double*>(residualNorms->specialBuffer()),
        reinterpret_cast<const double*>(qjlMatrix->specialBuffer()),
        reinterpret_cast<const double*>(values->specialBuffer()),
        reinterpret_cast<const double*>(maskPtr),
        reinterpret_cast<double*>(output->specialBuffer()),
        batch, numHeads, seqQ, seqK, headDim,
        scale, qjlCorrectionScale,
        maskRank, maskDim1, maskDim2, maskDim3);
  }

  // Mark output as device-authoritative
  output->tickWriteDevice();
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_turbo_quant_attention)
