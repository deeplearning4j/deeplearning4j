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
// CPU fallback implementations of fused LLM operations.
// These provide reference implementations when CUDA is not available.
//

#include <ops/declarable/helpers/fused_llm_ops.h>
#include <array/NDArray.h>
#include <helpers/Loops.h>
#include <cmath>
#include <random>

namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////////
// Fused GELU - x * sigmoid(1.702 * x)
//////////////////////////////////////////////////////////////////////////////

void fusedGELU(NDArray* input, NDArray* output, LaunchContext* context) {
  auto func = PRAGMA_THREADS_FOR {
    for (auto i = start; i < stop; i++) {
      float x = input->e<float>(i);
      // Fast GELU approximation: x * sigmoid(1.702 * x)
      float sig = 1.0f / (1.0f + std::exp(-1.702f * x));
      output->p(i, x * sig);
    }
  };

  samediff::Threads::parallel_for(func, 0, input->lengthOf());
}

void fusedGELUBackward(NDArray* input, NDArray* gradOut, NDArray* gradIn, LaunchContext* context) {
  auto func = PRAGMA_THREADS_FOR {
    for (auto i = start; i < stop; i++) {
      float x = input->e<float>(i);
      float dout = gradOut->e<float>(i);

      // d/dx[x * sigmoid(1.702*x)] = sigmoid(1.702*x) + x * 1.702 * sigmoid(1.702*x) * (1 - sigmoid(1.702*x))
      float sig = 1.0f / (1.0f + std::exp(-1.702f * x));
      float grad = sig + x * 1.702f * sig * (1.0f - sig);
      gradIn->p(i, dout * grad);
    }
  };

  samediff::Threads::parallel_for(func, 0, input->lengthOf());
}

//////////////////////////////////////////////////////////////////////////////
// Fused Layer Norm with Welford's algorithm
//////////////////////////////////////////////////////////////////////////////

void fusedLayerNorm(NDArray* input, NDArray* gain, NDArray* bias, NDArray* output,
                    float epsilon, LaunchContext* context) {
  const int rank = input->rankOf();
  const LongType numRows = input->lengthOf() / input->sizeAt(-1);
  const LongType rowLen = input->sizeAt(-1);

  auto func = PRAGMA_THREADS_FOR {
    for (auto row = start; row < stop; row++) {
      // Welford's online algorithm for mean and variance
      float mean = 0.0f;
      float M2 = 0.0f;
      float count = 0.0f;

      for (LongType i = 0; i < rowLen; i++) {
        float val = input->e<float>(row * rowLen + i);
        count += 1.0f;
        float delta = val - mean;
        mean += delta / count;
        float delta2 = val - mean;
        M2 += delta * delta2;
      }

      float variance = M2 / count;
      float invStd = 1.0f / std::sqrt(variance + epsilon);

      // Normalize, scale and shift
      for (LongType i = 0; i < rowLen; i++) {
        float val = input->e<float>(row * rowLen + i);
        float normalized = (val - mean) * invStd;
        float g = gain->e<float>(i);
        float result = normalized * g;
        if (bias != nullptr) {
          result += bias->e<float>(i);
        }
        output->p(row * rowLen + i, result);
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, numRows);
}

//////////////////////////////////////////////////////////////////////////////
// Fused RoPE
//////////////////////////////////////////////////////////////////////////////

void fusedRoPE(NDArray* input, NDArray* output, int positionOffset,
               float freqBase, float freqScale, int ropeType, LaunchContext* context,
               int rotaryDims) {
  auto rank = input->rankOf();
  auto batch = input->sizeAt(0);
  auto seqLen = input->sizeAt(1);
  auto numHeads = (rank >= 4) ? input->sizeAt(2) : static_cast<LongType>(1);
  auto headDim = (rank >= 4) ? input->sizeAt(3) : input->sizeAt(2);

  // Effective rotation dimensions: 0 means rotate all
  LongType rotateDims = (rotaryDims > 0 && rotaryDims < headDim) ? rotaryDims : headDim;
  LongType halfRotate = rotateDims / 2;

  // Copy input to output first (preserves unrotated dims when rotateDims < headDim)
  output->assign(input);

  auto func = PRAGMA_THREADS_FOR {
    for (auto b = start; b < stop; b++) {
      for (LongType s = 0; s < seqLen; s++) {
        LongType pos = positionOffset + s;
        for (LongType h = 0; h < numHeads; h++) {
          for (LongType i = 0; i < halfRotate; i++) {
            float theta = static_cast<float>(pos) * freqScale /
                          std::pow(freqBase, (2.0f * i) / rotateDims);
            float cosTheta = std::cos(theta);
            float sinTheta = std::sin(theta);

            LongType base = ((b * seqLen + s) * numHeads + h) * headDim;
            LongType idx1, idx2;
            if (ropeType == 0) {  // Standard (LLaMA)
              idx1 = base + i;
              idx2 = base + i + halfRotate;
            } else if (ropeType == 1) {  // NeoX
              idx1 = base + i * 2;
              idx2 = base + i * 2 + 1;
            } else {  // GPT-J
              idx1 = base + i;
              idx2 = base + i + halfRotate;
            }

            float x1 = input->e<float>(idx1);
            float x2 = input->e<float>(idx2);

            output->p(idx1, x1 * cosTheta - x2 * sinTheta);
            output->p(idx2, x1 * sinTheta + x2 * cosTheta);
          }
        }
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, batch);
}

void fusedRoPEBackward(NDArray* gradOut, NDArray* gradIn, int positionOffset,
                       float freqBase, float freqScale, int ropeType, LaunchContext* context,
                       int rotaryDims) {
  auto rank = gradOut->rankOf();
  auto batch = gradOut->sizeAt(0);
  auto seqLen = gradOut->sizeAt(1);
  auto numHeads = (rank >= 4) ? gradOut->sizeAt(2) : static_cast<LongType>(1);
  auto headDim = (rank >= 4) ? gradOut->sizeAt(3) : gradOut->sizeAt(2);

  LongType rotateDims = (rotaryDims > 0 && rotaryDims < headDim) ? rotaryDims : headDim;
  LongType halfRotate = rotateDims / 2;

  // Copy gradOut to gradIn first (preserves unrotated dims)
  gradIn->assign(gradOut);

  auto func = PRAGMA_THREADS_FOR {
    for (auto b = start; b < stop; b++) {
      for (LongType s = 0; s < seqLen; s++) {
        LongType pos = positionOffset + s;
        for (LongType h = 0; h < numHeads; h++) {
          for (LongType i = 0; i < halfRotate; i++) {
            float theta = static_cast<float>(pos) * freqScale /
                          std::pow(freqBase, (2.0f * i) / rotateDims);
            float cosTheta = std::cos(theta);
            float sinTheta = std::sin(theta);

            LongType base = ((b * seqLen + s) * numHeads + h) * headDim;
            LongType idx1, idx2;
            if (ropeType == 0) {
              idx1 = base + i;
              idx2 = base + i + halfRotate;
            } else if (ropeType == 1) {
              idx1 = base + i * 2;
              idx2 = base + i * 2 + 1;
            } else {
              idx1 = base + i;
              idx2 = base + i + halfRotate;
            }

            float g1 = gradOut->e<float>(idx1);
            float g2 = gradOut->e<float>(idx2);

            // Inverse rotation for backward pass
            gradIn->p(idx1, g1 * cosTheta + g2 * sinTheta);
            gradIn->p(idx2, -g1 * sinTheta + g2 * cosTheta);
          }
        }
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, batch);
}

//////////////////////////////////////////////////////////////////////////////
// Fused RoPE with pre-computed cos/sin (cached variant)
//////////////////////////////////////////////////////////////////////////////

void fusedRoPECached(NDArray* input, NDArray* cosValues, NDArray* sinValues,
                     NDArray* output, int ropeType, LaunchContext* context) {
  auto rank = input->rankOf();
  auto batch = input->sizeAt(0);
  auto seqLen = input->sizeAt(1);
  auto numHeads = (rank >= 4) ? input->sizeAt(2) : static_cast<LongType>(1);
  auto headDim = (rank >= 4) ? input->sizeAt(3) : input->sizeAt(2);
  auto halfDim = headDim / 2;

  output->assign(input);

  // Determine cos/sin indexing based on rank
  auto cosRank = cosValues->rankOf();

  auto func = PRAGMA_THREADS_FOR {
    for (auto b = start; b < stop; b++) {
      for (LongType s = 0; s < seqLen; s++) {
        for (LongType h = 0; h < numHeads; h++) {
          for (LongType i = 0; i < halfDim; i++) {
            // Get cos/sin values - handle different input ranks
            float cosVal, sinVal;
            if (cosRank == 2) {
              cosVal = cosValues->e<float>(s * halfDim + i);
              sinVal = sinValues->e<float>(s * halfDim + i);
            } else if (cosRank == 3) {
              cosVal = cosValues->e<float>((b * seqLen + s) * halfDim + i);
              sinVal = sinValues->e<float>((b * seqLen + s) * halfDim + i);
            } else {
              // rank 4: [B, S, 1, halfDim]
              cosVal = cosValues->e<float>(((b * seqLen + s) * 1) * halfDim + i);
              sinVal = sinValues->e<float>(((b * seqLen + s) * 1) * halfDim + i);
            }

            LongType idx1, idx2;
            if (ropeType == 0) {
              idx1 = ((b * seqLen + s) * numHeads + h) * headDim + i;
              idx2 = ((b * seqLen + s) * numHeads + h) * headDim + i + halfDim;
            } else if (ropeType == 1) {
              idx1 = ((b * seqLen + s) * numHeads + h) * headDim + i * 2;
              idx2 = ((b * seqLen + s) * numHeads + h) * headDim + i * 2 + 1;
            } else {
              idx1 = ((b * seqLen + s) * numHeads + h) * headDim + i;
              idx2 = ((b * seqLen + s) * numHeads + h) * headDim + i + halfDim;
            }

            float x1 = input->e<float>(idx1);
            float x2 = input->e<float>(idx2);

            output->p(idx1, x1 * cosVal - x2 * sinVal);
            output->p(idx2, x1 * sinVal + x2 * cosVal);
          }
        }
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, batch);
}

//////////////////////////////////////////////////////////////////////////////
// Fused Bias + Dropout + Residual
//////////////////////////////////////////////////////////////////////////////

void fusedBiasDropoutResidual(NDArray* input, NDArray* bias, NDArray* residual,
                              NDArray* output, float dropoutProb, LongType seed,
                              bool training, LaunchContext* context) {
  auto totalElements = input->lengthOf();
  auto biasLen = bias != nullptr ? bias->lengthOf() : 1;

  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  auto func = PRAGMA_THREADS_FOR {
    std::mt19937_64 localRng(seed + start);  // Thread-local RNG
    std::uniform_real_distribution<float> localDist(0.0f, 1.0f);

    for (auto i = start; i < stop; i++) {
      float val = input->e<float>(i);

      // Add bias (broadcast along last dimension)
      if (bias != nullptr) {
        val += bias->e<float>(i % biasLen);
      }

      // Apply dropout if training
      if (training && dropoutProb > 0.0f) {
        float rand = localDist(localRng);
        if (rand < dropoutProb) {
          val = 0.0f;
        } else {
          val /= (1.0f - dropoutProb);
        }
      }

      // Add residual
      if (residual != nullptr) {
        val += residual->e<float>(i);
      }

      output->p(i, val);
    }
  };

  samediff::Threads::parallel_for(func, 0, totalElements);
}

//////////////////////////////////////////////////////////////////////////////
// Fused RMS Norm + SwiGLU (placeholder - full fusion not implemented for CPU)
//////////////////////////////////////////////////////////////////////////////

void fusedRmsNormSwiGLU(NDArray* input, NDArray* gamma, NDArray* wGate, NDArray* wUp,
                        NDArray* output, float epsilon, LaunchContext* context) {
  // For CPU, decompose into separate operations
  // This is a fallback - the full fused kernel is only available on CUDA
  THROW_EXCEPTION("fusedRmsNormSwiGLU: Use separate ops on CPU");
}

void fusedRmsNormSwiGLUBackward(NDArray* input, NDArray* gamma, NDArray* wGate, NDArray* wUp,
                                 NDArray* gradOut, NDArray* gradInput, NDArray* gradGamma,
                                 NDArray* gradWGate, NDArray* gradWUp, float epsilon,
                                 LaunchContext* context) {
  THROW_EXCEPTION("fusedRmsNormSwiGLUBackward: Use separate ops on CPU");
}

void fusedLayerNormBackward(NDArray* input, NDArray* gain, NDArray* gradOut,
                             NDArray* gradInput, NDArray* gradGain, NDArray* gradBias,
                             float epsilon, LaunchContext* context) {
  const LongType numRows = input->lengthOf() / input->sizeAt(-1);
  const LongType rowLen = input->sizeAt(-1);

  // Zero out gradients
  double zero = 0.0;
  gradGain->assign(zero);
  if (gradBias != nullptr) {
    gradBias->assign(zero);
  }

  auto func = PRAGMA_THREADS_FOR {
    for (auto row = start; row < stop; row++) {
      // Recompute mean and variance
      float mean = 0.0f;
      float M2 = 0.0f;
      float count = 0.0f;

      for (LongType i = 0; i < rowLen; i++) {
        float val = input->e<float>(row * rowLen + i);
        count += 1.0f;
        float delta = val - mean;
        mean += delta / count;
        float delta2 = val - mean;
        M2 += delta * delta2;
      }

      float variance = M2 / count;
      float invStd = 1.0f / std::sqrt(variance + epsilon);

      // Compute gradients
      float dvar = 0.0f;
      float dmean = 0.0f;

      for (LongType i = 0; i < rowLen; i++) {
        float val = input->e<float>(row * rowLen + i);
        float normalized = (val - mean) * invStd;
        float dout = gradOut->e<float>(row * rowLen + i);
        float g = gain->e<float>(i);

        // Gradient for gain
        gradGain->p(i, gradGain->e<float>(i) + dout * normalized);

        // Gradient for bias
        if (gradBias != nullptr) {
          gradBias->p(i, gradBias->e<float>(i) + dout);
        }

        // Accumulate for input gradient
        float dnorm = dout * g;
        dvar += dnorm * (val - mean) * (-0.5f) * invStd * invStd * invStd;
        dmean += dnorm * (-invStd);
      }

      dmean += dvar * (-2.0f / count) * (0.0f); // mean of (x - mean) is 0

      // Compute input gradient
      for (LongType i = 0; i < rowLen; i++) {
        float val = input->e<float>(row * rowLen + i);
        float dout = gradOut->e<float>(row * rowLen + i);
        float g = gain->e<float>(i);
        float dnorm = dout * g;

        float dx = dnorm * invStd + dvar * 2.0f * (val - mean) / count + dmean / count;
        gradInput->p(row * rowLen + i, dx);
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, numRows);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
