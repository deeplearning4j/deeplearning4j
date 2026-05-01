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
  const int rank = input->rankOf();
  const LongType batch    = input->sizeAt(0);
  const LongType seqLen   = input->sizeAt(1);
  const LongType numHeads = (rank >= 4) ? input->sizeAt(2) : static_cast<LongType>(1);
  const LongType headDim  = (rank >= 4) ? input->sizeAt(3) : input->sizeAt(2);

  const LongType rotateDims = (rotaryDims > 0 && rotaryDims < headDim) ? rotaryDims : headDim;
  const LongType halfRotate = rotateDims / 2;

  NDArray::preparePrimaryUse({output}, {input});

  // Expand strides to length-4 arrays regardless of actual rank.
  // shape::stride() returns shapeInfo[1+rank], which is the first stride entry.
  LongType xS[4] = {0, 0, 0, 1};
  LongType zS[4] = {0, 0, 0, 1};
  {
    const LongType* xs = shape::stride(input->shapeInfo());
    const LongType* zs = shape::stride(output->shapeInfo());
    if (rank == 4) {
      xS[0]=xs[0]; xS[1]=xs[1]; xS[2]=xs[2]; xS[3]=xs[3];
      zS[0]=zs[0]; zS[1]=zs[1]; zS[2]=zs[2]; zS[3]=zs[3];
    } else if (rank == 3) {
      xS[0]=xs[0]; xS[1]=xs[1]; xS[2]=xs[2]; xS[3]=1;
      zS[0]=zs[0]; zS[1]=zs[1]; zS[2]=zs[2]; zS[3]=1;
    } else {
      xS[0]=xs[0]; xS[1]=xs[1]; xS[2]=1; xS[3]=1;
      zS[0]=zs[0]; zS[1]=zs[1]; zS[2]=1; zS[3]=1;
    }
  }

  // Pre-compute inverse-frequency table: invFreq[i] = freqScale / freqBase^(2i/rotateDims)
  // Only halfRotate entries (<=512 for any practical head_dim).
  float invFreq[512];
  for (LongType i = 0; i < halfRotate; ++i) {
    invFreq[i] = freqScale / std::pow(freqBase, (2.0f * static_cast<float>(i)) / static_cast<float>(rotateDims));
  }

  // Parallelise over (batch * seqLen * numHeads)
  const LongType outerSize = batch * seqLen * numHeads;
  const DataType dtype = input->dataType();

  // Typed dispatch — avoids per-element type conversion overhead
  auto applyRoPE = [&](auto* xBuf, auto* zBuf) {
    using T = std::remove_pointer_t<decltype(xBuf)>;

    auto func = PRAGMA_THREADS_FOR {
      for (auto idx = start; idx < stop; ++idx) {
        const LongType h   = idx % numHeads;
        const LongType tmp = idx / numHeads;
        const LongType s   = tmp % seqLen;
        const LongType b   = tmp / seqLen;

        const float posF = static_cast<float>(static_cast<LongType>(positionOffset) + s);

        const T* xPtr = xBuf + b * xS[0] + s * xS[1] + h * xS[2];
        T*       zPtr = zBuf + b * zS[0] + s * zS[1] + h * zS[2];

        if (ropeType == 1) {  // NeoX interleaved
          PRAGMA_OMP_SIMD
          for (LongType i = 0; i < halfRotate; ++i) {
            const float theta = posF * invFreq[i];
            const float cosT  = std::cos(theta);
            const float sinT  = std::sin(theta);
            const float x0 = static_cast<float>(xPtr[(2 * i)     * xS[3]]);
            const float x1 = static_cast<float>(xPtr[(2 * i + 1) * xS[3]]);
            zPtr[(2 * i)     * zS[3]] = static_cast<T>(x0 * cosT - x1 * sinT);
            zPtr[(2 * i + 1) * zS[3]] = static_cast<T>(x0 * sinT + x1 * cosT);
          }
          // Copy unrotated tail for NeoX
          for (LongType i = rotateDims; i < headDim; ++i) {
            zPtr[i * zS[3]] = xPtr[i * xS[3]];
          }
        } else {  // Standard (LLaMA / GPT-J)
          PRAGMA_OMP_SIMD
          for (LongType i = 0; i < halfRotate; ++i) {
            const float theta = posF * invFreq[i];
            const float cosT  = std::cos(theta);
            const float sinT  = std::sin(theta);
            const float x0 = static_cast<float>(xPtr[i               * xS[3]]);
            const float x1 = static_cast<float>(xPtr[(i + halfRotate) * xS[3]]);
            zPtr[i                * zS[3]] = static_cast<T>(x0 * cosT - x1 * sinT);
            zPtr[(i + halfRotate) * zS[3]] = static_cast<T>(x0 * sinT + x1 * cosT);
          }
          // Copy unrotated tail
          for (LongType i = rotateDims; i < headDim; ++i) {
            zPtr[i * zS[3]] = xPtr[i * xS[3]];
          }
        }
      }
    };
    samediff::Threads::parallel_for(func, 0, outerSize);
  };

  if (dtype == DataType::FLOAT32) {
    applyRoPE(input->bufferAsT<float>(), output->bufferAsT<float>());
  } else if (dtype == DataType::HALF) {
    applyRoPE(input->bufferAsT<float16>(), output->bufferAsT<float16>());
  } else if (dtype == DataType::BFLOAT16) {
    applyRoPE(input->bufferAsT<bfloat16>(), output->bufferAsT<bfloat16>());
  } else if (dtype == DataType::DOUBLE) {
    applyRoPE(input->bufferAsT<double>(), output->bufferAsT<double>());
  } else {
    // Fallback for other types via virtual dispatch (rare)
    output->assign(input);
    const LongType outerSizeFb = batch * seqLen * numHeads;
    auto func = PRAGMA_THREADS_FOR {
      for (auto idx = start; idx < stop; ++idx) {
        const LongType h   = idx % numHeads;
        const LongType tmp = idx / numHeads;
        const LongType s   = tmp % seqLen;
        const LongType b   = tmp / seqLen;
        const float posF = static_cast<float>(static_cast<LongType>(positionOffset) + s);
        const LongType base = (b * seqLen + s) * numHeads * headDim + h * headDim;
        for (LongType i = 0; i < halfRotate; ++i) {
          const float theta = posF * invFreq[i];
          const float cosT  = std::cos(theta);
          const float sinT  = std::sin(theta);
          LongType idx1, idx2;
          if (ropeType == 1) { idx1 = base + i * 2; idx2 = base + i * 2 + 1; }
          else                { idx1 = base + i;     idx2 = base + i + halfRotate; }
          const float x0 = input->e<float>(idx1);
          const float x1 = input->e<float>(idx2);
          output->p(idx1, x0 * cosT - x1 * sinT);
          output->p(idx2, x0 * sinT + x1 * cosT);
        }
      }
    };
    samediff::Threads::parallel_for(func, 0, outerSizeFb);
  }

  NDArray::registerPrimaryUse({output}, {input});
}

void fusedRoPEBackward(NDArray* gradOut, NDArray* gradIn, int positionOffset,
                       float freqBase, float freqScale, int ropeType, LaunchContext* context,
                       int rotaryDims) {
  const int rank = gradOut->rankOf();
  const LongType batch    = gradOut->sizeAt(0);
  const LongType seqLen   = gradOut->sizeAt(1);
  const LongType numHeads = (rank >= 4) ? gradOut->sizeAt(2) : static_cast<LongType>(1);
  const LongType headDim  = (rank >= 4) ? gradOut->sizeAt(3) : gradOut->sizeAt(2);

  const LongType rotateDims = (rotaryDims > 0 && rotaryDims < headDim) ? rotaryDims : headDim;
  const LongType halfRotate = rotateDims / 2;

  NDArray::preparePrimaryUse({gradIn}, {gradOut});

  // Strides
  LongType gS[4] = {0, 0, 0, 1};
  LongType oS[4] = {0, 0, 0, 1};
  {
    const LongType* gs = shape::stride(gradOut->shapeInfo());
    const LongType* os = shape::stride(gradIn->shapeInfo());
    if (rank == 4) {
      gS[0]=gs[0]; gS[1]=gs[1]; gS[2]=gs[2]; gS[3]=gs[3];
      oS[0]=os[0]; oS[1]=os[1]; oS[2]=os[2]; oS[3]=os[3];
    } else if (rank == 3) {
      gS[0]=gs[0]; gS[1]=gs[1]; gS[2]=gs[2]; gS[3]=1;
      oS[0]=os[0]; oS[1]=os[1]; oS[2]=os[2]; oS[3]=1;
    } else {
      gS[0]=gs[0]; gS[1]=gs[1]; gS[2]=1; gS[3]=1;
      oS[0]=os[0]; oS[1]=os[1]; oS[2]=1; oS[3]=1;
    }
  }

  // Pre-compute invFreq
  float invFreq[512];
  for (LongType i = 0; i < halfRotate; ++i) {
    invFreq[i] = freqScale / std::pow(freqBase, (2.0f * static_cast<float>(i)) / static_cast<float>(rotateDims));
  }

  const LongType outerSize = batch * seqLen * numHeads;
  const DataType dtype = gradOut->dataType();

  auto applyBwd = [&](auto* gBuf, auto* oBuf) {
    using T = std::remove_pointer_t<decltype(gBuf)>;

    auto func = PRAGMA_THREADS_FOR {
      for (auto idx = start; idx < stop; ++idx) {
        const LongType h   = idx % numHeads;
        const LongType tmp = idx / numHeads;
        const LongType s   = tmp % seqLen;
        const LongType b   = tmp / seqLen;
        const float posF = static_cast<float>(static_cast<LongType>(positionOffset) + s);

        const T* gPtr = gBuf + b * gS[0] + s * gS[1] + h * gS[2];
        T*       oPtr = oBuf + b * oS[0] + s * oS[1] + h * oS[2];

        if (ropeType == 1) {  // NeoX
          PRAGMA_OMP_SIMD
          for (LongType i = 0; i < halfRotate; ++i) {
            const float theta = posF * invFreq[i];
            const float cosT  = std::cos(theta);
            const float sinT  = std::sin(theta);
            const float g0 = static_cast<float>(gPtr[(2 * i)     * gS[3]]);
            const float g1 = static_cast<float>(gPtr[(2 * i + 1) * gS[3]]);
            oPtr[(2 * i)     * oS[3]] = static_cast<T>(g0 * cosT + g1 * sinT);
            oPtr[(2 * i + 1) * oS[3]] = static_cast<T>(-g0 * sinT + g1 * cosT);
          }
          for (LongType i = rotateDims; i < headDim; ++i) oPtr[i * oS[3]] = gPtr[i * gS[3]];
        } else {  // Standard
          PRAGMA_OMP_SIMD
          for (LongType i = 0; i < halfRotate; ++i) {
            const float theta = posF * invFreq[i];
            const float cosT  = std::cos(theta);
            const float sinT  = std::sin(theta);
            const float g0 = static_cast<float>(gPtr[i               * gS[3]]);
            const float g1 = static_cast<float>(gPtr[(i + halfRotate) * gS[3]]);
            oPtr[i                * oS[3]] = static_cast<T>(g0 * cosT + g1 * sinT);
            oPtr[(i + halfRotate) * oS[3]] = static_cast<T>(-g0 * sinT + g1 * cosT);
          }
          for (LongType i = rotateDims; i < headDim; ++i) oPtr[i * oS[3]] = gPtr[i * gS[3]];
        }
      }
    };
    samediff::Threads::parallel_for(func, 0, outerSize);
  };

  if (dtype == DataType::FLOAT32) {
    applyBwd(gradOut->bufferAsT<float>(), gradIn->bufferAsT<float>());
  } else if (dtype == DataType::HALF) {
    applyBwd(gradOut->bufferAsT<float16>(), gradIn->bufferAsT<float16>());
  } else if (dtype == DataType::BFLOAT16) {
    applyBwd(gradOut->bufferAsT<bfloat16>(), gradIn->bufferAsT<bfloat16>());
  } else if (dtype == DataType::DOUBLE) {
    applyBwd(gradOut->bufferAsT<double>(), gradIn->bufferAsT<double>());
  } else {
    gradIn->assign(gradOut);
    auto func = PRAGMA_THREADS_FOR {
      for (auto b = start; b < stop; ++b) {
        for (LongType s = 0; s < seqLen; s++) {
          const LongType pos = positionOffset + s;
          const float posF = static_cast<float>(pos);
          for (LongType h = 0; h < numHeads; h++) {
            const LongType base = (b * seqLen + s) * numHeads * headDim + h * headDim;
            for (LongType i = 0; i < halfRotate; i++) {
              const float theta = posF * invFreq[i];
              const float cosT  = std::cos(theta);
              const float sinT  = std::sin(theta);
              LongType idx1, idx2;
              if (ropeType == 1) { idx1 = base + i * 2; idx2 = base + i * 2 + 1; }
              else                { idx1 = base + i;     idx2 = base + i + halfRotate; }
              const float g0 = gradOut->e<float>(idx1);
              const float g1 = gradOut->e<float>(idx2);
              gradIn->p(idx1, g0 * cosT + g1 * sinT);
              gradIn->p(idx2, -g0 * sinT + g1 * cosT);
            }
          }
        }
      }
    };
    samediff::Threads::parallel_for(func, 0, batch);
  }

  NDArray::registerPrimaryUse({gradIn}, {gradOut});
}

//////////////////////////////////////////////////////////////////////////////
// Fused RoPE with pre-computed cos/sin (cached variant)
//////////////////////////////////////////////////////////////////////////////

void fusedRoPECached(NDArray* input, NDArray* cosValues, NDArray* sinValues,
                     NDArray* output, int ropeType, LaunchContext* context) {
  const int rank    = input->rankOf();
  const LongType batch    = input->sizeAt(0);
  const LongType seqLen   = input->sizeAt(1);
  const LongType numHeads = (rank >= 4) ? input->sizeAt(2) : static_cast<LongType>(1);
  const LongType headDim  = (rank >= 4) ? input->sizeAt(3) : input->sizeAt(2);
  const LongType halfDim  = headDim / 2;

  NDArray::preparePrimaryUse({output}, {input, cosValues, sinValues});

  // Input strides
  LongType xS[4] = {0, 0, 0, 1};
  LongType zS[4] = {0, 0, 0, 1};
  {
    const LongType* xs = shape::stride(input->shapeInfo());
    const LongType* zs = shape::stride(output->shapeInfo());
    if (rank == 4) {
      xS[0]=xs[0]; xS[1]=xs[1]; xS[2]=xs[2]; xS[3]=xs[3];
      zS[0]=zs[0]; zS[1]=zs[1]; zS[2]=zs[2]; zS[3]=zs[3];
    } else if (rank == 3) {
      xS[0]=xs[0]; xS[1]=xs[1]; xS[2]=xs[2]; xS[3]=1;
      zS[0]=zs[0]; zS[1]=zs[1]; zS[2]=zs[2]; zS[3]=1;
    } else {
      xS[0]=xs[0]; xS[1]=xs[1]; xS[2]=1; xS[3]=1;
      zS[0]=zs[0]; zS[1]=zs[1]; zS[2]=1; zS[3]=1;
    }
  }

  // cos/sin strides (always FLOAT32 from ONNX/cache)
  // Cast if needed to float for consistent access
  NDArray* cosF = cosValues;
  NDArray* sinF = sinValues;
  NDArray* cosCast = nullptr;
  NDArray* sinCast = nullptr;
  if (cosValues->dataType() != DataType::FLOAT32) {
    cosCast = cosValues->cast(DataType::FLOAT32);
    cosF = cosCast;
  }
  if (sinValues->dataType() != DataType::FLOAT32) {
    sinCast = sinValues->cast(DataType::FLOAT32);
    sinF = sinCast;
  }

  const int cosRank = cosF->rankOf();
  const float* cosPtr = cosF->bufferAsT<float>();
  const float* sinPtr = sinF->bufferAsT<float>();

  // cos/sin shape: [S, halfDim] (rank2), [B, S, halfDim] (rank3), or [B, S, 1, halfDim] (rank4)
  // In all cases the flat offset for position (b,s) is:
  //   rank2: s * halfDim
  //   rank3: (b * seqLen + s) * halfDim
  //   rank4: same as rank3 (the '1' head dim is trivially squeezed)

  const LongType outerSize = batch * seqLen * numHeads;

  const DataType dtype = input->dataType();

  auto applyCached = [&](auto* xBuf, auto* zBuf) {
    using T = std::remove_pointer_t<decltype(xBuf)>;

    auto func = PRAGMA_THREADS_FOR {
      for (auto idx = start; idx < stop; ++idx) {
        const LongType h   = idx % numHeads;
        const LongType tmp = idx / numHeads;
        const LongType s   = tmp % seqLen;
        const LongType b   = tmp / seqLen;

        // Offset into cos/sin tables
        const LongType csOff = (cosRank == 2) ? (s * halfDim)
                                               : ((b * seqLen + s) * halfDim);
        const float* cPtr = cosPtr + csOff;
        const float* sPtr = sinPtr + csOff;

        const T* xPtr = xBuf + b * xS[0] + s * xS[1] + h * xS[2];
        T*       zPtr = zBuf + b * zS[0] + s * zS[1] + h * zS[2];

        if (ropeType == 1) {  // NeoX interleaved
          PRAGMA_OMP_SIMD
          for (LongType i = 0; i < halfDim; ++i) {
            const float cosT = cPtr[i];
            const float sinT = sPtr[i];
            const float x0 = static_cast<float>(xPtr[(2 * i)     * xS[3]]);
            const float x1 = static_cast<float>(xPtr[(2 * i + 1) * xS[3]]);
            zPtr[(2 * i)     * zS[3]] = static_cast<T>(x0 * cosT - x1 * sinT);
            zPtr[(2 * i + 1) * zS[3]] = static_cast<T>(x0 * sinT + x1 * cosT);
          }
        } else {  // Standard (LLaMA / GPT-J)
          PRAGMA_OMP_SIMD
          for (LongType i = 0; i < halfDim; ++i) {
            const float cosT = cPtr[i];
            const float sinT = sPtr[i];
            const float x0 = static_cast<float>(xPtr[i           * xS[3]]);
            const float x1 = static_cast<float>(xPtr[(i + halfDim) * xS[3]]);
            zPtr[i           * zS[3]] = static_cast<T>(x0 * cosT - x1 * sinT);
            zPtr[(i + halfDim) * zS[3]] = static_cast<T>(x0 * sinT + x1 * cosT);
          }
        }
      }
    };
    samediff::Threads::parallel_for(func, 0, outerSize);
  };

  if (dtype == DataType::FLOAT32) {
    applyCached(input->bufferAsT<float>(), output->bufferAsT<float>());
  } else if (dtype == DataType::HALF) {
    applyCached(input->bufferAsT<float16>(), output->bufferAsT<float16>());
  } else if (dtype == DataType::BFLOAT16) {
    applyCached(input->bufferAsT<bfloat16>(), output->bufferAsT<bfloat16>());
  } else if (dtype == DataType::DOUBLE) {
    applyCached(input->bufferAsT<double>(), output->bufferAsT<double>());
  } else {
    // Fallback
    output->assign(input);
    auto func = PRAGMA_THREADS_FOR {
      for (auto b = start; b < stop; ++b) {
        for (LongType s = 0; s < seqLen; s++) {
          for (LongType h = 0; h < numHeads; h++) {
            const LongType csOff = (cosRank == 2) ? (s * halfDim) : ((b * seqLen + s) * halfDim);
            for (LongType i = 0; i < halfDim; i++) {
              const float cosT = cosPtr[csOff + i];
              const float sinT = sinPtr[csOff + i];
              LongType idx1, idx2;
              if (ropeType == 1) {
                idx1 = ((b * seqLen + s) * numHeads + h) * headDim + i * 2;
                idx2 = idx1 + 1;
              } else {
                idx1 = ((b * seqLen + s) * numHeads + h) * headDim + i;
                idx2 = idx1 + halfDim;
              }
              const float x0 = input->e<float>(idx1);
              const float x1 = input->e<float>(idx2);
              output->p(idx1, x0 * cosT - x1 * sinT);
              output->p(idx2, x0 * sinT + x1 * cosT);
            }
          }
        }
      }
    };
    samediff::Threads::parallel_for(func, 0, batch);
  }

  if (cosCast != nullptr) delete cosCast;
  if (sinCast != nullptr) delete sinCast;

  NDArray::registerPrimaryUse({output}, {input, cosValues, sinValues});
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
