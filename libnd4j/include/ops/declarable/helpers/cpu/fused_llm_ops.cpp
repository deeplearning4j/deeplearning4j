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
#include <helpers/MmulHelper.h>
#include <execution/Threads.h>
#include <system/type_boilerplate.h>
#include <cmath>
#include <random>

namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////////
// Fused GELU - x * sigmoid(1.702 * x)
//////////////////////////////////////////////////////////////////////////////

template <typename T>
static void fusedGELU_(NDArray* input, NDArray* output) {
  const LongType len = input->lengthOf();
  const T* xBuf = input->bufferAsT<T>();
  T*       zBuf = output->bufferAsT<T>();

  // Read real strides — input/output may be views.
  const LongType xStride = (input->rankOf() == 1)  ? input->strideAt(0)  : 1;
  const LongType zStride = (output->rankOf() == 1) ? output->strideAt(0) : 1;

  if (xStride == 1 && zStride == 1) {
    // Contiguous fast path
    auto func = PRAGMA_THREADS_FOR {
      PRAGMA_OMP_SIMD
      for (auto i = start; i < stop; i++) {
        const float x   = static_cast<float>(xBuf[i]);
        const float sig = 1.0f / (1.0f + std::exp(-1.702f * x));
        zBuf[i] = static_cast<T>(x * sig);
      }
    };
    samediff::Threads::parallel_for(func, 0, len);
  } else {
    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        const float x   = static_cast<float>(xBuf[i * xStride]);
        const float sig = 1.0f / (1.0f + std::exp(-1.702f * x));
        zBuf[i * zStride] = static_cast<T>(x * sig);
      }
    };
    samediff::Threads::parallel_for(func, 0, len);
  }
}

void fusedGELU(NDArray* input, NDArray* output, LaunchContext* context) {
  NDArray::preparePrimaryUse({output}, {input});
  BUILD_SINGLE_SELECTOR(input->dataType(), fusedGELU_, (input, output), SD_FLOAT_TYPES);
  NDArray::registerPrimaryUse({output}, {input});
}

template <typename T>
static void fusedGELUBackward_(NDArray* input, NDArray* gradOut, NDArray* gradIn) {
  const LongType len = input->lengthOf();
  const T* xBuf  = input->bufferAsT<T>();
  const T* doBuf = gradOut->bufferAsT<T>();
  T*       diBuf = gradIn->bufferAsT<T>();

  const LongType xStride  = (input->rankOf()   == 1) ? input->strideAt(0)   : 1;
  const LongType doStride = (gradOut->rankOf()  == 1) ? gradOut->strideAt(0) : 1;
  const LongType diStride = (gradIn->rankOf()   == 1) ? gradIn->strideAt(0)  : 1;

  if (xStride == 1 && doStride == 1 && diStride == 1) {
    auto func = PRAGMA_THREADS_FOR {
      PRAGMA_OMP_SIMD
      for (auto i = start; i < stop; i++) {
        const float x    = static_cast<float>(xBuf[i]);
        const float dout = static_cast<float>(doBuf[i]);
        // d/dx[x * sigmoid(1.702*x)] = sigmoid(1.702*x) + x * 1.702 * sigmoid(1.702*x) * (1 - sigmoid(1.702*x))
        const float sig  = 1.0f / (1.0f + std::exp(-1.702f * x));
        diBuf[i] = static_cast<T>(dout * (sig + x * 1.702f * sig * (1.0f - sig)));
      }
    };
    samediff::Threads::parallel_for(func, 0, len);
  } else {
    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        const float x    = static_cast<float>(xBuf[i * xStride]);
        const float dout = static_cast<float>(doBuf[i * doStride]);
        const float sig  = 1.0f / (1.0f + std::exp(-1.702f * x));
        diBuf[i * diStride] = static_cast<T>(dout * (sig + x * 1.702f * sig * (1.0f - sig)));
      }
    };
    samediff::Threads::parallel_for(func, 0, len);
  }
}

void fusedGELUBackward(NDArray* input, NDArray* gradOut, NDArray* gradIn, LaunchContext* context) {
  NDArray::preparePrimaryUse({gradIn}, {input, gradOut});
  BUILD_SINGLE_SELECTOR(input->dataType(), fusedGELUBackward_, (input, gradOut, gradIn), SD_FLOAT_TYPES);
  NDArray::registerPrimaryUse({gradIn}, {input, gradOut});
}

//////////////////////////////////////////////////////////////////////////////
// Fused Layer Norm with Welford's algorithm
//////////////////////////////////////////////////////////////////////////////

template <typename T>
static void fusedLayerNorm_(NDArray* input, NDArray* gain, NDArray* bias, NDArray* output,
                            float epsilon) {
  const LongType numRows = input->lengthOf() / input->sizeAt(-1);
  const LongType rowLen  = input->sizeAt(-1);

  const T* xBuf  = input->bufferAsT<T>();
  T*       zBuf  = output->bufferAsT<T>();
  const T* gBuf  = gain->bufferAsT<T>();
  const T* bBuf  = (bias != nullptr) ? bias->bufferAsT<T>() : nullptr;

  // Strides along the innermost (feature) dimension.
  // For a contiguous [B..., rowLen] tensor the last stride is 1.
  // For a non-contiguous view we must respect the actual stride.
  const LongType xRankM1  = input->rankOf()  - 1;
  const LongType zRankM1  = output->rankOf() - 1;
  const LongType gRankM1  = gain->rankOf()   - 1;
  const LongType xS1 = input->strideAt(xRankM1);
  const LongType zS1 = output->strideAt(zRankM1);
  const LongType gS1 = gain->strideAt(gRankM1);
  const LongType bS1 = (bias != nullptr) ? bias->strideAt(bias->rankOf() - 1) : 1;

  // Row stride: how many T elements to advance xBuf/zBuf per row.
  // For a rank-1 input this equals rowLen*xS1 but we compute it directly from
  // the second-to-last stride when available (handles arbitrary views).
  const LongType xRowStride = (input->rankOf()  >= 2) ? input->strideAt(xRankM1 - 1)
                                                        : rowLen * xS1;
  const LongType zRowStride = (output->rankOf() >= 2) ? output->strideAt(zRankM1 - 1)
                                                        : rowLen * zS1;

  auto func = PRAGMA_THREADS_FOR {
    for (auto row = start; row < stop; row++) {
      const T* xRow = xBuf + row * xRowStride;
      T*       zRow = zBuf + row * zRowStride;

      // Welford's online algorithm for mean and variance (accumulate in float)
      float mean  = 0.0f;
      float M2    = 0.0f;
      float count = 0.0f;
      for (LongType i = 0; i < rowLen; i++) {
        const float val = static_cast<float>(xRow[i * xS1]);
        count += 1.0f;
        const float delta  = val - mean;
        mean  += delta / count;
        const float delta2 = val - mean;
        M2    += delta * delta2;
      }
      const float variance = M2 / count;
      const float invStd   = 1.0f / std::sqrt(variance + epsilon);

      // Normalize, scale and shift
      PRAGMA_OMP_SIMD
      for (LongType i = 0; i < rowLen; i++) {
        const float val        = static_cast<float>(xRow[i * xS1]);
        const float normalized = (val - mean) * invStd;
        const float g          = static_cast<float>(gBuf[i * gS1]);
        float result = normalized * g;
        if (bBuf != nullptr) result += static_cast<float>(bBuf[i * bS1]);
        zRow[i * zS1] = static_cast<T>(result);
      }
    }
  };

  samediff::Threads::parallel_tad(func, 0, numRows);
}

void fusedLayerNorm(NDArray* input, NDArray* gain, NDArray* bias, NDArray* output,
                    float epsilon, LaunchContext* context) {
  NDArray::preparePrimaryUse({output}, {input, gain, bias});

  // Cast gain/bias to input dtype if needed (CPU: cast rather than dual template)
  NDArray* gainToUse  = gain;
  NDArray* biasToUse  = bias;
  NDArray* gainCast   = nullptr;
  NDArray* biasCast   = nullptr;
  if (gain != nullptr && gain->dataType() != input->dataType()) {
    gainCast  = gain->cast(input->dataType());
    gainToUse = gainCast;
  }
  if (bias != nullptr && bias->dataType() != input->dataType()) {
    biasCast  = bias->cast(input->dataType());
    biasToUse = biasCast;
  }

  BUILD_SINGLE_SELECTOR(input->dataType(), fusedLayerNorm_,
                         (input, gainToUse, biasToUse, output, epsilon), SD_FLOAT_TYPES);

  if (gainCast  != nullptr) delete gainCast;
  if (biasCast  != nullptr) delete biasCast;

  NDArray::registerPrimaryUse({output}, {input, gain, bias});
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
  // Heap-allocated to support any headDim without stack overflow (halfRotate = rotateDims/2).
  float* invFreq = new float[halfRotate];
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

  delete[] invFreq;
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

  // Pre-compute invFreq — heap-allocated to support any headDim without stack overflow.
  float* invFreq = new float[halfRotate];
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

  delete[] invFreq;
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

  // Extract actual strides from cos/sin NDArrays.
  // cos/sin shape: [S, halfDim] (rank2), [B, S, halfDim] (rank3), or [B, S, 1, halfDim] (rank4).
  // These may be slices of a larger cache (non-contiguous), so we MUST use the real
  // NDArray strides rather than assuming contiguous layout (the regression bug).
  LongType cosStride0 = 0;  // batch stride (0 = broadcast across batch for rank-2)
  LongType cosStride1 = 0;  // seq stride
  LongType cosStride2 = 1;  // innermost (halfDim element) stride
  if (cosRank == 2) {
    // [S, halfDim] — no batch dim; broadcast across batch by keeping cosStride0 = 0
    cosStride0 = 0;
    cosStride1 = cosF->strideAt(0);
    cosStride2 = cosF->strideAt(1);
  } else if (cosRank == 3) {
    // [B, S, halfDim]
    cosStride0 = cosF->strideAt(0);
    cosStride1 = cosF->strideAt(1);
    cosStride2 = cosF->strideAt(2);
  } else if (cosRank >= 4) {
    // [B, S, 1, halfDim] — skip the broadcast head dim (stride index 3 is innermost)
    cosStride0 = cosF->strideAt(0);
    cosStride1 = cosF->strideAt(1);
    cosStride2 = cosF->strideAt(3);
  }

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

        // Offset into cos/sin tables using real strides (handles non-contiguous slices).
        const LongType csOff = b * cosStride0 + s * cosStride1;
        const float* cPtr = cosPtr + csOff;
        const float* sPtr = sinPtr + csOff;

        const T* xPtr = xBuf + b * xS[0] + s * xS[1] + h * xS[2];
        T*       zPtr = zBuf + b * zS[0] + s * zS[1] + h * zS[2];

        if (ropeType == 1) {  // NeoX interleaved
          for (LongType i = 0; i < halfDim; ++i) {
            const float cosT = cPtr[i * cosStride2];
            const float sinT = sPtr[i * cosStride2];
            const float x0 = static_cast<float>(xPtr[(2 * i)     * xS[3]]);
            const float x1 = static_cast<float>(xPtr[(2 * i + 1) * xS[3]]);
            zPtr[(2 * i)     * zS[3]] = static_cast<T>(x0 * cosT - x1 * sinT);
            zPtr[(2 * i + 1) * zS[3]] = static_cast<T>(x0 * sinT + x1 * cosT);
          }
        } else {  // Standard (LLaMA / GPT-J)
          for (LongType i = 0; i < halfDim; ++i) {
            const float cosT = cPtr[i * cosStride2];
            const float sinT = sPtr[i * cosStride2];
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
            // Use real strides for non-contiguous cos/sin (same fix as the typed path above)
            const LongType csOff = b * cosStride0 + s * cosStride1;
            for (LongType i = 0; i < halfDim; i++) {
              const float cosT = cosPtr[csOff + i * cosStride2];
              const float sinT = sinPtr[csOff + i * cosStride2];
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

template <typename T>
static void fusedBiasDropoutResidual_(NDArray* input, NDArray* bias, NDArray* residual,
                                      NDArray* output, float dropoutProb, LongType seed,
                                      bool training) {
  const LongType totalElements = input->lengthOf();
  const LongType biasLen       = (bias != nullptr) ? bias->lengthOf() : 1;

  const T* xBuf   = input->bufferAsT<T>();
  T*       zBuf   = output->bufferAsT<T>();
  const T* bBuf   = (bias     != nullptr) ? bias->bufferAsT<T>()     : nullptr;
  const T* rBuf   = (residual != nullptr) ? residual->bufferAsT<T>() : nullptr;

  // Innermost stride (1 for contiguous, else real stride)
  const LongType xS = (input->rankOf()    == 1) ? input->strideAt(0)    : 1;
  const LongType zS = (output->rankOf()   == 1) ? output->strideAt(0)   : 1;
  const LongType bS = (bias != nullptr && bias->rankOf() == 1) ? bias->strideAt(0) : 1;
  const LongType rS = (residual != nullptr && residual->rankOf() == 1) ? residual->strideAt(0) : 1;

  auto func = PRAGMA_THREADS_FOR {
    std::mt19937_64 localRng(static_cast<uint64_t>(seed) + static_cast<uint64_t>(start));
    std::uniform_real_distribution<float> localDist(0.0f, 1.0f);

    for (auto i = start; i < stop; i++) {
      float val = static_cast<float>(xBuf[i * xS]);

      if (bBuf != nullptr) {
        val += static_cast<float>(bBuf[(i % biasLen) * bS]);
      }

      if (training && dropoutProb > 0.0f) {
        const float r = localDist(localRng);
        if (r < dropoutProb) {
          val = 0.0f;
        } else {
          val /= (1.0f - dropoutProb);
        }
      }

      if (rBuf != nullptr) {
        val += static_cast<float>(rBuf[i * rS]);
      }

      zBuf[i * zS] = static_cast<T>(val);
    }
  };

  samediff::Threads::parallel_for(func, 0, totalElements);
}

void fusedBiasDropoutResidual(NDArray* input, NDArray* bias, NDArray* residual,
                              NDArray* output, float dropoutProb, LongType seed,
                              bool training, LaunchContext* context) {
  NDArray::preparePrimaryUse({output}, {input, bias, residual});
  BUILD_SINGLE_SELECTOR(input->dataType(), fusedBiasDropoutResidual_,
                         (input, bias, residual, output, dropoutProb, seed, training),
                         SD_FLOAT_TYPES);
  NDArray::registerPrimaryUse({output}, {input, bias, residual});
}

//////////////////////////////////////////////////////////////////////////////
// Fused RMS Norm + SwiGLU
// Computes: silu(rms_norm(x) @ W_gate) * (rms_norm(x) @ W_up)
//////////////////////////////////////////////////////////////////////////////

template <typename T>
static void fusedRmsNormSwiGLU_(NDArray* input, NDArray* gamma, NDArray* wGate, NDArray* wUp,
                                 NDArray* output, float epsilon, LaunchContext* context) {
  const LongType batchSize       = input->sizeAt(0);
  const LongType seqLen          = input->sizeAt(1);
  const LongType hiddenDim       = input->sizeAt(2);
  const LongType intermediateDim = wGate->sizeAt(1);
  const LongType numRows         = batchSize * seqLen;

  // --------------------------------------------------------------------------
  // Step 1: RMS norm + gamma scaling
  // Input layout: [batchSize, seqLen, hiddenDim] — assumed contiguous C-order.
  // normalized layout: same shape, contiguous.
  // --------------------------------------------------------------------------
  std::vector<LongType> normShape = {batchSize, seqLen, hiddenDim};
  NDArray normalized('c', normShape, input->dataType(), context);

  const T* xBuf  = input->bufferAsT<T>();
  T*       nBuf  = normalized.bufferAsT<T>();
  const T* gBuf  = gamma->bufferAsT<T>();

  auto normFunc = PRAGMA_THREADS_FOR {
    for (auto row = start; row < stop; row++) {
      const T* xRow = xBuf + row * hiddenDim;
      T*       nRow = nBuf + row * hiddenDim;

      // Single pass: sum-of-squares in float to avoid FP16 overflow
      float sumSq = 0.0f;
      for (LongType i = 0; i < hiddenDim; i++) {
        const float v = static_cast<float>(xRow[i]);
        sumSq += v * v;
      }
      const float invRms = 1.0f / std::sqrt(sumSq / static_cast<float>(hiddenDim) + epsilon);

      // Normalize and apply gamma
      PRAGMA_OMP_SIMD
      for (LongType i = 0; i < hiddenDim; i++) {
        nRow[i] = static_cast<T>(static_cast<float>(xRow[i]) * invRms * static_cast<float>(gBuf[i]));
      }
    }
  };
  samediff::Threads::parallel_tad(normFunc, 0, numRows);

  // --------------------------------------------------------------------------
  // Step 2: gate = normalized @ wGate   [numRows, hiddenDim] x [hiddenDim, intermediateDim]
  // Step 3: up   = normalized @ wUp     (same shape)
  // BLAS via MmulHelper — already optimal; no typed loop needed here.
  // --------------------------------------------------------------------------
  std::vector<LongType> gateShape = {batchSize, seqLen, intermediateDim};
  NDArray gate('c', gateShape, input->dataType(), context);
  MmulHelper::mmul(&normalized, wGate, &gate, 1.0, 0.0);

  std::vector<LongType> upShape = {batchSize, seqLen, intermediateDim};
  NDArray up('c', upShape, input->dataType(), context);
  MmulHelper::mmul(&normalized, wUp, &up, 1.0, 0.0);

  // --------------------------------------------------------------------------
  // Step 4: output = silu(gate) * up
  // Fused elementwise with typed buffers — no virtual dispatch.
  // --------------------------------------------------------------------------
  const LongType totalElements = numRows * intermediateDim;
  const T* gBufG = gate.bufferAsT<T>();
  const T* uBuf  = up.bufferAsT<T>();
  T*       oBuf  = output->bufferAsT<T>();

  auto siluFunc = PRAGMA_THREADS_FOR {
    PRAGMA_OMP_SIMD
    for (auto i = start; i < stop; i++) {
      const float g      = static_cast<float>(gBufG[i]);
      const float silu_g = g / (1.0f + std::exp(-g));  // g * sigmoid(g)
      oBuf[i] = static_cast<T>(silu_g * static_cast<float>(uBuf[i]));
    }
  };
  samediff::Threads::parallel_for(siluFunc, 0, totalElements);
}

void fusedRmsNormSwiGLU(NDArray* input, NDArray* gamma, NDArray* wGate, NDArray* wUp,
                        NDArray* output, float epsilon, LaunchContext* context) {
  NDArray::preparePrimaryUse({output}, {input, gamma, wGate, wUp});

  // Cast gamma/weights to input dtype when they differ (CPU: cast rather than dual template)
  NDArray* gammaToUse = gamma;
  NDArray* wGateToUse = wGate;
  NDArray* wUpToUse   = wUp;
  NDArray* gammaCast  = nullptr;
  NDArray* wGateCast  = nullptr;
  NDArray* wUpCast    = nullptr;

  if (gamma != nullptr && gamma->dataType() != input->dataType()) {
    gammaCast  = gamma->cast(input->dataType());
    gammaToUse = gammaCast;
  }
  if (wGate != nullptr && wGate->dataType() != input->dataType()) {
    wGateCast  = wGate->cast(input->dataType());
    wGateToUse = wGateCast;
  }
  if (wUp != nullptr && wUp->dataType() != input->dataType()) {
    wUpCast  = wUp->cast(input->dataType());
    wUpToUse = wUpCast;
  }

  BUILD_SINGLE_SELECTOR(input->dataType(), fusedRmsNormSwiGLU_,
                         (input, gammaToUse, wGateToUse, wUpToUse, output, epsilon, context),
                         SD_FLOAT_TYPES);

  if (gammaCast != nullptr) delete gammaCast;
  if (wGateCast != nullptr) delete wGateCast;
  if (wUpCast   != nullptr) delete wUpCast;

  NDArray::registerPrimaryUse({output}, {input, gamma, wGate, wUp});
}

void fusedRmsNormSwiGLUBackward(NDArray* input, NDArray* gamma, NDArray* wGate, NDArray* wUp,
                                 NDArray* gradOut, NDArray* gradInput, NDArray* gradGamma,
                                 NDArray* gradWGate, NDArray* gradWUp, float epsilon,
                                 LaunchContext* context) {
  THROW_EXCEPTION("fusedRmsNormSwiGLUBackward: CPU backward not yet implemented — use separate ops for training");
}

//////////////////////////////////////////////////////////////////////////////
// Fused Layer Norm Backward
//////////////////////////////////////////////////////////////////////////////

template <typename T>
static void fusedLayerNormBackward_(NDArray* input, NDArray* gain, NDArray* gradOut,
                                    NDArray* gradInput, NDArray* gradGain, NDArray* gradBias,
                                    float epsilon) {
  const LongType numRows = input->lengthOf() / input->sizeAt(-1);
  const LongType rowLen  = input->sizeAt(-1);

  // Zero out gradient accumulators
  gradGain->assign(0.0);
  if (gradBias != nullptr) gradBias->assign(0.0);

  const T* xBuf  = input->bufferAsT<T>();
  const T* gBuf  = gain->bufferAsT<T>();
  const T* doBuf = gradOut->bufferAsT<T>();
  T*       diBuf = gradInput->bufferAsT<T>();
  T*       dgBuf = gradGain->bufferAsT<T>();
  T*       dbBuf = (gradBias != nullptr) ? gradBias->bufferAsT<T>() : nullptr;

  // All tensors are assumed contiguous C-order (last stride = 1).
  // strideAt(-1) would require rank checking; simpler to rely on the last dim stride.
  const LongType xS  = input->strideAt(input->rankOf()   - 1);
  const LongType doS = gradOut->strideAt(gradOut->rankOf() - 1);
  const LongType diS = gradInput->strideAt(gradInput->rankOf() - 1);
  const LongType gS  = gain->strideAt(gain->rankOf()     - 1);
  const LongType dgS = gradGain->strideAt(gradGain->rankOf() - 1);
  const LongType dbS = (gradBias != nullptr) ? gradBias->strideAt(gradBias->rankOf() - 1) : 1;

  // Row strides
  const LongType xRowS  = (input->rankOf()    >= 2) ? input->strideAt(input->rankOf()    - 2) : rowLen * xS;
  const LongType doRowS = (gradOut->rankOf()   >= 2) ? gradOut->strideAt(gradOut->rankOf()  - 2) : rowLen * doS;
  const LongType diRowS = (gradInput->rankOf() >= 2) ? gradInput->strideAt(gradInput->rankOf() - 2) : rowLen * diS;

  // NOTE: gradGain and gradBias accumulate across rows — they must be updated atomically
  // or with per-thread buffers merged at the end. Since numRows can be large, we use a
  // simple serial accumulation with a parallel row loop for the input gradient, then
  // do a separate parallel reduction for gain/bias gradients. This matches what the
  // reference PyTorch backward does (sum over batch+seq dims).

  // Allocate per-row temp buffers for gain/bias gradient accumulation to avoid
  // atomic contention. We store [numRows, rowLen] for gain and (optionally) bias.
  // This is safe because rowLen is typically small (e.g. 4096).
  std::vector<float> dgAcc(static_cast<size_t>(numRows * rowLen), 0.0f);
  std::vector<float> dbAcc;
  if (gradBias != nullptr) dbAcc.resize(static_cast<size_t>(numRows * rowLen), 0.0f);

  auto func = PRAGMA_THREADS_FOR {
    for (auto row = start; row < stop; row++) {
      const T* xRow  = xBuf  + row * xRowS;
      const T* doRow = doBuf + row * doRowS;
      T*       diRow = diBuf + row * diRowS;

      // Welford mean/variance recomputation (float for numerical stability)
      float mean  = 0.0f;
      float M2    = 0.0f;
      float count = 0.0f;
      for (LongType i = 0; i < rowLen; i++) {
        const float val = static_cast<float>(xRow[i * xS]);
        count += 1.0f;
        const float delta  = val - mean;
        mean  += delta / count;
        M2    += delta * (val - mean);
      }
      const float variance = M2 / count;
      const float invStd   = 1.0f / std::sqrt(variance + epsilon);

      // Accumulate dvar and dmean, and fill per-row gain/bias grad accumulators
      float dvar  = 0.0f;
      float dmean = 0.0f;
      for (LongType i = 0; i < rowLen; i++) {
        const float val        = static_cast<float>(xRow[i * xS]);
        const float normalized = (val - mean) * invStd;
        const float dout       = static_cast<float>(doRow[i * doS]);
        const float g          = static_cast<float>(gBuf[i * gS]);

        // Per-element gain gradient (accumulate over rows later)
        dgAcc[static_cast<size_t>(row * rowLen + i)] = dout * normalized;
        if (gradBias != nullptr) {
          dbAcc[static_cast<size_t>(row * rowLen + i)] = dout;
        }

        const float dnorm = dout * g;
        dvar  += dnorm * (val - mean) * (-0.5f) * invStd * invStd * invStd;
        dmean += dnorm * (-invStd);
      }
      // mean(x - mean) == 0, so the dvar contribution to dmean vanishes
      // dmean += dvar * (-2.0f / count) * 0.0f;  // omitted — always zero

      // Compute input gradient
      PRAGMA_OMP_SIMD
      for (LongType i = 0; i < rowLen; i++) {
        const float val   = static_cast<float>(xRow[i * xS]);
        const float dout  = static_cast<float>(doRow[i * doS]);
        const float g     = static_cast<float>(gBuf[i * gS]);
        const float dnorm = dout * g;
        const float dx    = dnorm * invStd
                           + dvar * 2.0f * (val - mean) / count
                           + dmean / count;
        diRow[i * diS] = static_cast<T>(dx);
      }
    }
  };
  samediff::Threads::parallel_tad(func, 0, numRows);

  // Reduce gain/bias gradients across rows: dgBuf[i] = sum_row dgAcc[row*rowLen + i]
  auto gainReduceFunc = PRAGMA_THREADS_FOR {
    for (auto i = start; i < stop; i++) {
      float acc = 0.0f;
      for (LongType row = 0; row < numRows; row++) {
        acc += dgAcc[static_cast<size_t>(row * rowLen + i)];
      }
      dgBuf[i * dgS] = static_cast<T>(acc);
    }
  };
  samediff::Threads::parallel_for(gainReduceFunc, 0, rowLen);

  if (gradBias != nullptr) {
    auto biasReduceFunc = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        float acc = 0.0f;
        for (LongType row = 0; row < numRows; row++) {
          acc += dbAcc[static_cast<size_t>(row * rowLen + i)];
        }
        dbBuf[i * dbS] = static_cast<T>(acc);
      }
    };
    samediff::Threads::parallel_for(biasReduceFunc, 0, rowLen);
  }
}

void fusedLayerNormBackward(NDArray* input, NDArray* gain, NDArray* gradOut,
                             NDArray* gradInput, NDArray* gradGain, NDArray* gradBias,
                             float epsilon, LaunchContext* context) {
  NDArray::preparePrimaryUse({gradInput, gradGain, gradBias}, {input, gain, gradOut});

  // Cast gain to input dtype if needed
  NDArray* gainToUse = gain;
  NDArray* gainCast  = nullptr;
  if (gain != nullptr && gain->dataType() != input->dataType()) {
    gainCast  = gain->cast(input->dataType());
    gainToUse = gainCast;
  }

  BUILD_SINGLE_SELECTOR(input->dataType(), fusedLayerNormBackward_,
                         (input, gainToUse, gradOut, gradInput, gradGain, gradBias, epsilon),
                         SD_FLOAT_TYPES);

  if (gainCast != nullptr) delete gainCast;

  NDArray::registerPrimaryUse({gradInput, gradGain, gradBias}, {input, gain, gradOut});
}

//////////////////////////////////////////////////////////////////////////////
// Fused attention output projection
// output = reshape(attentionOutput, [B*S, H*D]) @ Wo  [+ bias]
//////////////////////////////////////////////////////////////////////////////

template <typename T>
static void biasAdd_(NDArray* output, NDArray* bias) {
  // output: [batch, seq_len, out_dim] (contiguous C-order after mmul)
  // bias:   [out_dim]
  const LongType totalRows = output->lengthOf() / output->sizeAt(-1);
  const LongType outDim    = output->sizeAt(-1);

  T*       oBuf = output->bufferAsT<T>();
  const T* bBuf = bias->bufferAsT<T>();

  const LongType oS = output->strideAt(output->rankOf() - 1);
  const LongType bS = bias->strideAt(bias->rankOf() - 1);

  auto func = PRAGMA_THREADS_FOR {
    for (auto row = start; row < stop; row++) {
      T* oRow = oBuf + row * outDim * oS;
      PRAGMA_OMP_SIMD
      for (LongType i = 0; i < outDim; i++) {
        oRow[i * oS] = static_cast<T>(
            static_cast<float>(oRow[i * oS]) + static_cast<float>(bBuf[i * bS]));
      }
    }
  };
  samediff::Threads::parallel_tad(func, 0, totalRows);
}

void fusedAttentionProjection(NDArray* attentionOutput, NDArray* Wo, NDArray* bias,
                               NDArray* output, LaunchContext* context) {
  NDArray::preparePrimaryUse({output}, {attentionOutput, Wo, bias});

  const int rank         = attentionOutput->rankOf();
  const LongType batch   = attentionOutput->sizeAt(0);
  const LongType seqLen  = attentionOutput->sizeAt(1);

  // Compute hidden_dim: either H*D (rank-4) or last dim (rank-3)
  LongType hiddenDim;
  if (rank == 4) {
    hiddenDim = attentionOutput->sizeAt(2) * attentionOutput->sizeAt(3);
  } else {
    hiddenDim = attentionOutput->sizeAt(rank - 1);
  }

  // Cast Wo to input dtype if needed
  NDArray* woToUse = Wo;
  NDArray* woCast  = nullptr;
  if (Wo->dataType() != attentionOutput->dataType()) {
    woCast  = Wo->cast(attentionOutput->dataType());
    woToUse = woCast;
  }

  // Cast bias to input dtype if needed
  NDArray* biasToUse = bias;
  NDArray* biasCast  = nullptr;
  if (bias != nullptr && bias->dataType() != attentionOutput->dataType()) {
    biasCast  = bias->cast(attentionOutput->dataType());
    biasToUse = biasCast;
  }

  // Step 1: reshape attention output to 2D [B*S, hidden_dim]
  // reshape() returns a new NDArray view; it does not copy data if the layout allows it.
  std::vector<LongType> flatShape = {batch * seqLen, hiddenDim};
  NDArray* attnFlat  = attentionOutput->reshape('c', flatShape);

  // Step 2: mmul [B*S, hidden_dim] x [hidden_dim, out_dim] -> [B*S, out_dim]
  // Output is [batch, seq_len, out_dim] so we need a 2D view of it as well.
  const LongType outDim = Wo->sizeAt(1);
  std::vector<LongType> outFlat2D = {batch * seqLen, outDim};
  NDArray* outFlat = output->reshape('c', outFlat2D);

  MmulHelper::mmul(attnFlat, woToUse, outFlat, 1.0, 0.0);

  delete attnFlat;
  delete outFlat;

  // Step 3: add bias if provided
  if (biasToUse != nullptr) {
    BUILD_SINGLE_SELECTOR(output->dataType(), biasAdd_, (output, biasToUse), SD_FLOAT_TYPES);
  }

  if (woCast  != nullptr) delete woCast;
  if (biasCast != nullptr) delete biasCast;

  NDArray::registerPrimaryUse({output}, {attentionOutput, Wo, bias});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
