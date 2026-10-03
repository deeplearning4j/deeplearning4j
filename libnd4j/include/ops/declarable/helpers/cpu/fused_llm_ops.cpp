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
#include <helpers/shape.h>
#include <execution/Threads.h>
#include <math/templatemath.h>
#include <ops/op_types.h>
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

  // Check contiguity — use EWS-free stride check for all ranks.
  // For C-contiguous arrays (the common case), last-dim stride == 1 and we can
  // use the flat buffer directly. For non-contiguous views, fall back to strided access.
  const bool xContig = shape::strideDescendingCAscendingF(input->shapeInfo());
  const bool zContig = shape::strideDescendingCAscendingF(output->shapeInfo());
  const LongType xStride = xContig ? 1 : input->strideAt(input->rankOf() - 1);
  const LongType zStride = zContig ? 1 : output->strideAt(output->rankOf() - 1);

  if (xStride == 1 && zStride == 1) {
    // Contiguous fast path
    auto func = PRAGMA_THREADS_FOR {
      PRAGMA_OMP_SIMD
      for (auto i = start; i < stop; i++) {
        const float x   = static_cast<float>(xBuf[i]);
        const float sig = 1.0f / (1.0f + sd::math::sd_exp<float, float>(-1.702f * x));
        zBuf[i] = static_cast<T>(x * sig);
      }
    };
    samediff::Threads::parallel_for(func, 0, len);
  } else {
    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        const float x   = static_cast<float>(xBuf[i * xStride]);
        const float sig = 1.0f / (1.0f + sd::math::sd_exp<float, float>(-1.702f * x));
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

  const bool xContig  = shape::strideDescendingCAscendingF(input->shapeInfo());
  const bool doContig = shape::strideDescendingCAscendingF(gradOut->shapeInfo());
  const bool diContig = shape::strideDescendingCAscendingF(gradIn->shapeInfo());
  const LongType xStride  = xContig  ? 1 : input->strideAt(input->rankOf() - 1);
  const LongType doStride = doContig ? 1 : gradOut->strideAt(gradOut->rankOf() - 1);
  const LongType diStride = diContig ? 1 : gradIn->strideAt(gradIn->rankOf() - 1);

  if (xStride == 1 && doStride == 1 && diStride == 1) {
    auto func = PRAGMA_THREADS_FOR {
      PRAGMA_OMP_SIMD
      for (auto i = start; i < stop; i++) {
        const float x    = static_cast<float>(xBuf[i]);
        const float dout = static_cast<float>(doBuf[i]);
        // d/dx[x * sigmoid(1.702*x)] = sigmoid(1.702*x) + x * 1.702 * sigmoid(1.702*x) * (1 - sigmoid(1.702*x))
        const float sig  = 1.0f / (1.0f + sd::math::sd_exp<float, float>(-1.702f * x));
        diBuf[i] = static_cast<T>(dout * (sig + x * 1.702f * sig * (1.0f - sig)));
      }
    };
    samediff::Threads::parallel_for(func, 0, len);
  } else {
    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        const float x    = static_cast<float>(xBuf[i * xStride]);
        const float dout = static_cast<float>(doBuf[i * doStride]);
        const float sig  = 1.0f / (1.0f + sd::math::sd_exp<float, float>(-1.702f * x));
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

// The layer norm kernels walk dense C-order rows (row r at r * rowLen) and read the gain and bias as dense vectors, all
// in the input's type, as the CUDA kernels do: any other layout or type goes through a copy. Rows taken at multiples
// of the second-to-last stride were wrong for every layout whose leading dimensions are not one run (an F-ordered
// rank-3 array, a permuted view). The strides decide whether a layout is dense row-major (a view's offset is already
// in bufferAsT()); the order flag does not.
//
// `a` as a dense array of the given type: `a` itself when it is one, else a copy the caller deletes.
static NDArray* denseInType(NDArray* a, DataType dataType) {
  if (a == nullptr) return nullptr;
  NDArray* typed = a->dataType() == dataType ? a : a->cast(dataType);
  if (shape::isDenseRowMajor(typed->shapeInfo())) return typed;
  NDArray* dense = typed->dup('c');
  if (typed != a) delete typed;
  return dense;
}

// The array a kernel writes in place of `a`: `a` itself when it is dense and of the given type, else a dense
// temporary the caller assigns to `a` and deletes.
static NDArray* denseOutputInType(NDArray* a, DataType dataType, LaunchContext* context) {
  if (a == nullptr || (a->dataType() == dataType && shape::isDenseRowMajor(a->shapeInfo()))) return a;
  std::vector<LongType> dims(a->shapeOf(), a->shapeOf() + a->rankOf());
  return new NDArray('c', dims, dataType, context);
}

template <typename T>
static void fusedLayerNorm_(NDArray* input, NDArray* gain, NDArray* bias, NDArray* output,
                            float epsilon) {
  // statistics in the type's aggregate type: float for the 16-bit types, the type itself otherwise (a double row
  // accumulated in float lost its precision)
  using Acc = typename simdOps::AggregateType<T>::type;
  const LongType numRows = input->lengthOf() / input->sizeAt(-1);
  const LongType rowLen  = input->sizeAt(-1);

  const T* xBuf  = input->bufferAsT<T>();
  T*       zBuf  = output->bufferAsT<T>();
  const T* gBuf  = gain->bufferAsT<T>();
  const T* bBuf  = (bias != nullptr) ? bias->bufferAsT<T>() : nullptr;

  auto func = PRAGMA_THREADS_FOR {
    for (auto row = start; row < stop; row++) {
      const T* xRow = xBuf + row * rowLen;
      T*       zRow = zBuf + row * rowLen;

      // Welford's online algorithm for mean and variance
      Acc mean  = 0;
      Acc M2    = 0;
      Acc count = 0;
      for (LongType i = 0; i < rowLen; i++) {
        const Acc val = static_cast<Acc>(xRow[i]);
        count += static_cast<Acc>(1);
        const Acc delta  = val - mean;
        mean  += delta / count;
        const Acc delta2 = val - mean;
        M2    += delta * delta2;
      }
      const Acc variance = M2 / count;
      const Acc invStd   = static_cast<Acc>(1) / sd::math::sd_sqrt<Acc, Acc>(variance + static_cast<Acc>(epsilon));

      // Normalize, scale and shift
      PRAGMA_OMP_SIMD
      for (LongType i = 0; i < rowLen; i++) {
        const Acc normalized = (static_cast<Acc>(xRow[i]) - mean) * invStd;
        Acc result = normalized * static_cast<Acc>(gBuf[i]);
        if (bBuf != nullptr) result += static_cast<Acc>(bBuf[i]);
        zRow[i] = static_cast<T>(result);
      }
    }
  };

  samediff::Threads::parallel_tad(func, 0, numRows);
}

void fusedLayerNorm(NDArray* originalInput, NDArray* originalGain, NDArray* originalBias, NDArray* originalOutput,
                    float epsilon, LaunchContext* context) {
  if (originalInput->lengthOf() == 0) return;
  const auto dataType = originalInput->dataType();

  NDArray::preparePrimaryUse({originalOutput}, {originalInput, originalGain, originalBias});

  // The CUDA helper's staging: the kernel gets dense arrays in the input's type (the op takes any float type for
  // the gain, the bias and the output), and a copy is assigned to an output that is not one.
  NDArray* input = denseInType(originalInput, dataType);
  NDArray* gain = denseInType(originalGain, dataType);
  NDArray* bias = denseInType(originalBias, dataType);
  NDArray* output = denseOutputInType(originalOutput, dataType, context);

  BUILD_SINGLE_SELECTOR(dataType, fusedLayerNorm_, (input, gain, bias, output, epsilon), SD_FLOAT_TYPES);

  if (output != originalOutput) {
    originalOutput->assign(output);
    delete output;
  }
  if (input != originalInput) delete input;
  if (gain != originalGain) delete gain;
  if (bias != originalBias) delete bias;

  NDArray::registerPrimaryUse({originalOutput}, {originalInput, originalGain, originalBias});
}

//////////////////////////////////////////////////////////////////////////////
// Fused RoPE
//////////////////////////////////////////////////////////////////////////////

void fusedRoPE(NDArray* input, NDArray* output, NDArray* positionArr,
               float freqBase, float freqScale, int ropeType, LaunchContext* context,
               int rotaryDims) {

  // On CPU, safe to read position scalar directly.
  LongType positionOffset = positionArr->e<LongType>(0);

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
    invFreq[i] = freqScale / sd::math::sd_pow<float, float, float>(freqBase, (2.0f * static_cast<float>(i)) / static_cast<float>(rotateDims));
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
            const float cosT  = sd::math::sd_cos<float, float>(theta);
            const float sinT  = sd::math::sd_sin<float, float>(theta);
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
            const float cosT  = sd::math::sd_cos<float, float>(theta);
            const float sinT  = sd::math::sd_sin<float, float>(theta);
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
          const float cosT  = sd::math::sd_cos<float, float>(theta);
          const float sinT  = sd::math::sd_sin<float, float>(theta);
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
    invFreq[i] = freqScale / sd::math::sd_pow<float, float, float>(freqBase, (2.0f * static_cast<float>(i)) / static_cast<float>(rotateDims));
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
            const float cosT  = sd::math::sd_cos<float, float>(theta);
            const float sinT  = sd::math::sd_sin<float, float>(theta);
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
            const float cosT  = sd::math::sd_cos<float, float>(theta);
            const float sinT  = sd::math::sd_sin<float, float>(theta);
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
              const float cosT  = sd::math::sd_cos<float, float>(theta);
              const float sinT  = sd::math::sd_sin<float, float>(theta);
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

  // Check contiguity using stride-based check (not EWS which is unreliable for views)
  const bool xContig = shape::strideDescendingCAscendingF(input->shapeInfo());
  const bool zContig = shape::strideDescendingCAscendingF(output->shapeInfo());
  const bool bContig = (bias != nullptr) ? shape::strideDescendingCAscendingF(bias->shapeInfo()) : true;
  const bool rContig = (residual != nullptr) ? shape::strideDescendingCAscendingF(residual->shapeInfo()) : true;
  const LongType xS = xContig ? 1 : input->strideAt(input->rankOf() - 1);
  const LongType zS = zContig ? 1 : output->strideAt(output->rankOf() - 1);
  const LongType bS = (bias != nullptr && !bContig) ? bias->strideAt(bias->rankOf() - 1) : 1;
  const LongType rS = (residual != nullptr && !rContig) ? residual->strideAt(residual->rankOf() - 1) : 1;

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
      const float invRms = 1.0f / sd::math::sd_sqrt<float, float>(sumSq / static_cast<float>(hiddenDim) + epsilon);

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
      const float silu_g = g / (1.0f + sd::math::sd_exp<float, float>(-g));  // g * sigmoid(g)
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
  // in the type's aggregate type, as the forward pass
  using Acc = typename simdOps::AggregateType<T>::type;
  const LongType numRows = input->lengthOf() / input->sizeAt(-1);
  const LongType rowLen  = input->sizeAt(-1);

  const T* xBuf  = input->bufferAsT<T>();
  const T* gBuf  = gain->bufferAsT<T>();
  const T* doBuf = gradOut->bufferAsT<T>();
  T*       diBuf = gradInput->bufferAsT<T>();
  T*       dgBuf = gradGain->bufferAsT<T>();
  T*       dbBuf = (gradBias != nullptr) ? gradBias->bufferAsT<T>() : nullptr;

  // every operand is dense, in the input's type (fusedLayerNormBackward stages any other layout or type): the rows
  // of input, gradOut and gradInput are numRows runs of rowLen elements, and gain and the gradients of the gain and
  // the bias are rowLen elements

  // each row's mean and 1 / std, for the column pass
  std::vector<Acc> rowMean(static_cast<size_t>(numRows));
  std::vector<Acc> rowInvStd(static_cast<size_t>(numRows));

  // rows: the statistics, then dx = invStd * (dnorm - mean(dnorm) - xhat * mean(dnorm * xhat)), dnorm = dy * gain
  auto rowsFunc = PRAGMA_THREADS_FOR {
    for (auto row = start; row < stop; row++) {
      const T* xRow  = xBuf  + row * rowLen;
      const T* doRow = doBuf + row * rowLen;
      T*       diRow = diBuf + row * rowLen;

      Acc mean  = 0;
      Acc M2    = 0;
      Acc count = 0;
      for (LongType i = 0; i < rowLen; i++) {
        const Acc val = static_cast<Acc>(xRow[i]);
        count += static_cast<Acc>(1);
        const Acc delta  = val - mean;
        mean  += delta / count;
        M2    += delta * (val - mean);
      }
      const Acc invStd = static_cast<Acc>(1) / sd::math::sd_sqrt<Acc, Acc>(M2 / count + static_cast<Acc>(epsilon));
      rowMean[static_cast<size_t>(row)] = mean;
      rowInvStd[static_cast<size_t>(row)] = invStd;

      Acc sumDnorm = 0;
      Acc sumDnormXhat = 0;
      for (LongType i = 0; i < rowLen; i++) {
        const Acc xhat  = (static_cast<Acc>(xRow[i]) - mean) * invStd;
        const Acc dnorm = static_cast<Acc>(doRow[i]) * static_cast<Acc>(gBuf[i]);
        sumDnorm     += dnorm;
        sumDnormXhat += dnorm * xhat;
      }

      PRAGMA_OMP_SIMD
      for (LongType i = 0; i < rowLen; i++) {
        const Acc xhat  = (static_cast<Acc>(xRow[i]) - mean) * invStd;
        const Acc dnorm = static_cast<Acc>(doRow[i]) * static_cast<Acc>(gBuf[i]);
        diRow[i] = static_cast<T>(invStd * (dnorm - sumDnorm / count - xhat * sumDnormXhat / count));
      }
    }
  };
  samediff::Threads::parallel_tad(rowsFunc, 0, numRows);

  // columns: the gain and bias gradients sum dy * xhat and dy over the rows, in row order
  auto columnsFunc = PRAGMA_THREADS_FOR {
    for (auto i = start; i < stop; i++) {
      Acc gainSum = 0;
      Acc biasSum = 0;
      for (LongType row = 0; row < numRows; row++) {
        const Acc dout = static_cast<Acc>(doBuf[row * rowLen + i]);
        gainSum += dout * (static_cast<Acc>(xBuf[row * rowLen + i]) - rowMean[static_cast<size_t>(row)]) *
                   rowInvStd[static_cast<size_t>(row)];
        biasSum += dout;
      }
      dgBuf[i] = static_cast<T>(gainSum);
      if (dbBuf != nullptr) dbBuf[i] = static_cast<T>(biasSum);
    }
  };
  samediff::Threads::parallel_for(columnsFunc, 0, rowLen);
}

void fusedLayerNormBackward(NDArray* originalInput, NDArray* originalGain, NDArray* originalGradOut,
                             NDArray* originalGradInput, NDArray* originalGradGain, NDArray* originalGradBias,
                             float epsilon, LaunchContext* context) {
  if (originalInput->lengthOf() == 0) return;
  const auto dataType = originalInput->dataType();

  NDArray::preparePrimaryUse({originalGradInput, originalGradGain, originalGradBias},
                             {originalInput, originalGain, originalGradOut});

  // The CUDA helper's staging: the kernel gets dense arrays in the input's type, and a copy is assigned to each
  // gradient that is not one (the gain and bias gradients keep their parameters' types).
  NDArray* input = denseInType(originalInput, dataType);
  NDArray* gain = denseInType(originalGain, dataType);
  NDArray* gradOut = denseInType(originalGradOut, dataType);
  NDArray* gradInput = denseOutputInType(originalGradInput, dataType, context);
  NDArray* gradGain = denseOutputInType(originalGradGain, dataType, context);
  NDArray* gradBias = denseOutputInType(originalGradBias, dataType, context);

  BUILD_SINGLE_SELECTOR(dataType, fusedLayerNormBackward_,
                         (input, gain, gradOut, gradInput, gradGain, gradBias, epsilon),
                         SD_FLOAT_TYPES);

  if (gradInput != originalGradInput) {
    originalGradInput->assign(gradInput);
    delete gradInput;
  }
  if (gradGain != originalGradGain) {
    originalGradGain->assign(gradGain);
    delete gradGain;
  }
  if (gradBias != originalGradBias) {
    originalGradBias->assign(gradBias);
    delete gradBias;
  }
  if (gain != originalGain) delete gain;
  if (gradOut != originalGradOut) delete gradOut;
  if (input != originalInput) delete input;

  NDArray::registerPrimaryUse({originalGradInput, originalGradGain, originalGradBias},
                              {originalInput, originalGain, originalGradOut});
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
  // copyToNewBuff=false: create a view sharing the same buffer when possible.
  std::vector<LongType> flatShape = {batch * seqLen, hiddenDim};
  NDArray* attnFlat  = attentionOutput->reshape('c', flatShape, false);

  // Step 2: mmul [B*S, hidden_dim] x [hidden_dim, out_dim] -> [B*S, out_dim]
  // Output is [batch, seq_len, out_dim] so we need a 2D view of it as well.
  const LongType outDim = Wo->sizeAt(1);
  std::vector<LongType> outFlat2D = {batch * seqLen, outDim};
  NDArray* outFlat = output->reshape('c', outFlat2D, false);

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
