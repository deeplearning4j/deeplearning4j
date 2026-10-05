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

// Operands of the loops below. Loops that walk their operands as dense row-major arrays of one type (the layer norm, the
// GELU, the bias + dropout + residual, the RMS norm + SwiGLU rows; row r of a layer norm at r * rowLen) take a copy of
// any other layout or type, as the CUDA kernels do: rows taken at multiples of the second-to-last stride, or elements
// at their offsets from the start of the buffer, were wrong for every layout whose leading dimensions are not one run
// (an F-ordered rank-3 array, a permuted view, a stepped view). The strides decide whether a layout is dense row-major
// (a view's offset is already in bufferAsT()); the order flag does not.
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

// The array loops that index their output through its own strides write in place of `a`: `a` itself when it is of the
// given type (the loops write that type), else a dense temporary the caller assigns to `a` and deletes. The typed loops
// are those of the four float types; for any other type the generic fallback writes through p(), which converts to
// the output's own type, so `a` is written as it is.
static NDArray* outputInType(NDArray* a, DataType dataType, LaunchContext* context) {
  const bool typedLoops = dataType == DataType::FLOAT32 || dataType == DataType::DOUBLE ||
                          dataType == DataType::HALF || dataType == DataType::BFLOAT16;
  if (a->dataType() == dataType || !typedLoops) return a;
  std::vector<LongType> dims(a->shapeOf(), a->shapeOf() + a->rankOf());
  return new NDArray('c', dims, dataType, context);
}

//////////////////////////////////////////////////////////////////////////////
// Fused GELU - x * sigmoid(1.702 * x)
//////////////////////////////////////////////////////////////////////////////

template <typename T>
static void fusedGELU_(NDArray* input, NDArray* output) {
  // in the type's aggregate type, as the CUDA kernel: float for the 16-bit types, the type itself otherwise (a double
  // computed in float lost its precision)
  using Acc = typename simdOps::AggregateType<T>::type;
  const LongType len = input->lengthOf();

  // The input and the output are dense row-major arrays of one type (fusedGELU stages any other layout or type): element
  // i of each is at offset i. The strides of a stepped view or an array of another order paired the elements of the two
  // by their offsets from the start of their buffers, which are not the same logical elements.
  const T* xBuf = input->bufferAsT<T>();
  T*       zBuf = output->bufferAsT<T>();

  auto func = PRAGMA_THREADS_FOR {
    PRAGMA_OMP_SIMD
    for (auto i = start; i < stop; i++) {
      const Acc x   = static_cast<Acc>(xBuf[i]);
      const Acc sig = static_cast<Acc>(1) /
                      (static_cast<Acc>(1) + sd::math::sd_exp<Acc, Acc>(-static_cast<Acc>(1.702) * x));
      zBuf[i] = static_cast<T>(x * sig);
    }
  };
  samediff::Threads::parallel_for(func, 0, len);
}

void fusedGELU(NDArray* originalInput, NDArray* originalOutput, LaunchContext* context) {
  if (originalInput->lengthOf() == 0) return;
  const auto dataType = originalInput->dataType();

  NDArray::preparePrimaryUse({originalOutput}, {originalInput});

  // The CUDA helper's staging: the loop gets dense row-major arrays of the input's type (in place on a dense array, the
  // input and the output are one array).
  NDArray* input = denseInType(originalInput, dataType);
  NDArray* output = denseOutputInType(originalOutput, dataType, context);

  BUILD_SINGLE_SELECTOR(dataType, fusedGELU_, (input, output), SD_FLOAT_TYPES);

  if (output != originalOutput) {
    originalOutput->assign(output);
    delete output;
  }
  if (input != originalInput) delete input;

  NDArray::registerPrimaryUse({originalOutput}, {originalInput});
}

template <typename T>
static void fusedGELUBackward_(NDArray* input, NDArray* gradOut, NDArray* gradIn) {
  using Acc = typename simdOps::AggregateType<T>::type;
  const LongType len = input->lengthOf();

  // dense row-major arrays of one type, as the forward pass (fusedGELUBackward stages any other layout or type)
  const T* xBuf  = input->bufferAsT<T>();
  const T* doBuf = gradOut->bufferAsT<T>();
  T*       diBuf = gradIn->bufferAsT<T>();

  auto func = PRAGMA_THREADS_FOR {
    PRAGMA_OMP_SIMD
    for (auto i = start; i < stop; i++) {
      const Acc x     = static_cast<Acc>(xBuf[i]);
      const Acc dout  = static_cast<Acc>(doBuf[i]);
      const Acc scale = static_cast<Acc>(1.702);
      // d/dx[x * sigmoid(1.702*x)] = sigmoid(1.702*x) + x * 1.702 * sigmoid(1.702*x) * (1 - sigmoid(1.702*x))
      const Acc sig   = static_cast<Acc>(1) / (static_cast<Acc>(1) + sd::math::sd_exp<Acc, Acc>(-scale * x));
      diBuf[i] = static_cast<T>(dout * (sig + x * scale * sig * (static_cast<Acc>(1) - sig)));
    }
  };
  samediff::Threads::parallel_for(func, 0, len);
}

void fusedGELUBackward(NDArray* originalInput, NDArray* originalGradOut, NDArray* originalGradIn,
                       LaunchContext* context) {
  if (originalInput->lengthOf() == 0) return;
  const auto dataType = originalInput->dataType();

  NDArray::preparePrimaryUse({originalGradIn}, {originalInput, originalGradOut});

  NDArray* input = denseInType(originalInput, dataType);
  NDArray* gradOut = denseInType(originalGradOut, dataType);
  NDArray* gradIn = denseOutputInType(originalGradIn, dataType, context);

  BUILD_SINGLE_SELECTOR(dataType, fusedGELUBackward_, (input, gradOut, gradIn), SD_FLOAT_TYPES);

  if (gradIn != originalGradIn) {
    originalGradIn->assign(gradIn);
    delete gradIn;
  }
  if (input != originalInput) delete input;
  if (gradOut != originalGradOut) delete gradOut;

  NDArray::registerPrimaryUse({originalGradIn}, {originalInput, originalGradOut});
}

//////////////////////////////////////////////////////////////////////////////
// Fused Layer Norm with Welford's algorithm
//////////////////////////////////////////////////////////////////////////////

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

void fusedRoPE(NDArray* input, NDArray* originalOutput, NDArray* positionArr,
               float freqBase, float freqScale, int ropeType, LaunchContext* context,
               int rotaryDims) {

  // On CPU, safe to read position scalar directly.
  LongType positionOffset = positionArr->e<LongType>(0);

  // The loops below write the input's type through the output's own strides: an output of another type is written in
  // a copy that is assigned to it.
  NDArray* output = outputInType(originalOutput, input->dataType(), context);

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
      // [batch, seq, head_dim] is one head: the head stride is never stepped through, and the element stride is the
      // last dimension's (1 only for a C-ordered, unstepped array: the F-ordered and stepped ones have another)
      xS[0]=xs[0]; xS[1]=xs[1]; xS[2]=0; xS[3]=xs[2];
      zS[0]=zs[0]; zS[1]=zs[1]; zS[2]=0; zS[3]=zs[2];
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
  if (output != originalOutput) {
    originalOutput->assign(output);
    delete output;
  }
  NDArray::registerPrimaryUse({originalOutput}, {input});
}

void fusedRoPEBackward(NDArray* gradOut, NDArray* originalGradIn, int positionOffset,
                       float freqBase, float freqScale, int ropeType, LaunchContext* context,
                       int rotaryDims) {
  // The loops below write the gradient's type through the result's own strides: a result of another type is written
  // in a copy that is assigned to it.
  NDArray* gradIn = outputInType(originalGradIn, gradOut->dataType(), context);

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
      // one head, and the element stride is the last dimension's, as in the forward pass
      gS[0]=gs[0]; gS[1]=gs[1]; gS[2]=0; gS[3]=gs[2];
      oS[0]=os[0]; oS[1]=os[1]; oS[2]=0; oS[3]=os[2];
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
  if (gradIn != originalGradIn) {
    originalGradIn->assign(gradIn);
    delete gradIn;
  }
  NDArray::registerPrimaryUse({originalGradIn}, {gradOut});
}

//////////////////////////////////////////////////////////////////////////////
// Fused RoPE with pre-computed cos/sin (cached variant)
//////////////////////////////////////////////////////////////////////////////

void fusedRoPECached(NDArray* input, NDArray* cosValues, NDArray* sinValues,
                     NDArray* originalOutput, int ropeType, LaunchContext* context) {
  // The loops below write the input's type through the output's own strides: an output of another type is written in
  // a copy that is assigned to it.
  NDArray* output = outputInType(originalOutput, input->dataType(), context);

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
      // [batch, seq, head_dim] is one head: the head stride is never stepped through, and the element stride is the
      // last dimension's (1 only for a C-ordered, unstepped array: the F-ordered and stepped ones have another)
      xS[0]=xs[0]; xS[1]=xs[1]; xS[2]=0; xS[3]=xs[2];
      zS[0]=zs[0]; zS[1]=zs[1]; zS[2]=0; zS[3]=zs[2];
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

  const float* cosPtr = cosF->bufferAsT<float>();
  const float* sinPtr = sinF->bufferAsT<float>();

  // Extract actual strides from the cos and sin NDArrays, each from its own array (the tables have one shape, but
  // their layouts may differ: a slice of a larger cache and a dense copy of the other table, say).
  // cos/sin shape: [S, halfDim] (rank2), [B, S, halfDim] (rank3), or [B, S, 1, halfDim] (rank4).
  // These may be slices of a larger cache (non-contiguous), so we MUST use the real
  // NDArray strides rather than assuming contiguous layout (the regression bug).
  auto tableStrides = [](NDArray* table, LongType& batchStride, LongType& seqStride, LongType& halfDimStride) {
    const int tableRank = table->rankOf();
    batchStride   = 0;  // batch stride (0 = broadcast across batch for rank-2)
    seqStride     = 0;  // seq stride
    halfDimStride = 1;  // innermost (halfDim element) stride
    if (tableRank == 2) {
      // [S, halfDim] — no batch dim; broadcast across batch by keeping the batch stride 0
      seqStride     = table->strideAt(0);
      halfDimStride = table->strideAt(1);
    } else if (tableRank == 3) {
      // [B, S, halfDim]
      batchStride   = table->strideAt(0);
      seqStride     = table->strideAt(1);
      halfDimStride = table->strideAt(2);
    } else if (tableRank >= 4) {
      // [B, S, 1, halfDim] — skip the broadcast head dim (stride index 3 is innermost)
      batchStride   = table->strideAt(0);
      seqStride     = table->strideAt(1);
      halfDimStride = table->strideAt(3);
    }
  };
  LongType cosStride0, cosStride1, cosStride2, sinStride0, sinStride1, sinStride2;
  tableStrides(cosF, cosStride0, cosStride1, cosStride2);
  tableStrides(sinF, sinStride0, sinStride1, sinStride2);

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

        // Offset into the cos and sin tables using each one's real strides (handles non-contiguous slices).
        const float* cPtr = cosPtr + b * cosStride0 + s * cosStride1;
        const float* sPtr = sinPtr + b * sinStride0 + s * sinStride1;

        const T* xPtr = xBuf + b * xS[0] + s * xS[1] + h * xS[2];
        T*       zPtr = zBuf + b * zS[0] + s * zS[1] + h * zS[2];

        if (ropeType == 1) {  // NeoX interleaved
          for (LongType i = 0; i < halfDim; ++i) {
            const float cosT = cPtr[i * cosStride2];
            const float sinT = sPtr[i * sinStride2];
            const float x0 = static_cast<float>(xPtr[(2 * i)     * xS[3]]);
            const float x1 = static_cast<float>(xPtr[(2 * i + 1) * xS[3]]);
            zPtr[(2 * i)     * zS[3]] = static_cast<T>(x0 * cosT - x1 * sinT);
            zPtr[(2 * i + 1) * zS[3]] = static_cast<T>(x0 * sinT + x1 * cosT);
          }
        } else {  // Standard (LLaMA / GPT-J)
          for (LongType i = 0; i < halfDim; ++i) {
            const float cosT = cPtr[i * cosStride2];
            const float sinT = sPtr[i * sinStride2];
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
            const LongType cosOff = b * cosStride0 + s * cosStride1;
            const LongType sinOff = b * sinStride0 + s * sinStride1;
            for (LongType i = 0; i < halfDim; i++) {
              const float cosT = cosPtr[cosOff + i * cosStride2];
              const float sinT = sinPtr[sinOff + i * sinStride2];
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

  if (output != originalOutput) {
    originalOutput->assign(output);
    delete output;
  }
  NDArray::registerPrimaryUse({originalOutput}, {input, cosValues, sinValues});
}

//////////////////////////////////////////////////////////////////////////////
// Fused Bias + Dropout + Residual
//////////////////////////////////////////////////////////////////////////////

template <typename T>
static void fusedBiasDropoutResidual_(NDArray* input, NDArray* bias, NDArray* residual,
                                      NDArray* output, float dropoutProb, LongType seed,
                                      bool training) {
  // sums in the type's aggregate type: float for the 16-bit types, the type itself otherwise (a double sum rounded
  // through float lost its precision)
  using Acc = typename simdOps::AggregateType<T>::type;
  const LongType totalElements = input->lengthOf();
  const LongType biasLen       = (bias != nullptr) ? bias->lengthOf() : 1;

  // Every operand is a dense row-major array of one type (fusedBiasDropoutResidual stages any other layout or type):
  // element i of the input, the residual and the output is at offset i, and the bias repeats along them.
  const T* xBuf   = input->bufferAsT<T>();
  T*       zBuf   = output->bufferAsT<T>();
  const T* bBuf   = (bias     != nullptr) ? bias->bufferAsT<T>()     : nullptr;
  const T* rBuf   = (residual != nullptr) ? residual->bufferAsT<T>() : nullptr;

  auto func = PRAGMA_THREADS_FOR {
    std::mt19937_64 localRng(static_cast<uint64_t>(seed) + static_cast<uint64_t>(start));
    std::uniform_real_distribution<float> localDist(0.0f, 1.0f);

    for (auto i = start; i < stop; i++) {
      Acc val = static_cast<Acc>(xBuf[i]);

      if (bBuf != nullptr) {
        val += static_cast<Acc>(bBuf[i % biasLen]);
      }

      if (training && dropoutProb > 0.0f) {
        const float r = localDist(localRng);
        if (r < dropoutProb) {
          val = static_cast<Acc>(0);
        } else {
          val /= static_cast<Acc>(1.0f - dropoutProb);
        }
      }

      if (rBuf != nullptr) {
        val += static_cast<Acc>(rBuf[i]);
      }

      zBuf[i] = static_cast<T>(val);
    }
  };

  samediff::Threads::parallel_for(func, 0, totalElements);
}

void fusedBiasDropoutResidual(NDArray* originalInput, NDArray* originalBias, NDArray* originalResidual,
                              NDArray* originalOutput, float dropoutProb, LongType seed,
                              bool training, LaunchContext* context) {
  const auto dataType = originalInput->dataType();
  NDArray::preparePrimaryUse({originalOutput}, {originalInput, originalBias, originalResidual});

  // The CUDA helper's staging: the loops get dense row-major arrays of the input's type (a stepped view, an F-ordered
  // or permuted array, or an operand of another type goes through a copy; so does an output that is not one).
  NDArray* input = denseInType(originalInput, dataType);
  NDArray* bias = denseInType(originalBias, dataType);
  NDArray* residual = denseInType(originalResidual, dataType);
  NDArray* output = denseOutputInType(originalOutput, dataType, context);

  BUILD_SINGLE_SELECTOR(dataType, fusedBiasDropoutResidual_,
                         (input, bias, residual, output, dropoutProb, seed, training),
                         SD_FLOAT_TYPES);

  if (output != originalOutput) {
    originalOutput->assign(output);
    delete output;
  }
  if (input != originalInput) delete input;
  if (bias != originalBias) delete bias;
  if (residual != originalResidual) delete residual;

  NDArray::registerPrimaryUse({originalOutput}, {originalInput, originalBias, originalResidual});
}

//////////////////////////////////////////////////////////////////////////////
// Fused RMS Norm + SwiGLU
// Computes: silu(rms_norm(x) @ W_gate) * (rms_norm(x) @ W_up)
//////////////////////////////////////////////////////////////////////////////

// input [batch, seq_len, hidden_dim], gamma [hidden_dim] and output [batch, seq_len, intermediate_dim] are dense
// row-major arrays of the input's type here (fusedRmsNormSwiGLU stages any other layout or type), wGate and wUp
// [hidden_dim, intermediate_dim] are of that type and go to the matmuls in whatever layout they have. The normalized
// rows, the gate and the up projection are stored in the input's type, and every sum and the SiLU are computed in its
// aggregate type: float for the 16-bit types, the type itself otherwise (a double row summed and normalized in float
// lost its precision).
template <typename T>
static void fusedRmsNormSwiGLU_(NDArray* input, NDArray* gamma, NDArray* wGate, NDArray* wUp,
                                 NDArray* output, float epsilon, LaunchContext* context) {
  using Acc = typename simdOps::AggregateType<T>::type;

  const LongType hiddenDim       = input->sizeAt(2);
  const LongType intermediateDim = wGate->sizeAt(1);
  const LongType numRows         = input->sizeAt(0) * input->sizeAt(1);

  // --------------------------------------------------------------------------
  // Step 1: RMS norm + gamma scaling
  // normalized: [numRows, hiddenDim], contiguous.
  // --------------------------------------------------------------------------
  std::vector<LongType> normShape = {numRows, hiddenDim};
  NDArray normalized('c', normShape, input->dataType(), context);

  const T* xBuf  = input->bufferAsT<T>();
  T*       nBuf  = normalized.bufferAsT<T>();
  const T* gBuf  = gamma->bufferAsT<T>();

  auto normFunc = PRAGMA_THREADS_FOR {
    for (auto row = start; row < stop; row++) {
      const T* xRow = xBuf + row * hiddenDim;
      T*       nRow = nBuf + row * hiddenDim;

      // Single pass: sum-of-squares in the aggregate type (float for FP16: no overflow)
      Acc sumSq = 0;
      for (LongType i = 0; i < hiddenDim; i++) {
        const Acc v = static_cast<Acc>(xRow[i]);
        sumSq += v * v;
      }
      const Acc invRms = static_cast<Acc>(1) /
                         sd::math::sd_sqrt<Acc, Acc>(sumSq / static_cast<Acc>(hiddenDim) + static_cast<Acc>(epsilon));

      // Normalize and apply gamma
      PRAGMA_OMP_SIMD
      for (LongType i = 0; i < hiddenDim; i++) {
        nRow[i] = static_cast<T>(static_cast<Acc>(xRow[i]) * invRms * static_cast<Acc>(gBuf[i]));
      }
    }
  };
  samediff::Threads::parallel_tad(normFunc, 0, numRows);

  // --------------------------------------------------------------------------
  // Step 2: gate = normalized @ wGate   [numRows, hiddenDim] x [hiddenDim, intermediateDim]
  // Step 3: up   = normalized @ wUp     (same shape)
  // BLAS via MmulHelper — already optimal; no typed loop needed here.
  // --------------------------------------------------------------------------
  std::vector<LongType> projectedShape = {numRows, intermediateDim};
  NDArray gate('c', projectedShape, input->dataType(), context);
  MmulHelper::mmul(&normalized, wGate, &gate, 1.0, 0.0);

  NDArray up('c', projectedShape, input->dataType(), context);
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
      const Acc g      = static_cast<Acc>(gBufG[i]);
      const Acc silu_g = g / (static_cast<Acc>(1) + sd::math::sd_exp<Acc, Acc>(-g));  // g * sigmoid(g)
      oBuf[i] = static_cast<T>(silu_g * static_cast<Acc>(uBuf[i]));
    }
  };
  samediff::Threads::parallel_for(siluFunc, 0, totalElements);
}

void fusedRmsNormSwiGLU(NDArray* originalInput, NDArray* originalGamma, NDArray* originalWGate,
                        NDArray* originalWUp, NDArray* originalOutput, float epsilon, LaunchContext* context) {
  if (originalInput->lengthOf() == 0 || originalOutput->lengthOf() == 0) return;
  const auto dataType = originalInput->dataType();

  NDArray::preparePrimaryUse({originalOutput}, {originalInput, originalGamma, originalWGate, originalWUp});

  // The CUDA helper's staging: the loops get dense row-major arrays in the input's type (the op takes any float type
  // for the gamma, the weights and the output), the weights in that type in whatever layout (the matmuls deal with
  // layouts themselves).
  NDArray* input = denseInType(originalInput, dataType);
  NDArray* gamma = denseInType(originalGamma, dataType);
  NDArray* wGate = originalWGate->dataType() == dataType ? originalWGate : originalWGate->cast(dataType);
  NDArray* wUp   = originalWUp->dataType() == dataType ? originalWUp : originalWUp->cast(dataType);
  NDArray* output = denseOutputInType(originalOutput, dataType, context);

  BUILD_SINGLE_SELECTOR(dataType, fusedRmsNormSwiGLU_,
                         (input, gamma, wGate, wUp, output, epsilon, context),
                         SD_FLOAT_TYPES);

  if (output != originalOutput) {
    originalOutput->assign(output);
    delete output;
  }
  if (input != originalInput) delete input;
  if (gamma != originalGamma) delete gamma;
  if (wGate != originalWGate) delete wGate;
  if (wUp != originalWUp) delete wUp;

  NDArray::registerPrimaryUse({originalOutput}, {originalInput, originalGamma, originalWGate, originalWUp});
}

// Backward of the fused RMS norm + SwiGLU, the CUDA helper's algorithm. The forward pass is recomputed from the input
// (normalized rows, gate and up projections) and the gradients follow in closed form: with y = silu(gate) * up,
// gate = n @ wGate, up = n @ wUp, n = x * invRms * gamma and dy the gradient of y,
//   dUp = dy * silu(gate)   dGate = dy * up * silu'(gate)
//   dWGate = n^T @ dGate    dWUp = n^T @ dUp    dn = dGate @ wGate^T + dUp @ wUp^T
//   dGamma = sum over rows of dn * x * invRms    dx = invRms * (dn * gamma - x * invRms^2 * mean(dn * gamma * x))
// Everything is computed in the aggregate type of the input's (float for HALF and BFLOAT16, the type itself otherwise):
// the operands are dense copies in that type where they are not already, and each gradient is rounded to its own type
// once, as it is assigned back.
template <typename T>
static void fusedRmsNormSwiGLUBackward_(NDArray* originalInput, NDArray* originalGamma, NDArray* originalWGate,
                                        NDArray* originalWUp, NDArray* originalGradOut, NDArray* originalGradInput,
                                        NDArray* originalGradGamma, NDArray* originalGradWGate,
                                        NDArray* originalGradWUp, float epsilon, LaunchContext* context) {
  using Acc = typename simdOps::AggregateType<T>::type;
  const DataType accType = DataTypeUtils::fromT<Acc>();

  const LongType hiddenDim       = originalInput->sizeAt(2);
  const LongType intermediateDim = originalWGate->sizeAt(1);
  const LongType numRows         = originalInput->sizeAt(0) * originalInput->sizeAt(1);
  const LongType totalElements   = numRows * intermediateDim;

  NDArray* input     = denseInType(originalInput, accType);
  NDArray* gamma     = denseInType(originalGamma, accType);
  NDArray* gradOut   = denseInType(originalGradOut, accType);
  NDArray* wGate     = originalWGate->dataType() == accType ? originalWGate : originalWGate->cast(accType);
  NDArray* wUp       = originalWUp->dataType() == accType ? originalWUp : originalWUp->cast(accType);
  NDArray* gradInput = denseOutputInType(originalGradInput, accType, context);
  NDArray* gradGamma = denseOutputInType(originalGradGamma, accType, context);
  NDArray* gradWGate = denseOutputInType(originalGradWGate, accType, context);
  NDArray* gradWUp   = denseOutputInType(originalGradWUp, accType, context);

  std::vector<LongType> rowsShape      = {numRows, hiddenDim};
  std::vector<LongType> projectedShape = {numRows, intermediateDim};
  NDArray normalized('c', rowsShape, accType, context);
  NDArray gate('c', projectedShape, accType, context);
  NDArray up('c', projectedShape, accType, context);
  NDArray gradNormalized('c', rowsShape, accType, context);

  const Acc* xBuf  = input->bufferAsT<Acc>();
  const Acc* gBuf  = gamma->bufferAsT<Acc>();
  const Acc* doBuf = gradOut->bufferAsT<Acc>();
  Acc*       nBuf  = normalized.bufferAsT<Acc>();

  // the normalized rows and each row's 1 / rms
  std::vector<Acc> invRms(static_cast<size_t>(numRows));
  auto normFunc = PRAGMA_THREADS_FOR {
    for (auto row = start; row < stop; row++) {
      const Acc* xRow = xBuf + row * hiddenDim;
      Acc*       nRow = nBuf + row * hiddenDim;

      Acc sumSq = 0;
      for (LongType i = 0; i < hiddenDim; i++) sumSq += xRow[i] * xRow[i];
      const Acc inv = static_cast<Acc>(1) /
                      sd::math::sd_sqrt<Acc, Acc>(sumSq / static_cast<Acc>(hiddenDim) + static_cast<Acc>(epsilon));
      invRms[static_cast<size_t>(row)] = inv;

      PRAGMA_OMP_SIMD
      for (LongType i = 0; i < hiddenDim; i++) nRow[i] = xRow[i] * inv * gBuf[i];
    }
  };
  samediff::Threads::parallel_tad(normFunc, 0, numRows);

  // gate = normalized @ W_gate, up = normalized @ W_up
  MmulHelper::mmul(&normalized, wGate, &gate, 1.0, 0.0);
  MmulHelper::mmul(&normalized, wUp, &up, 1.0, 0.0);

  // gate and up become dGate and dUp (each element depends on its own values only)
  Acc* gateBuf = gate.bufferAsT<Acc>();
  Acc* upBuf   = up.bufferAsT<Acc>();
  auto swigluFunc = PRAGMA_THREADS_FOR {
    PRAGMA_OMP_SIMD
    for (auto i = start; i < stop; i++) {
      const Acc g  = gateBuf[i];
      const Acc u  = upBuf[i];
      const Acc dy = doBuf[i];
      const Acc s  = static_cast<Acc>(1) / (static_cast<Acc>(1) + sd::math::sd_exp<Acc, Acc>(-g));
      upBuf[i]   = dy * g * s;
      gateBuf[i] = dy * u * s * (static_cast<Acc>(1) + g * (static_cast<Acc>(1) - s));
    }
  };
  samediff::Threads::parallel_for(swigluFunc, 0, totalElements);

  // dWGate = normalized^T @ dGate, dWUp = normalized^T @ dUp
  MmulHelper::matmul(&normalized, &gate, gradWGate, true, false, 1.0, 0.0);
  MmulHelper::matmul(&normalized, &up, gradWUp, true, false, 1.0, 0.0);

  // dn = dGate @ W_gate^T + dUp @ W_up^T
  MmulHelper::matmul(&gate, wGate, &gradNormalized, false, true, 1.0, 0.0);
  MmulHelper::matmul(&up, wUp, &gradNormalized, false, true, 1.0, 1.0);

  // dx: one row at a time
  const Acc* dnBuf = gradNormalized.bufferAsT<Acc>();
  Acc*       dxBuf = gradInput->bufferAsT<Acc>();
  auto rowsFunc = PRAGMA_THREADS_FOR {
    for (auto row = start; row < stop; row++) {
      const Acc* xRow  = xBuf + row * hiddenDim;
      const Acc* dnRow = dnBuf + row * hiddenDim;
      Acc*       dxRow = dxBuf + row * hiddenDim;
      const Acc  inv   = invRms[static_cast<size_t>(row)];

      Acc sum = 0;
      for (LongType i = 0; i < hiddenDim; i++) sum += dnRow[i] * gBuf[i] * xRow[i];
      const Acc coefficient = inv * inv * sum / static_cast<Acc>(hiddenDim);

      PRAGMA_OMP_SIMD
      for (LongType i = 0; i < hiddenDim; i++) dxRow[i] = inv * (dnRow[i] * gBuf[i] - xRow[i] * coefficient);
    }
  };
  samediff::Threads::parallel_tad(rowsFunc, 0, numRows);

  // dGamma sums dn * x * invRms over the rows, in row order: each thread owns a run of columns and walks the rows
  // (a contiguous stretch of each row), so the order, and so the result, does not depend on the number of threads
  Acc* dgBuf = gradGamma->bufferAsT<Acc>();
  auto columnsFunc = PRAGMA_THREADS_FOR {
    for (auto i = start; i < stop; i++) dgBuf[i] = 0;
    for (LongType row = 0; row < numRows; row++) {
      const Acc* xRow  = xBuf + row * hiddenDim;
      const Acc* dnRow = dnBuf + row * hiddenDim;
      const Acc  inv   = invRms[static_cast<size_t>(row)];
      for (auto i = start; i < stop; i++) dgBuf[i] += dnRow[i] * xRow[i] * inv;
    }
  };
  samediff::Threads::parallel_for(columnsFunc, 0, hiddenDim);

  // the gradients that were computed in a dense copy, in the input's aggregate type, go to their own arrays
  if (gradInput != originalGradInput) {
    originalGradInput->assign(gradInput);
    delete gradInput;
  }
  if (gradGamma != originalGradGamma) {
    originalGradGamma->assign(gradGamma);
    delete gradGamma;
  }
  if (gradWGate != originalGradWGate) {
    originalGradWGate->assign(gradWGate);
    delete gradWGate;
  }
  if (gradWUp != originalGradWUp) {
    originalGradWUp->assign(gradWUp);
    delete gradWUp;
  }
  if (input != originalInput) delete input;
  if (gamma != originalGamma) delete gamma;
  if (gradOut != originalGradOut) delete gradOut;
  if (wGate != originalWGate) delete wGate;
  if (wUp != originalWUp) delete wUp;
}

void fusedRmsNormSwiGLUBackward(NDArray* input, NDArray* gamma, NDArray* wGate, NDArray* wUp,
                                 NDArray* gradOut, NDArray* gradInput, NDArray* gradGamma,
                                 NDArray* gradWGate, NDArray* gradWUp, float epsilon,
                                 LaunchContext* context) {
  if (input->lengthOf() == 0 || gradOut->lengthOf() == 0) return;

  NDArray::preparePrimaryUse({gradInput, gradGamma, gradWGate, gradWUp}, {input, gamma, wGate, wUp, gradOut});
  BUILD_SINGLE_SELECTOR(input->dataType(), fusedRmsNormSwiGLUBackward_,
                         (input, gamma, wGate, wUp, gradOut, gradInput, gradGamma, gradWGate, gradWUp, epsilon,
                          context),
                         SD_FLOAT_TYPES);
  NDArray::registerPrimaryUse({gradInput, gradGamma, gradWGate, gradWUp}, {input, gamma, wGate, wUp, gradOut});
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
  // sums in the type's aggregate type, as the CUDA kernel
  using Acc = typename simdOps::AggregateType<T>::type;
  // output: dense [batch, seq_len, out_dim] (fusedAttentionProjection stages any other layout or type), after the mmul
  // bias:   dense [out_dim] of the same type
  const LongType outDim    = output->sizeAt(-1);
  const LongType totalRows = output->lengthOf() / outDim;

  T*       oBuf = output->bufferAsT<T>();
  const T* bBuf = bias->bufferAsT<T>();

  auto func = PRAGMA_THREADS_FOR {
    for (auto row = start; row < stop; row++) {
      T* oRow = oBuf + row * outDim;
      PRAGMA_OMP_SIMD
      for (LongType i = 0; i < outDim; i++) {
        oRow[i] = static_cast<T>(static_cast<Acc>(oRow[i]) + static_cast<Acc>(bBuf[i]));
      }
    }
  };
  samediff::Threads::parallel_tad(func, 0, totalRows);
}

void fusedAttentionProjection(NDArray* originalAttentionOutput, NDArray* Wo, NDArray* originalBias,
                               NDArray* originalOutput, LaunchContext* context) {
  NDArray::preparePrimaryUse({originalOutput}, {originalAttentionOutput, Wo, originalBias});

  // The CUDA helper's staging: the reshapes below are views of dense arrays (the reshape of any other layout returns a
  // copy, and the product written into a copy of the output would be lost) and the bias loop walks the output and the
  // bias as dense arrays of the output's type, so any other layout or type goes through a dense copy.
  NDArray* attentionOutput = denseInType(originalAttentionOutput, originalAttentionOutput->dataType());
  NDArray* output = denseOutputInType(originalOutput, originalOutput->dataType(), context);
  NDArray* bias = denseInType(originalBias, originalOutput->dataType());

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

  // Step 1: reshape attention output to 2D [B*S, hidden_dim]
  // copyToNewBuff=false: create a view sharing the same buffer (the attention output is dense here).
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
  if (bias != nullptr) {
    BUILD_SINGLE_SELECTOR(output->dataType(), biasAdd_, (output, bias), SD_FLOAT_TYPES);
  }

  if (woCast  != nullptr) delete woCast;

  if (output != originalOutput) {
    originalOutput->assign(output);
    delete output;
  }
  if (attentionOutput != originalAttentionOutput) delete attentionOutput;
  if (bias != originalBias) delete bias;

  NDArray::registerPrimaryUse({originalOutput}, {originalAttentionOutput, Wo, originalBias});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
