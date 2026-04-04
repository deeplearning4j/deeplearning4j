/* ******************************************************************************
 *
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
// Dual RoPE (Gemma 4) CUDA implementation
//

#include <cuda_runtime.h>
#include <helpers/DebugHelper.h>
#include <array/NDArray.h>
#include <execution/cuda/LaunchDims.h>
#include <types/float16.h>
#include <ops/declarable/helpers/dual_rope.h>

namespace sd {
namespace ops {
namespace helpers {

///////////////////////////////////////////////////////////////////////////////
// CUDA kernel: each thread handles one (batch, seq, head, dim_pair)
///////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void dualRoPEKernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const LongType batch,
    const LongType seqLen,
    const LongType numHeads,
    const LongType headDim,
    const int positionOffset,
    const double freqBase,
    const double freqScale,
    const LongType xStride0, const LongType xStride1,
    const LongType xStride2, const LongType xStride3,
    const LongType zStride0, const LongType zStride1,
    const LongType zStride2, const LongType zStride3) {

  const LongType halfDim = headDim / 2;
  const LongType totalPairs = batch * seqLen * numHeads * halfDim;

  // Grid-stride loop
  for (LongType tid = blockIdx.x * blockDim.x + threadIdx.x;
       tid < totalPairs;
       tid += gridDim.x * blockDim.x) {

    // Decompose linear index -> (b, s, h, i)
    const LongType i   = tid % halfDim;
    const LongType tmp1 = tid / halfDim;
    const LongType h   = tmp1 % numHeads;
    const LongType tmp2 = tmp1 / numHeads;
    const LongType s   = tmp2 % seqLen;
    const LongType b   = tmp2 / seqLen;

    const LongType position = static_cast<LongType>(positionOffset) + s;

    // theta_i = freq_base ^ (-2i / head_dim) * freq_scale
    const double exponent = -2.0 * static_cast<double>(i) / static_cast<double>(headDim);
    const double theta_i = pow(freqBase, exponent) * freqScale;
    const double angle = static_cast<double>(position) * theta_i;

    const T cosVal = static_cast<T>(cos(angle));
    const T sinVal = static_cast<T>(sin(angle));

    // Read input pair using strides
    const LongType xOffset = b * xStride0 + s * xStride1 + h * xStride2;
    const T x0 = input[xOffset + 2 * i * xStride3];
    const T x1 = input[xOffset + (2 * i + 1) * xStride3];

    // Write output pair using strides
    const LongType zOffset = b * zStride0 + s * zStride1 + h * zStride2;
    output[zOffset + 2 * i * zStride3]       = x0 * cosVal - x1 * sinVal;
    output[zOffset + (2 * i + 1) * zStride3] = x0 * sinVal + x1 * cosVal;
  }
}

///////////////////////////////////////////////////////////////////////////////
// Launcher
///////////////////////////////////////////////////////////////////////////////
template <typename T>
static void launchDualRoPEKernel(
    const T* input, T* output,
    LongType batch, LongType seqLen, LongType numHeads, LongType headDim,
    int positionOffset, double freqBase, double freqScale,
    LongType xStride0, LongType xStride1, LongType xStride2, LongType xStride3,
    LongType zStride0, LongType zStride1, LongType zStride2, LongType zStride3,
    cudaStream_t stream) {

  const LongType halfDim = headDim / 2;
  const LongType totalPairs = batch * seqLen * numHeads * halfDim;

  dim3 launchDims = getLaunchDims("dual_rope");
  int threadsPerBlock = launchDims.y;
  int blocksPerGrid = static_cast<int>((totalPairs + threadsPerBlock - 1) / threadsPerBlock);
  // Cap to the configured grid size
  if (blocksPerGrid > static_cast<int>(launchDims.x)) {
    blocksPerGrid = static_cast<int>(launchDims.x);
  }

  dualRoPEKernel<T><<<blocksPerGrid, threadsPerBlock, launchDims.z, stream>>>(
      input, output,
      batch, seqLen, numHeads, headDim,
      positionOffset, freqBase, freqScale,
      xStride0, xStride1, xStride2, xStride3,
      zStride0, zStride1, zStride2, zStride3);

  DebugHelper::checkGlobalErrorCode("dualRoPEKernel failed");
}

// Explicit instantiations
template void launchDualRoPEKernel<float>(
    const float*, float*,
    LongType, LongType, LongType, LongType,
    int, double, double,
    LongType, LongType, LongType, LongType,
    LongType, LongType, LongType, LongType,
    cudaStream_t);

template void launchDualRoPEKernel<double>(
    const double*, double*,
    LongType, LongType, LongType, LongType,
    int, double, double,
    LongType, LongType, LongType, LongType,
    LongType, LongType, LongType, LongType,
    cudaStream_t);

template void launchDualRoPEKernel<float16>(
    const float16*, float16*,
    LongType, LongType, LongType, LongType,
    int, double, double,
    LongType, LongType, LongType, LongType,
    LongType, LongType, LongType, LongType,
    cudaStream_t);

///////////////////////////////////////////////////////////////////////////////
// Public interface
///////////////////////////////////////////////////////////////////////////////
void dualRoPE(LaunchContext* context, NDArray* input, NDArray* output,
              int attentionType, int positionOffset,
              double localFreqBase, double globalFreqBase,
              double localFreqScale, double globalFreqScale) {

  // Select freq_base and freq_scale based on attention type
  const double freqBase  = (attentionType == 0) ? localFreqBase : globalFreqBase;
  const double freqScale = (attentionType == 0) ? localFreqScale : globalFreqScale;

  const LongType batch    = input->sizeAt(0);
  const LongType seqLen   = input->sizeAt(1);
  const LongType numHeads = input->sizeAt(2);
  const LongType headDim  = input->sizeAt(3);

  NDArray::prepareSpecialUse({output}, {input});

  auto stream = context->getCudaStream();
  auto dtype = input->dataType();

  if (dtype == DataType::FLOAT32) {
    launchDualRoPEKernel<float>(
        reinterpret_cast<const float*>(input->specialBuffer()),
        reinterpret_cast<float*>(output->specialBuffer()),
        batch, seqLen, numHeads, headDim,
        positionOffset, freqBase, freqScale,
        input->strideAt(0), input->strideAt(1), input->strideAt(2), input->strideAt(3),
        output->strideAt(0), output->strideAt(1), output->strideAt(2), output->strideAt(3),
        *stream);
  } else if (dtype == DataType::DOUBLE) {
    launchDualRoPEKernel<double>(
        reinterpret_cast<const double*>(input->specialBuffer()),
        reinterpret_cast<double*>(output->specialBuffer()),
        batch, seqLen, numHeads, headDim,
        positionOffset, freqBase, freqScale,
        input->strideAt(0), input->strideAt(1), input->strideAt(2), input->strideAt(3),
        output->strideAt(0), output->strideAt(1), output->strideAt(2), output->strideAt(3),
        *stream);
  } else if (dtype == DataType::HALF) {
    launchDualRoPEKernel<float16>(
        reinterpret_cast<const float16*>(input->specialBuffer()),
        reinterpret_cast<float16*>(output->specialBuffer()),
        batch, seqLen, numHeads, headDim,
        positionOffset, freqBase, freqScale,
        input->strideAt(0), input->strideAt(1), input->strideAt(2), input->strideAt(3),
        output->strideAt(0), output->strideAt(1), output->strideAt(2), output->strideAt(3),
        *stream);
  } else {
    THROW_EXCEPTION("dualRoPE CUDA: Unsupported data type");
  }

  NDArray::registerSpecialUse({output}, {input});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
