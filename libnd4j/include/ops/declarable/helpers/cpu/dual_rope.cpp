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
// Dual RoPE (Gemma 4) CPU implementation
//

#include <execution/Threads.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/dual_rope.h>

#if NOT_EXCLUDED(OP_dual_rope)
namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void dualRoPE_(NDArray* input, NDArray* output, int attentionType, int positionOffset,
                      double localFreqBase, double globalFreqBase,
                      double localFreqScale, double globalFreqScale) {
  // Input shape: [batch, seq_len, num_heads, head_dim]
  const LongType batch    = input->sizeAt(0);
  const LongType seqLen   = input->sizeAt(1);
  const LongType numHeads = input->sizeAt(2);
  const LongType headDim  = input->sizeAt(3);
  const LongType halfDim  = headDim / 2;

  // Select freq_base and freq_scale based on attention type
  const double freqBase  = (attentionType == 0) ? localFreqBase : globalFreqBase;
  const double freqScale = (attentionType == 0) ? localFreqScale : globalFreqScale;

  const T* x = input->bufferAsT<T>();
  T* z = output->bufferAsT<T>();

  // Strides for input and output
  const LongType xStride0 = input->strideAt(0);   // batch stride
  const LongType xStride1 = input->strideAt(1);   // seq stride
  const LongType xStride2 = input->strideAt(2);   // head stride
  const LongType xStride3 = input->strideAt(3);   // dim stride

  const LongType zStride0 = output->strideAt(0);
  const LongType zStride1 = output->strideAt(1);
  const LongType zStride2 = output->strideAt(2);
  const LongType zStride3 = output->strideAt(3);

  // Parallel over batch * seq_len * num_heads
  const LongType outerSize = batch * seqLen * numHeads;

  auto func = PRAGMA_THREADS_FOR {
    for (auto idx = start; idx < stop; ++idx) {
      // Decompose linear index -> (b, s, h)
      const LongType h = idx % numHeads;
      const LongType tmp = idx / numHeads;
      const LongType s = tmp % seqLen;
      const LongType b = tmp / seqLen;

      const LongType position = static_cast<LongType>(positionOffset) + s;

      const T* xPtr = x + b * xStride0 + s * xStride1 + h * xStride2;
      T* zPtr = z + b * zStride0 + s * zStride1 + h * zStride2;

      for (LongType i = 0; i < halfDim; ++i) {
        // theta_i = freq_base ^ (-2i / head_dim) * freq_scale
        const double exponent = -2.0 * static_cast<double>(i) / static_cast<double>(headDim);
        const double theta_i = sd::math::sd_pow<double, double, double>(freqBase, exponent) * freqScale;
        const double angle = static_cast<double>(position) * theta_i;

        const T cosVal = static_cast<T>(sd::math::sd_cos<double, double>(angle));
        const T sinVal = static_cast<T>(sd::math::sd_sin<double, double>(angle));

        const T x0 = xPtr[2 * i * xStride3];
        const T x1 = xPtr[(2 * i + 1) * xStride3];

        zPtr[2 * i * zStride3]       = x0 * cosVal - x1 * sinVal;
        zPtr[(2 * i + 1) * zStride3] = x0 * sinVal + x1 * cosVal;
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, outerSize);
}

void dualRoPE(LaunchContext* context, NDArray* input, NDArray* output,
              int attentionType, int positionOffset,
              double localFreqBase, double globalFreqBase,
              double localFreqScale, double globalFreqScale) {
  NDArray::preparePrimaryUse({output}, {input});

  BUILD_SINGLE_SELECTOR(input->dataType(), dualRoPE_,
                         (input, output, attentionType, positionOffset,
                          localFreqBase, globalFreqBase, localFreqScale, globalFreqScale),
                         SD_FLOAT_TYPES);

  NDArray::registerPrimaryUse({output}, {input});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
