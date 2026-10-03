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
// Created by raver119 on 30.11.17.
//

#include <execution/Threads.h>
#include <ops/declarable/helpers/col2im.h>
#include <ops/op_types.h>
#if NOT_EXCLUDED(OP_col2im)

namespace sd {
namespace ops {
namespace helpers {

// [bS, iC, kH, kW, oH, oW] is de-convoluted to [bS, iC, iH, iW]. Every image element is written: it is the sum, in
// AggregateType<T>, of the column entries im2col reads from it (0 when no window covers it), so the result does not
// depend on what the output held before (conv2d_bp passes its gradI output uninitialized). The sum runs over the
// windows in row-major order, as the CUDA kernel's does.
template <typename T>
static void col2im_(sd::LaunchContext& context, NDArray* input, NDArray* output, const LongType sH, const LongType sW,
                    const LongType pH, const LongType pW, const LongType iH, const LongType iW, const LongType dH, const LongType dW) {
  if(input->rankOf() != 6) {
    THROW_EXCEPTION("ops::helpers::col2im: input array must have rank = 6");
  }

  if(output->rankOf() != 4) {
    THROW_EXCEPTION("ops::helpers::col2im: output array must have rank = 4");
  }

  if (output->sizeAt(2) != iH || output->sizeAt(3) != iW) {
    THROW_EXCEPTION("ops::helpers::col2im: the image size must equal the output's height and width");
  }

  using AccT = typename simdOps::AggregateType<T>::type;

  NDArray::preparePrimaryUse({output}, {input});

  const T* colBuff = input->bufferAsT<T>();
  T* imBuff = output->bufferAsT<T>();
  const LongType* colShape = shape::shapeOf(input->shapeInfo());
  const LongType* colStride = shape::stride(input->shapeInfo());
  const LongType* imShape = shape::shapeOf(output->shapeInfo());
  const LongType* imStride = shape::stride(output->shapeInfo());

  const LongType bS = imShape[0];
  const LongType iC = imShape[1];
  const LongType kH = colShape[2];
  const LongType kW = colShape[3];
  const LongType oH = colShape[4];
  const LongType oW = colShape[5];
  // extent of a dilated window
  const LongType ekH = dH * (kH - 1) + 1;
  const LongType ekW = dW * (kW - 1) + 1;

  auto func = PRAGMA_THREADS_FOR {
    for (auto plane = start; plane < stop; plane += increment) {
      const LongType b = plane / iC;
      const LongType c = plane % iC;
      const T* colPlane = colBuff + b * colStride[0] + c * colStride[1];
      T* imPlane = imBuff + b * imStride[0] + c * imStride[1];
      for (LongType h = 0; h < iH; ++h) {
        // windows colH that cover padded row h + pH: colH * sH <= h + pH < colH * sH + ekH
        const LongType imH = h + pH;
        const LongType colHstart = imH < ekH ? 0 : (imH - ekH) / sH + 1;
        const LongType colHend = sd::math::sd_min<LongType>(imH / sH + 1, oH);
        for (LongType w = 0; w < iW; ++w) {
          const LongType imW = w + pW;
          const LongType colWstart = imW < ekW ? 0 : (imW - ekW) / sW + 1;
          const LongType colWend = sd::math::sd_min<LongType>(imW / sW + 1, oW);
          AccT sum = static_cast<AccT>(0);
          for (LongType colH = colHstart; colH < colHend; ++colH) {
            const LongType kRowOffset = imH - colH * sH;
            if (kRowOffset % dH != 0) continue;
            const T* colRow = colPlane + (kRowOffset / dH) * colStride[2] + colH * colStride[4];
            for (LongType colW = colWstart; colW < colWend; ++colW) {
              const LongType kColOffset = imW - colW * sW;
              if (kColOffset % dW != 0) continue;
              sum += static_cast<AccT>(colRow[(kColOffset / dW) * colStride[3] + colW * colStride[5]]);
            }
          }
          imPlane[h * imStride[2] + w * imStride[3]] = static_cast<T>(sum);
        }
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, bS * iC);

  NDArray::registerPrimaryUse({output}, {input});
}
void col2im(LaunchContext& context,  NDArray* input, NDArray* output, const LongType sH, const LongType sW, const LongType pH,
            const LongType pW, const LongType iH, const LongType iW, const LongType dH, const LongType dW) {
  BUILD_SINGLE_SELECTOR(input->dataType(), col2im_, (context, input, output, sH, sW, pH, pW, iH, iW, dH, dW),
                        SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif