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
// @author raver119@gmail.com
// @author Yurii Shyrma (iuriish@yahoo.com)
//
#include <execution/Threads.h>
#include <helpers/ConstantTadHelper.h>
#include <ops/declarable/helpers/adjust_hue.h>
#if NOT_EXCLUDED(OP_adjust_hue)
namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void adjustHue_(NDArray *input, NDArray *deltaScalarArr, NDArray *output, const sd::LongType dimC) {
  const T delta = deltaScalarArr->e<T>(0);
  const int rank = input->rankOf();

  const T *x = input->bufferAsT<T>();
  T *z = output->bufferAsT<T>();

  if (dimC == rank - 1 && input->ews() == 1 && output->ews() == 1 && input->ordering() == 'c' &&
      output->ordering() == 'c') {
    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i += increment) {
        T h, s, v;

        rgbToHsv<T>(x[i], x[i + 1], x[i + 2], h, s, v);

        h += delta;
        if (h > (T)1)
          h -= (T)1;
        else if (h < 0)
          h += (T)1;

        hsvToRgb<T>(h, s, v, z[i], z[i + 1], z[i + 2]);
      }
    };

    samediff::Threads::parallel_for(func, 0, input->lengthOf(), 3);
  } else {
    auto packX = sd::ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimC,true);
    auto packZ = sd::ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimC,true);

    const sd::LongType numOfTads = packX->numberOfTads();
    const sd::LongType xDimCstride = input->stridesOf()[dimC];
    const sd::LongType zDimCstride = output->stridesOf()[dimC];

    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        const T *xTad = x + packX->platformOffsets()[i];
        T *zTad = z + packZ->platformOffsets()[i];

        T h, s, v;

        rgbToHsv<T>(xTad[0], xTad[xDimCstride], xTad[2 * xDimCstride], h, s, v);

        h += delta;
        if (h > (T)1)
          h -= (T)1;
        else if (h < 0)
          h += (T)1;

        hsvToRgb<T>(h, s, v, zTad[0], zTad[zDimCstride], zTad[2 * zDimCstride]);
      }
    };

    samediff::Threads::parallel_tad(func, 0, numOfTads);
  }
}

void adjustHue(sd::LaunchContext *context, NDArray *input, NDArray *deltaScalarArr, NDArray *output,
               const sd::LongType dimC) {
  BUILD_SINGLE_SELECTOR(input->dataType(), adjustHue_, (input, deltaScalarArr, output, dimC), SD_FLOAT_TYPES);
}



}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif