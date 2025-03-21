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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 19.04.2018
// @author raver119@gmail.com
//
#include <execution/Threads.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/helpers/activations.h>

#include <numeric>
#if NOT_EXCLUDED(OP_softmax)
namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void softMaxForVector_(void const* input, sd::LongType const* inShapeInfo, void* output,
                              sd::LongType const* outShapeInfo) {
  auto inBuff = reinterpret_cast<T const*>(input);
  auto outBuff = reinterpret_cast<T*>(output);

  T max = -DataTypeUtils::max<T>();
  T sum = 0.;
  int inEWS = shape::elementWiseStride(inShapeInfo);
  int outEWS = shape::elementWiseStride(outShapeInfo);
  int length = shape::length(inShapeInfo);

  if (inEWS >= 1 && outEWS >= 1) {
    if (inEWS == 1 && outEWS == 1) {
      for (int i = 0; i < length; i++) max = sd::math::sd_max<T>(max, inBuff[i]);

      for (int i = 0; i < length; i++) {
        outBuff[i] = sd::math::sd_exp<T, T>(inBuff[i] - max);
        sum += outBuff[i];
      }

      for (int i = 0; i < length; i++) outBuff[i] /= sum;
    } else {
      for (int i = 0; i < length; i++) max = sd::math::sd_max<T>(max, inBuff[i * inEWS]);

      for (int i = 0; i < length; i++) {
        T r = sd::math::sd_exp<T, T>(inBuff[i * inEWS] - max);
        outBuff[i * outEWS] = r;
        sum += r;
      }

      for (int i = 0; i < length; i++) outBuff[i * outEWS] /= sum;
    }
  }
}

///////////////////////////////////////////////////////////////////
void softMaxForVector(sd::LaunchContext* context, NDArray& input, NDArray& output) {
  if (!input.isVector() || !output.isVector())
    THROW_EXCEPTION("ops::helpers::softMaxForVector function: input and output arrays must be vectors !");

  auto xType = input.dataType();
  BUILD_SINGLE_SELECTOR(xType, softMaxForVector_,
                        (input.buffer(), input.shapeInfo(), output.buffer(), output.shapeInfo()), SD_FLOAT_TYPES);
}

template <typename T>
void softmax_loop(const T* input, T* output, const sd::LongType* offsets, sd::LongType numOfSubArrs, uint32_t tadLen);

#if defined(_OPENMP)
template <>
SD_INLINE void softmax_loop(const float* input, float* output, const sd::LongType* offsets, sd::LongType numOfSubArrs,
                            uint32_t tadLen) {
#pragma omp parallel for default(shared)
  for (sd::LongType i = 0; i < numOfSubArrs; i++) {
    auto inBuff = input + offsets[i];
    auto outBuff = output + offsets[i];

    float max = -DataTypeUtils::max<float>();
    float sum = 0.f;

#pragma omp simd reduction(max : max)
    for (sd::LongType j = 0; j < tadLen; ++j) max = sd::math::sd_max<float>(max, inBuff[j]);

#pragma omp simd reduction(+ : sum)
    for (sd::LongType j = 0; j < tadLen; ++j) {
      float temp = sd::math::sd_exp<float, float>(inBuff[j] - max);
      outBuff[j] = temp;
      sum += temp;
    }

    for (sd::LongType j = 0; j < tadLen; ++j) outBuff[j] /= sum;
  }
}
#else
template <>
SD_INLINE void softmax_loop(const float* input, float* output, const sd::LongType* offsets, sd::LongType numOfSubArrs,
                            uint32_t tadLen) {
  auto func = PRAGMA_THREADS_FOR {
    for (auto i = start; i < stop; i++) {
      auto inBuff = input + offsets[i];
      auto outBuff = output + offsets[i];

      float max = -DataTypeUtils::max<float>();
      float sum = 0.f;

      for (sd::LongType j = 0; j < tadLen; ++j) max = sd::math::sd_max<float>(max, inBuff[j]);

      for (sd::LongType j = 0; j < tadLen; ++j) {
        float temp = sd::math::sd_exp<float, float>(inBuff[j] - max);
        outBuff[j] = temp;
        sum += temp;
      }

      for (sd::LongType j = 0; j < tadLen; ++j) outBuff[j] /= sum;
    }
  };

  samediff::Threads::parallel_tad(func, 0, numOfSubArrs);
}

#endif

template <typename T>
SD_INLINE void softmax_loop(const T* input, T* output, const sd::LongType* offsets, sd::LongType numOfSubArrs,
                            uint32_t tadLen) {
  auto func = PRAGMA_THREADS_FOR {
    for (auto i = start; i < stop; i++) {
      auto inBuff = input + offsets[i];
      auto outBuff = output + offsets[i];

      T max = -DataTypeUtils::max<T>();
      T sum(0.f);

      for (sd::LongType j = 0; j < tadLen; ++j) max = sd::math::sd_max<T>(max, inBuff[j]);
      for (sd::LongType j = 0; j < tadLen; ++j) {
        T temp = sd::math::sd_exp<T, T>(inBuff[j] - max);
        outBuff[j] = temp;
        sum += temp;
      }

      for (sd::LongType j = 0; j < tadLen; ++j) outBuff[j] /= sum;
    }
  };

  samediff::Threads::parallel_tad(func, 0, numOfSubArrs);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void softmax_(sd::LaunchContext* context, NDArray* input, NDArray* output, const int dimension) {
  const int rank = input->rankOf();

  if (input->isVector()) {
    if (rank == 1 || input->sizeAt(dimension) != 1)
      softMaxForVector_<T>(input->buffer(), input->shapeInfo(), output->buffer(), output->shapeInfo());
    else
      *output = 1.;
  } else if (input->isSameShapeStrict(*output)) {
    TadPack *tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(),
                                                                             dimension);
    auto tadShapeInfo = tadPack->primaryShapeInfo();
    auto tadOffsets = tadPack->primaryOffsets();
    const sd::LongType numOfSubArrs = tadPack->numberOfTads();
    const sd::LongType tadLen = shape::length(tadShapeInfo);

    if (shape::elementWiseStride(tadShapeInfo) == 1) {
      auto inBuff = input->bufferAsT<T>();
      T* outBuff = output->bufferAsT<T>();

      softmax_loop(inBuff, outBuff, tadOffsets, numOfSubArrs, tadLen);
    } else {
      sd::LongType inShapeInfoCast[SD_MAX_RANK];
      bool canCast = sd::DataTypeUtils::castShapeInfo(tadShapeInfo, inShapeInfoCast);

      auto offsets = new sd::LongType[tadLen];
      shape::calcOffsets(tadShapeInfo, offsets);

      auto func = PRAGMA_THREADS_FOR {
        for (auto i = start; i < stop; i++) {
          auto inBuff = input->bufferAsT<T>() + tadOffsets[i];
          auto outBuff = output->bufferAsT<T>() + tadOffsets[i];

          T max = -DataTypeUtils::max<T>();
          T sum = 0.f;

          for (sd::LongType j = 0; j < tadLen; ++j) max = sd::math::sd_max<T>(max, inBuff[offsets[j]]);

          for (sd::LongType j = 0; j < tadLen; ++j) {
            T temp = sd::math::sd_exp<T, T>(inBuff[offsets[j]] - max);
            outBuff[offsets[j]] = temp;
            sum += temp;
          }

          for (sd::LongType j = 0; j < tadLen; ++j) outBuff[offsets[j]] /= sum;
        }
      };

      samediff::Threads::parallel_tad(func, 0, numOfSubArrs);

      delete[] offsets;
    }
  } else {
    std::vector<sd::LongType> dimensionVec = {dimension};
    NDArray max = input->reduceAlongDimension(sd::reduce::Max, &dimensionVec, true);
    input->applyTrueBroadcast(sd::BroadcastOpsTuple::Subtract(), &max, output, false);
    output->applyTransform(sd::transform::Exp, output);
    NDArray sum = output->reduceAlongDimension(sd::reduce::Sum, &dimensionVec, true);
    *output /= sum;
  }
}

///////////////////////////////////////////////////////////////////
void softmax(LaunchContext* context, NDArray* input, NDArray* output, const int dimension) {
  BUILD_SINGLE_SELECTOR(input->dataType(), softmax_, (context, input, output, dimension), SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif