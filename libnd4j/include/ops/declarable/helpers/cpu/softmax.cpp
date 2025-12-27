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

#include <cmath>
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
 T sum = static_cast<T>(0.);
 int length = shape::length(inShapeInfo);

 sd::LongType inRank = shape::rank(inShapeInfo);
 sd::LongType outRank = shape::rank(outShapeInfo);
 sd::LongType *inShape = shape::shapeOf(inShapeInfo);
 sd::LongType *outShape = shape::shapeOf(outShapeInfo);
 sd::LongType *inStride = shape::stride(inShapeInfo);
 sd::LongType *outStride = shape::stride(outShapeInfo);

 sd::LongType coords[SD_MAX_RANK];

 // Clamp value for numerical stability - prevents Inf from propagating
 // exp(88) ≈ 1.6e38 which is close to float max, exp(89) overflows
 const T clampMax = static_cast<T>(88.0f);
 const T clampMin = static_cast<T>(-88.0f);

 // Find max (skip Inf/NaN values)
 for (int i = 0; i < length; i++) {
   INDEX2COORDS(i, inRank, inShape, coords);
   sd::LongType inOffset;
   COORDS2INDEX(inRank, inStride, coords, inOffset);
   T val = inBuff[inOffset];
   // Skip Inf and NaN when finding max
   if (!std::isinf(val) && !std::isnan(val)) {
     max = sd::math::sd_max<T>(max, val);
   }
 }

 // If max is still at initial value (all values were Inf/NaN), use 0
 if (max == -DataTypeUtils::max<T>()) {
   max = static_cast<T>(0.0f);
 }

 // Calculate exp and sum
 for (int i = 0; i < length; i++) {
   INDEX2COORDS(i, inRank, inShape, coords);
   sd::LongType inOffset, outOffset;
   COORDS2INDEX(inRank, inStride, coords, inOffset);
   COORDS2INDEX(outRank, outStride, coords, outOffset);

   T val = inBuff[inOffset];
   // Handle Inf/NaN inputs - treat as very large/small values
   if (std::isinf(val) || std::isnan(val)) {
     val = (val > 0 || std::isnan(val)) ? clampMax + max : clampMin + max;
   }
   // Clamp the difference to prevent overflow in exp
   T diff = val - max;
   diff = sd::math::sd_max<T>(clampMin, sd::math::sd_min<T>(clampMax, diff));
   T r = sd::math::sd_exp<T, T>(diff);
   outBuff[outOffset] = r;
   sum += r;
 }

 // Add small epsilon to prevent division by zero
 sum = sd::math::sd_max<T>(sum, static_cast<T>(1e-6f));

 // Normalize
 for (int i = 0; i < length; i++) {
   INDEX2COORDS(i, outRank, outShape, coords);
   sd::LongType outOffset;
   COORDS2INDEX(outRank, outStride, coords, outOffset);
   outBuff[outOffset] /= sum;
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

// Clamp constants for numerical stability
static constexpr float SOFTMAX_CLAMP_MAX = 88.0f;
static constexpr float SOFTMAX_CLAMP_MIN = -88.0f;
static constexpr float SOFTMAX_SUM_EPS = 1e-6f;

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

   // Find max (skip Inf/NaN)
   for (sd::LongType j = 0; j < tadLen; ++j) {
     float val = inBuff[j];
     if (!std::isinf(val) && !std::isnan(val)) {
       max = sd::math::sd_max<float>(max, val);
     }
   }
   if (max == -DataTypeUtils::max<float>()) max = 0.0f;

   for (sd::LongType j = 0; j < tadLen; ++j) {
     float val = inBuff[j];
     if (std::isinf(val) || std::isnan(val)) {
       val = (val > 0 || std::isnan(val)) ? SOFTMAX_CLAMP_MAX + max : SOFTMAX_CLAMP_MIN + max;
     }
     float diff = val - max;
     diff = sd::math::sd_max<float>(SOFTMAX_CLAMP_MIN, sd::math::sd_min<float>(SOFTMAX_CLAMP_MAX, diff));
     float temp = sd::math::sd_exp<float, float>(diff);
     outBuff[j] = temp;
     sum += temp;
   }

   sum = sd::math::sd_max<float>(sum, SOFTMAX_SUM_EPS);
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

     // Find max (skip Inf/NaN)
     for (sd::LongType j = 0; j < tadLen; ++j) {
       float val = inBuff[j];
       if (!std::isinf(val) && !std::isnan(val)) {
         max = sd::math::sd_max<float>(max, val);
       }
     }
     if (max == -DataTypeUtils::max<float>()) max = 0.0f;

     for (sd::LongType j = 0; j < tadLen; ++j) {
       float val = inBuff[j];
       if (std::isinf(val) || std::isnan(val)) {
         val = (val > 0 || std::isnan(val)) ? SOFTMAX_CLAMP_MAX + max : SOFTMAX_CLAMP_MIN + max;
       }
       float diff = val - max;
       diff = sd::math::sd_max<float>(SOFTMAX_CLAMP_MIN, sd::math::sd_min<float>(SOFTMAX_CLAMP_MAX, diff));
       float temp = sd::math::sd_exp<float, float>(diff);
       outBuff[j] = temp;
       sum += temp;
     }

     sum = sd::math::sd_max<float>(sum, SOFTMAX_SUM_EPS);
     for (sd::LongType j = 0; j < tadLen; ++j) outBuff[j] /= sum;
   }
 };

 samediff::Threads::parallel_tad(func, 0, numOfSubArrs);
}

#endif

template <typename T>
SD_INLINE void softmax_loop(const T* input, T* output, const sd::LongType* offsets, sd::LongType numOfSubArrs,
                           uint32_t tadLen) {
 const T clampMax = static_cast<T>(SOFTMAX_CLAMP_MAX);
 const T clampMin = static_cast<T>(SOFTMAX_CLAMP_MIN);
 const T sumEps = static_cast<T>(SOFTMAX_SUM_EPS);

 auto func = PRAGMA_THREADS_FOR {
   for (auto i = start; i < stop; i++) {
     auto inBuff = input + offsets[i];
     auto outBuff = output + offsets[i];

     T max = -DataTypeUtils::max<T>();
     T sum(0.f);

     // Find max (skip Inf/NaN)
     for (sd::LongType j = 0; j < tadLen; ++j) {
       T val = inBuff[j];
       if (!std::isinf(static_cast<float>(val)) && !std::isnan(static_cast<float>(val))) {
         max = sd::math::sd_max<T>(max, val);
       }
     }
     if (max == -DataTypeUtils::max<T>()) max = static_cast<T>(0.0f);

     for (sd::LongType j = 0; j < tadLen; ++j) {
       T val = inBuff[j];
       if (std::isinf(static_cast<float>(val)) || std::isnan(static_cast<float>(val))) {
         val = (val > 0 || std::isnan(static_cast<float>(val))) ? clampMax + max : clampMin + max;
       }
       T diff = val - max;
       diff = sd::math::sd_max<T>(clampMin, sd::math::sd_min<T>(clampMax, diff));
       T temp = sd::math::sd_exp<T, T>(diff);
       outBuff[j] = temp;
       sum += temp;
     }

     sum = sd::math::sd_max<T>(sum, sumEps);
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
   auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(),
                                                                            dimension);
   auto tadShapeInfo = tadPack->primaryShapeInfo();
   auto tadOffsets = tadPack->primaryOffsets();
   const sd::LongType numOfSubArrs = tadPack->numberOfTads();
   const sd::LongType tadLen = shape::length(tadShapeInfo);

   // Remove element-wise stride check, always use coordinate-based approach
   sd::LongType tadRank = shape::rank(tadShapeInfo);
   sd::LongType *tadShape = shape::shapeOf(tadShapeInfo);
   sd::LongType *tadStride = shape::stride(tadShapeInfo);

   // Clamp value for numerical stability - prevents Inf from propagating
   // exp(88) ≈ 1.6e38 which is close to float max, exp(89) overflows
   const T clampMax = static_cast<T>(88.0f);
   const T clampMin = static_cast<T>(-88.0f);

   auto func = PRAGMA_THREADS_FOR {
     sd::LongType tadCoords[SD_MAX_RANK];

     for (auto i = start; i < stop; i++) {
       auto inBuff = input->bufferAsT<T>() + tadOffsets[i];
       auto outBuff = output->bufferAsT<T>() + tadOffsets[i];

       T max = -DataTypeUtils::max<T>();
       T sum = static_cast<T>(0.f);

       // Find max using INDEX2COORDS/COORDS2INDEX (skip Inf/NaN values)
       for (sd::LongType j = 0; j < tadLen; ++j) {
         INDEX2COORDS(j, tadRank, tadShape, tadCoords);
         sd::LongType offset;
         COORDS2INDEX(tadRank, tadStride, tadCoords, offset);
         T val = inBuff[offset];
         if (!std::isinf(val) && !std::isnan(val)) {
           max = sd::math::sd_max<T>(max, val);
         }
       }

       // If max is still at initial value (all values were Inf/NaN), use 0
       if (max == -DataTypeUtils::max<T>()) {
         max = static_cast<T>(0.0f);
       }

       // Calculate exp and sum using INDEX2COORDS/COORDS2INDEX
       for (sd::LongType j = 0; j < tadLen; ++j) {
         INDEX2COORDS(j, tadRank, tadShape, tadCoords);
         sd::LongType offset;
         COORDS2INDEX(tadRank, tadStride, tadCoords, offset);
         T val = inBuff[offset];
         // Handle Inf/NaN inputs
         if (std::isinf(val) || std::isnan(val)) {
           val = (val > 0 || std::isnan(val)) ? clampMax + max : clampMin + max;
         }
         // Clamp the difference to prevent overflow in exp
         T diff = val - max;
         diff = sd::math::sd_max<T>(clampMin, sd::math::sd_min<T>(clampMax, diff));
         T temp = sd::math::sd_exp<T, T>(diff);
         outBuff[offset] = temp;
         sum += temp;
       }

       // Add small epsilon to prevent division by zero
       sum = sd::math::sd_max<T>(sum, static_cast<T>(1e-6f));

       // Normalize using INDEX2COORDS/COORDS2INDEX
       for (sd::LongType j = 0; j < tadLen; ++j) {
         INDEX2COORDS(j, tadRank, tadShape, tadCoords);
         sd::LongType offset;
         COORDS2INDEX(tadRank, tadStride, tadCoords, offset);
         outBuff[offset] /= sum;
       }
     }
   };

   samediff::Threads::parallel_tad(func, 0, numOfSubArrs);

 } else {
   std::vector<sd::LongType> dimensionVec = {dimension};
   NDArray *max = input->reduceAlongDimension(sd::reduce::Max, &dimensionVec, true);
   input->applyTrueBroadcast(sd::BroadcastOpsTuple::Subtract(), max, output, false);
   output->applyTransform(sd::transform::Exp, output);
   NDArray *sum = output->reduceAlongDimension(sd::reduce::Sum, &dimensionVec, true);
   *output /= *sum;
   delete sum;
   delete max;

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
