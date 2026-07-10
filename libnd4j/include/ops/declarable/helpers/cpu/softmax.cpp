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
#include <ops/op_types.h>

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

 // Accumulate max/exp-sum in AggregateType (float for T=HALF/BF16): a HALF-typed
 // exp-sum loses ~1e-2 of row mass on long rows. Matches the CUDA kernels.
 using AccT = typename simdOps::AggregateType<T>::type;
 AccT max = -DataTypeUtils::max<AccT>();
 AccT sum = static_cast<AccT>(0.);
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
 const AccT clampMax = static_cast<AccT>(88.0f);
 const AccT clampMin = static_cast<AccT>(-88.0f);

 // Find max (skip Inf/NaN values)
 for (int i = 0; i < length; i++) {
   INDEX2COORDS(i, inRank, inShape, coords);
   sd::LongType inOffset;
   COORDS2INDEX(inRank, inStride, coords, inOffset);
   AccT val = static_cast<AccT>(inBuff[inOffset]);
   // Skip Inf and NaN when finding max
   if (!std::isinf(static_cast<float>(val)) && !std::isnan(static_cast<float>(val))) {
     max = sd::math::sd_max(max, val);
   }
 }

 // If max is still at initial value (all values were Inf/NaN), use 0
 if (max == -DataTypeUtils::max<AccT>()) {
   max = static_cast<AccT>(0.0f);
 }

 // Calculate exp and sum
 for (int i = 0; i < length; i++) {
   INDEX2COORDS(i, inRank, inShape, coords);
   sd::LongType inOffset, outOffset;
   COORDS2INDEX(inRank, inStride, coords, inOffset);
   COORDS2INDEX(outRank, outStride, coords, outOffset);

   AccT val = static_cast<AccT>(inBuff[inOffset]);
   // Handle Inf/NaN inputs - treat as very large/small values
   if (std::isinf(static_cast<float>(val)) || std::isnan(static_cast<float>(val))) {
     val = (val > 0 || std::isnan(static_cast<float>(val))) ? clampMax + max : clampMin + max;
   }
   // Clamp the difference to prevent overflow in exp
   AccT diff = val - max;
   diff = sd::math::sd_max(clampMin, sd::math::sd_min(clampMax, diff));
   AccT r = sd::math::sd_exp<AccT, AccT>(diff);
   outBuff[outOffset] = static_cast<T>(r);
   sum += r;
 }

 // Add small epsilon to prevent division by zero
 sum = sd::math::sd_max(sum, static_cast<AccT>(1e-6f));

 // Normalize
 for (int i = 0; i < length; i++) {
   INDEX2COORDS(i, outRank, outShape, coords);
   sd::LongType outOffset;
   COORDS2INDEX(outRank, outStride, coords, outOffset);
   outBuff[outOffset] = static_cast<T>(static_cast<AccT>(outBuff[outOffset]) / sum);
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

// Optimized float softmax - assumes no Inf/NaN (fast path for inference)
// Falls back to safe version if needed
template <>
SD_INLINE void softmax_loop(const float* input, float* output, const sd::LongType* offsets, sd::LongType numOfSubArrs,
                           uint32_t tadLen) {
  auto func = PRAGMA_THREADS_FOR {
    for (auto i = start; i < stop; i++) {
      auto inBuff = input + offsets[i];
      auto outBuff = output + offsets[i];

      // Fast path: find max without Inf/NaN checks (common case in inference)
      float max = inBuff[0];
      for (uint32_t j = 1; j < tadLen; ++j) {
        if (inBuff[j] > max) max = inBuff[j];
      }

      // Compute exp and sum in single pass
      float sum = 0.f;
      for (uint32_t j = 0; j < tadLen; ++j) {
        float diff = inBuff[j] - max;
        // Clamp to prevent overflow (exp(88) ≈ 1.6e38)
        if (diff < SOFTMAX_CLAMP_MIN) diff = SOFTMAX_CLAMP_MIN;
        float temp = sd::math::sd_exp<float, float>(diff);
        outBuff[j] = temp;
        sum += temp;
      }

      // Normalize
      float invSum = 1.0f / sum;
      for (uint32_t j = 0; j < tadLen; ++j) {
        outBuff[j] *= invSum;
      }
    }
  };

  samediff::Threads::parallel_tad(func, 0, numOfSubArrs);
}

template <typename T>
SD_INLINE void softmax_loop(const T* input, T* output, const sd::LongType* offsets, sd::LongType numOfSubArrs,
                           uint32_t tadLen) {
 // Accumulate in AggregateType (float for T=HALF/BF16) — see softMaxForVector_.
 using AccT = typename simdOps::AggregateType<T>::type;
 const AccT clampMax = static_cast<AccT>(SOFTMAX_CLAMP_MAX);
 const AccT clampMin = static_cast<AccT>(SOFTMAX_CLAMP_MIN);
 const AccT sumEps = static_cast<AccT>(SOFTMAX_SUM_EPS);

 auto func = PRAGMA_THREADS_FOR {
   for (auto i = start; i < stop; i++) {
     auto inBuff = input + offsets[i];
     auto outBuff = output + offsets[i];

     AccT max = -DataTypeUtils::max<AccT>();
     AccT sum(0.f);

     // Find max (skip Inf/NaN)
     for (sd::LongType j = 0; j < tadLen; ++j) {
       AccT val = static_cast<AccT>(inBuff[j]);
       if (!std::isinf(static_cast<float>(val)) && !std::isnan(static_cast<float>(val))) {
         max = sd::math::sd_max(max, val);
       }
     }
     if (max == -DataTypeUtils::max<AccT>()) max = static_cast<AccT>(0.0f);

     for (sd::LongType j = 0; j < tadLen; ++j) {
       AccT val = static_cast<AccT>(inBuff[j]);
       if (std::isinf(static_cast<float>(val)) || std::isnan(static_cast<float>(val))) {
         val = (val > 0 || std::isnan(static_cast<float>(val))) ? clampMax + max : clampMin + max;
       }
       AccT diff = val - max;
       diff = sd::math::sd_max(clampMin, sd::math::sd_min(clampMax, diff));
       AccT temp = sd::math::sd_exp<AccT, AccT>(diff);
       outBuff[j] = static_cast<T>(temp);
       sum += temp;
     }

     sum = sd::math::sd_max(sum, sumEps);
     for (sd::LongType j = 0; j < tadLen; ++j)
       outBuff[j] = static_cast<T>(static_cast<AccT>(outBuff[j]) / sum);
   }
 };

 samediff::Threads::parallel_tad(func, 0, numOfSubArrs);
}

//////////////////////////////////////////////////////////////////////////
// True only for a plain dense C-order array backed by its whole buffer with no base
// offset — i.e. one where flat index i maps to buffer element i. Used only to gate the
// fast linear softmax_loop; every other case goes through the coordinate-indexed path,
// which handles arbitrary strides/offsets correctly.
template <typename T>
static bool isPlainDenseCOrder(NDArray* arr) {
  if (arr->ordering() != 'c') return false;
  if (arr->offset() != 0) return false;
  const int rank = arr->rankOf();
  sd::LongType expected = 1;
  for (int i = rank - 1; i >= 0; --i) {
    if (arr->sizeAt(i) != 1 && arr->strideAt(i) != expected) return false;
    expected *= arr->sizeAt(i);
  }
  return true;
}

template <typename T>
static void softmax_(sd::LaunchContext* context, NDArray* input, NDArray* output, const int dimension) {
 const int rank = input->rankOf();
 // Normalize negative dimension to its positive equivalent before passing to tadForDimensions.
 // tadForDimensions treats -1 as a sentinel meaning "entire array", not "last dimension".
 const int dim = (dimension < 0) ? (rank + dimension) : dimension;

 if (input->isVector()) {
   if (rank == 1 || input->sizeAt(dim) != 1)
     softMaxForVector_<T>(input->buffer(), input->shapeInfo(), output->buffer(), output->shapeInfo());
   else
     *output = 1.;
   return;
 }

 // TAD softmax over `dim`. Input and output are indexed through their OWN TAD shapeInfo —
 // separate offset tables AND separate strides, resolved per element via
 // INDEX2COORDS/COORDS2INDEX. This is correct for strided/offset VIEWS on either side
 // (permute, slice), which the old code broke by reusing the input's TAD offsets/strides
 // for the output's writes. (bufferAsT<T>() already folds in each array's base offset.)
 auto inTadPack  = sd::ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(),  (sd::LongType)dim);
 auto outTadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), (sd::LongType)dim);
 auto inTadShapeInfo  = inTadPack->primaryShapeInfo();
 auto outTadShapeInfo = outTadPack->primaryShapeInfo();
 auto inTadOffsets  = inTadPack->primaryOffsets();
 auto outTadOffsets = outTadPack->primaryOffsets();
 const sd::LongType numOfSubArrs = inTadPack->numberOfTads();
 const sd::LongType tadLen = shape::length(inTadShapeInfo);

 // Fast linear path: only when BOTH sides are plain dense C-order (offset 0, default
 // strides) — then every TAD is unit-stride contiguous and the two offset tables
 // coincide, so softmax_loop's linear indexing is exact.
 if (isPlainDenseCOrder<T>(input) && isPlainDenseCOrder<T>(output) && input->isSameShapeStrict(*output)) {
   softmax_loop<T>(input->bufferAsT<T>(), output->bufferAsT<T>(), inTadOffsets, numOfSubArrs,
                   static_cast<uint32_t>(tadLen));
   return;
 }

 // General coordinate-indexed path. Input and output share the same logical TAD shape
 // (same rank/sizes), so one coordinate decomposition drives both — only the stride used
 // to turn coordinates into a buffer offset differs between them.
 const sd::LongType tadRank = shape::rank(inTadShapeInfo);
 sd::LongType *tadShape     = shape::shapeOf(inTadShapeInfo);
 sd::LongType *inTadStride  = shape::stride(inTadShapeInfo);
 sd::LongType *outTadStride = shape::stride(outTadShapeInfo);

 // Clamp value for numerical stability - prevents Inf from propagating
 // exp(88) ≈ 1.6e38 which is close to float max, exp(89) overflows
 // Accumulate in AggregateType (float for T=HALF/BF16) — see softMaxForVector_.
 using AccT = typename simdOps::AggregateType<T>::type;
 const AccT clampMax = static_cast<AccT>(88.0f);
 const AccT clampMin = static_cast<AccT>(-88.0f);

 auto func = PRAGMA_THREADS_FOR {
   sd::LongType tadCoords[SD_MAX_RANK];

   for (auto i = start; i < stop; i++) {
     auto inBuff  = input->bufferAsT<T>()  + inTadOffsets[i];
     auto outBuff = output->bufferAsT<T>() + outTadOffsets[i];

     AccT max = -DataTypeUtils::max<AccT>();
     AccT sum = static_cast<AccT>(0.f);

     // Find max using INDEX2COORDS/COORDS2INDEX (skip Inf/NaN values)
     for (sd::LongType j = 0; j < tadLen; ++j) {
       INDEX2COORDS(j, tadRank, tadShape, tadCoords);
       sd::LongType inOffset;
       COORDS2INDEX(tadRank, inTadStride, tadCoords, inOffset);
       AccT val = static_cast<AccT>(inBuff[inOffset]);
       if (!std::isinf(static_cast<float>(val)) && !std::isnan(static_cast<float>(val))) {
         max = sd::math::sd_max(max, val);
       }
     }

     // If max is still at initial value (all values were Inf/NaN), use 0
     if (max == -DataTypeUtils::max<AccT>()) {
       max = static_cast<AccT>(0.0f);
     }

     // Calculate exp and sum; read via input stride, write via output stride
     for (sd::LongType j = 0; j < tadLen; ++j) {
       INDEX2COORDS(j, tadRank, tadShape, tadCoords);
       sd::LongType inOffset, outOffset;
       COORDS2INDEX(tadRank, inTadStride, tadCoords, inOffset);
       COORDS2INDEX(tadRank, outTadStride, tadCoords, outOffset);
       AccT val = static_cast<AccT>(inBuff[inOffset]);
       // Handle Inf/NaN inputs
       if (std::isinf(static_cast<float>(val)) || std::isnan(static_cast<float>(val))) {
         val = (val > 0 || std::isnan(static_cast<float>(val))) ? clampMax + max : clampMin + max;
       }
       // Clamp the difference to prevent overflow in exp
       AccT diff = val - max;
       diff = sd::math::sd_max(clampMin, sd::math::sd_min(clampMax, diff));
       AccT temp = sd::math::sd_exp<AccT, AccT>(diff);
       outBuff[outOffset] = static_cast<T>(temp);
       sum += temp;
     }

     // Add small epsilon to prevent division by zero
     sum = sd::math::sd_max(sum, static_cast<AccT>(1e-6f));

     // Normalize via output stride
     for (sd::LongType j = 0; j < tadLen; ++j) {
       INDEX2COORDS(j, tadRank, tadShape, tadCoords);
       sd::LongType outOffset;
       COORDS2INDEX(tadRank, outTadStride, tadCoords, outOffset);
       outBuff[outOffset] = static_cast<T>(static_cast<AccT>(outBuff[outOffset]) / sum);
     }
   }
 };

 samediff::Threads::parallel_tad(func, 0, numOfSubArrs);
}

///////////////////////////////////////////////////////////////////
void softmax(LaunchContext* context, NDArray* input, NDArray* output, const int dimension) {
 BUILD_SINGLE_SELECTOR(input->dataType(), softmax_, (context, input, output, dimension), SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
