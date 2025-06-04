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

namespace sd {
namespace ops {
namespace helpers {

///////////////////////////////////////////////////////////////////
template <typename T>
void static _softMaxDerivForVector(sd::LaunchContext* context, const void* input, const sd::LongType* inShapeInfo,
                                   void* output) {
  const T* inBuff = reinterpret_cast<const T*>(input);
  T* outBuff = reinterpret_cast<T*>(output);

  T max = -DataTypeUtils::max<T>();
  T sum = static_cast<T>(0.);
  const sd::LongType length = shape::length(inShapeInfo);

  const sd::LongType rank = shape::rank(inShapeInfo);
  const sd::LongType* shape = shape::shapeOf(inShapeInfo);
  const sd::LongType* stride = shape::stride(inShapeInfo);

  LongType coords[SD_MAX_RANK];
  LongType offset;

  // Find the maximum value in the vector
  for (sd::LongType i = 0; i < length; i++) {
    INDEX2COORDS(i, rank, shape, coords);
    COORDS2INDEX(rank, stride, coords, offset);
    max = sd::math::sd_max<T>(max, inBuff[offset]);
  }

  // Calculate exponentials and sum
  for (sd::LongType i = 0; i < length; i++) {
    INDEX2COORDS(i, rank, shape, coords);
    COORDS2INDEX(rank, stride, coords, offset);
    outBuff[offset] = sd::math::sd_exp<T, T>(inBuff[offset] - max);
    sum += outBuff[offset];
  }

  // Compute softmax derivatives
  for (sd::LongType i = 0; i < length; i++) {
    INDEX2COORDS(i, rank, shape, coords);
    COORDS2INDEX(rank, stride, coords, offset);
    outBuff[offset] /= sum;
    outBuff[offset] *= (1.f - outBuff[offset]);  // derivative
  }
}

///////////////////////////////////////////////////////////////////
void softmaxDerivative(sd::LaunchContext* context, NDArray& input, NDArray& output, const int dimension) {
  const int rank = input.rankOf();
  sd::LongType temp;

  if (shape::isCommonVector(input.shapeInfo(), temp)) {
    BUILD_SINGLE_SELECTOR(input.dataType(), _softMaxDerivForVector,
                          (context, input.buffer(), input.shapeInfo(), output.buffer()), SD_FLOAT_TYPES);
  } else {
    std::vector<sd::LongType> dimVec = {dimension};
    auto maxAlongDim = const_cast<NDArray&>(input).reduceAlongDimension(reduce::Max, &dimVec, true);
    (input - maxAlongDim).applyTransform(transform::Exp, &output);  // output contains exponents temporarily
    auto sumAlongDim = output.reduceAlongDimension(reduce::Sum, &dimVec, true);
    output /= sumAlongDim;
    output *= (1.f - output);  // derivative
  }
}

///////////////////////////////////////////////////////////////////
template <typename T>
void logSoftMaxForVector_(void const* input, sd::LongType const* inShapeInfo, void* output,
                          sd::LongType const* outShapeInfo) {
  auto inBuff = reinterpret_cast<T const*>(input);
  auto outBuff = reinterpret_cast<T*>(output);

  T max = -DataTypeUtils::max<T>();
  T sum = static_cast<T>(0);

  auto length = shape::length(inShapeInfo);
  sd::LongType  inRank = shape::rank(inShapeInfo);
  sd::LongType *inShape = shape::shapeOf(inShapeInfo);
  sd::LongType *inStrides = shape::stride(inShapeInfo);

  sd::LongType *outShape = shape::shapeOf(outShapeInfo);
  sd::LongType *outStrides = shape::stride(outShapeInfo);
  sd::LongType outRank = shape::rank(outShapeInfo);
  sd::LongType inIndices[length];
  sd::LongType outIndices[length];
  PRAGMA_OMP_SIMD
  for (sd::LongType i2 = 0; i2 < length; i2++) {
    LongType coords[SD_MAX_RANK];
    sd::LongType  idx2;
    INDEX2COORDS(i2,inRank, inShape, coords);
    COORDS2INDEX(inRank, inStrides, coords, idx2);
    max = sd::math::sd_max<T,T>(max, inBuff[idx2]);
    inIndices[i2] = idx2;
  }

  PRAGMA_OMP_SIMD
  for (sd::LongType i2 = 0; i2 < length; i2++) {
    LongType coords[SD_MAX_RANK];
    sd::LongType  idx2;
    INDEX2COORDS(i2,outRank, outShape, coords);
    COORDS2INDEX(outRank, outStrides, coords, idx2);
    outBuff[idx2] = sd::math::sd_exp<T, T>(inBuff[inIndices[i2]] - max);
    sum += outBuff[idx2];
  }

  PRAGMA_OMP_SIMD
  for (sd::LongType i = 0; i < length; i++) {
    outBuff[outIndices[i]] /= sum;
    outBuff[outIndices[i]] = sd::math::sd_log<T, T>(outBuff[outIndices[i]]);
  }
}

///////////////////////////////////////////////////////////////////
void logSoftMaxForVector(sd::LaunchContext* context, NDArray& input, NDArray& output) {
  if (!input.isVector() || !output.isVector())
    THROW_EXCEPTION("ops::helpers::logSoftMaxForVector function input and output arrays must be vectors !");

  auto xType = input.dataType();
  BUILD_SINGLE_SELECTOR(xType, logSoftMaxForVector_,
                        (input.buffer(), input.shapeInfo(), output.buffer(), output.shapeInfo()), SD_FLOAT_TYPES);
}

//////////////////////////////////////////////////////////////////////////
void prelu(LaunchContext* context, NDArray* input, NDArray* alpha, NDArray* output) {
  const sd::LongType inputLen = input->lengthOf();
  const sd::LongType* inputShapeInfo = input->shapeInfo();
  const sd::LongType* alphaShapeInfo = alpha->shapeInfo();

  auto func = PRAGMA_THREADS_FOR {
    for (sd::LongType i = start; i < stop; i++) {
      // FIXME: double!
      double x = input->e<double>(i);
      if (x < 0.0) {
        // FIXME: double
        output->p(i, (x * alpha->e<double>(shape::subArrayIndex(i, inputShapeInfo, alphaShapeInfo))));
      } else
        output->p(i, x);
    }
  };

  samediff::Threads::parallel_for(func, 0, inputLen);
}

//////////////////////////////////////////////////////////////////////////
void preluBP(LaunchContext* context, NDArray* input, NDArray* alpha, NDArray* dLdO, NDArray* dLdI,
             NDArray* dLdA) {
  const sd::LongType inputLen = input->lengthOf();
  const sd::LongType* inputShapeInfo = input->shapeInfo();
  const sd::LongType* alphaShapeInfo = alpha->shapeInfo();
  float zero = 0.f;
  dLdA->assign(zero);

  for (sd::LongType i = 0; i < inputLen; ++i) {
    // FIXME: double
    double x = input->e<double>(i);
    double grO =  dLdO->isScalar() ?  dLdO->e<double>(0) : dLdO->e<double>(i);
    if (x < 0.0) {
      sd::LongType alphaInd = shape::subArrayIndex(i, inputShapeInfo, alphaShapeInfo);
      dLdI->p(i, grO * alpha->e<double>(alphaInd));
      double prevVal = dLdA->e<double>(alphaInd);
      prevVal += (grO * x);
      dLdA->p(alphaInd, prevVal);
    } else
      dLdI->p(i, grO);
  }
}

bool checkAlphaShapeLen(std::vector<sd::LongType> const& expectedShape, sd::LongType shapeLen) {
  sd::LongType expectedAlphaLen =
      std::accumulate(expectedShape.cbegin(), expectedShape.cend(), 1, std::multiplies<sd::LongType>());
  return expectedAlphaLen == shapeLen;
}
template <typename T>
static void thresholdRelu_(NDArray *input, double threshold, NDArray* output) {
  auto routine = LAMBDA_T(_x, threshold) { return _x > (T)threshold ? _x : (T)0.f; });
  input->applyLambda<T>(routine, output);
}

void thresholdRelu(LaunchContext* context, NDArray* input, double threshold, NDArray* output) {
  BUILD_SINGLE_SELECTOR(input->dataType(), thresholdRelu_, (input, threshold, output), SD_FLOAT_TYPES);
}

template <typename T>
static void thresholdReluDerivative_(sd::LaunchContext* context, NDArray* input, double theta, NDArray* dLdO,
                                     NDArray* output) {
  auto derivative = LAMBDA_TT(_x, grO, theta) {
    if (_x > theta)
      return grO;
    else
      return static_cast<T>(0);
  });

  input->applyPairwiseLambda<T>(dLdO, derivative, output);
}

void thresholdReluDerivative(sd::LaunchContext* context, NDArray* input, double threshold, NDArray* dLdO,
                             NDArray* output) {
  BUILD_SINGLE_SELECTOR(input->dataType(), thresholdReluDerivative_, (context, input, threshold, dLdO, output),
                        SD_FLOAT_TYPES);
}

///////////////////////////////////////////////////////////////////
void logSoftmax(LaunchContext* context, NDArray* input, NDArray* output, const int dimension) {
  const int rank = input->rankOf();

  if (input->isVector()) {
    if (rank == 1 || input->sizeAt(dimension) != 1) {
      BUILD_SINGLE_SELECTOR(input->dataType(), logSoftMaxForVector_,
                            (input->buffer(), input->shapeInfo(), output->buffer(), output->shapeInfo()), SD_FLOAT_TYPES);
    } else
      *output = 0.;
  } else {
    std::vector<sd::LongType> dimVector = {dimension};
    auto maxAlongDim = input->reduceAlongDimension(reduce::Max, &dimVector, true);
    auto maxMinusDim = *input - maxAlongDim;
    maxMinusDim.applyTransform(transform::Exp, output);  // output contains exponents temporarily
    auto sumAlongDim = output->reduceAlongDimension(reduce::Sum, &dimVector, true);
    *output /= sumAlongDim;
    output->applyTransform(transform::Log, output);
  }
}

BUILD_SINGLE_TEMPLATE(template void thresholdReluDerivative_,
                      (sd::LaunchContext * context, NDArray* input, double threshold, NDArray* dLdO, NDArray* output),
                      SD_FLOAT_TYPES);
BUILD_SINGLE_TEMPLATE(template void logSoftMaxForVector_,
                      (void const* input, sd::LongType const* inShapeInfo, void* output,
                          sd::LongType const* outShapeInfo),
                      SD_FLOAT_TYPES);
BUILD_SINGLE_TEMPLATE(template void _softMaxDerivForVector,
                      (sd::LaunchContext * context, const void* input, const sd::LongType* inShapeInfo, void* output),
                      SD_FLOAT_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
