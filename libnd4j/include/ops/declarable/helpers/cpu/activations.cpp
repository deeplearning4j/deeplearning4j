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
  T sum = 0.;
  sd::LongType length = shape::length(inShapeInfo);

  for (sd::LongType i = 0; i < length; i++) {
    const sd::LongType offset = shape::getIndexOffset(i, inShapeInfo);
    max = sd::math::sd_max<T>(max, inBuff[offset]);
  }

  for (sd::LongType i = 0; i < length; i++) {
    const sd::LongType offset = shape::getIndexOffset(i, inShapeInfo);
    outBuff[offset] = sd::math::sd_exp<T, T>(inBuff[offset] - max);
    sum += outBuff[offset];
  }

  for (sd::LongType i = 0; i < length; i++) {
    const sd::LongType offset = shape::getIndexOffset(i, inShapeInfo);
    outBuff[offset] /= sum;
    outBuff[offset] *= (1.f - outBuff[offset]);  // derivative
  }
}

///////////////////////////////////////////////////////////////////
void softmaxDerivative(sd::LaunchContext* context, const NDArray& input, NDArray& output, const int dimension) {
  const int rank = input.rankOf();
  sd::LongType temp;

  if (shape::isCommonVector(input.shapeInfo(), temp)) {
    BUILD_SINGLE_SELECTOR(input.dataType(), _softMaxDerivForVector,
                          (context, input.buffer(), input.shapeInfo(), output.buffer()), SD_FLOAT_TYPES);
  } else {
    std::vector<sd::LongType> dimVec = {dimension};
    auto maxAlongDim = const_cast<NDArray&>(input).reduceAlongDimension(reduce::Max, &dimVec, true);
    (input - maxAlongDim).applyTransform(transform::Exp, output);  // output contains exponents temporarily
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
  T sum = 0;

  auto inEWS = shape::elementWiseStride(inShapeInfo);
  auto length = shape::length(inShapeInfo);

  if (inEWS == 1) {
    for (sd::LongType i = 0; i < length; i++) max = sd::math::sd_max<T>(max, inBuff[i]);
    PRAGMA_OMP_SIMD_SUM(sum)
    for (sd::LongType i = 0; i < length; i++) {
      outBuff[i] = sd::math::sd_exp<T, T>(inBuff[i] - max);
      sum += outBuff[i];
    }

    PRAGMA_OMP_SIMD
    for (sd::LongType i = 0; i < length; i++) {
      outBuff[i] /= sum;
      outBuff[i] = sd::math::sd_log<T, T>(outBuff[i]);
    }
  } else if (inEWS > 1) {
    PRAGMA_OMP_SIMD_MAX(max)

    for (sd::LongType i = 0; i < length; i++) max = sd::math::sd_max<T>(max, inBuff[i * inEWS]);

    PRAGMA_OMP_SIMD_SUM(sum)
    for (sd::LongType i = 0; i < length; i++) {
      outBuff[i * inEWS] = sd::math::sd_exp<T, T>(inBuff[i * inEWS] - max);
      sum += outBuff[i * inEWS];
    }

    PRAGMA_OMP_SIMD
    for (sd::LongType i = 0; i < length; i++) {
      outBuff[i * inEWS] /= sum;
      outBuff[i * inEWS] = sd::math::sd_log<T, T>(outBuff[i * inEWS]);
    }
  }
}

///////////////////////////////////////////////////////////////////
void logSoftMaxForVector(sd::LaunchContext* context, const NDArray& input, NDArray& output) {
  if (!input.isVector() || !output.isVector())
    THROW_EXCEPTION("ops::helpers::logSoftMaxForVector function input and output arrays must be vectors !");

  auto xType = input.dataType();
  BUILD_SINGLE_SELECTOR(xType, logSoftMaxForVector_,
                        (input.buffer(), input.shapeInfo(), output.buffer(), output.shapeInfo()), SD_FLOAT_TYPES);
}

//////////////////////////////////////////////////////////////////////////
void prelu(sd::LaunchContext* context, const NDArray& input, const NDArray& alpha, NDArray& output) {
  const sd::LongType inputLen = input.lengthOf();
  const sd::LongType* inputShapeInfo = input.shapeInfo();
  const sd::LongType* alphaShapeInfo = alpha.shapeInfo();

  auto func = PRAGMA_THREADS_FOR {
    for (sd::LongType i = start; i < stop; i++) {
      // FIXME: double!
      double x = input.e<double>(i);
      if (x < 0.0) {
        // FIXME: double
        output.p(i, (x * alpha.e<double>(shape::subArrayIndex(i, inputShapeInfo, alphaShapeInfo))));
      } else
        output.p(i, x);
    }
  };

  samediff::Threads::parallel_for(func, 0, inputLen);
}

//////////////////////////////////////////////////////////////////////////
void preluBP(sd::LaunchContext* context, const NDArray& input, const NDArray& alpha, const NDArray& dLdO, NDArray& dLdI,
             NDArray& dLdA) {
  const sd::LongType inputLen = input.lengthOf();
  const sd::LongType* inputShapeInfo = input.shapeInfo();
  const sd::LongType* alphaShapeInfo = alpha.shapeInfo();

  dLdA.assign(0.0f);

  for (sd::LongType i = 0; i < inputLen; ++i) {
    // FIXME: double
    double x = input.e<double>(i);
    double grO =  dLdO.isScalar() ?  dLdO.e<double>(0) : dLdO.e<double>(i);
    if (x < 0.0) {
      sd::LongType alphaInd = shape::subArrayIndex(i, inputShapeInfo, alphaShapeInfo);
      dLdI.p(i, grO * alpha.e<double>(alphaInd));
      double prevVal = dLdA.e<double>(alphaInd);
      prevVal += (grO * x);
      dLdA.p(alphaInd, prevVal);
    } else
      dLdI.p(i, grO);
  }
}


template <typename T>
static void thresholdRelu_(NDArray const& input, double threshold, NDArray& output) {
  auto routine = LAMBDA_T(_x, threshold) { return _x > (T)threshold ? _x : (T)0.f; };
  const_cast<NDArray&>(input).applyLambda<T>(routine, output);
}

void thresholdRelu(sd::LaunchContext* context, NDArray const& input, double threshold, NDArray& output) {
  BUILD_SINGLE_SELECTOR(input.dataType(), thresholdRelu_, (input, threshold, output), SD_FLOAT_TYPES);
}

template <typename T>
static void thresholdReluDerivative_(sd::LaunchContext* context, NDArray* input, double theta, NDArray* dLdO,
                                     NDArray* output) {
  auto derivative = LAMBDA_TT(_x, grO, theta) {
    if (_x > theta)
      return grO;
    else
      return static_cast<T>(0);
  };

  input->applyPairwiseLambda<T>(*dLdO, derivative, *output);
}

void thresholdReluDerivative(sd::LaunchContext* context, NDArray* input, double threshold, NDArray* dLdO,
                             NDArray* output) {
  BUILD_SINGLE_SELECTOR(input->dataType(), thresholdReluDerivative_, (context, input, threshold, dLdO, output),
                        SD_FLOAT_TYPES);
}

///////////////////////////////////////////////////////////////////
void logSoftmax(sd::LaunchContext* context, const NDArray& input, NDArray& output, const int dimension) {
  const int rank = input.rankOf();

  if (input.isVector()) {
    if (rank == 1 || input.sizeAt(dimension) != 1) {
      BUILD_SINGLE_SELECTOR(input.dataType(), logSoftMaxForVector_,
                            (input.buffer(), input.shapeInfo(), output.buffer(), output.shapeInfo()), SD_FLOAT_TYPES);
    } else
      output = 0.;
  } else {
    std::vector<sd::LongType> dimVector = {dimension};
    auto maxAlongDim = const_cast<NDArray&>(input).reduceAlongDimension(reduce::Max, &dimVector, true);
    (input - maxAlongDim).applyTransform(transform::Exp, output);  // output contains exponents temporarily
    auto sumAlongDim = output.reduceAlongDimension(reduce::Sum, &dimVector, true);
    output /= sumAlongDim;
    output.applyTransform(transform::Log, output);
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
