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

/*
 * summarystatsreduce.h
 *
 *  Created on: Jan 19, 2016
 *      Author: agibsonccc
 */

#ifndef SUMMARYSTATSREDUCE_H_
#define SUMMARYSTATSREDUCE_H_
#include <helpers/shape.h>
#include <math/templatemath.h>

#ifdef __JNI__
#include <jni.h>
#endif
#include <loops/legacy_ops.h>
#include <ops/ops.h>
#include <system/op_boilerplate.h>

namespace functions {
namespace summarystats {

// This example computes several statistical properties of a data
// series in a single reduction.  The algorithm is described in detail here:
// http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
//
// Thanks to Joseph Rhoads for contributing this example

// structure used to accumulate the moments and other
// statistical properties encountered so far.
template <typename X>
class SummaryStatsData {
 public:
  double n;
  double min;
  double max;
  double mean;
  double M2;
  double M3;
  double M4;
  double bias;

  SD_HOST_DEVICE SummaryStatsData() { initialize(); }

  // initialize to the identity element

  SD_HOST_DEVICE void initialize() { n = mean = M2 = M3 = M4 = bias = 0; }

  SD_HOST_DEVICE void initWithValue(X val) {
    n = 1;
    min = val;
    max = val;
    mean = val;
    M2 = 0;
    M3 = 0;
    M4 = 0;
    bias = 0;
  }

  SD_HOST_DEVICE void setValues(SummaryStatsData<X>* target) {
    n = target->n;
    min = target->min;
    max = target->max;
    mean = target->mean;
    M2 = target->M2;
    M3 = target->M3;
    M4 = target->M4;
    bias = target->bias;
  }

  SD_HOST_DEVICE double variance() {
    if (n <= 1.0) return 0.0;
    return M2 / (n);
  }

  SD_HOST_DEVICE double varianceBiasCorrected() {
    if (this->n <= 1.0) {
      return 0.0;
    }

    return M2 / (n - 1.0);
  }

  SD_HOST_DEVICE double variance_n() {
    if (n <= 1.0) return 0.0;
    return M2 / n;
  }

  SD_HOST_DEVICE double skewness() {
    return M2 > 0.0 ? sd::math::sd_sqrt<double, double>(n) * M3 / sd::math::sd_pow<double, double, double>(M2, 1.5)
                    : 0.0;
  }

  SD_HOST_DEVICE double kurtosis() { return M2 > 0.0 ? n * M4 / (M2 * M2) : 0; }

  SD_HOST_DEVICE double getM2() { return M2; }

  SD_HOST_DEVICE void setM2(X m2) { M2 = m2; }

  SD_HOST_DEVICE double getM3() { return M3; }

  SD_HOST_DEVICE void setM3(X m3) { M3 = m3; }

  SD_HOST_DEVICE double getM4() { return M4; }

  SD_HOST_DEVICE void setM4(X m4) { M4 = m4; }

  SD_HOST_DEVICE double getMax() { return max; }

  SD_HOST_DEVICE void setMax(X max) { this->max = max; }

  SD_HOST_DEVICE double getMean() { return mean; }

  SD_HOST_DEVICE void setMean(X mean) { this->mean = mean; }

  SD_HOST_DEVICE double getMin() { return min; }

  SD_HOST_DEVICE void setMin(X min) { this->min = min; }

  SD_HOST_DEVICE double getN() { return n; }

  SD_HOST_DEVICE void setN(X n) { this->n = n; }
};

#ifdef __CUDACC__
// This is the un-specialized struct.  Note that we prevent instantiation of this
// struct by putting an undefined symbol in the function body so it won't compile.
template <typename T>
struct SharedSummaryStatsData {
  // Ensure that we won't compile any un-specialized types
  SD_DEVICE T* getPointer() {
    extern SD_DEVICE void error(void);
    error();
    return 0;
  }
};

// Following are the specializations for the following types.
// int, sd::Unsigned, char, uchar, short, ushort, long long, ulong long, bool, float, and double
// One could also specialize it for user-defined types.

template <>
struct SharedSummaryStatsData<float> {
  SD_DEVICE SummaryStatsData<float>* getPointer() {
    extern __shared__ SummaryStatsData<float> s_int2[];
    return s_int2;
  }
};
// Following are the specializations for the following types.
// int, sd::Unsigned, char, uchar, short, ushort, long long, ulong long, bool, float, and double
// One could also specialize it for user-defined types.

template <>
struct SharedSummaryStatsData<double> {
  SD_DEVICE SummaryStatsData<double>* getPointer() {
    extern __shared__ SummaryStatsData<double> s_int6[];
    return s_int6;
  }
};
#endif

/**
 * Standard deviation or variance 1 pass
 */
template <typename X, typename Z>
class SummaryStatsReduce {
 public:
  // calculate an update of the reduce operation
  SD_HOST_DEVICE static SummaryStatsData<X> update(SummaryStatsData<X> x, SummaryStatsData<X> y, void* extraParams) {
    if ((long)x.n == 0 && (long)y.n > 0)
      return y;
    else if ((long)x.n > 0 && (long)y.n == 0)
      return x;
    SummaryStatsData<X> vz;
    double n = x.n + y.n;
    double n2 = n * n;
    double n3 = n2 * n;

    double delta = y.mean - x.mean;
    double delta2 = delta * delta;
    double delta3 = delta2 * delta;
    double delta4 = delta3 * delta;

    // Basic number of samples (n), min, and max
    vz.n = n;
    vz.min = sd::math::sd_min(x.min, y.min);
    vz.max = sd::math::sd_max(x.max, y.max);
    double meanD = x.mean + delta * y.n / n;
    vz.mean = meanD;
    double M2D = x.M2 + y.M2;
    M2D += delta2 * x.n * y.n / n;
    vz.M2 = M2D;
    vz.M3 = x.M3 + y.M3;
    vz.M3 += delta3 * x.n * y.n * (x.n - y.n) / n2;
    vz.M3 += 3.0 * delta * (x.n * y.M2 - y.n * x.M2) / n;

    vz.M4 = x.M4 + y.M4;
    vz.M4 += delta4 * x.n * y.n * (x.n * x.n - x.n * y.n + y.n * y.n) / n3;
    vz.M4 += 6.0 * delta2 * (x.n * x.n * y.M2 + y.n * y.n * x.M2) / n2;
    vz.M4 += 4.0 * delta * (x.n * y.M3 - y.n * x.M3) / n;

    return vz;
  }

#ifdef __CUDACC__

  static SD_INLINE SD_DEVICE Z startingValue(X * input) { return static_cast<Z>(0); }

  template <typename OpType>
  static SD_DEVICE void aggregatePartials(SummaryStatsData<X>* sPartials, sd::LongType tid, sd::LongType numElements,
                                          void* extraParams);

  template <typename OpType>
  static SD_DEVICE void transform(void * dx, sd::LongType * xShapeInfo, void* extraParams, void* vz,
                                  sd::LongType * zShapeInfo, sd::LongType* dimension,
                                  long long int dimensionLength,
                                  int postProcessOrNot, sd::LongType* allocationBuffer, void* reductionBuffer,
                                  sd::LongType * tadOnlyShapeInfo, sd::LongType * tadOffsets);

  static SD_DEVICE void transform( int opNum, void * dx, sd::LongType * xShapeInfo, void* extraParams,
                                  void* vz, sd::LongType * zShapeInfo, sd::LongType* dimension,
                                  sd::LongType dimensionLength,
                                  int postProcessOrNot, sd::LongType* allocationBuffer, void* reductionBuffer,
                                  sd::LongType * tadOnlyShapeInfo, sd::LongType * tadOffsets);

  static SD_HOST void execSummaryStatsReduceScalar(dim3& launchDims, cudaStream_t* stream, int opNum, void * x,
                                                   sd::LongType * xShapeInfo, sd::LongType * hxShapeInfo,
                                                   void* extraParams, void* vz, sd::LongType * zShapeInfo,
                                                   sd::LongType * hzShapeInfo, sd::LongType * tadShapeInfo,
                                                   sd::LongType * tadOffsets, bool biasCorrected,
                                                   void* reductionBuffer);
  static SD_HOST void execSummaryStatsReduce(dim3& launchDims, cudaStream_t* stream, int opNum, void * x,
                                             sd::LongType * xShapeInfo, sd::LongType * hxShapeInfo,
                                             void* extraParams, void* vz, sd::LongType * zShapeInfo,
                                             sd::LongType * hzShapeInfo, sd::LongType * tadShapeInfo,
                                             sd::LongType * tadOffsets, bool biasCorrected, void* reductionBuffer);
#else

  static Z execScalar(int opNum, bool biasCorrected,  void *x,  sd::LongType *xShapeInfo, void *extraParams);

  static void execScalar(int opNum, bool biasCorrected,  void *x,  sd::LongType *xShapeInfo,
                         void *extraParams, void *vz,  sd::LongType *resultShapeInfoBuffer);

  static void exec(int opNum, bool biasCorrected,  void *x,  sd::LongType *xShapeInfo, void *extraParams,
                   void *vz,  sd::LongType *resultShapeInfoBuffer, long long int *dimension,
                   long long int dimensionLength);

  template <typename OpType>
  static Z execScalar(bool biasCorrected,  void *x,  sd::LongType *xShapeInfo, void *extraParams);

  template <typename OpType>
  static void execScalar(bool biasCorrected,  void *x,  sd::LongType *xShapeInfo, void *extraParams, void *vz,
                          sd::LongType *resultShapeInfoBuffer);

  template <typename OpType>
  static void exec(bool biasCorrected,  void *x,  sd::LongType *xShapeInfo, void *extraParams, void *vz,
                    sd::LongType *resultShapeInfoBuffer, sd::LongType *dimension, sd::LongType dimensionLength);
#endif
};
}  // namespace summarystats
}  // namespace functions

#endif /* SUMMARYSTATSREDUCE_H_ */
