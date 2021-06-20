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
 // @author AbdelRauf 
 //
#include <type_traits>
#include <cmath>
#include <stdexcept>
#include <memory>
#include <execution/Threads.h>
#include <execution/ThreadPool.h>
#include <math/platformmath.h>

#include <math/templatemath.h>
#include <helpers/LoopsCoordsHelper.h>
#include <ops/declarable/helpers/reductions.h>
#include <vector>
#if 1
#define  LOG_CALLS(X) 
#else
#define  LOG_CALLS(X)  nd4j_printf("___%s_________%d+\n", __PRETTY_FUNCTION__, X); 
#endif
#define USE_REDUCED_DIV 1
namespace sd {
    namespace ops {
        namespace helpers {

            constexpr int threadingThreshold = 4096 * 2;
            constexpr int vectorizationThreshold = 64;

            struct DeviationAggregate {
                double n;
                double mean;
                double M2;
            };

            template <typename X, typename Z, bool Standard = false>
            class Deviation {
            public:
                using aggregate_type = DeviationAggregate;

                template <bool S = Standard>
                static FORCEINLINE typename std::enable_if<S == true, Z>::type
                    getDeviation(const aggregate_type& a, bool biasCorrected) {
                    if (a.n <= 1.0) {
                        return  static_cast<Z>(0.0);
                    }
                    double ret;
                    if (biasCorrected) {
                        ret = a.M2 / (a.n - 1.0);
                        if (ret < 0.0)  ret = a.M2 / a.n;
                    }
                    else {
                        ret = a.M2 / a.n;
                    }
                    return sd::math::nd4j_sqrt<double, Z>(ret);
                }

                template <bool S = Standard>
                static FORCEINLINE typename std::enable_if<S == false, Z>::type
                    getDeviation(const aggregate_type& a, bool biasCorrected) {
                    if (a.n <= 1.0) {
                        return  static_cast<Z>(0.0);
                    }
                    double ret;
                    if (biasCorrected) {
                        ret = a.M2 / (a.n - 1.0);
                        if (ret < 0.0)  ret = a.M2 / a.n;
                    }
                    else {
                        ret = a.M2 / a.n;
                    }
                    return static_cast<Z>(ret);
                }

                template<bool InitializeFromAggregate = false>
                static FORCEINLINE void updateInnerLoop1b(const X* buffer, Nd4jLong length, aggregate_type& a) {
                    double xn = InitializeFromAggregate ? a.n : 0;
                    double xmean = InitializeFromAggregate ? a.mean : 0;
                    double xM2 = InitializeFromAggregate ? a.M2 : 0;
                    for (Nd4jLong i = 0; i < length; i++) {
                        double n = xn + 1;
                        double delta = buffer[i] - xmean;
                        double delta2 = delta * delta;
                        double delta_n = delta / n;
                        xmean = xmean + delta_n;
                        xM2 += delta2 * xn / n;
                        xn = n;
                    }
                    a = { xn, xmean, xM2 };
                }

                template<bool InitializeFromAggregate = false>
                static FORCEINLINE void updateInnerLoop1b(const X* buffer, Nd4jLong length, Nd4jLong stride, aggregate_type& a) {
                    double xn = InitializeFromAggregate ? a.n : 0;
                    double xmean = InitializeFromAggregate ? a.mean : 0;
                    double xM2 = InitializeFromAggregate ? a.M2 : 0;
                    for (Nd4jLong i = 0; i < length; i++) {
                        double n = xn + 1;
                        double delta = buffer[i * stride] - xmean;
                        double delta2 = delta * delta;
                        double delta_n = delta / n;
                        xmean = xmean + delta_n;
                        xM2 += delta2 * xn / n;
                        xn = n;
                    }
                    a = { xn, xmean, xM2 };
                }

                //WARNING: a.n a1.n a2.n a3.n should be equal
                template<bool InitializeFromAggregate = false>
                static FORCEINLINE void updateInnerLoop4b(const X* buffer, const X* buffer1, const X* buffer2,
                    const X* buffer3, Nd4jLong length,
                    aggregate_type& a, aggregate_type& a1, aggregate_type& a2, aggregate_type& a3) {
                    double xn, x0mean, x0M2, x1mean, x1M2, x2mean, x2M2, x3mean, x3M2;
                    xn = InitializeFromAggregate ? a.n : 0;
                    x0mean = InitializeFromAggregate ? a.mean : 0;
                    x1mean = InitializeFromAggregate ? a1.mean : 0;
                    x2mean = InitializeFromAggregate ? a2.mean : 0;
                    x3mean = InitializeFromAggregate ? a3.mean : 0;
                    x0M2 = InitializeFromAggregate ? a.M2 : 0;
                    x1M2 = InitializeFromAggregate ? a1.M2 : 0;
                    x2M2 = InitializeFromAggregate ? a2.M2 : 0;
                    x3M2 = InitializeFromAggregate ? a3.M2 : 0;
                    for (Nd4jLong i = 0; i < length; i++) {
                        double n = xn + 1;
                        double delta0 = buffer[i] - x0mean;
                        double delta1 = buffer1[i] - x1mean;
                        double delta2 = buffer2[i] - x2mean;
                        double delta3 = buffer3[i] - x3mean;
#if		 defined(USE_REDUCED_DIV)
                        double delta_nj0 = delta0 / n;
                        double delta_nj1 = delta1 / n;
                        double delta_nj2 = delta2 / n;
                        double delta_nj3 = delta3 / n;
                        x0mean = x0mean + delta_nj0;
                        x1mean = x1mean + delta_nj1;
                        x2mean = x2mean + delta_nj2;
                        x3mean = x3mean + delta_nj3;
                        x0M2 += delta0 * delta_nj0 * xn;
                        x1M2 += delta1 * delta_nj1 * xn;
                        x2M2 += delta2 * delta_nj2 * xn;
                        x3M2 += delta3 * delta_nj3 * xn;
#else
                        double delta02 = delta0 * delta0;
                        double delta12 = delta1 * delta1;
                        double delta22 = delta2 * delta2;
                        double delta32 = delta3 * delta3;
                        x0mean = x0mean + delta0 / n;
                        x1mean = x1mean + delta1 / n;
                        x2mean = x2mean + delta2 / n;
                        x3mean = x3mean + delta3 / n;
                        x0M2 += delta02 * xn / n;
                        x1M2 += delta12 * xn / n;
                        x2M2 += delta22 * xn / n;
                        x3M2 += delta32 * xn / n;
#endif
                        xn = n;
                    }
                    a =  { xn, x0mean, x0M2 };
                    a1 = { xn, x1mean, x1M2 };
                    a2 = { xn, x2mean, x2M2 };
                    a3 = { xn, x3mean, x3M2 };
                }

                //WARNING: a.n a1.n a2.n a3.n should be equal
                template<bool InitializeFromAggregate = false>
                static FORCEINLINE void updateInnerLoop4b(const X* buffer, const X* buffer1, const X* buffer2,
                    const X* buffer3, Nd4jLong length, Nd4jLong stride,
                    aggregate_type& a, aggregate_type& a1, aggregate_type& a2, aggregate_type& a3) {
                    double xn, x0mean, x0M2, x1mean, x1M2, x2mean, x2M2, x3mean, x3M2;
                    xn = InitializeFromAggregate ? a.n : 0;
                    x0mean = InitializeFromAggregate ? a.mean : 0;
                    x1mean = InitializeFromAggregate ? a1.mean : 0;
                    x2mean = InitializeFromAggregate ? a2.mean : 0;
                    x3mean = InitializeFromAggregate ? a3.mean : 0;
                    x0M2 = InitializeFromAggregate ? a.M2 : 0;
                    x1M2 = InitializeFromAggregate ? a1.M2 : 0;
                    x2M2 = InitializeFromAggregate ? a2.M2 : 0;
                    x3M2 = InitializeFromAggregate ? a3.M2 : 0;
                    //nd4j_printf("++----%f    \n", xn);
                    for (Nd4jLong i = 0; i < length; i++) {
                        double n = xn + 1;
                        double delta0 = buffer[i * stride] - x0mean;
                        double delta1 = buffer1[i * stride] - x1mean;
                        double delta2 = buffer2[i * stride] - x2mean;
                        double delta3 = buffer3[i * stride] - x3mean;
#if		 defined(USE_REDUCED_DIV)
                        double delta_nj0 = delta0 / n;
                        double delta_nj1 = delta1 / n;
                        double delta_nj2 = delta2 / n;
                        double delta_nj3 = delta3 / n;
                        x0mean = x0mean + delta_nj0;
                        x1mean = x1mean + delta_nj1;
                        x2mean = x2mean + delta_nj2;
                        x3mean = x3mean + delta_nj3;
                        x0M2 += delta0 * delta_nj0 * xn;
                        x1M2 += delta1 * delta_nj1 * xn;
                        x2M2 += delta2 * delta_nj2 * xn;
                        x3M2 += delta3 * delta_nj3 * xn;
#else
                        double delta02 = delta0 * delta0;
                        double delta12 = delta1 * delta1;
                        double delta22 = delta2 * delta2;
                        double delta32 = delta3 * delta3;
                        x0mean = x0mean + delta0 / n;
                        x1mean = x1mean + delta1 / n;
                        x2mean = x2mean + delta2 / n;
                        x3mean = x3mean + delta3 / n;
                        x0M2 += delta02 * xn / n;
                        x1M2 += delta12 * xn / n;
                        x2M2 += delta22 * xn / n;
                        x3M2 += delta32 * xn / n;
#endif
                        xn = n;
                    }
                    a = { xn, x0mean, x0M2 };
                    a1 = { xn, x1mean, x1M2 };
                    a2 = { xn, x2mean, x2M2 };
                    a3 = { xn, x3mean, x3M2 };
                }

                //WARNING: length required to to be length/8. 
                static FORCEINLINE void updateInnerLoop1b_vec8(const X* buffer, Nd4jLong length_8th, double(&xn)[8], double(&xmean)[8], double(&xM2)[8]) {
                    double n[8] = {};
                    for (int i = 0; i < length_8th; i++) {
                        const X* bufferX = &(buffer[i * 8]);
                        #pragma omp simd
                        for (int j = 0; j < 8; j++) {
                            n[j] = xn[j] + 1.0;
#if  defined(USE_REDUCED_DIV)
                            double delta = bufferX[j] - xmean[j];
                            double delta_nj = delta / n[j];
                            xmean[j] = xmean[j] + delta_nj;
                            xM2[j] = xM2[j] + delta * delta_nj * xn[j];
#else
                            double delta = bufferX[j] - xmean[j];
                            double delta2 = delta * delta;
                            xmean[j] = xmean[j] + delta / n[j];
                            xM2[j] = xM2[j] + delta2 * xn[j] / n[j];
#endif
                            xn[j] = n[j];
                        }
                    }
                }

                //WARNING: length required to to be length/8.
                //WARNING: a.n a1.n a2.n a3.n should be equal
                static FORCEINLINE void updateInnerLoop4b_vec8(const X* buffer, const X* buffer1, const X* buffer2,
                    const X* buffer3, Nd4jLong length_8th, double(&xn)[8], double(&x0mean)[8], double(&x1mean)[8],
                    double(&x2mean)[8], double(&x3mean)[8], double(&x0M2)[8], double(&x1M2)[8], double(&x2M2)[8], double(&x3M2)[8]) {
                    double n[8] = {};
                    for (Nd4jLong i = 0; i < length_8th; i++) {
                        const X* bufferX = &(buffer[i * 8]);
                        const X* buffer1X = &(buffer1[i * 8]);
                        const X* buffer2X = &(buffer2[i * 8]);
                        const X* buffer3X = &(buffer3[i * 8]);
                        #pragma omp simd
                        for (int j = 0; j < 8; j++) {
                            n[j] = xn[j] + 1.0;
                            double delta0 = bufferX[j] - x0mean[j];
                            double delta1 = buffer1X[j] - x1mean[j];
                            double delta2 = buffer2X[j] - x2mean[j];
                            double delta3 = buffer3X[j] - x3mean[j];
                            double delta0_nj = delta0 / n[j];
                            double delta1_nj = delta1 / n[j];
                            double delta2_nj = delta2 / n[j];
                            double delta3_nj = delta3 / n[j];
                            x0mean[j] = x0mean[j] + delta0_nj;
                            x1mean[j] = x1mean[j] + delta1_nj;
                            x2mean[j] = x2mean[j] + delta2_nj;
                            x3mean[j] = x3mean[j] + delta3_nj;
                            x0M2[j] = x0M2[j] + xn[j] * delta0 * delta0_nj;
                            x1M2[j] = x1M2[j] + xn[j] * delta1 * delta1_nj;
                            x2M2[j] = x2M2[j] + xn[j] * delta2 * delta2_nj;
                            x3M2[j] = x3M2[j] + xn[j] * delta3 * delta3_nj;
                            xn[j] = n[j];
                        }
                    }
                }

                template<size_t constRank, bool LastIndexFaster = true>
                static FORCEINLINE void reduceConstRankLoop4b(const X* buff, const X* buff1, const X* buff2, const X* buff3,
                    Z* output0, Z* output1, Z* output2, Z* output3, const Nd4jLong* bases, const Nd4jLong* strides,
                    const Nd4jLong& outerLoopCount, const Nd4jLong& innerLoopCount, bool biasCorrected)
                {
                    //skip 1 from the beginning or end depending the Order 
                    constexpr size_t updated_index = LastIndexFaster ? 0 : 1;
                    constexpr size_t updated_rank = constRank - 1;
                    sd::CoordsState<updated_rank - 1> cst;
                    //we skip 1  
                    size_t offset = sd::init_coords<updated_rank, 0, LastIndexFaster>(cst, 0, bases + updated_index, strides + updated_index);
                    aggregate_type agg0 = {};
                    aggregate_type agg1 = {};
                    aggregate_type agg2 = {};
                    aggregate_type agg3 = {};
                    if (innerLoopCount >= vectorizationThreshold) {
                        LOG_CALLS(0)
                            //use vector
                        const Nd4jLong loopCount = innerLoopCount & (-8);
                        const Nd4jLong tail = innerLoopCount & 7;
                        const auto loopCount_8th = loopCount / 8;
                        double xn[8] = {};
                        double x0mean[8] = {}; double x1mean[8] = {}; double x2mean[8] = {}; double x3mean[8] = {};
                        double x0M2[8] = {}; double x1M2[8] = {}; double x2M2[8] = {}; double x3M2[8] = {};
                        for (Nd4jLong i = 0; i < outerLoopCount; i++) {
                            const X* buffPtr0 = &(buff[offset]);
                            const X* buffPtr1 = &(buff1[offset]);
                            const X* buffPtr2 = &(buff2[offset]);
                            const X* buffPtr3 = &(buff3[offset]);
                            updateInnerLoop4b_vec8(buffPtr0, buffPtr1, buffPtr2, buffPtr3, loopCount_8th, xn, x0mean, x1mean, x2mean, x3mean, x0M2, x1M2, x2M2, x3M2);
                            if (tail > 0) {
                                //tails
                                updateInnerLoop4b<true>(&(buffPtr0[loopCount]), &(buffPtr1[loopCount]), &(buffPtr2[loopCount]), &(buffPtr3[loopCount]), tail, agg0, agg1, agg2, agg3);
                            }
                            offset = sd::inc_coords<updated_rank, 0, LastIndexFaster>(cst, offset);
                        }
                        //merge vector and tails
                        auto merged0 = mergeAggregates(xn[0], x0mean, x0M2);
                        auto merged1 = mergeAggregates(xn[0], x1mean, x1M2);
                        auto merged2 = mergeAggregates(xn[0], x2mean, x2M2);
                        auto merged3 = mergeAggregates(xn[0], x3mean, x3M2);
                        //tail with merged3
                        agg0 = mergeAggregates(merged0, agg0);
                        agg1 = mergeAggregates(merged1, agg1);
                        agg2 = mergeAggregates(merged2, agg2);
                        agg3 = mergeAggregates(merged3, agg3);
                    }
                    else {
                        LOG_CALLS(1)
                            for (Nd4jLong i = 0; i < outerLoopCount; i++) {
                                updateInnerLoop4b<true>(&(buff[offset]), &(buff1[offset]), &(buff2[offset]), &(buff3[offset]), innerLoopCount, agg0, agg1, agg2, agg3);
                                offset = sd::inc_coords<updated_rank, 0, LastIndexFaster>(cst, offset);
                            }
                    }
                    *output0 = getDeviation(agg0, biasCorrected);
                    *output1 = getDeviation(agg1, biasCorrected);
                    *output2 = getDeviation(agg2, biasCorrected);
                    *output3 = getDeviation(agg3, biasCorrected);
                    return;
                }

                template<size_t constRank, bool LastIndexFaster = true>
                static FORCEINLINE void reduceConstRankLoop4b(const X* buff, const X* buff1, const X* buff2, const X* buff3,
                    Z* output0, Z* output1, Z* output2, Z* output3, const Nd4jLong* bases, const Nd4jLong* strides,
                    const Nd4jLong& outerLoopCount, const Nd4jLong& innerLoopCount, const Nd4jLong& inner_stride, bool biasCorrected)
                {
                    LOG_CALLS(2)
                        //skip 1 from the beginning or end depending the Order 
                    constexpr size_t updated_index = LastIndexFaster ? 0 : 1;
                    constexpr size_t updated_rank = constRank - 1;
                    sd::CoordsState<updated_rank - 1> cst;
                    //we skip 1  
                    size_t offset = sd::init_coords<updated_rank, 0, LastIndexFaster>(cst, 0, bases + updated_index, strides + updated_index);
                    aggregate_type agg = { 0.0, 0.0, 0.0 };
                    aggregate_type agg1 = { 0.0, 0.0, 0.0 };
                    aggregate_type agg2 = { 0.0, 0.0, 0.0 };
                    aggregate_type agg3 = { 0.0, 0.0, 0.0 };
                    //LOG_CALLS(0)
                    for (Nd4jLong i = 0; i < outerLoopCount; i++) {
                        updateInnerLoop4b<true>(&(buff[offset]), &(buff1[offset]), &(buff2[offset]), &(buff3[offset]),
                            innerLoopCount, inner_stride, agg, agg1, agg2, agg3);
                        offset = sd::inc_coords<updated_rank, 0, LastIndexFaster>(cst, offset);
                    }
                    *output0 = getDeviation(agg, biasCorrected);
                    *output1 = getDeviation(agg1, biasCorrected);
                    *output2 = getDeviation(agg2, biasCorrected);
                    *output3 = getDeviation(agg3, biasCorrected);
                    return;
                }

                template<size_t constRank, bool LastIndexFaster = true>
                static FORCEINLINE void reduceConstRankLoop1b(const X* buff, Z* output, const Nd4jLong* bases, const Nd4jLong* strides,
                    const Nd4jLong& outerLoopCount, const Nd4jLong& innerLoopCount, bool biasCorrected)
                {
                    //skip 1 from the beginning or end depending the Order 
                    constexpr size_t updated_index = LastIndexFaster ? 0 : 1;
                    constexpr size_t updated_rank = constRank - 1;
                    sd::CoordsState<updated_rank - 1> cst;
                    //we skip 1  
                    size_t offset = sd::init_coords<updated_rank, 0, LastIndexFaster>(cst, 0, bases + updated_index, strides + updated_index);
                    aggregate_type agg = { 0.0, 0.0, 0.0 };
                    if (innerLoopCount >= vectorizationThreshold) {
                        LOG_CALLS(0)
                            //use vector
                            const Nd4jLong loopCount = innerLoopCount & (-8);
                        const Nd4jLong tail = innerLoopCount & 7;
                        const auto loopCount_8th = loopCount / 8;
                        double xn[8] = {};
                        double xmean[8] = {};
                        double xM2[8] = {};
                        for (Nd4jLong i = 0; i < outerLoopCount; i++) {
                            const X* buffPtr0 = &(buff[offset]);
                            updateInnerLoop1b_vec8(buffPtr0, loopCount_8th, xn, xmean, xM2);
                            if (tail > 0) {
                                updateInnerLoop1b<true>(&(buffPtr0[loopCount]), tail, agg);
                            }
                            offset = sd::inc_coords<updated_rank, 0, LastIndexFaster>(cst, offset);
                        }
                        //merge vector between and with the tail agg
                        auto merged = mergeAggregates(xn[0], xmean, xM2);
                        agg = mergeAggregates(agg, merged);
                    }
                    else {
                        LOG_CALLS(1)
                            for (Nd4jLong i = 0; i < outerLoopCount; i++) {
                                updateInnerLoop1b<true>(&(buff[offset]), innerLoopCount, agg);
                                offset = sd::inc_coords<updated_rank, 0, LastIndexFaster>(cst, offset);
                            }
                    }
                    *output = getDeviation(agg, biasCorrected);
                    return;
                }

                template<size_t constRank, bool LastIndexFaster = true>
                static FORCEINLINE void reduceConstRankLoop1b(const X* buff, Z* output, const Nd4jLong* bases, const Nd4jLong* strides,
                    const Nd4jLong& outerLoopCount, const Nd4jLong& innerLoopCount, const Nd4jLong& inner_stride, bool biasCorrected)
                {
                    LOG_CALLS(2)
                    //skip 1 from the beginning or end depending the Order 
                    constexpr size_t updated_index = LastIndexFaster ? 0 : 1;
                    constexpr size_t updated_rank = constRank - 1;
                    sd::CoordsState<updated_rank - 1> cst;
                    //we skip 1  
                    size_t offset = sd::init_coords<updated_rank, 0, LastIndexFaster>(cst, 0, bases + updated_index, strides + updated_index);
                    aggregate_type agg = { 0.0, 0.0, 0.0 };
                    //LOG_CALLS(0)
                    for (Nd4jLong i = 0; i < outerLoopCount; i++) {
                        updateInnerLoop1b<true>(&(buff[offset]), innerLoopCount, inner_stride, agg);
                        offset = sd::inc_coords<updated_rank, 0, LastIndexFaster>(cst, offset);
                    }
                    *output = getDeviation(agg, biasCorrected);
                    return;
                }

                template<bool LastIndexFaster = true>
                static FORCEINLINE void updateGeneralLoop1b(int rank, const X* buff, DeviationAggregate& agg, const Nd4jLong* bases, const Nd4jLong* strides,
                    const Nd4jLong& outerLoopStart, const Nd4jLong& outerLoopStop, const Nd4jLong& innerLoopCount)
                {
                    agg = {};
                    size_t offset = 0;
                    Nd4jLong outerLoopCount = outerLoopStop - outerLoopStart;
                    Nd4jLong coords[MAX_RANK] = {};
                    Nd4jLong* ptr_coords = (Nd4jLong*)&coords;
                    if (outerLoopStart > 0) {
                        if (LastIndexFaster) {
                            sd::index2coords_C(outerLoopStart, rank - 1, bases, ptr_coords);
                        }
                        else {
                            //skip first base 
                            sd::index2coords_F(outerLoopStart, rank - 1, &(bases[1]), ptr_coords);
                        }
                        offset = sd::offset_from_coords(strides, ptr_coords, rank);
                    }
                    if (innerLoopCount >= vectorizationThreshold) {
                        LOG_CALLS(88)
                        //use vector
                        const Nd4jLong loopCount = innerLoopCount & (-8);
                        const Nd4jLong tail = innerLoopCount & 7;
                        const auto loopCount_8th = loopCount / 8;
                        double xn[8] = {};
                        double x0mean[8] = {};
                        double x0M2[8] = {};
                        for (Nd4jLong i = 0; i < outerLoopCount; i++) {
                            const X* buffPtr0 = &(buff[offset]);
                            updateInnerLoop1b_vec8(buffPtr0, loopCount_8th, xn, x0mean, x0M2);
                            if (tail > 0) {
                                //tails
                                updateInnerLoop1b<true>(&(buffPtr0[loopCount]), tail, agg);
                            }
                            offset = inc_coords<LastIndexFaster>(bases, strides, ptr_coords, offset, rank, 1);
                        }
                        //merge vector and tails
                        auto merged0 = mergeAggregates(xn[0], x0mean, x0M2);
                        //tail with merged3
                        agg = mergeAggregates(merged0, agg);
                    }
                    else {
                        LOG_CALLS(89)
                        for (Nd4jLong i = 0; i < outerLoopCount; i++) {
                            updateInnerLoop1b<true>(&(buff[offset]), innerLoopCount, agg);
                            offset = inc_coords<LastIndexFaster>(bases, strides, ptr_coords, offset, rank, 1);
                        }
                    }
                }

                template< bool LastIndexFaster = true>
                static FORCEINLINE void updateGeneralLoop1b(int rank, const X* buff, DeviationAggregate& agg, const Nd4jLong* bases, const Nd4jLong* strides,
                    const Nd4jLong& outerLoopStart, const Nd4jLong& outerLoopStop, const Nd4jLong& innerLoopCount, const Nd4jLong& inner_stride)
                {
                    agg = {};
                    size_t offset = 0;
                    Nd4jLong outerLoopCount = outerLoopStop - outerLoopStart;
                    Nd4jLong coords[MAX_RANK] = {};
                    Nd4jLong* ptr_coords = (Nd4jLong*)&coords;
                    if (outerLoopStart > 0) {
                        if (LastIndexFaster) {
                            sd::index2coords_C(outerLoopStart, rank - 1, bases, ptr_coords);
                        }
                        else {
                            //skip first base 
                            sd::index2coords_F(outerLoopStart, rank - 1, &(bases[1]), ptr_coords);
                        }
                        offset = sd::offset_from_coords(strides, ptr_coords, rank);
                    }
                    LOG_CALLS(90)
                    for (Nd4jLong i = 0; i < outerLoopCount; i++) {
                        updateInnerLoop1b<true>(&(buff[offset]), innerLoopCount, inner_stride, agg);
                        offset = inc_coords<LastIndexFaster>(bases, strides, ptr_coords, offset, rank, 1);
                    }
                }

                static FORCEINLINE aggregate_type mergeAggregates(const aggregate_type& x, const  aggregate_type& y) {
                    if ((long)x.n == 0 && (long)y.n > 0)
                        return y;
                    else if ((long)x.n > 0 && (long)y.n == 0)
                        return x;
                    double n = x.n + y.n;
                    double delta = y.mean - x.mean;
                    double delta2 = delta * delta;
                    double meanD = x.mean + delta * y.n / n;
                    double M2D = x.M2 + y.M2;
                    M2D += delta2 * x.n * y.n / n;
                    return { n, meanD, M2D };
                }

                static  FORCEINLINE aggregate_type  mergeAggregates(double xn, double(&xmean)[8], double(&xM2)[8])
                {
                    auto arg0 = mergeAggregates(xn, xn, xmean[0], xmean[1], xM2[0], xM2[1]);
                    auto arg1 = mergeAggregates(xn, xn, xmean[2], xmean[3], xM2[2], xM2[3]);
                    auto arg2 = mergeAggregates(xn, xn, xmean[4], xmean[5], xM2[4], xM2[5]);
                    auto arg3 = mergeAggregates(xn, xn, xmean[6], xmean[7], xM2[6], xM2[7]);
                    auto arg01 = mergeAggregates(arg0, arg1);
                    auto arg23 = mergeAggregates(arg2, arg3);
                    return mergeAggregates(arg01, arg23);
                }

                static  FORCEINLINE aggregate_type  mergeAggregates(double(&xn)[8], double(&xmean)[8], double(&xM2)[8])
                {
                    auto arg0 = mergeAggregates(xn[0], xn[1], xmean[0], xmean[1], xM2[0], xM2[1]);
                    auto arg1 = mergeAggregates(xn[2], xn[3], xmean[2], xmean[3], xM2[2], xM2[3]);
                    auto arg2 = mergeAggregates(xn[4], xn[5], xmean[4], xmean[5], xM2[4], xM2[5]);
                    auto arg3 = mergeAggregates(xn[6], xn[7], xmean[6], xmean[7], xM2[6], xM2[7]);
                    auto arg01 = mergeAggregates(arg0, arg1);
                    auto arg23 = mergeAggregates(arg2, arg3);
                    return mergeAggregates(arg01, arg23);
                }

                static FORCEINLINE aggregate_type mergeAggregates(double xn, double yn, double xmean, double ymean, double xM2, double yM2) {
                    double n = xn + yn;
                    double delta = ymean - xmean;
                    double delta2 = delta * delta;
                    double meanD = xmean + delta * yn / n;
                    double M2D = xM2 + yM2;
                    M2D += delta2 * xn * yn / n;
                    return { n, meanD, M2D };
                }
            };

            template<typename X, typename Z, typename DeviationOp, bool LastIndexFaster = true>
            void reductionCase1Scalar(const  int& second_rank, const Nd4jLong* inner_bases, const Nd4jLong* inner_strides, const  X* bufferX, Z* outputZ, bool biasCorrected)
            {
                using AggType = typename DeviationOp::aggregate_type;
                Nd4jLong inner_total;
                Nd4jLong inner_last = 0;
                int maxThreads = sd::Environment::getInstance().maxMasterThreads();
                if (second_rank == 1) {
                    inner_total = inner_bases[0];
                    if (inner_total < threadingThreshold) {
                        maxThreads = 1;
                    }
                    else {
                        auto gen = inner_total / threadingThreshold + 1;
                        maxThreads = gen > maxThreads ? maxThreads : gen;
                        //nd4j_printf("%ld %ld  mth %d %d\n", inner_total, threadingThreshold,  maxThreads, gen);
                    }
                }
                else {
                    inner_total = getLength<LastIndexFaster>(inner_bases, second_rank, 1, inner_last);
                    if (inner_total * inner_last < threadingThreshold) {
                        maxThreads = 1;
                    }
                    else {
                        auto gen = inner_total * inner_last / threadingThreshold + 1;
                        maxThreads = gen > maxThreads ? maxThreads : gen;
                        //nd4j_printf("%ld %ld  mth %d %d\n", inner_total, threadingThreshold, maxThreads, gen);
                    }
                }
#define BLOCKX4 1
                std::unique_ptr<AggType[]> aggs(new AggType[maxThreads]);
                AggType* ptrAggs = aggs.get();
                auto func = [ptrAggs, inner_last, second_rank, inner_bases, inner_strides, bufferX](uint64_t thread_id, int64_t start, int64_t stop, int64_t increment) -> void {
                    //LOG_CALLS(0)
                    const Nd4jLong inner_stride = LastIndexFaster ? inner_strides[second_rank - 1] : inner_strides[0];
                    Z argCurrent; X current;
                    if (second_rank == 1) {
                        const Nd4jLong loopTotal = stop - start;
#if defined(BLOCKX4)
                        if (loopTotal >= 2048) {
                            AggType agg, agg1, agg2, agg3;
                            if (inner_stride == 1) {
                                //use vector version
                                Nd4jLong loopTotal4_32 = loopTotal & (-32);
                                auto loopCount4 = loopTotal4_32 / 4;
                                auto loopCount4_8th = loopCount4 / 8;
                                auto tail = (loopTotal & 31);
                                const X* buffer0 = bufferX + start * 1;
                                const X* buffer1 = buffer0 + 1 * loopCount4;
                                const X* buffer2 = buffer1 + 1 * loopCount4;
                                const X* buffer3 = buffer2 + 1 * loopCount4;
                                double xn[8] = {};
                                double x0mean[8] = {}; double x1mean[8] = {}; double x2mean[8] = {}; double x3mean[8] = {};
                                double x0M2[8] = {}; double x1M2[8] = {}; double x2M2[8] = {}; double x3M2[8] = {};
                                DeviationOp::updateInnerLoop4b_vec8(buffer0, buffer1, buffer2, buffer3, loopCount4_8th, xn, x0mean, x1mean, x2mean, x3mean, x0M2, x1M2, x2M2, x3M2);
                                //merge vectors
                                agg = DeviationOp::mergeAggregates(xn[0], x0mean, x0M2);
                                agg1 = DeviationOp::mergeAggregates(xn[0], x1mean, x1M2);
                                agg2 = DeviationOp::mergeAggregates(xn[0], x2mean, x2M2);
                                agg3 = DeviationOp::mergeAggregates(xn[0], x3mean, x3M2);
                                //tail merge to one of the aggs
                                if (tail > 0) {

                                    DeviationOp::template updateInnerLoop1b<true>(&(buffer3[loopCount4]), tail, agg3);
                                }
                            }
                            else {
                                auto loopCount4 = loopTotal / 4;
                                auto tail = (loopTotal & 3);
                                const X* buffer0 = bufferX + start * inner_stride;
                                const X* buffer1 = buffer0 + inner_stride * loopCount4;
                                const X* buffer2 = buffer1 + inner_stride * loopCount4;
                                const X* buffer3 = buffer2 + inner_stride * loopCount4;
                                DeviationOp::updateInnerLoop4b(buffer0, buffer1, buffer2, buffer3, loopCount4, inner_stride, agg, agg1, agg2, agg3);
                                //tail
                                if (tail > 0) {

                                    DeviationOp::template updateInnerLoop1b<true>(&(buffer3[loopCount4 * inner_stride]), tail, inner_stride, agg3);
                                }
                            }
                            //merge all
                            auto merged = DeviationOp::mergeAggregates(agg, agg1);
                            merged = DeviationOp::mergeAggregates(merged, agg2);
                            merged = DeviationOp::mergeAggregates(merged, agg3);
                            ptrAggs[thread_id] = merged;
                        }
                        else {
#endif
                            if (inner_stride == 1) {
                                const X* buffer0 = bufferX + start;
                                if (loopTotal > vectorizationThreshold) {
                                    auto length8 = loopTotal / 8;
                                    auto bufferTail = buffer0 + (loopTotal & (-8));
                                    auto tail = loopTotal & 7;
                                    double xn[8] = {};
                                    double xmean[8] = {};
                                    double xM2[8] = {};
                                    DeviationOp::updateInnerLoop1b_vec8(buffer0, length8, xn, xmean, xM2);
                                    auto agg = DeviationOp::mergeAggregates(xn[0], xmean, xM2);
                                    //add tail into
                                    if (tail > 0) {

                                        DeviationOp::template updateInnerLoop1b<true>(bufferTail, tail, agg);
                                    }
                                    ptrAggs[thread_id] = agg;
                                }
                                else {
                                    DeviationOp::updateInnerLoop1b(buffer0, loopTotal, ptrAggs[thread_id]);
                                }
                            }
                            else {
                                DeviationOp::updateInnerLoop1b(&(bufferX[start * inner_stride]), loopTotal, inner_stride, ptrAggs[thread_id]);
                            }
#if defined(BLOCKX4)
                        }
#endif
                    }
                    else {
                        //just lets do general case
                        if (inner_stride == 1) {

                            DeviationOp::template updateGeneralLoop1b<LastIndexFaster>(second_rank, bufferX, ptrAggs[thread_id], inner_bases, inner_strides, start, stop, inner_last, inner_stride);
                        }
                        else {

                            DeviationOp::template updateGeneralLoop1b<LastIndexFaster>(second_rank, bufferX, ptrAggs[thread_id], inner_bases, inner_strides, start, stop, inner_last, inner_stride);
                        }
                    }
                };
#if 0
                int Count = 0;
                func(0, 0, inner_total, 1);
#else
                int Count = samediff::Threads::parallel_tad(func, 0, inner_total, 1, maxThreads);
#endif
                auto current = ptrAggs[0];
                for (int i = 1; i < Count; i++) {
                    current = DeviationOp::mergeAggregates(current, ptrAggs[i]);
                }
                *outputZ = DeviationOp::getDeviation(current, biasCorrected);
            }

            template<typename X, typename Z, typename DeviationOp, bool LastIndexFaster = true, typename Movement>
            void reductionCases(Movement& movement, Nd4jLong loopTotal, const int& second_rank, const Nd4jLong* inner_bases, const Nd4jLong* inner_strides, const X* bufferX, Z* outputZ, bool biasCorrected)
            {
                using AggType = typename DeviationOp::aggregate_type;
                Nd4jLong inner_stride = LastIndexFaster ? inner_strides[second_rank - 1] : inner_strides[0];
                Nd4jLong loopTotal_K = loopTotal / 4;
                Nd4jLong loopTotalTail = loopTotal & 3;
                if (inner_stride == 1) {
                    if (second_rank == 1) {
                        LOG_CALLS(0)
                        Nd4jLong inner_total = getLength<LastIndexFaster>(inner_bases, second_rank);
                        auto loopCount4 = inner_total & (-8);
                        auto loopCount4_8th = inner_total / 8;
                        auto tail = inner_total & 7;
                        bool use_vector = loopCount4_8th > 16;
                        //nd4j_printf("++ %d %d %d \n", loopCount4, loopCount4_8th, tail);
                        for (Nd4jLong i = 0; i < loopTotal_K; i++) {
                            AggType agg0, agg1, agg2, agg3;
                            const X* buff0 = &(bufferX[movement.First()]);
                            Z* output0 = &(outputZ[movement.Second()]);
                            movement.increment();
                            const X* buff1 = &(bufferX[movement.First()]);
                            Z* output1 = &(outputZ[movement.Second()]);
                            movement.increment();
                            const X* buff2 = &(bufferX[movement.First()]);
                            Z* output2 = &(outputZ[movement.Second()]);
                            movement.increment();
                            const X* buff3 = &(bufferX[movement.First()]);
                            Z* output3 = &(outputZ[movement.Second()]);
                            movement.increment();
                            if (use_vector) {
                                double xn[8] = {};
                                double x0mean[8] = {}; double x1mean[8] = {}; double x2mean[8] = {}; double x3mean[8] = {};
                                double x0M2[8] = {}; double x1M2[8] = {}; double x2M2[8] = {}; double x3M2[8] = {};
                                DeviationOp::updateInnerLoop4b_vec8(buff0, buff1, buff2, buff3, loopCount4_8th, xn, x0mean, x1mean, x2mean, x3mean, x0M2, x1M2, x2M2, x3M2);
                                //merge vectors
                                agg0 = DeviationOp::mergeAggregates(xn[0], x0mean, x0M2);
                                agg1 = DeviationOp::mergeAggregates(xn[0], x1mean, x1M2);
                                agg2 = DeviationOp::mergeAggregates(xn[0], x2mean, x2M2);
                                agg3 = DeviationOp::mergeAggregates(xn[0], x3mean, x3M2);
                                if (tail > 0) {
                                    //tails into merged , this time for each

                                    DeviationOp::template updateInnerLoop4b<true>(&(buff0[loopCount4]), &(buff1[loopCount4]), &(buff2[loopCount4]), &(buff3[loopCount4]), tail, agg0, agg1, agg2, agg3);
                                }
                                //nd4j_printf("~~~ %f %f %f \n", agg0.n, agg0.mean, agg0.M2);
                            }
                            else {
                                DeviationOp::updateInnerLoop4b(buff0, buff1, buff2, buff3, inner_total, agg0, agg1, agg2, agg3);
                            }
                            *output0 = DeviationOp::getDeviation(agg0, biasCorrected);
                            *output1 = DeviationOp::getDeviation(agg1, biasCorrected);
                            *output2 = DeviationOp::getDeviation(agg2, biasCorrected);
                            *output3 = DeviationOp::getDeviation(agg3, biasCorrected);
                        }
                        for (Nd4jLong i = 0; i < loopTotalTail; i++) {
                            AggType agg0;
                            const X* buff0 = &(bufferX[movement.First()]);
                            Z* output0 = &(outputZ[movement.Second()]);
                            if(use_vector){
                                double xn[8] = {};
                                double x0mean[8] = {};
                                double x0M2[8] = {};
                                DeviationOp::updateInnerLoop1b_vec8(buff0, loopCount4_8th, xn, x0mean, x0M2);
                                //merge vectors
                                agg0 = DeviationOp::mergeAggregates(xn[0], x0mean, x0M2); 
                                if (tail > 0) {
                                    //tails into merged

                                    DeviationOp::template updateInnerLoop1b<true>(&(buff0[loopCount4]), tail, agg0);
                                }
                            }
                            else {
                                DeviationOp::updateInnerLoop1b(buff0,  inner_total, agg0);
                            }
                            movement.increment();
                            *output0 = DeviationOp::getDeviation(agg0, biasCorrected);
                        }
                    }
                    else {
                        Nd4jLong inner_last;
                        Nd4jLong inner_loop = getLength<LastIndexFaster>(inner_bases, second_rank, 1, inner_last);
                        if (second_rank == 2) {
                            LOG_CALLS(11)
                                //nd4j_printf("%d %d %d\n", inner_loop, inner_last, inner_stride);
                                for (Nd4jLong i = 0; i < loopTotal_K; i++) {
                                    const X* buffer0 = &(bufferX[movement.First()]);
                                    Z* output0 = &(outputZ[movement.Second()]);
                                    movement.increment();
                                    const X* buffer1 = &(bufferX[movement.First()]);
                                    Z* output1 = &(outputZ[movement.Second()]);
                                    movement.increment();
                                    const X* buffer2 = &(bufferX[movement.First()]);
                                    Z* output2 = &(outputZ[movement.Second()]);
                                    movement.increment();
                                    const X* buffer3 = &(bufferX[movement.First()]);
                                    Z* output3 = &(outputZ[movement.Second()]);
                                    movement.increment();

                                    DeviationOp::template reduceConstRankLoop4b<2, LastIndexFaster>(buffer0, buffer1, buffer2, buffer3, output0, output1, output2, output3, inner_bases, inner_strides,
                                        inner_loop, inner_last, biasCorrected);
                                }
                            for (Nd4jLong i = 0; i < loopTotalTail; i++) {
                                const X* buffer0 = &(bufferX[movement.First()]);

                                DeviationOp::template reduceConstRankLoop1b < 2, LastIndexFaster >(buffer0, &(outputZ[movement.Second()]), inner_bases, inner_strides,
                                    inner_loop, inner_last, biasCorrected);
                                movement.increment();
                            }
                        }
                        else if (second_rank == 3) {
                            LOG_CALLS(12)
                                for (Nd4jLong i = 0; i < loopTotal_K; i++) {
                                    const X* buffer0 = &(bufferX[movement.First()]);
                                    Z* output0 = &(outputZ[movement.Second()]);
                                    movement.increment();
                                    const X* buffer1 = &(bufferX[movement.First()]);
                                    Z* output1 = &(outputZ[movement.Second()]);
                                    movement.increment();
                                    const X* buffer2 = &(bufferX[movement.First()]);
                                    Z* output2 = &(outputZ[movement.Second()]);
                                    movement.increment();
                                    const X* buffer3 = &(bufferX[movement.First()]);
                                    Z* output3 = &(outputZ[movement.Second()]);
                                    movement.increment();

                                    DeviationOp::template reduceConstRankLoop4b<3, LastIndexFaster>(buffer0, buffer1, buffer2, buffer3, output0, output1, output2, output3, inner_bases, inner_strides,
                                        inner_loop, inner_last, biasCorrected);
                                }
                            for (Nd4jLong i = 0; i < loopTotalTail; i++) {
                                const X* buffer0 = &(bufferX[movement.First()]);

                                DeviationOp::template reduceConstRankLoop1b<3, LastIndexFaster>(buffer0, &(outputZ[movement.Second()]), inner_bases, inner_strides,
                                    inner_loop, inner_last, biasCorrected);
                                movement.increment();
                            }
                        }
                        else {
                            LOG_CALLS(13)
                            AggType agg;
                            for (Nd4jLong i = 0; i < loopTotal; i++) {
                                const X* buffer0 = &(bufferX[movement.First()]);

                                DeviationOp::template updateGeneralLoop1b<LastIndexFaster>(second_rank, buffer0, agg, inner_bases, inner_strides, 0,
                                    inner_loop, inner_last);
                                outputZ[movement.Second()] = DeviationOp::getDeviation(agg, biasCorrected);
                                movement.increment();
                            }
                        }
                    }
                }
                else {
                    if (second_rank == 1) {
                        LOG_CALLS(20)
                        Nd4jLong inner_total = getLength<LastIndexFaster>(inner_bases, second_rank);
                        for (Nd4jLong i = 0; i < loopTotal_K; i++) {
                            AggType agg, agg1, agg2, agg3;
                            const X* buffer0 = &(bufferX[movement.First()]);
                            Z* output0 = &(outputZ[movement.Second()]);
                            movement.increment();
                            const X* buffer1 = &(bufferX[movement.First()]);
                            Z* output1 = &(outputZ[movement.Second()]);
                            movement.increment();
                            const X* buffer2 = &(bufferX[movement.First()]);
                            Z* output2 = &(outputZ[movement.Second()]);
                            movement.increment();
                            const X* buffer3 = &(bufferX[movement.First()]);
                            Z* output3 = &(outputZ[movement.Second()]);
                            movement.increment();
                            DeviationOp::updateInnerLoop4b(buffer0, buffer1, buffer2, buffer3, inner_total, inner_stride, agg, agg1, agg2, agg3);
                            *output0 = DeviationOp::getDeviation(agg, biasCorrected);
                            *output1 = DeviationOp::getDeviation(agg1, biasCorrected);
                            *output2 = DeviationOp::getDeviation(agg2, biasCorrected);
                            *output3 = DeviationOp::getDeviation(agg3, biasCorrected);
                        }
                        for (Nd4jLong i = 0; i < loopTotalTail; i++) {
                            AggType agg;
                            const X* buffer0 = &(bufferX[movement.First()]);
                            Z* output0 = &(outputZ[movement.Second()]);
                            DeviationOp::updateInnerLoop1b(buffer0, inner_total, inner_stride, agg);
                            movement.increment();
                            *output0 = DeviationOp::getDeviation(agg, biasCorrected);
                        }
                    }
                    else {
                        Nd4jLong inner_last;
                        Nd4jLong inner_loop = getLength<LastIndexFaster>(inner_bases, second_rank, 1, inner_last);
                        if (second_rank == 2) {
                            LOG_CALLS(21)
                            for (Nd4jLong i = 0; i < loopTotal_K; i++) {
                                const X* buffer0 = &(bufferX[movement.First()]);
                                Z* output0 = &(outputZ[movement.Second()]);
                                movement.increment();
                                const X* buffer1 = &(bufferX[movement.First()]);
                                Z* output1 = &(outputZ[movement.Second()]);
                                movement.increment();
                                const X* buffer2 = &(bufferX[movement.First()]);
                                Z* output2 = &(outputZ[movement.Second()]);
                                movement.increment();
                                const X* buffer3 = &(bufferX[movement.First()]);
                                Z* output3 = &(outputZ[movement.Second()]);
                                movement.increment();

                                DeviationOp::template reduceConstRankLoop4b<2, LastIndexFaster>(buffer0, buffer1, buffer2, buffer3, output0, output1, output2, output3, inner_bases, inner_strides,
                                        inner_loop, inner_last, inner_stride, biasCorrected);
                            }
                            for (Nd4jLong i = 0; i < loopTotalTail; i++) {
                                const X* buffer0 = &(bufferX[movement.First()]);

                                DeviationOp::template reduceConstRankLoop1b < 2, LastIndexFaster >(buffer0, &(outputZ[movement.Second()]), inner_bases, inner_strides,
                                    inner_loop, inner_last, inner_stride, biasCorrected);
                                movement.increment();
                            }
                        }
                        else if (second_rank == 3) {
                            LOG_CALLS(22)
                            for (Nd4jLong i = 0; i < loopTotal_K; i++) {
                                const X* buffer0 = &(bufferX[movement.First()]);
                                Z* output0 = &(outputZ[movement.Second()]);
                                movement.increment();
                                const X* buffer1 = &(bufferX[movement.First()]);
                                Z* output1 = &(outputZ[movement.Second()]);
                                movement.increment();
                                const X* buffer2 = &(bufferX[movement.First()]);
                                Z* output2 = &(outputZ[movement.Second()]);
                                movement.increment();
                                const X* buffer3 = &(bufferX[movement.First()]);
                                Z* output3 = &(outputZ[movement.Second()]);
                                movement.increment();

                                DeviationOp::template reduceConstRankLoop4b<3, LastIndexFaster>(buffer0, buffer1, buffer2, buffer3, output0, output1, output2, output3, inner_bases, inner_strides,
                                        inner_loop, inner_last, inner_stride, biasCorrected);
                            }
                            for (Nd4jLong i = 0; i < loopTotalTail; i++) {
                                const X* buffer0 = &(bufferX[movement.First()]);

                                DeviationOp::template reduceConstRankLoop1b<3, LastIndexFaster>(buffer0, &(outputZ[movement.Second()]), inner_bases, inner_strides,
                                    inner_loop, inner_last, inner_stride, biasCorrected);
                                movement.increment();
                            }
                        }
                        else {
                            LOG_CALLS(23)
                            AggType agg;
                            for (Nd4jLong i = 0; i < loopTotal; i++) {
                                const X* buffer0 = &(bufferX[movement.First()]);

                                DeviationOp::template updateGeneralLoop1b<LastIndexFaster>(second_rank, buffer0, agg, inner_bases, inner_strides, 0,
                                    inner_loop, inner_last, inner_stride);
                                outputZ[movement.Second()] = DeviationOp::getDeviation(agg, biasCorrected);
                                movement.increment();
                            }
                        }
                    }
                }
            }

            template<typename X, typename Z, typename DeviationOp, bool LastIndexFaster = true>
            void reductionCaseNonScalar(const  int& first_rank, const int& output_rank, bool squashed, const  int& second_rank,
                const Nd4jLong*& outer_bases, const Nd4jLong* outer_strides, const Nd4jLong* output_strides, const Nd4jLong& output_stride,
                const Nd4jLong*& inner_bases, const Nd4jLong* inner_strides, const X* bufferX, Z* outputZ, bool biasCorrected)
            {
                Nd4jLong total = getLength<LastIndexFaster>(outer_bases, first_rank);
                Nd4jLong inner_stride = LastIndexFaster ? inner_strides[second_rank - 1] : inner_strides[0];
                Nd4jLong outer_stride = LastIndexFaster ? outer_strides[second_rank - 1] : outer_strides[0];
                auto func = [first_rank, output_rank, squashed, outer_bases, outer_strides, output_strides, output_stride, second_rank, inner_bases, inner_strides, bufferX, outputZ, biasCorrected](uint64_t thread_id, int64_t start, int64_t stop, int64_t increment) -> void {
                    Nd4jLong loopTotal = stop - start;
                    Nd4jLong stride = LastIndexFaster ? outer_strides[first_rank - 1] : outer_strides[0];
                    if (first_rank == 1) {
                        if (stride == 1) {
                            ZipGenericCoordsRank1Stride1 movement;
                            movement.init(nullptr, nullptr, nullptr, 0, start);
                            reductionCases<X, Z, DeviationOp, LastIndexFaster>(movement, loopTotal, second_rank, inner_bases, inner_strides, bufferX, outputZ, biasCorrected);
                        }
                        else {
                            ZipGenericCoordsRank1BothStrideN movement;
                            movement.init(nullptr, &stride, &output_stride, 0, start);
                            reductionCases<X, Z, DeviationOp, LastIndexFaster>(movement, loopTotal, second_rank, inner_bases, inner_strides, bufferX, outputZ, biasCorrected);
                        }
                    }
                    else if (squashed && first_rank <= output_rank) {
                        if (first_rank == 2) {
                            if (output_stride == 1) {
                                ZipGenericCoordsConstMovementSecondStride1<2, LastIndexFaster> movement;
                                movement.init(outer_bases, outer_strides, nullptr, first_rank, start);
                                reductionCases<X, Z, DeviationOp, LastIndexFaster>(movement, loopTotal, second_rank, inner_bases, inner_strides, bufferX, outputZ, biasCorrected);
                            }
                            else {
                                ZipGenericCoordsConstMovementSecondStrideN<2, LastIndexFaster> movement;
                                movement.init(outer_bases, outer_strides, &output_stride, first_rank, start);
                                reductionCases<X, Z, DeviationOp, LastIndexFaster>(movement, loopTotal, second_rank, inner_bases, inner_strides, bufferX, outputZ, biasCorrected);
                            }
                        }
                        else if (first_rank == 3) {
                            if (output_stride == 1) {
                                ZipGenericCoordsConstMovementSecondStride1<3, LastIndexFaster> movement;
                                movement.init(outer_bases, outer_strides, nullptr, first_rank, start);
                                reductionCases<X, Z, DeviationOp, LastIndexFaster>(movement, loopTotal, second_rank, inner_bases, inner_strides, bufferX, outputZ, biasCorrected);
                            }
                            else {
                                ZipGenericCoordsConstMovementSecondStrideN<3, LastIndexFaster> movement;
                                movement.init(outer_bases, outer_strides, &output_stride, first_rank, start);
                                reductionCases<X, Z, DeviationOp, LastIndexFaster>(movement, loopTotal, second_rank, inner_bases, inner_strides, bufferX, outputZ, biasCorrected);
                            }
                        }
                        else {
                            ZipGenericCoordsMovementSecondStrideN< LastIndexFaster> movement;
                            movement.init(outer_bases, outer_strides, &output_stride, first_rank, start);
                            reductionCases<X, Z, DeviationOp, LastIndexFaster>(movement, loopTotal, second_rank, inner_bases, inner_strides, bufferX, outputZ, biasCorrected);
                        }
                    }
                    else {
                        ZipGenericCoordsMovement<LastIndexFaster> movement;
                        movement.init(outer_bases, outer_strides, output_strides, first_rank, start);
                        reductionCases<X, Z, DeviationOp, LastIndexFaster>(movement, loopTotal, second_rank, inner_bases, inner_strides, bufferX, outputZ, biasCorrected);
                    }
                };
#if 0
                func(0, 0, total, 1);
#else
                //
                uint32_t numThreads = sd::Environment::getInstance().maxMasterThreads();
                Nd4jLong inner_total = getLength<LastIndexFaster>(inner_bases, second_rank);
                if (total * inner_total <= threadingThreshold) {
                    numThreads = 1;
                }
                else {
                    if (inner_stride > outer_stride && total <= 1024) {
                        auto desired = total > 4 ? (total / 4) : 1;
                        numThreads = numThreads > desired ? desired : numThreads;
                    }
                }
                samediff::Threads::parallel_tad(func, 0, total, 1, numThreads);
#endif
            }

            template<typename X, typename Z, typename DeviationOp>
            void  reduction_(const NDArray& input, NDArray& output, const std::vector<int>& dimensions, bool biasCorrected) {
                //nd4j_printf("___%s_________%d+\n", __PRETTY_FUNCTION__, 0);
                char input_order = input.ordering();
                bool try_squash_outer = (input_order == output.ordering()) && output.ews() != 0;
                auto input_shapeInfo = input.shapeInfo();
                auto output_shapeInfo = output.shapeInfo();
                const Nd4jLong  rank = input_shapeInfo[0];
                const Nd4jLong* input_bases = &(input_shapeInfo[1]);
                const Nd4jLong* input_strides = &(input_shapeInfo[rank + 1]);
                const Nd4jLong  output_rank = output_shapeInfo[0];
                const Nd4jLong* output_strides = &(output_shapeInfo[output_rank + 1]);
                Nd4jLong new_bases[MAX_RANK];
                Nd4jLong new_strides[MAX_RANK];
                int first_begin, first_end, second_begin, second_end;
                //rePartition into two parts based on the selection
                rePartition(input_order, dimensions, rank, input_bases, input_strides, new_bases, new_strides, first_begin, first_end, second_begin, second_end, try_squash_outer, true);
                int first_rank = first_end - first_begin; //the first rank can be 0 for scalar cases
                int second_rank = second_end - second_begin;
                auto bufferX = input.bufferAsT<X>();
                auto outputZ = output.bufferAsT<Z>();
                const Nd4jLong* outer_bases = &(new_bases[first_begin]);
                const Nd4jLong* outer_strides = &(new_strides[first_begin]);
                const Nd4jLong* inner_bases = &(new_bases[second_begin]);
                const Nd4jLong* inner_strides = &(new_strides[second_begin]);
                const Nd4jLong output_stride = output.ordering() == 'c' ? output_strides[output_rank - 1] : output_strides[0];
                if (input_order == 'c') {
                    if (first_rank == 0) {
                        reductionCase1Scalar<X, Z, DeviationOp>(second_rank, inner_bases, inner_strides, bufferX, outputZ, biasCorrected);
                    }
                    else {
                        reductionCaseNonScalar<X, Z, DeviationOp>(first_rank, output_rank, try_squash_outer, second_rank, outer_bases, outer_strides, output_strides,
                            output_stride, inner_bases, inner_strides, bufferX, outputZ, biasCorrected);
                    }
                }
                else {
                    if (first_rank == 0) {
                        LOG_CALLS(100);
                        reductionCase1Scalar<X, Z, DeviationOp, false>(second_rank, inner_bases, inner_strides, bufferX, outputZ, biasCorrected);
                    }
                    else {
                        LOG_CALLS(101);
                        reductionCaseNonScalar<X, Z, DeviationOp, false>(first_rank, output_rank, try_squash_outer, second_rank, outer_bases, outer_strides, output_strides,
                            output_stride, inner_bases, inner_strides, bufferX, outputZ, biasCorrected);
                    }
                }
            }


            template<typename X, typename Z>
            void  variance_(const NDArray& input, NDArray& output, const std::vector<int>& dimensions, bool biasCorrected) {
                return reduction_<X, Z, Deviation<X, Z>>(input, output, dimensions, biasCorrected);
            }

            template<typename X, typename Z>
            void  standardDeviation_(const NDArray& input, NDArray& output, const std::vector<int>& dimensions, bool biasCorrected) {
                return  reduction_<X, Z, Deviation<X, Z, true>>(input, output, dimensions, biasCorrected);
            }
        }
    }
}

