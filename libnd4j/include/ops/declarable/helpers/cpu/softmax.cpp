/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
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

#include <ops/declarable/helpers/activations.h>
#include <helpers/ShapeUtils.h>
#include <numeric>
#include <helpers/ConstantTadHelper.h>
#include <execution/Threads.h>

namespace sd {
    namespace ops {
        namespace helpers {

            template <typename T>
            static void softMaxForVector_(void const* input, Nd4jLong const* inShapeInfo, void *output, Nd4jLong const* outShapeInfo) {

                auto inBuff  = reinterpret_cast<T const*>(input);
                auto outBuff = reinterpret_cast<T *>(output);

                T max = -DataTypeUtils::max<T>();
                T sum = 0.;
                int inEWS = shape::elementWiseStride(inShapeInfo);
                int outEWS = shape::elementWiseStride(outShapeInfo);
                int length = shape::length(inShapeInfo);

                if (inEWS >= 1 && outEWS >= 1) {

                    if (inEWS == 1 && outEWS == 1) {

                        for (int i = 0; i < length; i++)
                            max = sd::math::nd4j_max<T>(max, inBuff[i]);

                        for (int i = 0; i < length; i++) {
                            outBuff[i] = sd::math::nd4j_exp<T, T>(inBuff[i] - max);
                            sum += outBuff[i];
                        }

                        for (int i = 0; i < length; i++)
                            outBuff[i] /= sum;
                    }
                    else {

                        for (int i = 0; i < length; i++)
                            max = sd::math::nd4j_max<T>(max, inBuff[i * inEWS]);

                        for (int i = 0; i < length; i++) {
                            T r = sd::math::nd4j_exp<T, T>(inBuff[i * inEWS] - max);
                            outBuff[i * outEWS] = r;
                            sum += r;
                        }

                        for (int i = 0; i < length; i++)
                            outBuff[i * outEWS] /= sum;
                    }
                }
            }

            ///////////////////////////////////////////////////////////////////
            void softMaxForVector(sd::LaunchContext * context, const NDArray& input, NDArray& output) {

                if(!input.isVector() || !output.isVector())
                    throw std::runtime_error("ops::helpers::softMaxForVector function: input and output arrays must be vectors !");

                auto xType = input.dataType();
                BUILD_SINGLE_SELECTOR(xType, softMaxForVector_, (input.buffer(), input.shapeInfo(), output.buffer(), output.shapeInfo()), FLOAT_TYPES);
            }

            template <typename T>
            void softmax_loop(const T* input, T *output, const Nd4jLong * offsets, Nd4jLong numOfSubArrs, uint32_t tadLen);

#ifdef _OPENMP
            template <>
            FORCEINLINE void softmax_loop(const float* input, float *output, const Nd4jLong * offsets, Nd4jLong numOfSubArrs, uint32_t tadLen) {
#pragma omp parallel for default(shared)
                    for (Nd4jLong i = 0; i < numOfSubArrs; i++) {
                        auto inBuff = input + offsets[i];
                        auto outBuff = output + offsets[i];

                        float max = -DataTypeUtils::max<float>();
                        float sum = 0.f;

#pragma omp simd reduction(max:max)
                        for (uint j = 0; j < tadLen; ++j)
                            max = sd::math::nd4j_max<float>(max, inBuff[j]);

#pragma omp simd reduction(+:sum)
                        for (uint j = 0; j < tadLen; ++j) {
                            float temp = sd::math::nd4j_exp<float, float>(inBuff[j] - max);
                            outBuff[j] = temp;
                            sum += temp;
                        }

                        for (uint j = 0; j < tadLen; ++j)
                            outBuff[j] /= sum;
                    }
            }
#else
            template <>
            FORCEINLINE void softmax_loop(const float *input, float *output, const Nd4jLong *offsets, Nd4jLong numOfSubArrs, uint32_t tadLen) {
                auto func = PRAGMA_THREADS_FOR {
                    for (auto i = start; i < stop; i++) {
                        auto inBuff = input + offsets[i];
                        auto outBuff = output + offsets[i];

                        float max = -DataTypeUtils::max<float>();
                        float sum = 0.f;

                        for (uint j = 0; j < tadLen; ++j)
                            max = sd::math::nd4j_max<float>(max, inBuff[j]);

                        for (uint j = 0; j < tadLen; ++j) {
                            float temp = sd::math::nd4j_exp<float, float>(inBuff[j] - max);
                            outBuff[j] = temp;
                            sum += temp;
                        }

                        for (uint j = 0; j < tadLen; ++j)
                            outBuff[j] /= sum;
                    }
                };

                samediff::Threads::parallel_tad(func,0, numOfSubArrs);
            }

#endif


            template <typename T>
            FORCEINLINE void softmax_loop(const T *input, T *output, const Nd4jLong *offsets, Nd4jLong numOfSubArrs, uint32_t tadLen) {
                auto func = PRAGMA_THREADS_FOR {
                    for (auto i = start; i < stop; i++) {
                        auto inBuff = input + offsets[i];
                        auto outBuff = output + offsets[i];

                        T max = -DataTypeUtils::max<T>();
                        T sum(0.f);

#pragma omp simd reduction(maxT:max)
                        for (uint j = 0; j < tadLen; ++j)
                            max = sd::math::nd4j_max<T>(max, inBuff[j]);

#pragma omp simd reduction(sumT:sum)
                        for (uint j = 0; j < tadLen; ++j) {
                            T temp = sd::math::nd4j_exp<T, T>(inBuff[j] - max);
                            outBuff[j] = temp;
                            sum += temp;
                        }

                        for (uint j = 0; j < tadLen; ++j)
                            outBuff[j] /= sum;
                    }
                };

                samediff::Threads::parallel_tad(func,0, numOfSubArrs);
            }

//////////////////////////////////////////////////////////////////////////
            template <typename T>
            static void softmax_(sd::LaunchContext * context, const NDArray& input, NDArray& output, const int dimension) {

                const int rank = input.rankOf();

                if(input.isVector()) {

                    if(rank == 1 || input.sizeAt(dimension) != 1)
                        softMaxForVector_<T>(input.buffer(), input.shapeInfo(), output.buffer(), output.shapeInfo());
                    else
                        output = 1.;
                }
                else if(input.isSameShapeStrict(output)) {

                    TadPack tadPack  = sd::ConstantTadHelper::getInstance().tadForDimensions(input.shapeInfo(), dimension);
                    auto tadShapeInfo  = tadPack.primaryShapeInfo();
                    auto tadOffsets    = tadPack.primaryOffsets();
                    const uint numOfSubArrs = tadPack.numberOfTads();
                    const uint tadLen       = shape::length(tadShapeInfo);

                    if(shape::elementWiseStride(tadShapeInfo) == 1){
                        auto inBuff = input.bufferAsT<T>();
                        T *outBuff = output.bufferAsT<T>();

                        softmax_loop(inBuff, outBuff, tadOffsets, numOfSubArrs, tadLen);
                    }
                    else {

                        uint inShapeInfoCast[MAX_RANK];
                        bool canCast = sd::DataTypeUtils::castShapeInfo(tadShapeInfo, inShapeInfoCast);

                        auto offsets = new Nd4jLong[tadLen];
                        shape::calcOffsets(tadShapeInfo, offsets);

                        auto func = PRAGMA_THREADS_FOR {
                            for (auto i = start; i < stop; i++) {
                                auto inBuff = input.bufferAsT<T>() + tadOffsets[i];
                                auto outBuff = output.bufferAsT<T>() + tadOffsets[i];

                                T max = -DataTypeUtils::max<T>();
                                T sum = 0.f;

                                for (uint j = 0; j < tadLen; ++j)
                                    max = sd::math::nd4j_max<T>(max, inBuff[offsets[j]]);

                                for (uint j = 0; j < tadLen; ++j) {
                                    T temp = sd::math::nd4j_exp<T, T>(inBuff[offsets[j]] - max);
                                    outBuff[offsets[j]] = temp;
                                    sum += temp;
                                }

                                for (uint j = 0; j < tadLen; ++j)
                                    outBuff[offsets[j]] /= sum;
                            }
                        };

                        samediff::Threads::parallel_tad(func, 0, numOfSubArrs);

                        delete []offsets;
                    }
                }
                else {
                    NDArray max = input.reduceAlongDimension(sd::reduce::Max, {dimension}, true);
                    input.applyTrueBroadcast(sd::BroadcastOpsTuple::Subtract(), max, output, false);
                    output.applyTransform(sd::transform::Exp, output);
                    NDArray sum = output.reduceAlongDimension(sd::reduce::Sum, {dimension}, true);
                    output /= sum;
                }
            }


            ///////////////////////////////////////////////////////////////////
            void softmax(sd::LaunchContext * context, const NDArray& input, NDArray& output, const int dimension) {

                BUILD_SINGLE_SELECTOR(input.dataType(), softmax_, (context, input, output, dimension), FLOAT_TYPES);
            }

        }
    }
}