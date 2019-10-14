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
//  @author raver119@gmail.com
//  @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <types/types.h>
#include <ShapeUtils.h>
#include <op_boilerplate.h>
#include <loops/reduce_float.h>
#include <loops/legacy_ops.h>
#include <OmpLaunchHelper.h>
#include <helpers/Loops.h>
#include <helpers/ConstantTadHelper.h>

using namespace simdOps;

namespace functions {
    namespace reduce {
        template <typename X, typename Z>
        template <typename OpType>
        void _CUDA_H ReduceFloatFunction<X,Z>::execScalar(void *vx,
                                                Nd4jLong *xShapeInfo,
                                                void *vextraParams,
                                                void *vz,
                                                Nd4jLong *zShapeInfo) {
            auto x = reinterpret_cast<X *>(vx);
            auto z = reinterpret_cast<Z *>(vz);
            auto extraParams = reinterpret_cast<Z *>(vextraParams);

            const Nd4jLong length = shape::length(xShapeInfo);
            auto xEws = shape::elementWiseStride(xShapeInfo);

            if (shape::isEmpty(xShapeInfo)) {
                if (std::is_same<OpType, simdOps::Mean<X,Z>>::value) {
                    z[0] = nd4j::DataTypeUtils::nanOrZero<Z>();
                } else {
                    z[0] = OpType::startingValue(x);
                }
                return;
            }

             if(nd4j::ArrayOptions::arrayType(xShapeInfo) == nd4j::ArrayType::EMPTY) {
                if(nd4j::ArrayOptions::arrayType(zShapeInfo) == nd4j::ArrayType::EMPTY)
                    return;
                const auto startingVal = OpType::startingValue(x);

                for (uint i = 0; i < length; i++)
                    z[i] = startingVal;

                return;
            }

            if (xEws > 0) {
                z[0] = execScalar<OpType>(x, xEws, length, extraParams);
            }
            else {
                auto startingValue = OpType::startingValue(x);
                uint xShapeInfoCast[MAX_RANK];
                const bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);

                for (auto i = 0; i < length; i++)
                    startingValue = OpType::update(startingValue, OpType::op(x[shape::indexOffset(i, xShapeInfo, xShapeInfoCast, canCastX)], extraParams), extraParams);

                z[0] = OpType::postProcess(startingValue, length, extraParams);
            }
        }


        template <typename X, typename Z>
        template <typename OpType>
            Z _CUDA_H ReduceFloatFunction<X, Z>::execScalar(void *vx, Nd4jLong *xShapeInfo, void *vextraParams) {
                auto x = reinterpret_cast<X *>(vx);
                auto extraParams = reinterpret_cast<Z *>(vextraParams);

                const Nd4jLong length = shape::length(xShapeInfo);
                int xEws = shape::elementWiseStride(xShapeInfo);

                if (xEws > 0) {
                    return execScalar<OpType>(x, xEws, length, extraParams);
                }
                else {
                    auto startingValue = OpType::startingValue(x);
                    uint xShapeInfoCast[MAX_RANK];
                    bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);

                    for (auto i = 0; i < length; i++)
                        startingValue = OpType::update(startingValue, OpType::op(x[shape::indexOffset(i, xShapeInfo, xShapeInfoCast, canCastX)], extraParams), extraParams);

                    return OpType::postProcess(startingValue, length, extraParams);
                }
            }

        template <typename X, typename Y>
        Y ReduceFloatFunction<X, Y>::execScalar(const int opNum,
                void *x,
                Nd4jLong *xShapeInfo,
                void *extraParams) {
                RETURNING_DISPATCH_BY_OPNUM_TT(execScalar, PARAMS(x, xShapeInfo, extraParams), REDUCE_FLOAT_OPS);
        }

        template <typename X, typename Y>
        void ReduceFloatFunction<X, Y>::execScalar(const int opNum,
                                        void *x,
                                        Nd4jLong *xShapeInfo,
                                        void *extraParams,
                                        void *z,
                                        Nd4jLong *zShapeInfo) {
            DISPATCH_BY_OPNUM_TT(execScalar, PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo), REDUCE_FLOAT_OPS);
        }

        template <typename X, typename Y>
        void ReduceFloatFunction<X, Y>::exec(const int opNum,
                             void *x,
                             Nd4jLong *xShapeInfo,
                             void *extraParams,
                             void *z,
                             Nd4jLong *zShapeInfo,
                             int *dimension,
                             int dimensionLength,
                             Nd4jLong *tadShapeInfo,
                             Nd4jLong *tadOffset, int64_t start, int64_t stop) {
                DISPATCH_BY_OPNUM_TT(exec, PARAMS(x,
                                               xShapeInfo,
                                               extraParams,
                                               z,
                                               zShapeInfo,
                                               dimension,
                                               dimensionLength,
                                               tadShapeInfo,
                                               tadOffset, start, stop),
                                  REDUCE_FLOAT_OPS);
        }

        template <typename X, typename Z>
        template <typename OpType>
        void _CUDA_H ReduceFloatFunction<X,Z>::exec(void *vx,
                             Nd4jLong *xShapeInfo,
                             void *vextraParams,
                             void *vresult,
                             Nd4jLong *zShapeInfo,
                             int *dimension,
                             int dimensionLength,
                             Nd4jLong *tadShapeInfo,
                             Nd4jLong *tadOffset, int64_t start, int64_t stop) {

                auto x = reinterpret_cast<X *>(vx);
                auto z = reinterpret_cast<Z *>(vresult);
                auto extraParams = reinterpret_cast<Z *>(vextraParams);

                auto resultLength = shape::length(zShapeInfo);

                if(nd4j::ArrayOptions::arrayType(xShapeInfo) == nd4j::ArrayType::EMPTY) {
                    if(nd4j::ArrayOptions::arrayType(zShapeInfo) == nd4j::ArrayType::EMPTY)
                        return;
                    const auto startingVal = std::is_same<OpType, simdOps::Mean<X,Z>>::value ? nd4j::DataTypeUtils::nanOrZero<Z>() : static_cast<Z>(OpType::startingValue(x));

                    for (uint i = 0; i < resultLength; i++)
                        z[i] = startingVal;
                    return;
                }

                //pre squeezed: this is for keeping the pointer to the original
                //shape information for tad offset
                //the squeezed information doesn't render the right strides for
                //tad offset
                // || tad.wholeThing
                if (resultLength == 1 || dimension == nullptr || dimensionLength == shape::rank(xShapeInfo)) {
                    z[0] = execScalar<OpType>(x, xShapeInfo, extraParams);
                    return;
                }

                if (OpType::requiresSpecialAccumulation) {
                    OpType::execSpecial(x, xShapeInfo, extraParams, z, zShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffset);
                    return;
                }

                auto tadOnlyShapeInfo = tadShapeInfo;
                auto tadOffsets = tadOffset;

                if (tadOnlyShapeInfo == nullptr || tadOffsets == nullptr) {
                    if (dimensionLength < 0)
                        return;

                    auto tadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(xShapeInfo, dimension, dimensionLength);
                    tadOnlyShapeInfo = tadPack.primaryShapeInfo();
                    tadOffsets = tadPack.primaryOffsets();
                }

#ifdef INLINE_LOOPS
                nd4j::ReductionLoops<X,Z,Z>::template loopReduce<OpType>(x, xShapeInfo, z, zShapeInfo,  tadOnlyShapeInfo, tadOffsets, extraParams, start, stop);
#else
                nd4j::ReductionFloatLoops<X,Z>::template innerloopReduce<OpType>(x, xShapeInfo, z, zShapeInfo,  tadOnlyShapeInfo, tadOffsets, extraParams, start, stop);
#endif
            }


        template <typename X, typename Z>
        template<typename OpType>
        void _CUDA_H ReduceFloatFunction<X,Z>::exec(void *x,
                             Nd4jLong *xShapeInfo,
                             void *extraParams,
                             void *vresult,
                             Nd4jLong *resultShapeInfo) {
                // FIXME: wtf???
                auto z = reinterpret_cast<Z*>(vresult);
                z[0] = execScalar<OpType>(x, xShapeInfo, extraParams);
        }

        template <typename X, typename Z>
        template <typename OpType>
        Z _CUDA_H ReduceFloatFunction<X, Z>::execScalar(void *vx, Nd4jLong xEws, Nd4jLong length, void *vextraParams) {

            auto x = reinterpret_cast<X *>(vx);
            auto extraParams = reinterpret_cast<Z *>(vextraParams);
            auto startingVal = OpType::startingValue(x);

            if (xEws == 1) {
                for (auto i = 0; i < length; i++)
                    startingVal = OpType::update(startingVal, OpType::op(x[i], extraParams), extraParams);
            } else {
                for (auto i = 0; i < length; i++)
                    startingVal = OpType::update(startingVal, OpType::op(x[i * xEws], extraParams), extraParams);
            }

            return OpType::postProcess(startingVal, length, extraParams);
        }


        BUILD_DOUBLE_TEMPLATE(template class ND4J_EXPORT ReduceFloatFunction, , LIBND4J_TYPES, FLOAT_TYPES);
    }
}