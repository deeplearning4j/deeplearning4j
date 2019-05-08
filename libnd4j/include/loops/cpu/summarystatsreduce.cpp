/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
// Created by raver119 on 18.12.17.
//

#include <types/types.h>
#include <op_boilerplate.h>
#include <loops/summarystatsreduce.h>
#include <helpers/shape.h>
#include <helpers/TAD.h>
#include <helpers/ConstantTadHelper.h>

using namespace simdOps;

namespace functions {
    namespace summarystats {


        template <typename X, typename Y>
        Y SummaryStatsReduce<X,Y>::execScalar(const int opNum,
                const bool biasCorrected,
                void *x,
                Nd4jLong *xShapeInfo,
                void *extraParams) {
            RETURNING_DISPATCH_BY_OPNUM_TT(execScalar, PARAMS(biasCorrected, x, xShapeInfo, extraParams), SUMMARY_STATS_OPS);
        }

        template <typename X, typename Y>
        void SummaryStatsReduce<X,Y>::execScalar(const int opNum,
                                              const bool biasCorrected,
                                              void *x,
                                              Nd4jLong *xShapeInfo,
                                              void *extraParams,
                                              void *z,
                                              Nd4jLong *resultShapeInfoBuffer) {
            DISPATCH_BY_OPNUM_TT(execScalar, PARAMS(biasCorrected, x, xShapeInfo, extraParams, z, resultShapeInfoBuffer), SUMMARY_STATS_OPS);
        }

        template <typename X, typename Y>
        void SummaryStatsReduce<X,Y>::exec(const int opNum,
                const bool biasCorrected,
                void *x,
                Nd4jLong *xShapeInfo,
                void *extraParams,
                void *z,
                Nd4jLong *resultShapeInfoBuffer,
                int *dimension,
                int dimensionLength) {
            DISPATCH_BY_OPNUM_TT(exec, PARAMS(biasCorrected, x, xShapeInfo, extraParams, z, resultShapeInfoBuffer, dimension, dimensionLength), SUMMARY_STATS_OPS);
        }

        template <typename X, typename Z>
        template <typename OpType >
        void SummaryStatsReduce<X,Z>::execScalar(const bool biasCorrected,
                                              void *vx,
                                              Nd4jLong *xShapeInfo,
                                              void *vextraParams,
                                              void *vz,
                                              Nd4jLong *resultShapeInfoBuffer) {
            auto z = reinterpret_cast<Z*>(vz);
            z[0] = execScalar<OpType>(biasCorrected, vx, xShapeInfo, vextraParams);
        }

        template <typename X, typename Z>
        template <typename OpType >
        Z SummaryStatsReduce<X,Z>::execScalar(const bool biasCorrected, void *vx, Nd4jLong *xShapeInfo, void *vextraParams) {

            auto x = reinterpret_cast<X *>(vx);
            auto extraParams = reinterpret_cast<Z *>(vextraParams);

            SummaryStatsData<X> startingIndex;
            startingIndex.initialize();
            auto length = shape::length(xShapeInfo);
            
            uint xShapeInfoCast[MAX_RANK];
            const bool canCast = nd4j::DataTypeUtils::castShapeInfo<uint>(xShapeInfo, xShapeInfoCast);
            
            for (Nd4jLong i = 0; i < length; i++) {
                                        
                auto xOffset = shape::indexOffset(i, xShapeInfo, xShapeInfoCast, length, canCast);

                SummaryStatsData<X> curr;
                curr.initWithValue(x[xOffset]);
                startingIndex = update(startingIndex, curr, extraParams);
            }

            return OpType::getValue(biasCorrected, startingIndex);            
        }

        template <typename X, typename Z>
        template <typename OpType >
        void SummaryStatsReduce<X,Z>::exec(const bool biasCorrected,
                void *vx,
                Nd4jLong *xShapeInfo,
                void *vextraParams,
                void *vresult,
                Nd4jLong *resultShapeInfoBuffer,
                int *dimension,
                int dimensionLength) {
            auto x = reinterpret_cast<X *>(vx);
            auto z = reinterpret_cast<Z *>(vresult);
            auto extraParams = reinterpret_cast<Z *>(vextraParams);

            if (shape::isScalar(resultShapeInfoBuffer)) {
                z[0] = execScalar<OpType>(biasCorrected, x, xShapeInfo, extraParams);
                return;
            }



            //no-op
            if (dimensionLength < 1)
                return;

            auto tadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(xShapeInfo, dimension, dimensionLength);

            int resultLength = shape::length(resultShapeInfoBuffer);
            //pre squeezed: this is for keeping the pointer to the original
            //shape information for tad offset
            //the squeezed information doesn't render the right strides for
            //tad offset
            if (resultLength == 1 || dimensionLength == shape::rank(xShapeInfo) || tadPack.numberOfTads() == 1) {
                z[0] = execScalar<OpType>(biasCorrected, x, xShapeInfo, extraParams);
                return;
            }

            auto tadShapeShapeInfo = tadPack.primaryShapeInfo();
            auto tadLength = shape::length(tadPack.primaryShapeInfo());
            auto tadEWS = shape::elementWiseStride(tadPack.primaryShapeInfo());
            auto tadOrder = shape::order(tadPack.primaryShapeInfo());

            uint tadShapeShapeInfoCast[MAX_RANK];
            const bool canCast = tadEWS == 1 && tadOrder == 'c' ? false : nd4j::DataTypeUtils::castShapeInfo<uint>(tadShapeShapeInfo, tadShapeShapeInfoCast);

            PRAGMA_OMP_PARALLEL_FOR
            for (int r = 0; r < resultLength; r++) {
                        
                auto tadOffsetForBlock = tadPack.primaryOffsets()[r];
                auto tx = x + tadOffsetForBlock;
                SummaryStatsData<X> comp;
                comp.initWithValue(tx[0]);

                if (tadEWS == 1 && tadOrder == 'c') {
                    for (int i = 1; i < tadLength; i ++) {
                        SummaryStatsData <X> indexVal2;
                        indexVal2.initWithValue(tx[i]);

                        comp = update(comp, OpType::op(indexVal2, extraParams), extraParams);
                    }
                } 
                else {
                    for (int i = 1; i < tadLength; i ++) {
                        auto xOffset = shape::indexOffset(i, tadShapeShapeInfo, tadShapeShapeInfoCast, tadLength, canCast);

                        SummaryStatsData <X> indexVal2;
                        indexVal2.initWithValue(tx[xOffset]);

                        comp = update(comp, OpType::op(indexVal2, extraParams), extraParams);
                    }
                }

                z[r] = OpType::getValue(biasCorrected, comp);
            }     
        }


        BUILD_DOUBLE_TEMPLATE(template class ND4J_EXPORT SummaryStatsReduce, , LIBND4J_TYPES, FLOAT_TYPES);
    }
}