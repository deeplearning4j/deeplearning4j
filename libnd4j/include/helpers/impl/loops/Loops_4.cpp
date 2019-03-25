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

#include "helpers/Loops.hpp"

using namespace simdOps;

namespace nd4j {
    //////////////////////////////////////////////////////////////////////////////
    Loops::LoopKind Loops::deduceKindOfLoopXYZ(const Nd4jLong* xShapeInfo, const Nd4jLong* yShapeInfo, const Nd4jLong* zShapeInfo) {

        const int xRank = shape::rank(xShapeInfo);

        const Nd4jLong xEws = shape::elementWiseStride(xShapeInfo);
        const Nd4jLong yEws = shape::elementWiseStride(yShapeInfo);
        const Nd4jLong zEws = shape::elementWiseStride(zShapeInfo);

        int temp;
        const bool xVector = shape::isCommonVector(xShapeInfo, temp);
        const bool yVector = shape::isCommonVector(yShapeInfo, temp);
        const bool zVector = shape::isCommonVector(zShapeInfo, temp);

        const char xOrder = shape::order(xShapeInfo);
        const char yOrder = shape::order(yShapeInfo);
        const char zOrder = shape::order(zShapeInfo);

        const bool shapesSame = shape::shapeEquals(xShapeInfo, yShapeInfo) && shape::shapeEquals(xShapeInfo, zShapeInfo);

        if (xEws == 1 && yEws == 1 && zEws == 1 && ((xOrder == yOrder && xOrder == zOrder) || ((xVector || xOrder == 'c') && (yVector || yOrder == 'c') && (zVector || zOrder == 'c'))))
            return EWS1;

        if(xEws > 0 && yEws > 0 && zEws > 0     && ((xOrder == yOrder && xOrder == zOrder) || ((xVector || xOrder == 'c') && (yVector || yOrder == 'c') && (zVector || zOrder == 'c'))))
            return EWSNONZERO;

        if(xRank == 1 && shapesSame)
            return RANK1;

        if(xRank == 2 && shapesSame)
            return RANK2;

        if(xRank == 3 && shapesSame)
            return RANK3;

        if(xRank == 4 && shapesSame)
            return RANK4;

        if(xRank == 5 && shapesSame)
            return RANK5;

        return COMMON;
    }

//////////////////////////////////////////////////////////////////////////////
    Loops::LoopKind Loops::deduceKindOfLoopXZ(const Nd4jLong* xShapeInfo, const Nd4jLong* zShapeInfo) {


        const int xRank = shape::rank(xShapeInfo);

        const Nd4jLong xEws = shape::elementWiseStride(xShapeInfo);
        const Nd4jLong zEws = shape::elementWiseStride(zShapeInfo);

        const char xOrder = shape::order(xShapeInfo);
        const char zOrder = shape::order(zShapeInfo);

        int temp;
        const bool xVector = shape::isCommonVector(xShapeInfo, temp);
        const bool zVector = shape::isCommonVector(zShapeInfo, temp);

        const bool shapesSame = shape::shapeEquals(xShapeInfo, zShapeInfo);

        if(xEws == 1 && zEws == 1 && ((xOrder == zOrder) || ((xVector || xOrder == 'c') && (zVector || zOrder == 'c'))))
            return EWS1;

        if(xEws > 0 && zEws > 0   && ((xOrder == zOrder) || ((xVector || xOrder == 'c') && (zVector || zOrder == 'c'))))
            return EWSNONZERO;

        if(xRank == 1 && shapesSame)
            return RANK1;

        if(xRank == 2 && shapesSame)
            return RANK2;

        if(xRank == 3 && shapesSame)
            return RANK3;

        if(xRank == 4 && shapesSame)
            return RANK4;

        if(xRank == 5 && shapesSame)
            return RANK5;

        if(xEws > 0 && (xOrder == 'c' || xVector) && zEws == 0)
            return X_EWSNONZERO;

        if(zEws > 0 && (zOrder == 'c' || zVector) && xEws == 0)
            return Z_EWSNONZERO;

        return COMMON;
    }

//////////////////////////////////////////////////////////////////////////////
    Loops::LoopKind Loops::deduceKindOfLoopTadXZ(const Nd4jLong* xShapeInfo, const Nd4jLong* zShapeInfo, const Nd4jLong* tadShapeInfo) {

        const int tadRank = shape::rank(tadShapeInfo);

        const Nd4jLong tadEws = shape::elementWiseStride(tadShapeInfo);
        const Nd4jLong zEws = shape::elementWiseStride(zShapeInfo);

        const char xOrder = shape::order(tadShapeInfo);
        const char zOrder = shape::order(zShapeInfo);

        int temp;
        const bool xVector = shape::isCommonVector(tadShapeInfo, temp);
        const bool zVector = shape::isCommonVector(zShapeInfo, temp);

        if(shape::length(tadShapeInfo) * shape::length(zShapeInfo) <= Environment::getInstance()->elementwiseThreshold() && shape::rank(xShapeInfo) == 2 &&
            tadEws > 1 && zEws == 1 && ((xOrder == zOrder) || ((xVector || xOrder == 'c') && (zVector || zOrder == 'c'))))
            return SMALLARR2DX;

        if(tadEws == 1 && zEws == 1 && ((xOrder == zOrder) || ((xVector || xOrder == 'c') && (zVector || zOrder == 'c'))))
            return EWS1;

        if(tadEws > 0 && zEws > 0   && ((xOrder == zOrder) || ((xVector || xOrder == 'c') && (zVector || zOrder == 'c'))))
            return EWSNONZERO;

        if(tadRank == 1 && zEws == 1 && (zVector || zOrder == 'c'))
            return RANK1;

        if(tadRank == 2 && zEws == 1 && (zVector || zOrder == 'c'))
            return RANK2;

        if(tadRank == 3 && zEws == 1 && (zVector || zOrder == 'c'))
            return RANK3;

        if(tadRank == 4 && zEws == 1 && (zVector || zOrder == 'c'))
            return RANK4;

        if(tadRank == 5 && zEws == 1 && (zVector || zOrder == 'c'))
            return RANK5;

        if(tadEws > 0 && (xVector || xOrder == 'c') && zEws == 0)
            return X_EWSNONZERO;

        if(zEws > 0 && (zOrder == 'c' || zVector) && tadEws == 0)
            return Z_EWSNONZERO;

        return COMMON;
    }


    template <typename X>
    class IndexReduceWrapper {
    public:
        template <typename OpType>
        static void wrapper(const X *x, const Nd4jLong* xShapeInfo, Nd4jLong *z, const Nd4jLong *zShapeInfo, const Nd4jLong *tadShapeInfo, const Nd4jLong *tadOffset, X *extras) {
            Loops::loopIndexTadXZ<X, OpType>(x, xShapeInfo, z, zShapeInfo, tadShapeInfo, tadOffset, extras);
        }

        static void wrap(const int opNum, const X *x, const Nd4jLong* xShapeInfo, Nd4jLong *z, const Nd4jLong *zShapeInfo, const Nd4jLong *tadShapeInfo, const Nd4jLong *tadOffset, X *extras) {
            DISPATCH_BY_OPNUM_T(wrapper, PARAMS(x, xShapeInfo, z, zShapeInfo, tadShapeInfo, tadOffset, extras), INDEX_REDUCE_OPS);
        }
    };
    BUILD_SINGLE_TEMPLATE(template class IndexReduceWrapper, , LIBND4J_TYPES);
}