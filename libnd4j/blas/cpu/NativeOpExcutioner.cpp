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


#include <vector>
#include <pointercast.h>
#include "../NativeOpExcutioner.h"
#include <types/types.h>

#include <pairwise_bool.h>
#include <broadcasting_bool.h>
#include <scalar_bool.h>

#include <loops/transform_float.h>
#include <loops/transform_bool.h>
#include <loops/transform_any.h>
#include <loops/transform_same.h>
#include <loops/transform_strict.h>

#include <loops/reduce_float.h>
#include <loops/reduce_same.h>
#include <loops/reduce_bool.h>
#include <loops/reduce_long.h>

#include <loops/broadcasting.h>
#include <loops/indexreduce.h>
#include <loops/pairwise_transform.h>
#include <loops/reduce_float.h>
#include <loops/reduce3.h>
#include <loops/summarystatsreduce.h>
#include <loops/transform_same.h>
#include <loops/scalar.h>
#include <loops/random.h>
#include <pointercast.h>
#include <graph/exceptions/datatype_exception.h>
#include <loops/BroadcastScalarConverter.h>
#include <helpers/ConstantTadHelper.h>



////////////////////////////////////////////////////////////////////////
/**
*
* @param opNum
* @param x
* @param xShapeInfo
* @param extraParams
* @param result
* @param resultShapeInfo
*/
void NativeOpExcutioner::execIndexReduceScalar(int opNum, void *x, Nd4jLong *xShapeInfo, void *extraParams, void *vz, Nd4jLong *zShapeInfo) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
    auto z = reinterpret_cast<Nd4jLong*>(vz);

    BUILD_SINGLE_SELECTOR(xType, z[0] = functions::indexreduce::IndexReduce, ::execScalar(opNum, x,xShapeInfo,extraParams), LIBND4J_TYPES);
}

////////////////////////////////////////////////////////////////////////
/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 * @param result
 * @param resultShapeInfoBuffer
 * @param dimension
 * @param dimensionLength
 */

void NativeOpExcutioner::execIndexReduce(int opNum,
        void *x,
        Nd4jLong *xShapeInfo,
        void *extraParams,
        Nd4jLong *result,
        Nd4jLong *resultShapeInfoBuffer,
        int *dimension,
        int dimensionLength,
        Nd4jLong *tadShapeInfo,
        Nd4jLong *tadOffsets) {

    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);

    BUILD_SINGLE_SELECTOR(xType, functions::indexreduce::IndexReduce, ::exec(opNum, x, xShapeInfo, extraParams, result, resultShapeInfoBuffer, dimension, dimensionLength, tadShapeInfo, tadOffsets), LIBND4J_TYPES);
}

////////////////////////////////////////////////////////////////////////
/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param y
 * @param yShapeInfo
 * @param result
 * @param resultShapeInfo
 * @param dimension
 * @param dimensionLength
 */

void NativeOpExcutioner::execBroadcast(int opNum, void *x, Nd4jLong *xShapeInfo, void *y, Nd4jLong *yShapeInfo, void *result, Nd4jLong *resultShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadOnlyShapeInfoZ, Nd4jLong *tadOffsetsZ) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
    auto yType = nd4j::ArrayOptions::dataType(yShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(resultShapeInfo);

    if (!nd4j::Environment::getInstance()->isExperimentalBuild())
    if ((yType != xType && yType != nd4j::DataType::BOOL) || xType != zType)
        throw nd4j::datatype_exception::build("NativeOps::execBroadcast both operands must have same data type", xType, yType);

    auto xRank = shape::rank(xShapeInfo);
    auto yRank = shape::rank(yShapeInfo);
    auto zRank = shape::rank(resultShapeInfo);

    auto xOrder = shape::order(xShapeInfo);
    auto yOrder = shape::order(yShapeInfo);
    auto zOrder = shape::order(resultShapeInfo);

    auto xEws = shape::elementWiseStride(xShapeInfo);
    auto yEws = shape::elementWiseStride(yShapeInfo);
    auto zEws = shape::elementWiseStride(resultShapeInfo);

    int axis = 0;
    auto isVector = shape::isCommonVector(yShapeInfo, axis);
    auto isConvertible = nd4j::isConvertibleToScalar((nd4j::broadcast::Ops) opNum);

    // add column vector case for C ordered columnAdd
    if (xOrder == 'c' && zOrder == 'c' && xEws == 1 && zEws == 1 && yEws == 1 && xRank == 2 && isVector && dimensionLength == 1 && dimension[0] == 0 && isConvertible) {
        // invoke scalar along dimension here
        auto scalarOp = nd4j::convertToScalar((nd4j::broadcast::Ops) opNum);
        int newDim = 1;
        auto tadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(xShapeInfo, newDim);

#ifdef __ND4J_EXPERIMENTAL__
        BUILD_PAIRWISE_SELECTOR(xType, yType, zType, functions::scalar::ScalarTransform, ::transform(scalarOp, x, xShapeInfo, nullptr, result, resultShapeInfo, y, &newDim, dimensionLength, tadPack.primaryShapeInfo(), tadPack.primaryOffsets(), tadPack.primaryShapeInfo(), tadPack.primaryOffsets()), LIBND4J_TYPES, LIBND4J_TYPES);
#else
        BUILD_SINGLE_SELECTOR_THRICE(xType, functions::scalar::ScalarTransform, ::transform(scalarOp, x, xShapeInfo, nullptr, result, resultShapeInfo, y, &newDim, dimensionLength, tadPack.primaryShapeInfo(), tadPack.primaryOffsets(), tadPack.primaryShapeInfo(), tadPack.primaryOffsets()), LIBND4J_TYPES);
#endif

    } else if (xOrder == 'f' && zOrder == 'f' && xEws == 1 && zEws == 1 && yEws == 1 && xRank == 2 && isVector && dimensionLength == 1 && dimension[0] == 1 && isConvertible) {
        // add row vector case for F ordered rowAdd
        auto scalarOp = nd4j::convertToScalar((nd4j::broadcast::Ops) opNum);
        int newDim = 0;
        auto tadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(xShapeInfo, newDim);

#ifdef __ND4J_EXPERIMENTAL__
        BUILD_PAIRWISE_SELECTOR(xType, yType, zType, functions::scalar::ScalarTransform, ::transform(scalarOp, x, xShapeInfo, nullptr, result, resultShapeInfo, y, &newDim, dimensionLength, tadPack.primaryShapeInfo(), tadPack.primaryOffsets(), tadPack.primaryShapeInfo(), tadPack.primaryOffsets()), LIBND4J_TYPES, LIBND4J_TYPES);
#else
        BUILD_SINGLE_SELECTOR_THRICE(xType, functions::scalar::ScalarTransform, ::transform(scalarOp, x, xShapeInfo, nullptr, result, resultShapeInfo, y, &newDim, dimensionLength, tadPack.primaryShapeInfo(), tadPack.primaryOffsets(), tadPack.primaryShapeInfo(), tadPack.primaryOffsets()), LIBND4J_TYPES);
#endif

    }else {
        // default case

#ifdef __ND4J_EXPERIMENTAL__
        BUILD_PAIRWISE_SELECTOR(xType, yType, zType, functions::broadcast::Broadcast, ::exec(opNum, x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ), LIBND4J_TYPES, LIBND4J_TYPES);
#else
        BUILD_SINGLE_SELECTOR_THRICE(xType, functions::broadcast::Broadcast, ::exec(opNum, x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ,tadOffsetsZ), LIBND4J_TYPES);
#endif
    }
}


void NativeOpExcutioner::execInverseBroadcast(int opNum, void *x, Nd4jLong *xShapeInfo, void *y, Nd4jLong *yShapeInfo, void *result, Nd4jLong *resultShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadOnlyShapeInfoZ, Nd4jLong *tadOffsetsZ) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
    auto yType = nd4j::ArrayOptions::dataType(yShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(resultShapeInfo);

    if (!nd4j::Environment::getInstance()->isExperimentalBuild())
        if ((yType != xType && yType != nd4j::DataType::BOOL) || xType != zType)
            throw nd4j::datatype_exception::build("NativeOps::execBroadcast both operands must have same data type", xType, yType);

#ifdef __ND4J_EXPERIMENTAL__
        BUILD_PAIRWISE_SELECTOR(xType, yType, zType, functions::broadcast::Broadcast, ::execInverse(opNum, x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ), LIBND4J_TYPES, LIBND4J_TYPES);
#else
        BUILD_SINGLE_SELECTOR_THRICE(xType, functions::broadcast::Broadcast, ::execInverse(opNum, x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ,tadOffsetsZ), LIBND4J_TYPES);
#endif

}

void NativeOpExcutioner::execBroadcastBool(int opNum, void *x, Nd4jLong *xShapeInfo, void *y, Nd4jLong *yShapeInfo, void *result, Nd4jLong *resultShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadOnlyShapeInfoZ, Nd4jLong *tadOffsetsZ) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
    auto yType = nd4j::ArrayOptions::dataType(yShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(resultShapeInfo);

    if (!nd4j::Environment::getInstance()->isExperimentalBuild())
        if (yType != xType || nd4j::DataType::BOOL != zType)
            throw nd4j::datatype_exception::build("NativeOps::execBroadcastBool both operands must have same data type", xType, yType);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::broadcast::BroadcastBool, ::exec(opNum, x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ), LIBND4J_TYPES, BOOL_TYPES);
}

void NativeOpExcutioner::execInverseBroadcastBool(int opNum, void *x, Nd4jLong *xShapeInfo, void *y, Nd4jLong *yShapeInfo, void *result, Nd4jLong *resultShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadOnlyShapeInfoZ, Nd4jLong *tadOffsetsZ) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
    auto yType = nd4j::ArrayOptions::dataType(yShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(resultShapeInfo);

    if (!nd4j::Environment::getInstance()->isExperimentalBuild())
        if (yType != xType || nd4j::DataType::BOOL != zType)
            throw nd4j::datatype_exception::build("NativeOps::execInverseBroadcastBool both operands must have same data type", xType, yType);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::broadcast::BroadcastBool, ::execInverse(opNum, x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ), LIBND4J_TYPES, BOOL_TYPES);
}


////////////////////////////////////////////////////////////////////////
/**
*
* @param opNum
* @param dx
* @param xStride
* @param y
* @param yStride
* @param result
* @param resultStride
* @param extraParams
* @param n
*/
void NativeOpExcutioner::execPairwiseTransform(int opNum, void *dx, Nd4jLong *xShapeInfo, void *y, Nd4jLong *yShapeInfo, void *result, Nd4jLong *resultShapeInfo, void *extraParams) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
    auto yType = nd4j::ArrayOptions::dataType(yShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(resultShapeInfo);

    if (!nd4j::Environment::getInstance()->isExperimentalBuild()) {
        if ((yType != xType && yType != nd4j::DataType::BOOL))
            throw nd4j::datatype_exception::build("NativeOps::execPairwiseTransform both operands must have same data type", xType, yType);

        if (xType != zType)
            throw nd4j::datatype_exception::build("NativeOps::execPairwiseTransform result must have the same type as X", xType, zType);
    }

#ifdef __ND4J_EXPERIMENTAL__
    BUILD_PAIRWISE_SELECTOR(xType, yType, zType, functions::pairwise_transforms::PairWiseTransform, ::exec(opNum, dx, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, extraParams), LIBND4J_TYPES, LIBND4J_TYPES);
#else
    BUILD_SINGLE_SELECTOR_THRICE(xType, functions::pairwise_transforms::PairWiseTransform, ::exec(opNum, dx, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, extraParams), LIBND4J_TYPES);
#endif
}

void NativeOpExcutioner::execPairwiseBoolTransform(int opNum, void *dx, Nd4jLong *xShapeInfo, void *y, Nd4jLong *yShapeInfo, void *result, Nd4jLong *resultShapeInfo, void *extraParams) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
    auto yType = nd4j::ArrayOptions::dataType(yShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(resultShapeInfo);

    if (!nd4j::Environment::getInstance()->isExperimentalBuild()) {
        if (yType != xType)
            throw nd4j::datatype_exception::build(
                    "NativeOps::execPairwiseBoolTransform both operands must have same data type, and result must have bool type", xType, yType);

        if (nd4j::DataType::BOOL != zType)
            throw nd4j::datatype_exception::build("NativeOps::execPairwiseBoolTransform result must have bool type", zType);
    }

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::pairwise_transforms::PairWiseBoolTransform, ::exec(opNum, dx, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, extraParams), LIBND4J_TYPES, BOOL_TYPES);
}



////////////////////////////////////////////////////////////////////////
/**
*
* @param opNum
* @param x
* @param xShapeInfo
* @param extraParams
* @param result
* @param resultShapeInfo
*/
void NativeOpExcutioner::execReduceFloat(int opNum, void *x, Nd4jLong *xShapeInfo, void *extraParams, void *result, Nd4jLong *resultShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(resultShapeInfo);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce::ReduceFloatFunction, ::exec(opNum, x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets), LIBND4J_TYPES, FLOAT_TYPES);
}

void NativeOpExcutioner::execReduceSame(int opNum, void *x, Nd4jLong *xShapeInfo, void *extraParams, void *result, Nd4jLong *resultShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(resultShapeInfo);

    BUILD_SINGLE_SELECTOR(xType, functions::reduce::ReduceSameFunction, ::exec(opNum, x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets), LIBND4J_TYPES);
}

void NativeOpExcutioner::execReduceBool(int opNum, void *x, Nd4jLong *xShapeInfo, void *extraParams, void *result, Nd4jLong *resultShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(resultShapeInfo);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce::ReduceBoolFunction, ::exec(opNum, x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets), LIBND4J_TYPES, BOOL_TYPES);
}

void NativeOpExcutioner::execReduceLong(int opNum, void *x, Nd4jLong *xShapeInfo, void *extraParams, void *result, Nd4jLong *resultShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(resultShapeInfo);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce::ReduceLongFunction, ::exec(opNum, x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets), LIBND4J_TYPES, LONG_TYPES);
}

////////////////////////////////////////////////////////////////////////
/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 * @return
 */
void NativeOpExcutioner::execReduceFloatScalar(int opNum, void *x, Nd4jLong *xShapeInfo, void *extraParams, void *z, Nd4jLong *zShapeInfo) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(zShapeInfo);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce::ReduceFloatFunction, ::execScalar(opNum, x, xShapeInfo, extraParams, z, zShapeInfo), LIBND4J_TYPES, FLOAT_TYPES);
}

void NativeOpExcutioner::execReduceSameScalar(int opNum, void *x, Nd4jLong *xShapeInfo, void *extraParams, void *z, Nd4jLong *zShapeInfo) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);

    BUILD_SINGLE_SELECTOR(xType, functions::reduce::ReduceSameFunction, ::execScalar(opNum, x, xShapeInfo, extraParams, z, zShapeInfo), LIBND4J_TYPES);
}

void NativeOpExcutioner::execReduceBoolScalar(int opNum, void *x, Nd4jLong *xShapeInfo, void *extraParams, void *z, Nd4jLong *zShapeInfo) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(zShapeInfo);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce::ReduceBoolFunction, ::execScalar(opNum, x, xShapeInfo, extraParams, z, zShapeInfo), LIBND4J_TYPES, BOOL_TYPES);
}

void NativeOpExcutioner::execReduceLongScalar(int opNum, void *x, Nd4jLong *xShapeInfo, void *extraParams, void *z, Nd4jLong *zShapeInfo) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(zShapeInfo);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce::ReduceLongFunction, ::execScalar(opNum, x, xShapeInfo, extraParams, z, zShapeInfo), LIBND4J_TYPES, LONG_TYPES);
}


////////////////////////////////////////////////////////////////////////
/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParamsVals
 * @param y
 * @param yShapeInfo
 * @param result
 * @param resultShapeInfoBuffer
 * @param dimension
 * @param dimensionLength
 */
void NativeOpExcutioner::execReduce3Scalar(int opNum, void *x, Nd4jLong *xShapeInfo, void *extraParamsVals, void *y, Nd4jLong *yShapeInfo, void *z, Nd4jLong *zShapeInfo) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(zShapeInfo);


    BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce3::Reduce3, ::execScalar(opNum, x, xShapeInfo, extraParamsVals, y, yShapeInfo, z, zShapeInfo), LIBND4J_TYPES, FLOAT_TYPES);
}

////////////////////////////////////////////////////////////////////////
/**
*
* @param opNum
* @param x
* @param xShapeInfo
* @param extraParamsVals
* @param y
* @param yShapeInfo
* @param result
* @param resultShapeInfo
*/
void NativeOpExcutioner::execReduce3(int opNum, void *x, Nd4jLong *xShapeInfo, void *extraParamsVals, void *y, Nd4jLong *yShapeInfo, void *result, Nd4jLong *resultShapeInfo) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(resultShapeInfo);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce3::Reduce3, ::exec(opNum, x, xShapeInfo, extraParamsVals, y, yShapeInfo, result, resultShapeInfo, nullptr, 1), LIBND4J_TYPES, FLOAT_TYPES);

}

////////////////////////////////////////////////////////////////////////
void NativeOpExcutioner::execReduce3All(int opNum, void *x, Nd4jLong *xShapeInfo, void *extraParamsVals, void *y, Nd4jLong *yShapeInfo, void *result, Nd4jLong *resultShapeInfoBuffer, int *dimension, int dimensionLength, Nd4jLong *xTadShapeInfo, Nd4jLong *xOffsets, Nd4jLong *yTadShapeInfo, Nd4jLong *yOffsets) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(resultShapeInfoBuffer);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce3::Reduce3, ::execAll(opNum, x, xShapeInfo, extraParamsVals, y, yShapeInfo, result, resultShapeInfoBuffer, dimension, dimensionLength, xTadShapeInfo, xOffsets, yTadShapeInfo, yOffsets), LIBND4J_TYPES, FLOAT_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExcutioner::execReduce3TAD(int opNum, void *x, Nd4jLong *xShapeInfo, void *extraParamsVals, void *y, Nd4jLong *yShapeInfo, void *result, Nd4jLong *resultShapeInfoBuffer, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(resultShapeInfoBuffer);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce3::Reduce3, ::exec(opNum, x, xShapeInfo, extraParamsVals, y, yShapeInfo, result, resultShapeInfoBuffer, dimension, dimensionLength, tadShapeInfo, tadOffsets), LIBND4J_TYPES, FLOAT_TYPES);
}


////////////////////////////////////////////////////////////////////////
/**
*
* @param opNum
* @param x
* @param xStride
* @param result
* @param resultStride
* @param scalar
* @param extraParams
* @param n
*/
void NativeOpExcutioner::execScalar(int opNum, void *x, Nd4jLong *xShapeInfo, void *result, Nd4jLong *resultShapeInfo, void *scalar, Nd4jLong *scalarShapeInfo, void *extraParams) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
    auto yType = nd4j::ArrayOptions::dataType(scalarShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(resultShapeInfo);
    if (!nd4j::Environment::getInstance()->isExperimentalBuild()) {
        if ((yType != xType && yType != nd4j::DataType::BOOL) || zType != xType){
            throw nd4j::datatype_exception::build("NativeOpExecutioner::execScalar both operands must have same data type", xType, yType);
        }
    }

#ifdef __ND4J_EXPERIMENTAL__
        BUILD_PAIRWISE_SELECTOR(xType, yType, zType, functions::scalar::ScalarTransform, ::transform(opNum, x, xShapeInfo, result, resultShapeInfo, scalar, extraParams), LIBND4J_TYPES, LIBND4J_TYPES);
#else
        BUILD_SINGLE_SELECTOR_THRICE(xType, functions::scalar::ScalarTransform,
                                     ::transform(opNum, x, xShapeInfo, result, resultShapeInfo, scalar, extraParams),
                                     LIBND4J_TYPES);
#endif
}


////////////////////////////////////////////////////////////////////////
void NativeOpExcutioner::execScalar(int opNum, void *x, Nd4jLong *xShapeInfo, void *extraParams, void *z, Nd4jLong *zShapeInfo, void *scalars, Nd4jLong *scalarShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadShapeInfoZ, Nd4jLong *tadOffsetsZ) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
    auto yType = nd4j::ArrayOptions::dataType(scalarShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(zShapeInfo);

    if (!nd4j::Environment::getInstance()->isExperimentalBuild())
        if ((yType != xType && yType != nd4j::DataType::BOOL) || xType != zType)
            throw nd4j::datatype_exception::build("NativeOpExecutioner::execScalar both operands must have same data type", xType, yType);

#ifdef __ND4J_EXPERIMENTAL__
    BUILD_PAIRWISE_SELECTOR(xType, yType, zType, functions::scalar::ScalarTransform, ::transform(opNum, x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), LIBND4J_TYPES, LIBND4J_TYPES);
#else
    BUILD_SINGLE_SELECTOR_THRICE(xType, functions::scalar::ScalarTransform, ::transform(opNum, x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), LIBND4J_TYPES);
#endif
}

void NativeOpExcutioner::execScalarBool(int opNum, void *x, Nd4jLong *xShapeInfo, void *result, Nd4jLong *resultShapeInfo, void *scalar, Nd4jLong *scalarShapeInfo, void *extraParams) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(resultShapeInfo);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::scalar::ScalarBoolTransform, ::transform(opNum, x, xShapeInfo, result, resultShapeInfo, scalar, extraParams), LIBND4J_TYPES, BOOL_TYPES);
}


////////////////////////////////////////////////////////////////////////
void NativeOpExcutioner::execScalarBool(int opNum, void *x, Nd4jLong *xShapeInfo, void *extraParams, void *z, Nd4jLong *zShapeInfo, void *scalars, Nd4jLong *scalarShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadShapeInfoZ, Nd4jLong *tadOffsetsZ) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
    auto yType = nd4j::ArrayOptions::dataType(scalarShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(zShapeInfo);

    if (!nd4j::Environment::getInstance()->isExperimentalBuild())
        if (yType != xType || nd4j::DataType::BOOL != zType)
            throw nd4j::datatype_exception::build("NativeOpExecutioner::execScalarBool both operands must have same data type", xType, yType);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::scalar::ScalarBoolTransform, ::transform(opNum, x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), LIBND4J_TYPES, BOOL_TYPES);
}


////////////////////////////////////////////////////////////////////////
/**
*
* @param opNum
* @param x
* @param xShapeInfo
* @param extraParams
* @param result
* @param resultShapeInfo
*/
void NativeOpExcutioner::execSummaryStats(int opNum, void *x, Nd4jLong *xShapeInfo, void *extraParams, void *result, Nd4jLong *resultShapeInfo, bool biasCorrected) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(resultShapeInfo);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::summarystats::SummaryStatsReduce, ::exec(opNum, biasCorrected, x, xShapeInfo, extraParams, result, resultShapeInfo, nullptr, 1), LIBND4J_TYPES, FLOAT_TYPES);
}

////////////////////////////////////////////////////////////////////////
/**
*
* @param opNum
* @param x
* @param xShapeInfo
* @param extraParams
* @param result
* @param resultShapeInfo
*/
void NativeOpExcutioner::execSummaryStatsScalar(int opNum, void *x, Nd4jLong *xShapeInfo, void *extraParams, void *result, Nd4jLong *resultShapeInfo, bool biasCorrected) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(resultShapeInfo);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::summarystats::SummaryStatsReduce, ::execScalar(opNum, biasCorrected, x, xShapeInfo, extraParams, result, resultShapeInfo), LIBND4J_TYPES, FLOAT_TYPES);
}

////////////////////////////////////////////////////////////////////////
/**
*
* @param opNum
* @param x
* @param xShapeInfo
* @param extraParams
* @param result
* @param resultShapeInfoBuffer
* @param dimension
* @param dimensionLength
*/
void NativeOpExcutioner::execSummaryStats(int opNum, void *x, Nd4jLong *xShapeInfo, void *extraParams, void *result, Nd4jLong *resultShapeInfoBuffer, int *dimension, int dimensionLength, bool biasCorrected) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(resultShapeInfoBuffer);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::summarystats::SummaryStatsReduce, ::exec(opNum, biasCorrected, x, xShapeInfo, extraParams, result, resultShapeInfoBuffer, dimension, dimensionLength), LIBND4J_TYPES, FLOAT_TYPES);
}


////////////////////////////////////////////////////////////////////////
/**
*
* @param opNum
* @param dx
* @param xStride
* @param result
* @param resultStride
* @param extraParams
* @param n
*/
void NativeOpExcutioner::execTransformFloat(int opNum, void *dx, Nd4jLong *xShapeInfo, void *result, Nd4jLong *resultShapeInfo, void *extraParams, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(resultShapeInfo);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::transform::TransformFloat, ::exec(opNum, dx, xShapeInfo, result, resultShapeInfo, extraParams, tadShapeInfo, tadOffsets), LIBND4J_TYPES, FLOAT_TYPES);
}

void NativeOpExcutioner::execTransformBool(int opNum, void *dx, Nd4jLong *xShapeInfo, void *result, Nd4jLong *resultShapeInfo, void *extraParams, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(resultShapeInfo);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::transform::TransformBool, ::exec(opNum, dx, xShapeInfo, result, resultShapeInfo, extraParams, tadShapeInfo, tadOffsets), LIBND4J_TYPES, BOOL_TYPES);
}

void NativeOpExcutioner::execTransformAny(int opNum, void *dx, Nd4jLong *xShapeInfo, void *result, Nd4jLong *resultShapeInfo, void *extraParams, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(resultShapeInfo);

    if (opNum == nd4j::transform::Assign && xType == zType && shape::elementWiseStride(xShapeInfo) == 1 && shape::elementWiseStride(resultShapeInfo) == 1 && shape::order(xShapeInfo) == shape::order(resultShapeInfo) && shape::equalsTypesAndShapesSoft(xShapeInfo, resultShapeInfo)) {
        memcpy(result, dx, nd4j::DataTypeUtils::sizeOf(xType) * shape::length(xShapeInfo));
    } else {
        BUILD_DOUBLE_SELECTOR(xType, zType, functions::transform::TransformAny, ::exec(opNum, dx, xShapeInfo, result, resultShapeInfo, extraParams, tadShapeInfo, tadOffsets), LIBND4J_TYPES, LIBND4J_TYPES);
    }
}

void NativeOpExcutioner::execTransformSame(int opNum, void *dx, Nd4jLong *xShapeInfo, void *result, Nd4jLong *resultShapeInfo, void *extraParams, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(resultShapeInfo);

    BUILD_SINGLE_SELECTOR(xType, functions::transform::TransformSame, ::exec(opNum, dx, xShapeInfo, result, resultShapeInfo, extraParams, tadShapeInfo, tadOffsets), LIBND4J_TYPES);
}

void NativeOpExcutioner::execTransformStrict(int opNum, void *dx, Nd4jLong *xShapeInfo, void *result, Nd4jLong *resultShapeInfo, void *extraParams, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(resultShapeInfo);

    BUILD_SINGLE_SELECTOR(xType, functions::transform::TransformStrict, ::exec(opNum, dx, xShapeInfo, result, resultShapeInfo, extraParams, tadShapeInfo, tadOffsets), FLOAT_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExcutioner::execRandom(int opNum, Nd4jPointer state, void *z, Nd4jLong *zShapeInfo, void *extraArguments) {
    auto zType = nd4j::ArrayOptions::dataType(zShapeInfo);

    BUILD_SINGLE_SELECTOR(zType, functions::random::RandomFunction, ::execTransform(opNum, state, z, zShapeInfo, extraArguments), FLOAT_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExcutioner::execRandom(int opNum, Nd4jPointer state, void *x, Nd4jLong *xShapeInfo, void *z, Nd4jLong *zShapeInfo, void *extraArguments) {
    auto zType = nd4j::ArrayOptions::dataType(zShapeInfo);

    BUILD_SINGLE_SELECTOR(zType, functions::random::RandomFunction, ::execTransform(opNum, state, x, xShapeInfo, z, zShapeInfo, extraArguments), FLOAT_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExcutioner::execRandom(int opNum, Nd4jPointer state, void *x, Nd4jLong *xShapeInfo, void *y, Nd4jLong *yShapeBuffer, void *z, Nd4jLong *zShapeBuffer, void *extraArguments) {
    auto xType = nd4j::ArrayOptions::dataType(zShapeBuffer);

    BUILD_SINGLE_SELECTOR(xType, functions::random::RandomFunction, ::execTransform(opNum, state, x, xShapeInfo, y, yShapeBuffer, z, zShapeBuffer, extraArguments), FLOAT_TYPES);
}

void NativeOpExcutioner::execReduce3(int opNum, void *x, Nd4jLong *xShapeInfo, void *extraParamsVals, void *y, Nd4jLong *yShapeInfo, void *result, Nd4jLong *resultShapeInfoBuffer, int *dimension, int dimensionLength) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(resultShapeInfoBuffer);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce3::Reduce3, ::exec(opNum, x, xShapeInfo, extraParamsVals, y, yShapeInfo, result, resultShapeInfoBuffer, dimension, dimensionLength), LIBND4J_TYPES, FLOAT_TYPES);
}

