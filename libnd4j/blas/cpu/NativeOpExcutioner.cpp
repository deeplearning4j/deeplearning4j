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
#include <loops/transform_float.h>


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
Nd4jLong NativeOpExcutioner::execIndexReduceScalar(int opNum, void *x, Nd4jLong *xShapeInfo, void *extraParams, void *vz, Nd4jLong *zShapeInfo) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
    auto z = reinterpret_cast<Nd4jLong*>(vz);

    BUILD_SINGLE_SELECTOR(xType, z[0] = functions::indexreduce::IndexReduce, ::execScalar(opNum, x,xShapeInfo,extraParams), FLOAT_TYPES);
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

    BUILD_SINGLE_SELECTOR(xType, functions::indexreduce::IndexReduce, ::exec(opNum, x, xShapeInfo, extraParams, result, resultShapeInfoBuffer, dimension, dimensionLength, tadShapeInfo, tadOffsets);, FLOAT_TYPES);
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

    BUILD_PAIRWISE_SELECTOR(xType, yType, zType, functions::broadcast::Broadcast, ::exec(opNum, x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ);, LIBND4J_TYPES, LIBND4J_TYPES);
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

    BUILD_PAIRWISE_SELECTOR(xType, yType, zType, functions::pairwise_transforms::PairWiseTransform, ::exec(opNum, dx, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, extraParams), LIBND4J_TYPES, LIBND4J_TYPES);
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
void NativeOpExcutioner::execReduce(int opNum, void *x, Nd4jLong *xShapeInfo, void *extraParams, void *result, Nd4jLong *resultShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);

    // for LogSumExp reduction we need to have max stored in result
    if (opNum == 19) {
        BUILD_SINGLE_SELECTOR(xType, functions::reduce::ReduceFunction, ::exec(3, x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets), FLOAT_TYPES);
    }

    BUILD_SINGLE_SELECTOR(xType, functions::reduce::ReduceFunction, ::exec(opNum, x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets), FLOAT_TYPES);
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
void NativeOpExcutioner::execReduceScalar(int opNum, void *x, Nd4jLong *xShapeInfo, void *extraParams, void *z, Nd4jLong *zShapeInfo) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);

    if (opNum == 19) {
        BUILD_SINGLE_SELECTOR(xType, functions::reduce::ReduceFunction, ::execScalar(3, x, xShapeInfo, extraParams, z, zShapeInfo), FLOAT_TYPES);
        BUILD_SINGLE_SELECTOR(xType, functions::reduce::ReduceFunction, ::execScalar(opNum, x, xShapeInfo, z, z, zShapeInfo), FLOAT_TYPES);
        return;
    }

    BUILD_SINGLE_SELECTOR(xType, functions::reduce::ReduceFunction, ::execScalar(opNum, x, xShapeInfo, extraParams, z, zShapeInfo), FLOAT_TYPES);
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

    BUILD_PAIRWISE_SELECTOR(xType, yType, zType, functions::scalar::ScalarTransform, ::transform(opNum, x, xShapeInfo, result, resultShapeInfo, scalar, extraParams), LIBND4J_TYPES, LIBND4J_TYPES);
}


////////////////////////////////////////////////////////////////////////
void NativeOpExcutioner::execScalar(int opNum, void *x, Nd4jLong *xShapeInfo, void *extraParams, void *z, Nd4jLong *zShapeInfo, void *scalars, Nd4jLong *scalarShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadShapeInfoZ, Nd4jLong *tadOffsetsZ) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
    auto yType = nd4j::ArrayOptions::dataType(scalarShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(zShapeInfo);

    BUILD_PAIRWISE_SELECTOR(xType, yType, zType, functions::scalar::ScalarTransform, ::transform(opNum, x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), LIBND4J_TYPES, LIBND4J_TYPES);
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

    BUILD_SINGLE_SELECTOR(xType, functions::summarystats::SummaryStatsReduce, ::exec(opNum, biasCorrected, x, xShapeInfo, extraParams, result, resultShapeInfo, nullptr, 1), FLOAT_TYPES);
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
double NativeOpExcutioner::execSummaryStatsScalar(int opNum, void *x, Nd4jLong *xShapeInfo, void *extraParams, bool biasCorrected) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);

    BUILD_SINGLE_SELECTOR(xType, return functions::summarystats::SummaryStatsReduce, ::execScalar(opNum, biasCorrected, x, xShapeInfo, extraParams), FLOAT_TYPES);
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

    BUILD_SINGLE_SELECTOR(xType, functions::summarystats::SummaryStatsReduce, ::exec(opNum, biasCorrected, x, xShapeInfo, extraParams, result, resultShapeInfoBuffer, dimension, dimensionLength), FLOAT_TYPES);
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
void NativeOpExcutioner::execTransform(int opNum, void *dx, Nd4jLong *xShapeInfo, void *result, Nd4jLong *resultShapeInfo, void *extraParams, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(resultShapeInfo);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::transform::TransformFloat, ::exec(opNum, dx, xShapeInfo, result, resultShapeInfo, extraParams, tadShapeInfo, tadOffsets), LIBND4J_TYPES, FLOAT_TYPES);
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
void NativeOpExcutioner::execTransform(int opNum, void *dx, Nd4jLong *xShapeInfo, void *result, Nd4jLong *resultShapeInfo, void *extraParams, Nd4jLong *xIndexes, Nd4jLong *resultIndexes, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);

    //BUILD_SINGLE_SELECTOR(xType, functions::transform::Transform, ::exec(opNum, dx, xShapeInfo, result, resultShapeInfo, extraParams, xIndexes, resultIndexes, tadShapeInfo, tadOffsets), LIBND4J_TYPES);
}



////////////////////////////////////////////////////////////////////////
void NativeOpExcutioner::execRandom(int opNum, Nd4jPointer state, void *z, Nd4jLong *zShapeInfo, void *extraArguments) {
    auto zType = nd4j::ArrayOptions::dataType(zShapeInfo);

    BUILD_SINGLE_SELECTOR(zType, functions::random::RandomFunction, ::execTransform(opNum, state, z, zShapeInfo, extraArguments), FLOAT_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExcutioner::execRandom(int opNum, Nd4jPointer state, void *x, Nd4jLong *xShapeInfo, void *z, Nd4jLong *zShapeInfo, void *extraArguments) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);

    BUILD_SINGLE_SELECTOR(xType, functions::random::RandomFunction, ::execTransform(opNum, state, x, xShapeInfo, z, zShapeInfo, extraArguments), FLOAT_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExcutioner::execRandom(int opNum, Nd4jPointer state, void *x, Nd4jLong *xShapeInfo, void *y, Nd4jLong *yShapeBuffer, void *z, Nd4jLong *zShapeBuffer, void *extraArguments) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);

    BUILD_SINGLE_SELECTOR(xType, functions::random::RandomFunction, ::execTransform(opNum, state, x, xShapeInfo, y, yShapeBuffer, z, zShapeBuffer, extraArguments), FLOAT_TYPES);
}

void NativeOpExcutioner::execReduce3(int opNum, void *x, Nd4jLong *xShapeInfo, void *extraParamsVals, void *y, Nd4jLong *yShapeInfo, void *result, Nd4jLong *resultShapeInfoBuffer, int *dimension, int dimensionLength) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(resultShapeInfoBuffer);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce3::Reduce3, ::exec(opNum, x, xShapeInfo, extraParamsVals, y, yShapeInfo, result, resultShapeInfoBuffer, dimension, dimensionLength), LIBND4J_TYPES, FLOAT_TYPES);
}






