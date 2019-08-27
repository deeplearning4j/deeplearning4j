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

#include "../NativeOpExecutioner.h"
#include <cuda.h>
#include <op_boilerplate.h>
#include <helpers/DebugHelper.h>
#include <DataTypeUtils.h>
#include <exceptions/datatype_exception.h>
#include <exceptions/cuda_exception.h>
#include <helpers/CudaLaunchHelper.h>
#include <helpers/ShapeBuilders.h>
#include <PointersManager.h>

#include <array/ConstantDataBuffer.h>
#include <array/ShapeDescriptor.h>
#include <helpers/ConstantShapeHelper.h>

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
#include <loops/pairwise_bool.h>
#include <loops/broadcasting_bool.h>
#include <loops/reduce_float.h>
#include <loops/reduce3.h>
#include <loops/summarystatsreduce.h>
#include <loops/transform_same.h>
#include <loops/scalar.h>
#include <loops/random.h>
#include <loops/special_kernels.h>
#include <loops/scalar_bool.h>

using namespace nd4j;

/**
* This is utility kernel, that updates given special buffer with proper values in device memory
*/
extern "C" __global__ void prepareShapeBuffer(int *dimension, int *maxDimension, Nd4jLong *specialPointer, int rows, nd4j::DataType dataType) {
    Nd4jLong tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0)
        return;

    dimension[0] = 0;
    maxDimension[0] = 1;

    specialPointer[0] = 2;
    specialPointer[1] = rows;
    specialPointer[2] = 1;
    specialPointer[3] = 1;
    specialPointer[4] = 1;
    specialPointer[5] = 0;
    specialPointer[6] = 1;
    specialPointer[7] = 99;

    ArrayOptions::setDataType(specialPointer, dataType);

    //printf("special[0]: [%lld]\n", (long long) specialPointer[0]);
    //shape::printShapeInfoLinear("prepareShapeBuffer", specialPointer);
}


////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execPairwiseTransform(nd4j::LaunchContext  *lc,
                                    int opNum,
                                    void *hX, Nd4jLong *hXShapeInfo,
                                    void *dX, Nd4jLong *dXShapeInfo,
                                    void *hY, Nd4jLong *hYShapeInfo,
                                    void *dY, Nd4jLong *dYShapeInfo,
                                    void *hZ, Nd4jLong *hZShapeInfo,
                                    void *dZ, Nd4jLong *dZShapeInfo,
                                    void *extraParams) {

    auto stream = lc->getCudaStream();

    auto xType = nd4j::ArrayOptions::dataType(hXShapeInfo);
    auto yType = nd4j::ArrayOptions::dataType(hYShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(hZShapeInfo);

    if (xType != zType && yType != zType)
        throw std::runtime_error("NativeOpExecutioner::execPairwiseTransform requires Z operand to have either X or Y type");
    if (lc == nullptr)
        throw std::runtime_error("NativeOpExecutioner::execPairwiseTransform: launch context cannot be nullptr !");
    if (stream == nullptr)
        throw std::runtime_error("NativeOpExecutioner::execPairwiseTransform: CUDA stream cannot be nullptr !");

    dim3 launchDims(256, 1024, 8192);

#ifdef __ND4J_EXPERIMENTAL__
    BUILD_PAIRWISE_SELECTOR(xType, yType, zType, functions::pairwise_transforms::PairWiseTransform, ::executeCudaShaped(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo, extraParams), LIBND4J_TYPES, LIBND4J_TYPES)
#else
    BUILD_SINGLE_SELECTOR_THRICE(xType, functions::pairwise_transforms::PairWiseTransform, ::executeCudaShaped(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo, extraParams), LIBND4J_TYPES)
#endif

    // TODO: remove after the release
    auto res = cudaStreamSynchronize(*stream);
    if (res != 0)
        throw cuda_exception::build("execPairwiseTransform failed", res);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execPairwiseBoolTransform( nd4j::LaunchContext  *lc,
                                                    int opNum,
                                                    void *hX, Nd4jLong *hXShapeInfo,
                                                    void *dX, Nd4jLong *dXShapeInfo,
                                                    void *hY, Nd4jLong *hYShapeInfo,
                                                    void *dY, Nd4jLong *dYShapeInfo,
                                                    void *hZ, Nd4jLong *hZShapeInfo,
                                                    void *dZ, Nd4jLong *dZShapeInfo,
                                                    void *extraParams) {

	auto stream = lc->getCudaStream();

    auto xType = nd4j::ArrayOptions::dataType(hXShapeInfo);
    auto yType = nd4j::ArrayOptions::dataType(hYShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(hZShapeInfo);

    if (!DataTypeUtils::isB(zType))
		throw nd4j::datatype_exception::build("NativeOpExecutioner::execPairwiseBoolTransform wrong Z operand data type", nd4j::DataType::BOOL, zType);

    if (yType != xType)
        throw nd4j::datatype_exception::build("NativeOpExecutioner::execPairwiseBoolTransform both operands must have same data type", xType, yType);

    dim3 launchDims(256, 1024, 16384);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::pairwise_transforms::PairWiseBoolTransform, ::executeCudaShaped(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo, extraParams), LIBND4J_TYPES, BOOL_TYPES)

    // TODO: remove after the release
    auto res = cudaStreamSynchronize(*stream);
    if (res != 0)
        throw cuda_exception::build("execPairwiseBoolTransform failed", res);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execSummaryStatsScalar(nd4j::LaunchContext  *lc,
                                    int opNum,
                                    void *hX, Nd4jLong *hXShapeInfo,
                                    void *dX, Nd4jLong *dXShapeInfo,
                                    void *extraParams,
                                    void *hZ, Nd4jLong *hZShapeInfo,
                                    void *dZ, Nd4jLong *dZShapeInfo,
                                    bool biasCorrected) {

	auto stream = lc->getCudaStream();
    auto reductionPointer = lc->getReductionPointer();

    dim3 launchDims = dim3(256, 256, 32768);

	auto xType = nd4j::ArrayOptions::dataType(hXShapeInfo);
	auto zType = nd4j::ArrayOptions::dataType(hZShapeInfo);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::summarystats::SummaryStatsReduce, ::execSummaryStatsReduceScalar(launchDims, stream, opNum, dX, dXShapeInfo, hXShapeInfo, extraParams, dZ, dZShapeInfo, hZShapeInfo, nullptr, nullptr, biasCorrected, reductionPointer), LIBND4J_TYPES, FLOAT_TYPES);

    // TODO: remove after the release
    auto res = cudaStreamSynchronize(*stream);
    if (res != 0)
        throw cuda_exception::build("execSummaryStatsScalar failed", res);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execBroadcastBool(nd4j::LaunchContext  *lc,
                            int opNum,
                            void *hX, Nd4jLong *hXShapeInfo,
                            void *dX, Nd4jLong *dXShapeInfo,
                            void *hY, Nd4jLong *hYShapeInfo,
                            void *dY, Nd4jLong *dYShapeInfo,
                            void *hZ, Nd4jLong *hZShapeInfo,
                            void *dZ, Nd4jLong *dZShapeInfo,
                            int *dimension, int dimensionLength,
                            Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets,
                            Nd4jLong *tadOnlyShapeInfoZ,Nd4jLong *tadOffsetsZ) {

	auto stream = lc->getCudaStream();

	auto xType = nd4j::ArrayOptions::dataType(hXShapeInfo);
	auto yType = nd4j::ArrayOptions::dataType(hYShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(hZShapeInfo);

	if (!DataTypeUtils::isB(zType))
        throw std::runtime_error("NativeOpExecutioner::execBroadcastBool requires Z operand to have BOOL type");

    if (yType != xType)
        throw std::runtime_error("NativeOpExecutioner::execBroadcastBool requires both X & Y operands to have same type");

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("F3B opNum:[%i]\n", opNum);

	dim3 launchDims(256, 256, 1024);

	BUILD_DOUBLE_SELECTOR(xType, zType, functions::broadcast::BroadcastBool, ::execBroadcast(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ), LIBND4J_TYPES, BOOL_TYPES)

    // TODO: remove after the release
    auto res = cudaStreamSynchronize(*stream);
    if (res != 0)
        throw cuda_exception::build("execBroadcastBool failed", res);
}

void NativeOpExecutioner::execInverseBroadcastBool(nd4j::LaunchContext  *lc,
                                                   int opNum,
                                                   void *hX, Nd4jLong *hXShapeInfo,
                                                   void *dX, Nd4jLong *dXShapeInfo,
                                                   void *hY, Nd4jLong *hYShapeInfo,
                                                   void *dY, Nd4jLong *dYShapeInfo,
                                                   void *hZ, Nd4jLong *hZShapeInfo,
                                                   void *dZ, Nd4jLong *dZShapeInfo,
                                                   int *dimension, int dimensionLength,
                                                   Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets,
                                                   Nd4jLong *tadOnlyShapeInfoZ,Nd4jLong *tadOffsetsZ) {
    auto stream = lc->getCudaStream();

    auto xType = nd4j::ArrayOptions::dataType(hXShapeInfo);
    auto yType = nd4j::ArrayOptions::dataType(hYShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(hZShapeInfo);

    if (!DataTypeUtils::isB(zType))
        throw std::runtime_error("NativeOpExecutioner::execBroadcastBool requires Z operand to have BOOL type");

    if (yType != xType)
        throw std::runtime_error("NativeOpExecutioner::execBroadcastBool requires both X & Y operands to have same type");

    if (nd4j::Environment::getInstance()->isDebugAndVerbose())
        printf("F3BI opNum:[%i]\n", opNum);

    dim3 launchDims(256, 256, 1024);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::broadcast::BroadcastBool, ::execInverseBroadcast(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ), LIBND4J_TYPES, BOOL_TYPES)

    // TODO: remove after the release
    auto res = cudaStreamSynchronize(*stream);
    if (res != 0)
        throw cuda_exception::build("execInverseBroadcastBool failed", res);
}

////////////////////////////////////////////////////////////////////////
/**
 *
 * @param opNum
 * @param dX
 * @param dXShapeInfo
 * @param dY
 * @param dYShapeInfo
 * @param dZ
 * @param dZShapeInfo
 * @param dimension
 * @param dimensionLength
 */
void NativeOpExecutioner::execBroadcast(nd4j::LaunchContext  *lc,
		                              int opNum,
		                              void *hX, Nd4jLong *hXShapeInfo,
		                              void *dX, Nd4jLong *dXShapeInfo,
		                              void *hY, Nd4jLong *hYShapeInfo,
		                              void *dY, Nd4jLong *dYShapeInfo,
		                              void *hZ, Nd4jLong *hZShapeInfo,
		                              void *dZ, Nd4jLong *dZShapeInfo,
		                              int *dimension, int dimensionLength,
		                              Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets,
		                              Nd4jLong *tadOnlyShapeInfoZ,Nd4jLong *tadOffsetsZ) {

	auto stream = lc->getCudaStream();

	auto xType = nd4j::ArrayOptions::dataType(hXShapeInfo);
	auto yType = nd4j::ArrayOptions::dataType(hYShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(hZShapeInfo);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("F3 opNum:[%i]\n", opNum);

	dim3 launchDims(256, 256, 1024);

#ifdef __ND4J_EXPERIMENTAL__
	BUILD_PAIRWISE_SELECTOR(xType, yType, zType, functions::broadcast::Broadcast, ::execBroadcast(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ), LIBND4J_TYPES, LIBND4J_TYPES);
#else
    BUILD_SINGLE_SELECTOR_THRICE(xType, functions::broadcast::Broadcast, ::execBroadcast(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ), LIBND4J_TYPES);
#endif

    // TODO: remove after the release
    auto res = cudaStreamSynchronize(*stream);
    if (res != 0)
        throw cuda_exception::build("execBroadcast failed", res);
}

void NativeOpExecutioner::execInverseBroadcast(nd4j::LaunchContext  *lc,
                                               int opNum,
                                               void *hX, Nd4jLong *hXShapeInfo,
                                               void *dX, Nd4jLong *dXShapeInfo,
                                               void *hY, Nd4jLong *hYShapeInfo,
                                               void *dY, Nd4jLong *dYShapeInfo,
                                               void *hZ, Nd4jLong *hZShapeInfo,
                                               void *dZ, Nd4jLong *dZShapeInfo,
                                               int *dimension, int dimensionLength,
                                               Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets,
                                               Nd4jLong *tadOnlyShapeInfoZ,Nd4jLong *tadOffsetsZ) {

    auto stream = lc->getCudaStream();

    auto xType = nd4j::ArrayOptions::dataType(hXShapeInfo);
    auto yType = nd4j::ArrayOptions::dataType(hYShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(hZShapeInfo);

    if (nd4j::Environment::getInstance()->isDebugAndVerbose())
        printf("F3I opNum:[%i]\n", opNum);

    dim3 launchDims(256, 256, 1024);

#ifdef __ND4J_EXPERIMENTAL__
    BUILD_PAIRWISE_SELECTOR(xType, yType, zType, functions::broadcast::Broadcast, ::execInverseBroadcast(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ), LIBND4J_TYPES, LIBND4J_TYPES);
#else
    BUILD_SINGLE_SELECTOR_THRICE(xType, functions::broadcast::Broadcast, ::execInverseBroadcast(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ), LIBND4J_TYPES);
#endif

    // TODO: remove after the release
    auto res = cudaStreamSynchronize(*stream);
    if (res != 0)
        throw cuda_exception::build("execInverseBroadcast failed", res);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduceSame(nd4j::LaunchContext  *lc,
                            int opNum,
                            void *hX, Nd4jLong *hXShapeInfo,
                            void *dX, Nd4jLong *dXShapeInfo,
                            void *extraParams,
                            void *hZ, Nd4jLong *hZShapeInfo,
                            void *dZ, Nd4jLong *dZShapeInfo,
                            int *dimension, int dimensionLength,
                            Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {

	auto stream = lc->getCudaStream();
	auto reductionPointer = lc->getReductionPointer();

    if (nd4j::Environment::getInstance()->isDebugAndVerbose())
        printf("SF7 opNum:[%i]\n", opNum);

    auto xType = nd4j::ArrayOptions::dataType(hXShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(hZShapeInfo);
    auto xRank = shape::rank(hXShapeInfo);

    if (zType != xType)
        throw datatype_exception::build("NativeOpExecutioner::execReduceSame requires both X & Z operands to have same type", xType, zType);

    auto numBlocks = shape::length(hZShapeInfo);
    dim3 launchDims(numBlocks, 256, 8192);

    BUILD_SINGLE_SELECTOR(xType, functions::reduce::ReduceSameFunction, ::execReduceXD(launchDims, stream, opNum, xRank, dX, dXShapeInfo, hXShapeInfo, extraParams, dZ, dZShapeInfo, hZShapeInfo, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets), LIBND4J_TYPES);

    // TODO: remove after the release
    auto res = cudaStreamSynchronize(*stream);
    if (res != 0)
        throw cuda_exception::build("execReduceSame failed", res);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduceLong(nd4j::LaunchContext  *lc,
                            int opNum,
                            void *hX, Nd4jLong *hXShapeInfo,
                            void *dX, Nd4jLong *dXShapeInfo,
                            void *extraParams,
                            void *hZ, Nd4jLong *hZShapeInfo,
                            void *dZ, Nd4jLong *dZShapeInfo,
                            int *dimension,int dimensionLength,
                            Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {

	auto stream = lc->getCudaStream();
	auto reductionPointer = lc->getReductionPointer();

    if (nd4j::Environment::getInstance()->isDebugAndVerbose())
        printf("LF7 opNum:[%i]\n", opNum);

    auto xType = nd4j::ArrayOptions::dataType(hXShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(hZShapeInfo);

    if (zType != nd4j::DataType::INT64)
        throw datatype_exception::build("NativeOpExecutioner::execReduceLong wrong Z data type", nd4j::DataType::INT64, zType);

    auto xRank = shape::rank(hXShapeInfo);
    auto numBlocks = shape::length(hZShapeInfo);
    dim3 launchDims(numBlocks, 256, 32768);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce::ReduceLongFunction, ::execReduceXD(launchDims, stream, opNum, xRank, dX, dXShapeInfo, hXShapeInfo, extraParams, dZ, dZShapeInfo, hZShapeInfo, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets), LIBND4J_TYPES, LONG_TYPES);

    // TODO: remove after the release
    auto res = cudaStreamSynchronize(*stream);
    if (res != 0)
        throw cuda_exception::build("execReduceLong failed", res);

}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduceBool(nd4j::LaunchContext  *lc,
                            int opNum,
                            void *hX, Nd4jLong *hXShapeInfo,
                            void *dX, Nd4jLong *dXShapeInfo,
                            void *extraParams,
                            void *hZ, Nd4jLong *hZShapeInfo,
                            void *dZ, Nd4jLong *dZShapeInfo,
                            int *dimension, int dimensionLength,
                            Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {

	auto stream = lc->getCudaStream();
	auto reductionPointer = lc->getReductionPointer();

    if (nd4j::Environment::getInstance()->isDebugAndVerbose())
        printf("BF7 opNum:[%i]\n", opNum);

    auto xType = nd4j::ArrayOptions::dataType(hXShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(hZShapeInfo);

    if (zType != nd4j::DataType::BOOL)
        throw std::runtime_error("NativeOpExecutioner::execReduceBool requires Z operand to have BOOL type");

    auto xRank = shape::rank(hXShapeInfo);
    auto numBlocks = shape::length(hZShapeInfo);
    dim3 launchDims(numBlocks, 256, 32768);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce::ReduceBoolFunction, ::execReduceXD(launchDims, stream, opNum, xRank, dX, dXShapeInfo, hXShapeInfo, extraParams, dZ, dZShapeInfo, hZShapeInfo, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets), LIBND4J_TYPES, BOOL_TYPES);

    // TODO: remove after the release
    auto res = cudaStreamSynchronize(*stream);
    if (res != 0)
        throw cuda_exception::build("execReduceBool failed", res);
}

////////////////////////////////////////////////////////////////////////
/**
 *
 * @param opNum
 * @param dX
 * @param dXShapeInfo
 * @param extraParams
 * @param dZ
 * @param dZShapeInfo
 * @param dimension
 * @param dimensionLength
 */
void NativeOpExecutioner::execIndexReduce(nd4j::LaunchContext  *lc,
                                int opNum,
                                void *hX, Nd4jLong *hXShapeInfo,
                                void *dX, Nd4jLong *dXShapeInfo,
                                void *extraParams,
                                void *hZ, Nd4jLong *hZShapeInfo,
                                void *dZ, Nd4jLong *dZShapeInfo,
                                int *dimension, int dimensionLength,
                                Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {

	auto stream = lc->getCudaStream();
	auto reductionPointer = lc->getReductionPointer();
	auto allocationPointer = lc->getAllocationPointer();

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("F2 opNum:[%i]\n", opNum);

	auto xType = nd4j::ArrayOptions::dataType(hXShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(hZShapeInfo);
	auto numBlocks = shape::length(hZShapeInfo);
    dim3 launchDims(numBlocks, 256, 32768);

    if (zType != nd4j::DataType::INT64 && zType != nd4j::DataType::INT32)
        throw datatype_exception::build("NativeOpExecutioner::execIndexReduce requires Z operand to have INT32/INT64 type", zType);

	auto dz = reinterpret_cast<Nd4jLong*>(dZ);

	BUILD_DOUBLE_SELECTOR(xType, zType, functions::indexreduce::IndexReduce,  ::executeIndexReduce(launchDims, stream, opNum, dX, dXShapeInfo, shape::rank(hXShapeInfo), extraParams, dz, dZShapeInfo, shape::rank(hZShapeInfo), dimension, dimensionLength, 1, allocationPointer, reductionPointer, tadShapeInfo, tadOffsets), LIBND4J_TYPES, INDEXING_TYPES);

    // TODO: remove after the release
    auto res = cudaStreamSynchronize(*stream);
    if (res != 0)
        throw cuda_exception::build("execIndexReduce failed", res);
}

////////////////////////////////////////////////////////////////////////
/**
 *
 * @param opNum
 * @param dX
 * @param dXShapeInfo
 * @param extraParams
 * @param dZ
 * @param dZShapeInfo
 */
void  NativeOpExecutioner::execReduceFloat(nd4j::LaunchContext  *lc,
										int opNum,
										void *hX, Nd4jLong *hXShapeInfo,
        								void *dX, Nd4jLong *dXShapeInfo,
        								void *extraParams,
        								void *hZ, Nd4jLong *hZShapeInfo,
										void *dZ, Nd4jLong *dZShapeInfo,
										int *dimension,int dimensionLength,
										Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {

	auto stream = lc->getCudaStream();
	auto reductionPointer = lc->getReductionPointer();

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("F8 opNum:[%i]\n", opNum);

	auto xType = nd4j::ArrayOptions::dataType(hXShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(hZShapeInfo);

    auto xRank = shape::rank(hXShapeInfo);
    auto numBlocks = shape::length(hZShapeInfo);
    dim3 launchDims(numBlocks, 256, 32768);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce::ReduceFloatFunction, ::execReduceXD(launchDims, stream, opNum, xRank, dX, dXShapeInfo, hXShapeInfo, extraParams, dZ, dZShapeInfo, hZShapeInfo, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets), LIBND4J_TYPES, FLOAT_TYPES);

    // TODO: remove after the release
    auto res = cudaStreamSynchronize(*stream);
    if (res != 0)
        throw cuda_exception::build("execReduceFloat failed", res);
}


/**
 *
 * @param opNum
 * @param dX
 * @param dXShapeInfo
 * @param extraParams
 */
////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execIndexReduceScalar(nd4j::LaunchContext  *lc,
											int opNum,
											void *hX, Nd4jLong *hXShapeInfo,
        									void *dX, Nd4jLong *dXShapeInfo,
        									void *extraParams,
        									void *hZ, Nd4jLong *hZShapeInfo,
											void *dZ, Nd4jLong *dZShapeInfo){

	if (nd4j::Environment::getInstance()->isDebug())
		printf("F1 opNum:[%i]\n", opNum);

	auto stream = lc->getCudaStream();
	auto reductionPointer = lc->getReductionPointer();
	auto allocationPointer = lc->getAllocationPointer();

    auto xLength = shape::length(hXShapeInfo);
    auto blockWidth = 256;
    auto numBlocks = CudaLaunchHelper::getReductionBlocks(xLength, blockWidth);
    dim3 launchDims(numBlocks, blockWidth, 32768);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose() && launchDims.x == 1)
		printf("AF1 opNum:[%i]\n", opNum);

	auto xType = nd4j::ArrayOptions::dataType(hXShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(hZShapeInfo);

    // FIXME: we want Z to be one of integer types
	//if (!DataTypeUtils::isZ(zType))
	//    throw nd4j::datatype_exception("NativeOpExecutioner::execIndexReduceScalar requires Z operand to have one of integer types")
	if (zType != nd4j::DataType::INT64 && zType != nd4j::DataType::INT32)
        throw nd4j::datatype_exception::build("NativeOpExecutioner::execIndexReduceScalar requires Z operand to have INT32/INT64 data type", zType);

    auto dz = reinterpret_cast<Nd4jLong*>(dZ);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::indexreduce::IndexReduce, ::executeIndexReduceScalar(launchDims, stream,
                                                                                                opNum,
                                                                                                dX, dXShapeInfo, shape::rank(hXShapeInfo),
                                                                                                extraParams,
                                                                                                dz, dZShapeInfo, 0,
                                                                                                nullptr, 0,
                                                                                                1,
                                                                                                allocationPointer, reductionPointer,
                                                                                                nullptr, nullptr), LIBND4J_TYPES, INDEXING_TYPES);
    // TODO: remove after the release
    auto res = cudaStreamSynchronize(*stream);
    if (res != 0)
        throw cuda_exception::build("execIndexReduceScalar failed", res);
}


////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduceFloatScalar(nd4j::LaunchContext  *lc,
                                                int opNum,
                                                void *hX, Nd4jLong *hXShapeInfo,
                                                void *dX, Nd4jLong *dXShapeInfo,
                                                void *extraParams,
                                                void *hZ, Nd4jLong *hZShapeInfo,
                                                void *dZ, Nd4jLong *dZShapeInfo) {

    auto stream = lc->getCudaStream();
    auto reductionPointer = lc->getReductionPointer();

    auto xType = nd4j::ArrayOptions::dataType(hXShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(hZShapeInfo);

    auto xLength = shape::length(hXShapeInfo);
    auto blockWidth = 256;
    auto numBlocks = CudaLaunchHelper::getReductionBlocks(xLength, blockWidth);
    dim3 launchDims(numBlocks, blockWidth, 32768);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce::ReduceFloatFunction, ::execReduceScalar(launchDims, stream, opNum, dX,dXShapeInfo, hXShapeInfo, extraParams, dZ,dZShapeInfo, hZShapeInfo, nullptr, 0, reductionPointer, nullptr), LIBND4J_TYPES, FLOAT_TYPES);

    // TODO: remove after the release
    auto res = cudaStreamSynchronize(*stream);
    if (res != 0)
        throw cuda_exception::build("execReduceFloatScalar failed", res);
}


////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduceBoolScalar(nd4j::LaunchContext  *lc,
                                        int opNum,
                                        void *hX, Nd4jLong *hXShapeInfo,
                                        void *dX, Nd4jLong *dXShapeInfo,
                                        void *extraParams,
                                        void *hZ, Nd4jLong *hZShapeInfo,
                                        void *dZ, Nd4jLong *dZShapeInfo) {

    auto stream = lc->getCudaStream();
    auto reductionPointer = lc->getReductionPointer();

    auto xType = nd4j::ArrayOptions::dataType(hXShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(hZShapeInfo);

    if (zType != nd4j::DataType::BOOL)
        throw std::runtime_error("NativeOpExecutioner::execReduceBoolScalar requires Z operand to have BOOL type");

    auto xLength = shape::length(hXShapeInfo);
    auto blockWidth = 256;
    auto numBlocks = CudaLaunchHelper::getReductionBlocks(xLength, blockWidth);
    dim3 launchDims(numBlocks, blockWidth, 32768);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce::ReduceBoolFunction, ::execReduceScalar(launchDims, stream, opNum, dX, dXShapeInfo, hXShapeInfo, extraParams, dZ, dZShapeInfo, hZShapeInfo, nullptr, 0, reductionPointer, nullptr), LIBND4J_TYPES, BOOL_TYPES);

    // TODO: remove after the release
    auto res = cudaStreamSynchronize(*stream);
    if (res != 0)
        throw cuda_exception::build("execReduceBoolScalar failed", res);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduceSameScalar(nd4j::LaunchContext  *lc,
                                        int opNum,
                                        void *hX, Nd4jLong *hXShapeInfo,
                                        void *dX, Nd4jLong *dXShapeInfo,
                                        void *extraParams,
                                        void *hZ, Nd4jLong *hZShapeInfo,
                                        void *dZ, Nd4jLong *dZShapeInfo) {

    auto stream = lc->getCudaStream();
    auto reductionPointer = lc->getReductionPointer();

    auto xType = nd4j::ArrayOptions::dataType(hXShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(hZShapeInfo);

    if (zType != xType)
        throw datatype_exception::build("NativeOpExecutioner::execReduceSameScalar requires both X & Z operands to have same type", xType, zType);

    auto xLength = shape::length(hXShapeInfo);
    auto blockWidth = 256;
    auto numBlocks = CudaLaunchHelper::getReductionBlocks(xLength, blockWidth);
    dim3 launchDims(numBlocks, blockWidth, 32768);

    BUILD_SINGLE_SELECTOR(xType, functions::reduce::ReduceSameFunction, ::execReduceScalar(launchDims, stream, opNum, dX, dXShapeInfo, hXShapeInfo, extraParams, dZ, dZShapeInfo, hZShapeInfo, nullptr, 0, reductionPointer, nullptr), LIBND4J_TYPES);

    // TODO: remove after the release
    auto res = cudaStreamSynchronize(*stream);
    if (res != 0)
        throw cuda_exception::build("execReduceSameScalar failed", res);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduceLongScalar(nd4j::LaunchContext  *lc,
                                    int opNum,
                                    void *hX, Nd4jLong *hXShapeInfo,
                                    void *dX, Nd4jLong *dXShapeInfo,
                                    void *extraParams,
                                    void *hZ, Nd4jLong *hZShapeInfo,
                                    void *dZ, Nd4jLong *dZShapeInfo) {

    auto stream = lc->getCudaStream();
    auto reductionPointer = lc->getReductionPointer();

    auto xType = nd4j::ArrayOptions::dataType(hXShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(hZShapeInfo);

    if (zType != nd4j::DataType::INT64)
        throw datatype_exception::build("NativeOpExecutioner::execReduceLongScalar wrong Z data type", nd4j::DataType::INT64, zType);

    auto xLength = shape::length(hXShapeInfo);
    auto blockWidth = 256;
    auto numBlocks = CudaLaunchHelper::getReductionBlocks(xLength, blockWidth);
    dim3 launchDims(numBlocks, blockWidth, 32768);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce::ReduceLongFunction, ::execReduceScalar(launchDims, stream, opNum, dX, dXShapeInfo, hXShapeInfo, extraParams, dZ, dZShapeInfo, hZShapeInfo, nullptr, 0, reductionPointer, nullptr), LIBND4J_TYPES, LONG_TYPES);

    // TODO: remove after the release
    auto res = cudaStreamSynchronize(*stream);
    if (res != 0)
        throw cuda_exception::build("execReduceLongScalar failed", res);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execTransformSame(nd4j::LaunchContext  *lc,
									int opNum,
                                   	void *hX, Nd4jLong *hXShapeInfo,
                                   	void *dX, Nd4jLong *dXShapeInfo,
                                   	void *hZ, Nd4jLong *hZShapeInfo,
                                   	void *dZ, Nd4jLong *dZShapeInfo,
                                   	void *extraParams,
                                   	Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {

    auto stream = lc->getCudaStream();
    dim3 launchDims(512, 512, 16384);

    auto xRank = shape::rank(hXShapeInfo);
	auto zRank = shape::rank(hZShapeInfo);
	auto xType = ArrayOptions::dataType(hXShapeInfo);
    auto zType = ArrayOptions::dataType(hZShapeInfo);

    if (xType != zType)
        throw std::runtime_error("NativeOpExecutioner::execTransformSame requires X & Z to have same type");

    BUILD_SINGLE_SELECTOR(xType, functions::transform::TransformSame, ::executeTransformShaped(launchDims, stream, opNum, dX, dXShapeInfo, xRank, extraParams, dZ, dZShapeInfo, zRank, nullptr, nullptr, nullptr, nullptr), LIBND4J_TYPES);

    // TODO: remove after the release
    auto res = cudaStreamSynchronize(*stream);
    if (res != 0)
        throw cuda_exception::build("execTransformSame failed", res);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execTransformBool(nd4j::LaunchContext  *lc,
                                int opNum,
                                void *hX, Nd4jLong *hXShapeInfo,
                                void *dX, Nd4jLong *dXShapeInfo,
                                void *hZ, Nd4jLong *hZShapeInfo,
                                void *dZ, Nd4jLong *dZShapeInfo,
                                void *extraParams,
                                Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {

	auto stream = lc->getCudaStream();
	dim3 launchDims(512, 512, 16384);

	auto xRank = shape::rank(hXShapeInfo);
	auto zRank = shape::rank(hZShapeInfo);
	auto xType = ArrayOptions::dataType(hXShapeInfo);
    auto zType = ArrayOptions::dataType(hZShapeInfo);

    if (!DataTypeUtils::isB(zType))
        throw std::runtime_error("NativeOpExecutioner::execTransformBool requires Z to have same boolean type");

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::transform::TransformBool, ::executeTransformShaped(launchDims, stream, opNum, dX, dXShapeInfo, xRank, extraParams, dZ, dZShapeInfo, zRank, nullptr, nullptr, nullptr, nullptr), LIBND4J_TYPES, BOOL_TYPES);

    // TODO: remove after the release
    auto res = cudaStreamSynchronize(*stream);
    if (res != 0)
        throw cuda_exception::build("execTransformBool failed", res);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execTransformAny(nd4j::LaunchContext  *lc,
                                		int opNum,
                                		void *hX, Nd4jLong *hXShapeInfo,
                                		void *dX, Nd4jLong *dXShapeInfo,
                                		void *hZ, Nd4jLong *hZShapeInfo,
                                		void *dZ, Nd4jLong *dZShapeInfo,
                                		void *extraParams,
                                		Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, bool allowParallelism) {

	auto stream = lc->getCudaStream();

	auto xRank = shape::rank(hXShapeInfo);
	auto zRank = shape::rank(hZShapeInfo);
	auto xType = ArrayOptions::dataType(hXShapeInfo);
	auto zType = ArrayOptions::dataType(hZShapeInfo);

	dim3 launchDims(512, 512, 2048);

	BUILD_DOUBLE_SELECTOR(xType, zType, functions::transform::TransformAny, ::executeTransformShaped(launchDims, stream, opNum, dX, dXShapeInfo, xRank, extraParams, dZ, dZShapeInfo, zRank, nullptr, nullptr, nullptr, nullptr), LIBND4J_TYPES, LIBND4J_TYPES);

    // TODO: remove after the release
    auto res = cudaStreamSynchronize(*stream);
    if (res != 0)
        throw cuda_exception::build("execTransformAny failed", res);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execTransformStrict(nd4j::LaunchContext  *lc,
                                    int opNum,
                                    void *hX, Nd4jLong *hXShapeInfo,
                                    void *dX, Nd4jLong *dXShapeInfo,
                                    void *hZ, Nd4jLong *hZShapeInfo,
                                    void *dZ, Nd4jLong *dZShapeInfo,
                                    void *extraParams,
                                    Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {

    auto stream = lc->getCudaStream();
    dim3 launchDims(512, 512, 16384);

    auto xRank = shape::rank(hXShapeInfo);
    auto zRank = shape::rank(hZShapeInfo);
    auto xType = ArrayOptions::dataType(hXShapeInfo);
    auto zType = ArrayOptions::dataType(hZShapeInfo);

    if (xType != zType || !DataTypeUtils::isR(xType))
        throw datatype_exception::build("NativeOpExecutioner::execTransformStrict requires X & Z to have same floating point type", xType, zType);

    BUILD_SINGLE_SELECTOR(xType, functions::transform::TransformStrict, ::executeTransformShaped(launchDims, stream, opNum, dX, dXShapeInfo, xRank, extraParams, dZ, dZShapeInfo, zRank, nullptr, nullptr, nullptr, nullptr), FLOAT_TYPES);

    // TODO: remove after the release
    auto res = cudaStreamSynchronize(*stream);
    if (res != 0)
        throw cuda_exception::build("execTransformStrict failed", res);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execTransformFloat(nd4j::LaunchContext  *lc,
                                int opNum,
                                void *hX, Nd4jLong *hXShapeInfo,
                                void *dX, Nd4jLong *dXShapeInfo,
                                void *hZ, Nd4jLong *hZShapeInfo,
                                void *dZ, Nd4jLong *dZShapeInfo,
                                void *extraParams,
                                Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {

    auto stream = lc->getCudaStream();
    auto reductionPointer = lc->getReductionPointer();

    auto xRank = shape::rank(hXShapeInfo);
    auto zRank = shape::rank(hZShapeInfo);
    auto xType = ArrayOptions::dataType(hXShapeInfo);
    auto zType = ArrayOptions::dataType(hZShapeInfo);

    if (!DataTypeUtils::isR(zType))
        throw datatype_exception::build("NativeOpExecutioner::execTransformFloat requires Z to have floating point type", zType);

    dim3 launchDims(512, 512, 2048);
    BUILD_DOUBLE_SELECTOR(xType, zType, functions::transform::TransformFloat, ::executeTransformShaped(launchDims, stream, opNum, dX, dXShapeInfo, xRank, extraParams, dZ, dZShapeInfo, zRank, nullptr, nullptr, nullptr, nullptr), LIBND4J_TYPES, FLOAT_TYPES);

    // TODO: remove after the release
    auto res = cudaStreamSynchronize(*stream);
    if (res != 0)
        throw cuda_exception::build("execTransformFloat failed", res);
}


////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execSummaryStats(nd4j::LaunchContext  *lc,
                                int opNum,
                                void *hX, Nd4jLong *hXShapeInfo,
                                void *dX, Nd4jLong *dXShapeInfo,
                                void *extraParams,
                                void *hZ, Nd4jLong *hZShapeInfo,
                                void *dZ, Nd4jLong *dZShapeInfo,
                                bool biasCorrected) {

    auto stream = lc->getCudaStream();
    auto reductionPointer = lc->getReductionPointer();

    dim3 launchDims = dim3(256, 256, 32768);

	auto xType = nd4j::ArrayOptions::dataType(hXShapeInfo);
	auto zType = nd4j::ArrayOptions::dataType(hZShapeInfo);

    if (!DataTypeUtils::isR(zType))
        throw nd4j::datatype_exception::build("NativeOpExecutioner::execSummaryStats requires Z operand to have floating point data type", zType);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::summarystats::SummaryStatsReduce, ::execSummaryStatsReduce(launchDims, stream, opNum, dX, dXShapeInfo, hXShapeInfo, extraParams, dZ, dZShapeInfo, hZShapeInfo, nullptr, nullptr, biasCorrected, reductionPointer), LIBND4J_TYPES, FLOAT_TYPES);

    // TODO: remove after the release
    auto res = cudaStreamSynchronize(*stream);
    if (res != 0)
        throw cuda_exception::build("execSummaryStats A failed", res);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execSummaryStats(nd4j::LaunchContext  *lc,
                                			int opNum,
                                			void *hX, Nd4jLong *hXShapeInfo,
                                			void *dX, Nd4jLong *dXShapeInfo,
                                			void *extraParams,
                                			void *hZ, Nd4jLong *hZShapeInfo,
                                			void *dZ, Nd4jLong *dZShapeInfo,
                                			int *dimension, int dimensionLength,
                                            Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets,
                                			bool biasCorrected) {
	auto stream = lc->getCudaStream();
	auto reductionPointer = lc->getReductionPointer();

    dim3 launchDims = dim3(256, 256, 32768);

	auto xType = nd4j::ArrayOptions::dataType(hXShapeInfo);
	auto zType = nd4j::ArrayOptions::dataType(hZShapeInfo);

    if (!DataTypeUtils::isR(zType))
        throw nd4j::datatype_exception::build("NativeOpExecutioner::execSummaryStats requires Z operand to have floating point data type", zType);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::summarystats::SummaryStatsReduce, ::execSummaryStatsReduce(launchDims, stream, opNum, dX, dXShapeInfo, hXShapeInfo, extraParams, dZ, dZShapeInfo, hZShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, biasCorrected, reductionPointer), LIBND4J_TYPES, FLOAT_TYPES);

    // TODO: remove after the release
    auto res = cudaStreamSynchronize(*stream);
    if (res != 0)
        throw cuda_exception::build("execSummaryStats B failed", res);
}


////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduce3(nd4j::LaunchContext  *lc,
                            int opNum,
                            void *hX, Nd4jLong *hXShapeInfo,
                            void *dX, Nd4jLong *dXShapeInfo,
                            void *extraParams,
                            void *hY, Nd4jLong *hYShapeInfo,
                            void *dY, Nd4jLong *dYShapeInfo,
                            void *hZ, Nd4jLong *hZShapeInfo,
                            void *dZ, Nd4jLong *dZShapeInfo) {

	auto stream = lc->getCudaStream();
    auto reductionPointer = lc->getReductionPointer();
	auto allocationPointer = lc->getAllocationPointer();

    auto xType = nd4j::ArrayOptions::dataType(hXShapeInfo);
    auto yType = nd4j::ArrayOptions::dataType(hYShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(hZShapeInfo);

    auto blockWidth = 256;
    auto numBlocks = CudaLaunchHelper::getReductionBlocks(shape::length(hXShapeInfo), blockWidth);
    dim3 launchDims(numBlocks, blockWidth, 32768);

    if (xType != yType)
        throw nd4j::datatype_exception::build("NativeOpExecutioner::execReduce3 requires Y operand to have X type", xType, yType);

    if (!DataTypeUtils::isR(zType))
        throw nd4j::datatype_exception::build("NativeOpExecutioner::execReduce3 requires Z operand to have floating point data type", zType);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce3::Reduce3, ::execScalar(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, extraParams, dZ, dZShapeInfo, allocationPointer, reductionPointer, nullptr), LIBND4J_TYPES, FLOAT_TYPES);

    // TODO: remove after the release
    auto res = cudaStreamSynchronize(*stream);
    if (res != 0)
        throw cuda_exception::build("execReduce3 failed", res);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduce3(nd4j::LaunchContext  *lc,
                            int opNum,
                            void *hX, Nd4jLong *hXShapeInfo,
                            void *dX, Nd4jLong *dXShapeInfo,
                            void *extraParams,
                            void *hY, Nd4jLong *hYShapeInfo,
                            void *dY, Nd4jLong *dYShapeInfo,
                            void *hZ, Nd4jLong *hZShapeInfo,
                            void *dZ, Nd4jLong *dZShapeInfo,
                            int *dimension, int dimensionLength,
                            Nd4jLong* tadOnlyShapeInfo, Nd4jLong* tadOffsets,
                            Nd4jLong* yTadOnlyShapeInfo, Nd4jLong* yTadOffsets) {

    if(shape::isScalar(hZShapeInfo)) {
        NativeOpExecutioner::execReduce3(lc, opNum, hX, hXShapeInfo, dX, dXShapeInfo, extraParams, hY, hYShapeInfo, dY, dYShapeInfo, hZ, hZShapeInfo, dZ, dZShapeInfo);
        return;
    }

    auto stream = lc->getCudaStream();
    auto allocationPointer = lc->getAllocationPointer();

    auto xType = nd4j::ArrayOptions::dataType(hXShapeInfo);
    auto yType = nd4j::ArrayOptions::dataType(hYShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(hZShapeInfo);

     if (xType != yType)
        throw nd4j::datatype_exception::build("NativeOpExecutioner::execReduce3 requires Y operand to have X type", xType, yType);

    if (!DataTypeUtils::isR(zType))
        throw nd4j::datatype_exception::build("NativeOpExecutioner::execReduce3 requires Z operand to have floating point data type", zType);


    auto numBlocks = shape::length(hZShapeInfo);
    dim3 launchDims(numBlocks, 256, 32768);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce3::Reduce3, ::exec(launchDims, stream, opNum,
                                                                    dX, dXShapeInfo,
                                                                    dY, dYShapeInfo,
                                                                    extraParams,
                                                                    dZ, dZShapeInfo,
                                                                    dimension, dimensionLength,
                                                                    1,
                                                                    allocationPointer,
                                                                    tadOnlyShapeInfo, tadOffsets,
                                                                    yTadOnlyShapeInfo, yTadOffsets), LIBND4J_TYPES, FLOAT_TYPES);

    // TODO: remove after the release
    auto res = cudaStreamSynchronize(*stream);
    if (res != 0)
        throw cuda_exception::build("execReduce3 B failed", res);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduce3Scalar(nd4j::LaunchContext  *lc,
								  int opNum,
                                  void *hX, Nd4jLong *hXShapeInfo,
                                  void *dX, Nd4jLong *dXShapeInfo,
                                  void *extraParams,
                                  void *hY, Nd4jLong *hYShapeInfo,
                                  void *dY, Nd4jLong *dYShapeInfo,
                                  void *hZ, Nd4jLong *hZShapeInfo,
                                  void *dZ, Nd4jLong *dZShapeInfo) {


	auto stream 		   = lc->getCudaStream();
	auto allocationPointer = lc->getAllocationPointer();
	auto reductionPointer  = lc->getReductionPointer();

    auto xType = nd4j::ArrayOptions::dataType(hXShapeInfo);
    auto yType = nd4j::ArrayOptions::dataType(hYShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(hZShapeInfo);

    auto xLength = shape::length(hXShapeInfo);
    auto blockWidth = 256;
    auto numBlocks = CudaLaunchHelper::getReductionBlocks(xLength, blockWidth);
    dim3 launchDims(numBlocks, blockWidth, 32768);

    if (xType != yType)
        throw nd4j::datatype_exception::build("NativeOpExecutioner::execReduce3Scalar requires Y operand to have X type", xType, yType);

    if (!DataTypeUtils::isR(zType))
        throw nd4j::datatype_exception::build("NativeOpExecutioner::execReduce3Scalar requires Z operand to have floating point data type", zType);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce3::Reduce3, ::execScalar(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, extraParams, dZ, dZShapeInfo, allocationPointer, reductionPointer, nullptr), LIBND4J_TYPES, FLOAT_TYPES);

    // TODO: remove after the release
    auto res = cudaStreamSynchronize(*stream);
    if (res != 0)
        throw cuda_exception::build("execReduce3Scalar failed", res);
}


////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execScalarBool(nd4j::LaunchContext  *lc,
										int opNum,
										void *hX, Nd4jLong *hXShapeInfo,
										void *dX, Nd4jLong *dXShapeInfo,
										void *hZ, Nd4jLong *hZShapeInfo,
										void *dZ, Nd4jLong *dZShapeInfo,
										void *hScalar, Nd4jLong *hScalarShapeInfo,
										void *dScalar, Nd4jLong *dScalarShapeInfo,
										void *extraParams, bool allowParallelism) {

	auto stream = lc->getCudaStream();

	dim3 launchDims = dim3(256, 512, 8192);

	auto xType = nd4j::ArrayOptions::dataType(hXShapeInfo);
	auto yType = nd4j::ArrayOptions::dataType(hScalarShapeInfo);
	auto zType = nd4j::ArrayOptions::dataType(hZShapeInfo);

	if (xType != yType )
		throw std::runtime_error("NativeOpExecutioner::execScalarBool requires X & Y to have same type");

	if (!DataTypeUtils::isB(zType) )
		throw std::runtime_error("NativeOpExecutioner::execScalarBool requires Z operand to have BOOL type");

	BUILD_DOUBLE_SELECTOR(xType, zType, functions::scalar::ScalarBoolTransform, ::executeCudaShaped(launchDims, stream, opNum, dX, dXShapeInfo, dZ, dZShapeInfo, dScalar, extraParams), LIBND4J_TYPES, BOOL_TYPES);

    // TODO: remove after the release
    auto res = cudaStreamSynchronize(*stream);
    if (res != 0)
        throw cuda_exception::build("execScalarBool failed", res);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execScalarBool(nd4j::LaunchContext  *lc,
						   				int opNum,
						   				void *hX, Nd4jLong *hXShapeInfo,
						   				void *dX, Nd4jLong *dXShapeInfo,
                                        void *extraParams,
						   				void *hZ, Nd4jLong *hZShapeInfo,
						   				void *dZ, Nd4jLong *dZShapeInfo,
						   				void *hScalars, Nd4jLong *hScalarShapeInfo,
						   				void *dScalars, Nd4jLong *dScalarShapeInfo,
						   				int *dimension, int dimensionLength,
                           				Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets,
                           				Nd4jLong *tadShapeInfoZ, Nd4jLong *tadOffsetsZ) {

	auto stream = lc->getCudaStream();

	dim3 launchDims(256, 512, 8192);

	auto xType = nd4j::ArrayOptions::dataType(hXShapeInfo);
	auto yType = nd4j::ArrayOptions::dataType(hScalarShapeInfo);
	auto zType = nd4j::ArrayOptions::dataType(hZShapeInfo);

	if (xType != yType )
		throw std::runtime_error("NativeOpExecutioner::execScalarBool requires X & Y to have same type");

	if (!DataTypeUtils::isB(zType) )
		throw std::runtime_error("NativeOpExecutioner::execScalarBool requires Z operand to have BOOL type");

	BUILD_DOUBLE_SELECTOR(xType, zType, functions::scalar::ScalarBoolTransform, ::executeCudaAlongDimension(launchDims, stream, opNum, dX, dXShapeInfo, dZ, dZShapeInfo, dScalars, extraParams, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), LIBND4J_TYPES, BOOL_TYPES);

    // TODO: remove after the release
    auto res = cudaStreamSynchronize(*stream);
    if (res != 0)
        throw cuda_exception::build("execScalarBool B failed", res);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execScalar(nd4j::LaunchContext  *lc,
									int opNum,
									void *hX, Nd4jLong *hXShapeInfo,
									void *dX, Nd4jLong *dXShapeInfo,
									void *hZ, Nd4jLong *hZShapeInfo,
									void *dZ, Nd4jLong *dZShapeInfo,
									void *hScalar, Nd4jLong *hScalarShapeInfo,
									void *dScalar, Nd4jLong *dScalarShapeInfo,
									void *extraParams, bool allowParallelism) {

	auto stream = lc->getCudaStream();

	dim3 launchDims(256, 512, 8192);

	auto xType = nd4j::ArrayOptions::dataType(hXShapeInfo);
	auto yType = nd4j::ArrayOptions::dataType(hScalarShapeInfo);
	auto zType = nd4j::ArrayOptions::dataType(hZShapeInfo);


#ifdef __ND4J_EXPERIMENTAL__
	BUILD_PAIRWISE_SELECTOR(xType, yType, zType, functions::scalar::ScalarTransform, ::executeCudaShaped(launchDims, stream, opNum, dX, dXShapeInfo, hXShapeInfo, dZ, dZShapeInfo, hZShapeInfo, dScalar, extraParams), LIBND4J_TYPES, LIBND4J_TYPES);
#else
	BUILD_SINGLE_SELECTOR_THRICE(xType, functions::scalar::ScalarTransform, ::executeCudaShaped(launchDims, stream, opNum, dX, dXShapeInfo, hXShapeInfo, dZ, dZShapeInfo, hZShapeInfo, dScalar, extraParams), LIBND4J_TYPES);
#endif

    // TODO: remove after the release
    auto res = cudaStreamSynchronize(*stream);
    if (res != 0)
        throw cuda_exception::build("execScalar failed", res);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execScalar(nd4j::LaunchContext  *lc,
					 				int opNum,
					 				void *hX, Nd4jLong *hXShapeInfo,
                     				void *dX, Nd4jLong *dXShapeInfo,
                                    void *extraParams,
                     				void *hZ, Nd4jLong *hZShapeInfo,
                     				void *dZ, Nd4jLong *dZShapeInfo,
                     				void *hScalars, Nd4jLong *hScalarShapeInfo,
                     				void *dScalars, Nd4jLong *dScalarShapeInfo,
					 				int *dimension, int dimensionLength,
                     				Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets,
                     				Nd4jLong *tadShapeInfoZ, Nd4jLong *tadOffsetsZ) {

    auto stream = lc->getCudaStream();

    auto xType = nd4j::ArrayOptions::dataType(hXShapeInfo);
    auto yType = nd4j::ArrayOptions::dataType(hScalarShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(hZShapeInfo);

	dim3 launchDims(256, 256, 16384);

#ifdef __ND4J_EXPERIMENTAL__
    BUILD_PAIRWISE_SELECTOR(xType, yType, zType, functions::scalar::ScalarTransform, ::executeCudaAlongDimension(launchDims, stream, opNum, dX, dXShapeInfo, dZ, dZShapeInfo, dScalars, extraParams, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), LIBND4J_TYPES, LIBND4J_TYPES);
#else
	BUILD_SINGLE_SELECTOR_THRICE(xType, functions::scalar::ScalarTransform, ::executeCudaAlongDimension(launchDims, stream, opNum, dX, dXShapeInfo, dZ, dZShapeInfo, dScalars, extraParams, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), LIBND4J_TYPES);
#endif

    // TODO: remove after the release
    auto res = cudaStreamSynchronize(*stream);
    if (res != 0)
        throw cuda_exception::build("execScalar B failed", res);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execRandom(nd4j::LaunchContext  *lc,
						  int opNum,
                          Nd4jPointer stateHost,
                          void *hZ, Nd4jLong *hZShapeInfo,
                          void *dZ, Nd4jLong *dZShapeInfo,
                          void *extraArguments) {

    auto stream = lc->getCudaStream();
    auto sizeOf = sizeof(nd4j::graph::RandomGenerator);
    Nd4jPointer stateDevice;

    cudaError_t res = cudaMalloc(reinterpret_cast<void **>(&stateDevice), sizeOf);
    checkCudaErrors(cudaStreamSynchronize(*stream));
    checkCudaErrors(cudaMemcpyAsync(stateDevice, stateHost, sizeOf, cudaMemcpyHostToDevice, *stream));

    dim3 launchDims = dim3(512, 512, 32768);
    auto zType = nd4j::ArrayOptions::dataType(hZShapeInfo);

    auto rng = reinterpret_cast<nd4j::graph::RandomGenerator*>(stateHost);

    // functions::random::RandomFunction<float>::executeCudaSingle(launchDims, extraPointers, opNum, stateHost, dZ, dZShapeInfo, extraArguments),
    BUILD_SINGLE_SELECTOR(zType, functions::random::RandomFunction, ::executeCudaSingle(launchDims, stream, opNum, stateDevice, dZ, dZShapeInfo, extraArguments), FLOAT_TYPES);

    res = cudaStreamSynchronize(*stream);
    if (res != 0)
        throw cuda_exception::build("execRandom X failed", res);

    cudaFree(stateDevice);

    rng->rewindH(shape::length(hZShapeInfo));
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execRandom(nd4j::LaunchContext  *lc,
							int opNum,
							Nd4jPointer stateHost,
						   	void *hX, Nd4jLong *hXShapeInfo,
						   	void *dX, Nd4jLong *dXShapeInfo,
						   	void *hZ, Nd4jLong *hZShapeInfo,
						   	void *dZ, Nd4jLong *dZShapeInfo,
						   	void *extraArguments) {

    auto stream = lc->getCudaStream();

    auto sizeOf = sizeof(nd4j::graph::RandomGenerator);
    Nd4jPointer stateDevice;

    cudaError_t res = cudaMalloc(reinterpret_cast<void **>(&stateDevice), sizeOf);
    checkCudaErrors(cudaStreamSynchronize(*stream));
    checkCudaErrors(cudaMemcpyAsync(stateDevice, stateHost, sizeOf, cudaMemcpyHostToDevice, *stream));

    auto rng = reinterpret_cast<nd4j::graph::RandomGenerator*>(stateHost);

    dim3 launchDims = dim3(512, 512, 32768);
    auto xType = nd4j::ArrayOptions::dataType(hZShapeInfo);
    // functions::random::RandomFunction<float>::executeCudaDouble(launchDims, extraPointers, opNum, stateHost, dX, dXShapeInfo, dZ, dZShapeInfo, extraArguments);
    BUILD_SINGLE_SELECTOR(xType, functions::random::RandomFunction, ::executeCudaDouble(launchDims, stream, opNum, stateDevice, dX, dXShapeInfo, dZ, dZShapeInfo, extraArguments), FLOAT_TYPES);

    res = cudaStreamSynchronize(*stream);
    if (res != 0)
        throw cuda_exception::build("execRandom XY failed", res);

    cudaFree(stateDevice);

    rng->rewindH(shape::length(hZShapeInfo));
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execRandom(nd4j::LaunchContext  *lc,
							int opNum,
							Nd4jPointer stateHost,
							void *hX, Nd4jLong *hXShapeInfo,
							void *dX, Nd4jLong *dXShapeInfo,
							void *hY, Nd4jLong *hYShapeInfo,
							void *dY, Nd4jLong *dYShapeInfo,
							void *hZ, Nd4jLong *hZShapeInfo,
							void *dZ, Nd4jLong *dZShapeInfo,
							void *extraArguments) {

    auto stream = lc->getCudaStream();
    auto sizeOf = sizeof(nd4j::graph::RandomGenerator);
    Nd4jPointer stateDevice;

    cudaError_t res = cudaMalloc(reinterpret_cast<void **>(&stateDevice), sizeOf);
    checkCudaErrors(cudaStreamSynchronize(*stream));
    checkCudaErrors(cudaMemcpyAsync(stateDevice, stateHost, sizeOf, cudaMemcpyHostToDevice, *stream));

    auto rng = reinterpret_cast<nd4j::graph::RandomGenerator*>(stateHost);

    dim3 launchDims = dim3(512, 512, 32768);
    auto xType = nd4j::ArrayOptions::dataType(hZShapeInfo);
    // functions::random::RandomFunction<float>::executeCudaTriple(launchDims, extraPointers, opNum, stateHost, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo, extraArguments);
    BUILD_SINGLE_SELECTOR(xType, functions::random::RandomFunction, ::executeCudaTriple(launchDims, stream, opNum, stateDevice, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo, extraArguments), FLOAT_TYPES);

    res = cudaStreamSynchronize(*stream);
    if (res != 0)
        throw cuda_exception::build("execRandom XYZ failed", res);

    cudaFree(stateDevice);

    rng->rewindH(shape::length(hZShapeInfo));
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduce3All(nd4j::LaunchContext  *lc,
									int opNum,
									void *hX, Nd4jLong *hXShapeInfo,
                            		void *dX, Nd4jLong *dXShapeInfo,
                            		void *extraParamsVals,
									void *hY, Nd4jLong *hYShapeInfo,
                            		void *dY, Nd4jLong *dYShapeInfo,
                            		void *hZ, Nd4jLong *hZShapeInfo,
                            		void *dZ, Nd4jLong *dZShapeInfo,
									int *dimension, int dimensionLength,
									Nd4jLong *xTadShapeInfo, Nd4jLong *xOffsets,
									Nd4jLong *yTadShapeInfo, Nd4jLong *yOffsets) {

    auto stream = lc->getCudaStream();
    auto allocationPointer = lc->getAllocationPointer();
	auto reductionPointer  = lc->getReductionPointer();

    if (nd4j::Environment::getInstance()->isDebugAndVerbose())
        printf("D119 opNum:[%i]\n", opNum);

    dim3 launchDims(shape::length(hZShapeInfo), 256, 32768);

    if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
        printf("AD119 opNum:[%i]\n", opNum);

    auto xType = nd4j::ArrayOptions::dataType(hXShapeInfo);
    auto yType = nd4j::ArrayOptions::dataType(hYShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(hZShapeInfo);

    if (yType != xType)
        throw nd4j::datatype_exception::build("NativeOpExecutioner::execReduce3All both operands must have same data type", xType, yType);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce3::Reduce3, ::execAll(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, extraParamsVals, dZ, dZShapeInfo, dimension, dimensionLength, 1, allocationPointer, xTadShapeInfo, xOffsets, yTadShapeInfo, yOffsets), LIBND4J_TYPES, FLOAT_TYPES);

    // TODO: remove after the release
    auto res = cudaStreamSynchronize(*stream);
    if (res != 0)
        throw cuda_exception::build("execReduce3All failed", res);
}


////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduce3TAD(nd4j::LaunchContext  *lc,
                                            int opNum,
                                            void *hX, Nd4jLong *hXShapeInfo,
                                            void *dX, Nd4jLong *dXShapeInfo,
                                            void *extraParams,
                                            void *hY, Nd4jLong *hYShapeInfo,
                                            void *dY, Nd4jLong *dYShapeInfo,
                                            void *hZ, Nd4jLong *hZShapeInfo,
                                            void *dZ, Nd4jLong *dZShapeInfo,
                                            int *dimension, int dimensionLength,
                                            Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets,
                                            Nd4jLong *yTadShapeInfo, Nd4jLong *yTadOffsets) {

    if(shape::isScalar(hZShapeInfo)) {
        NativeOpExecutioner::execReduce3(lc, opNum, hX, hXShapeInfo, dX, dXShapeInfo, extraParams, hY, hYShapeInfo, dY, dYShapeInfo, hZ, hZShapeInfo, dZ, dZShapeInfo);
        return;
    }

    auto stream = lc->getCudaStream();
    auto allocationPointer = lc->getAllocationPointer();

    auto xType = nd4j::ArrayOptions::dataType(hXShapeInfo);
    auto yType = nd4j::ArrayOptions::dataType(hYShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(hZShapeInfo);

     if (xType != yType)
        throw nd4j::datatype_exception::build("NativeOpExecutioner::execReduce3TAD requires Y operand to have X type", xType, yType);

    if (!DataTypeUtils::isR(zType))
        throw nd4j::datatype_exception::build("NativeOpExecutioner::execReduce3TAD requires Z operand to have floating point data type", zType);

    auto numBlocks = shape::length(hZShapeInfo);
    dim3 launchDims(numBlocks, 256, 32768);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce3::Reduce3, ::exec(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, extraParams, dZ, dZShapeInfo, dimension, dimensionLength, 1, allocationPointer, tadShapeInfo, tadOffsets, yTadShapeInfo, yTadOffsets), LIBND4J_TYPES, FLOAT_TYPES);

    // TODO: remove after the release
    auto res = cudaStreamSynchronize(*stream);
    if (res != 0)
        throw cuda_exception::build("execReduce3TAD failed", res);
}

