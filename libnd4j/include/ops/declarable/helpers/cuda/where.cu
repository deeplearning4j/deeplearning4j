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
// CUDA implementation of Where op
// @author Adam Gibson
//

#include <array/NDArrayFactory.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/helpers/where.h>
#include <system/op_boilerplate.h>

#include "execution/cuda/LaunchDims.h"
#include "helpers/DebugHelper.h"

namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
// Kernel to count non-zero elements in condition array
template <typename T>
SD_KERNEL static void countTrueElementsKernel(const void* vx, const LongType* xShapeInfo,
                                               LongType* count) {
    const auto x = reinterpret_cast<const T*>(vx);

    __shared__ LongType xLen;
    __shared__ LongType xRank;
    __shared__ const LongType* xShape;
    __shared__ const LongType* xStride;
    __shared__ LongType localCount;

    if (threadIdx.x == 0) {
        xLen = shape::length(xShapeInfo);
        xRank = shape::rank(xShapeInfo);
        xShape = shape::shapeOf(xShapeInfo);
        xStride = shape::stride(xShapeInfo);
        localCount = 0;
    }
    __syncthreads();

    // Each thread counts its portion
    LongType threadCount = 0;

    for (LongType i = blockIdx.x * blockDim.x + threadIdx.x; i < xLen; i += blockDim.x * gridDim.x) {
        LongType coords[SD_MAX_RANK];
        LongType xOffset;

        INDEX2COORDS(i, xRank, xShape, coords);
        COORDS2INDEX(xRank, xStride, coords, xOffset);

        // Check if value is non-zero (true)
        if (x[xOffset] != static_cast<T>(0)) {
            threadCount++;
        }
    }

    // Reduce within block using atomics
    if (threadCount > 0) {
        sd::math::atomics::sd_atomicAdd(&localCount, threadCount);
    }
    __syncthreads();

    // First thread adds block count to global count
    if (threadIdx.x == 0 && localCount > 0) {
        sd::math::atomics::sd_atomicAdd(count, localCount);
    }
}

//////////////////////////////////////////////////////////////////////////
// Kernel to compute boolean flags: flags[i] = 1 if condition[i] != 0, else 0
template <typename T>
SD_KERNEL static void computeFlagsKernel(const void* vx, const LongType* xShapeInfo,
                                          LongType* flags) {
    const auto x = reinterpret_cast<const T*>(vx);

    __shared__ LongType xLen;
    __shared__ LongType xRank;
    __shared__ const LongType* xShape;
    __shared__ const LongType* xStride;

    if (threadIdx.x == 0) {
        xLen = shape::length(xShapeInfo);
        xRank = shape::rank(xShapeInfo);
        xShape = shape::shapeOf(xShapeInfo);
        xStride = shape::stride(xShapeInfo);
    }
    __syncthreads();

    for (LongType i = blockIdx.x * blockDim.x + threadIdx.x; i < xLen; i += blockDim.x * gridDim.x) {
        LongType coords[SD_MAX_RANK];
        LongType xOffset;

        INDEX2COORDS(i, xRank, xShape, coords);
        COORDS2INDEX(xRank, xStride, coords, xOffset);

        flags[i] = (x[xOffset] != static_cast<T>(0)) ? 1 : 0;
    }
}

//////////////////////////////////////////////////////////////////////////
// Kernel to write coordinates using pre-computed positions from prefix sum.
// positions[i] gives the output row index for element i (if it's a true element).
template <typename T, typename Z>
SD_KERNEL static void writeOrderedCoordinatesKernel(const void* vx, const LongType* xShapeInfo,
                                                     void* vz, const LongType* zShapeInfo,
                                                     const LongType* positions,
                                                     const LongType* flags) {
    const auto x = reinterpret_cast<const T*>(vx);
    auto z = reinterpret_cast<Z*>(vz);

    __shared__ LongType xLen;
    __shared__ LongType xRank;
    __shared__ LongType zRank;
    __shared__ LongType zNumRows;
    __shared__ const LongType* xShape;
    __shared__ const LongType* xStride;
    __shared__ const LongType* zShape;
    __shared__ const LongType* zStride;

    if (threadIdx.x == 0) {
        xLen = shape::length(xShapeInfo);
        xRank = shape::rank(xShapeInfo);
        zRank = shape::rank(zShapeInfo);
        xShape = shape::shapeOf(xShapeInfo);
        xStride = shape::stride(xShapeInfo);
        zShape = shape::shapeOf(zShapeInfo);
        zStride = shape::stride(zShapeInfo);
        zNumRows = zShape[0];
    }
    __syncthreads();

    for (LongType i = blockIdx.x * blockDim.x + threadIdx.x; i < xLen; i += blockDim.x * gridDim.x) {
        // Only process true elements
        if (flags[i] == 0) continue;

        LongType rowIdx = positions[i];
        if (rowIdx >= zNumRows) continue;

        LongType coords[SD_MAX_RANK];
        INDEX2COORDS(i, xRank, xShape, coords);

        // Write coordinates to output row
        for (LongType d = 0; d < xRank; d++) {
            LongType zCoords[2] = {rowIdx, d};
            LongType zOffset;
            COORDS2INDEX(zRank, zStride, zCoords, zOffset);
            z[zOffset] = static_cast<Z>(coords[d]);
        }
    }
}

//////////////////////////////////////////////////////////////////////////
// Kernel for 3-input where: condition ? x : y (element-wise)
template <typename T, typename X>
SD_KERNEL static void whereElementWiseKernel(const void* vcond, const LongType* condShapeInfo,
                                              const void* vx, const LongType* xShapeInfo,
                                              const void* vy, const LongType* yShapeInfo,
                                              void* vz, const LongType* zShapeInfo) {
    const auto cond = reinterpret_cast<const T*>(vcond);
    const auto x = reinterpret_cast<const X*>(vx);
    const auto y = reinterpret_cast<const X*>(vy);
    auto z = reinterpret_cast<X*>(vz);

    __shared__ LongType zLen;
    __shared__ LongType condRank, xRank, yRank, zRank;
    __shared__ const LongType* condShape;
    __shared__ const LongType* condStride;
    __shared__ const LongType* xShape;
    __shared__ const LongType* xStride;
    __shared__ const LongType* yShape;
    __shared__ const LongType* yStride;
    __shared__ const LongType* zShape;
    __shared__ const LongType* zStride;
    __shared__ bool sameShapes;

    if (threadIdx.x == 0) {
        zLen = shape::length(zShapeInfo);
        condRank = shape::rank(condShapeInfo);
        xRank = shape::rank(xShapeInfo);
        yRank = shape::rank(yShapeInfo);
        zRank = shape::rank(zShapeInfo);

        condShape = shape::shapeOf(condShapeInfo);
        condStride = shape::stride(condShapeInfo);
        xShape = shape::shapeOf(xShapeInfo);
        xStride = shape::stride(xShapeInfo);
        yShape = shape::shapeOf(yShapeInfo);
        yStride = shape::stride(yShapeInfo);
        zShape = shape::shapeOf(zShapeInfo);
        zStride = shape::stride(zShapeInfo);

        // Check if all shapes are the same for fast path
        sameShapes = shape::equalsSoft(condShapeInfo, xShapeInfo) &&
                     shape::equalsSoft(xShapeInfo, yShapeInfo) &&
                     shape::equalsSoft(xShapeInfo, zShapeInfo);
    }
    __syncthreads();

    for (LongType i = blockIdx.x * blockDim.x + threadIdx.x; i < zLen; i += blockDim.x * gridDim.x) {
        LongType zCoords[SD_MAX_RANK];
        LongType zOffset, condOffset, xOffset, yOffset;

        INDEX2COORDS(i, zRank, zShape, zCoords);
        COORDS2INDEX(zRank, zStride, zCoords, zOffset);

        if (sameShapes) {
            // Fast path: all arrays have same shape
            COORDS2INDEX(condRank, condStride, zCoords, condOffset);
            COORDS2INDEX(xRank, xStride, zCoords, xOffset);
            COORDS2INDEX(yRank, yStride, zCoords, yOffset);
        } else {
            // Slow path: handle broadcasting
            LongType condCoords[SD_MAX_RANK];
            LongType xCoords[SD_MAX_RANK];
            LongType yCoords[SD_MAX_RANK];

            // Map z coordinates to each input with broadcasting
            for (LongType d = 0; d < zRank; d++) {
                LongType condDim = d - (zRank - condRank);
                LongType xDim = d - (zRank - xRank);
                LongType yDim = d - (zRank - yRank);

                if (condDim >= 0 && condDim < condRank) {
                    condCoords[condDim] = (condShape[condDim] == 1) ? 0 : zCoords[d];
                }
                if (xDim >= 0 && xDim < xRank) {
                    xCoords[xDim] = (xShape[xDim] == 1) ? 0 : zCoords[d];
                }
                if (yDim >= 0 && yDim < yRank) {
                    yCoords[yDim] = (yShape[yDim] == 1) ? 0 : zCoords[d];
                }
            }

            COORDS2INDEX(condRank, condStride, condCoords, condOffset);
            COORDS2INDEX(xRank, xStride, xCoords, xOffset);
            COORDS2INDEX(yRank, yStride, yCoords, yOffset);
        }

        // Select x or y based on condition
        z[zOffset] = (cond[condOffset] != static_cast<T>(0)) ? x[xOffset] : y[yOffset];
    }
}

//////////////////////////////////////////////////////////////////////////
// Kernel for TAD-based where (condition is 1D, selecting entire TADs)
template <typename T, typename X>
SD_KERNEL static void whereTadKernel(const void* vcond, const LongType* condShapeInfo,
                                      const void* vx, const LongType* xTadShapeInfo, const LongType* xTadOffsets,
                                      const void* vy, const LongType* yTadShapeInfo, const LongType* yTadOffsets,
                                      void* vz, const LongType* zTadShapeInfo, const LongType* zTadOffsets,
                                      LongType numTads, LongType tadLen) {
    const auto cond = reinterpret_cast<const T*>(vcond);
    const auto x = reinterpret_cast<const X*>(vx);
    const auto y = reinterpret_cast<const X*>(vy);
    auto z = reinterpret_cast<X*>(vz);

    __shared__ LongType condRank;
    __shared__ const LongType* condStride;
    __shared__ LongType xTadRank, zTadRank;
    __shared__ const LongType* xTadShape;
    __shared__ const LongType* xTadStride;
    __shared__ const LongType* zTadShape;
    __shared__ const LongType* zTadStride;

    if (threadIdx.x == 0) {
        condRank = shape::rank(condShapeInfo);
        condStride = shape::stride(condShapeInfo);
        xTadRank = shape::rank(xTadShapeInfo);
        xTadShape = shape::shapeOf(xTadShapeInfo);
        xTadStride = shape::stride(xTadShapeInfo);
        zTadRank = shape::rank(zTadShapeInfo);
        zTadShape = shape::shapeOf(zTadShapeInfo);
        zTadStride = shape::stride(zTadShapeInfo);
    }
    __syncthreads();

    for (LongType tadIdx = blockIdx.x; tadIdx < numTads; tadIdx += gridDim.x) {
        // Get condition value for this TAD
        LongType condOffset = tadIdx * condStride[0];
        bool useTadX = (cond[condOffset] != static_cast<T>(0));

        const X* srcTad = useTadX ? (x + xTadOffsets[tadIdx]) : (y + yTadOffsets[tadIdx]);
        X* dstTad = z + zTadOffsets[tadIdx];

        // Copy TAD elements
        for (LongType e = threadIdx.x; e < tadLen; e += blockDim.x) {
            LongType coords[SD_MAX_RANK];
            LongType srcOffset, dstOffset;

            INDEX2COORDS(e, xTadRank, xTadShape, coords);
            COORDS2INDEX(xTadRank, xTadStride, coords, srcOffset);
            COORDS2INDEX(zTadRank, zTadStride, coords, dstOffset);

            dstTad[dstOffset] = srcTad[srcOffset];
        }
    }
}

//////////////////////////////////////////////////////////////////////////
// Host launcher for counting true elements
template <typename T>
static void countTrueElementsLauncher(const cudaStream_t* stream, const void* vx,
                                       const LongType* xShapeInfo, LongType* count,
                                       LongType length) {
    dim3 whereDims = getLaunchDims("where");
    countTrueElementsKernel<T><<<whereDims.y, whereDims.x, whereDims.z, *stream>>>(
        vx, xShapeInfo, count);
    DebugHelper::checkErrorCode(const_cast<cudaStream_t*>(stream), "countTrueElementsKernel failed");
}

//////////////////////////////////////////////////////////////////////////
// Host launcher for computing flags
template <typename T>
static void computeFlagsLauncher(const cudaStream_t* stream, const void* vx,
                                  const LongType* xShapeInfo, LongType* flags,
                                  LongType length) {
    dim3 whereDims = getLaunchDims("where");
    computeFlagsKernel<T><<<whereDims.y, whereDims.x, whereDims.z, *stream>>>(
        vx, xShapeInfo, flags);
    DebugHelper::checkErrorCode(const_cast<cudaStream_t*>(stream), "computeFlagsKernel failed");
}

//////////////////////////////////////////////////////////////////////////
// Host launcher for writing ordered coordinates
template <typename T, typename Z>
static void writeOrderedCoordinatesLauncher(const cudaStream_t* stream, const void* vx,
                                             const LongType* xShapeInfo, void* vz,
                                             const LongType* zShapeInfo,
                                             const LongType* positions,
                                             const LongType* flags,
                                             LongType length) {
    dim3 whereDims = getLaunchDims("where");
    writeOrderedCoordinatesKernel<T, Z><<<whereDims.y, whereDims.x, whereDims.z, *stream>>>(
        vx, xShapeInfo, vz, zShapeInfo, positions, flags);
    DebugHelper::checkErrorCode(const_cast<cudaStream_t*>(stream), "writeOrderedCoordinatesKernel failed");
}

//////////////////////////////////////////////////////////////////////////
// Host launcher for element-wise where
template <typename T, typename X>
static void whereElementWiseLauncher(const cudaStream_t* stream,
                                      const void* vcond, const LongType* condShapeInfo,
                                      const void* vx, const LongType* xShapeInfo,
                                      const void* vy, const LongType* yShapeInfo,
                                      void* vz, const LongType* zShapeInfo,
                                      LongType length) {
    dim3 whereDims = getLaunchDims("where");
    whereElementWiseKernel<T, X><<<whereDims.y, whereDims.x, whereDims.z, *stream>>>(
        vcond, condShapeInfo, vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo);
    DebugHelper::checkErrorCode(const_cast<cudaStream_t*>(stream), "whereElementWiseKernel failed");
}

//////////////////////////////////////////////////////////////////////////
// Host launcher for TAD-based where
template <typename T, typename X>
static void whereTadLauncher(const cudaStream_t* stream,
                              const void* vcond, const LongType* condShapeInfo,
                              const void* vx, const LongType* xTadShapeInfo, const LongType* xTadOffsets,
                              const void* vy, const LongType* yTadShapeInfo, const LongType* yTadOffsets,
                              void* vz, const LongType* zTadShapeInfo, const LongType* zTadOffsets,
                              LongType numTads, LongType tadLen) {
    dim3 whereDims = getLaunchDims("where");
    whereTadKernel<T, X><<<whereDims.y, whereDims.x, whereDims.z, *stream>>>(
        vcond, condShapeInfo, vx, xTadShapeInfo, xTadOffsets,
        vy, yTadShapeInfo, yTadOffsets, vz, zTadShapeInfo, zTadOffsets,
        numTads, tadLen);
    DebugHelper::checkErrorCode(const_cast<cudaStream_t*>(stream), "whereTadKernel failed");
}

//////////////////////////////////////////////////////////////////////////
// Main helper function - single input case (return coordinates of true elements)
// Uses prefix-sum approach to guarantee deterministic output ordering
// matching numpy.nonzero behavior (C-contiguous iteration order).
void _where(LaunchContext* context, NDArray& condition, NDArray& output, memory::Workspace* workspace) {
    // Early return if output is empty
    if (output.isEmpty() || output.lengthOf() == 0) {
        return;
    }

    // Early return if condition is empty
    if (condition.isEmpty() || condition.lengthOf() == 0) {
        return;
    }

    PointersManager manager(context, "where");
    auto stream = context->getCudaStream();
    LongType xLen = condition.lengthOf();

    // Allocate temporary buffers for flags and positions via CudaMemoryPool
    LongType* dBuffer = nullptr;
    ALLOCATE_SPECIAL(dBuffer, workspace, xLen, LongType);

    NDArray::prepareSpecialUse({&output}, {&condition});

    auto xType = condition.dataType();
    auto zType = output.dataType();

    // Step 1: Compute flags on GPU (flags[i] = 1 if condition[i] != 0, else 0)
    BUILD_SINGLE_SELECTOR(xType, computeFlagsLauncher,
                          (stream, condition.specialBuffer(), condition.specialShapeInfo(), dBuffer, xLen),
                          SD_COMMON_TYPES);

    // Step 2: Copy flags to host and compute exclusive prefix sum
    std::vector<LongType> hBuffer(xLen);
    cudaMemcpyAsync(hBuffer.data(), dBuffer, xLen * sizeof(LongType), cudaMemcpyDeviceToHost, *stream);
    cudaStreamSynchronize(*stream);

    // Exclusive prefix sum: positions[i] = number of true elements before index i
    LongType sum = 0;
    for (LongType i = 0; i < xLen; i++) {
        LongType flag = hBuffer[i];
        hBuffer[i] = sum;  // position for element i
        sum += flag;
    }

    // Allocate positions buffer and copy prefix sum to device
    LongType* dPositions = nullptr;
    ALLOCATE_SPECIAL(dPositions, workspace, xLen, LongType);
    cudaMemcpyAsync(dPositions, hBuffer.data(), xLen * sizeof(LongType), cudaMemcpyHostToDevice, *stream);

    // Recompute flags on GPU (since we overwrote hBuffer with positions)
    BUILD_SINGLE_SELECTOR(xType, computeFlagsLauncher,
                          (stream, condition.specialBuffer(), condition.specialShapeInfo(), dBuffer, xLen),
                          SD_COMMON_TYPES);

    // Step 3: Write coordinates at deterministic positions
    BUILD_DOUBLE_SELECTOR(xType, zType, writeOrderedCoordinatesLauncher,
                          (stream, condition.specialBuffer(), condition.specialShapeInfo(),
                           output.specialBuffer(), output.specialShapeInfo(),
                           dPositions, dBuffer, xLen),
                          SD_COMMON_TYPES, SD_INDEXING_TYPES);

    NDArray::registerSpecialUse({&output}, {&condition});

    RELEASE_SPECIAL(dBuffer, workspace);
    RELEASE_SPECIAL(dPositions, workspace);
    manager.synchronize();
}

//////////////////////////////////////////////////////////////////////////
// Helper for 3-input where (condition, x, y)
void _whereElementWise(LaunchContext* context, NDArray& condition, NDArray& x, NDArray& y,
                        NDArray& output) {
    // Early return if output is empty
    if (output.isEmpty() || output.lengthOf() == 0) {
        return;
    }

    NDArray::prepareSpecialUse({&output}, {&condition, &x, &y});

    auto condType = condition.dataType();
    auto xType = x.dataType();

    BUILD_DOUBLE_SELECTOR(condType, xType, whereElementWiseLauncher,
                          (context->getCudaStream(),
                           condition.specialBuffer(), condition.specialShapeInfo(),
                           x.specialBuffer(), x.specialShapeInfo(),
                           y.specialBuffer(), y.specialShapeInfo(),
                           output.specialBuffer(), output.specialShapeInfo(),
                           output.lengthOf()),
                          SD_COMMON_TYPES, SD_COMMON_TYPES);

    NDArray::registerSpecialUse({&output}, {&condition, &x, &y});
}

//////////////////////////////////////////////////////////////////////////
// Helper for TAD-based where
void _whereTad(LaunchContext* context, NDArray& condition, NDArray& x, NDArray& y,
               NDArray& output, const std::vector<LongType>& axis) {
    // Early return if output is empty
    if (output.isEmpty() || output.lengthOf() == 0) {
        return;
    }

    PointersManager manager(context, "whereTad");

    // Get TAD info for x, y, z along the given axis
    auto axisMutable = axis;
    auto packX = ConstantTadHelper::getInstance().tadForDimensions(x.shapeInfo(), &axisMutable);
    auto packY = ConstantTadHelper::getInstance().tadForDimensions(y.shapeInfo(), &axisMutable);
    auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output.shapeInfo(), &axisMutable);

    LongType numTads = packX->numberOfTads();
    LongType tadLen = shape::length(packX->primaryShapeInfo());

    NDArray::prepareSpecialUse({&output}, {&condition, &x, &y});

    auto condType = condition.dataType();
    auto xType = x.dataType();

    // Launch TAD kernel using launcher
    BUILD_DOUBLE_SELECTOR(condType, xType, whereTadLauncher,
                          (context->getCudaStream(),
                           condition.specialBuffer(), condition.specialShapeInfo(),
                           x.specialBuffer(), packX->specialShapeInfo(), packX->specialOffsets(),
                           y.specialBuffer(), packY->specialShapeInfo(), packY->specialOffsets(),
                           output.specialBuffer(), packZ->specialShapeInfo(), packZ->specialOffsets(),
                           numTads, tadLen),
                          SD_COMMON_TYPES, SD_COMMON_TYPES);

    NDArray::registerSpecialUse({&output}, {&condition, &x, &y});

    manager.synchronize();
}

//////////////////////////////////////////////////////////////////////////
// Count true elements on GPU (for shape function)
LongType countTrue(LaunchContext* context, NDArray& condition) {
    if (condition.isEmpty() || condition.lengthOf() == 0) {
        return 0;
    }

    PointersManager manager(context, "countTrue");

    // Allocate and initialize counter on device
    NDArray countArr(INT64, context, true);  // scalar
    countArr.assign(0);

    NDArray::prepareSpecialUse({&countArr}, {&condition});

    auto xType = condition.dataType();

    BUILD_SINGLE_SELECTOR(xType, countTrueElementsLauncher,
                          (context->getCudaStream(), condition.specialBuffer(), condition.specialShapeInfo(),
                           reinterpret_cast<LongType*>(countArr.specialBuffer()), condition.lengthOf()),
                          SD_COMMON_TYPES);

    NDArray::registerSpecialUse({&countArr}, {&condition});

    manager.synchronize();

    return countArr.e<LongType>(0);
}

// Template instantiations
BUILD_SINGLE_TEMPLATE(void computeFlagsLauncher,
                      (const cudaStream_t* stream, const void* vx, const LongType* xShapeInfo,
                       LongType* flags, LongType length),
                      SD_COMMON_TYPES);

BUILD_DOUBLE_TEMPLATE(void writeOrderedCoordinatesLauncher,
                      (const cudaStream_t* stream, const void* vx, const LongType* xShapeInfo,
                       void* vz, const LongType* zShapeInfo, const LongType* positions,
                       const LongType* flags, LongType length),
                      SD_COMMON_TYPES, SD_INDEXING_TYPES);

BUILD_DOUBLE_TEMPLATE(void whereElementWiseLauncher,
                      (const cudaStream_t* stream, const void* vcond, const LongType* condShapeInfo,
                       const void* vx, const LongType* xShapeInfo, const void* vy, const LongType* yShapeInfo,
                       void* vz, const LongType* zShapeInfo, LongType length),
                      SD_COMMON_TYPES, SD_COMMON_TYPES);

BUILD_SINGLE_TEMPLATE(void countTrueElementsLauncher,
                      (const cudaStream_t* stream, const void* vx, const LongType* xShapeInfo,
                       LongType* count, LongType length),
                      SD_COMMON_TYPES);

BUILD_DOUBLE_TEMPLATE(void whereTadLauncher,
                      (const cudaStream_t* stream,
                       const void* vcond, const LongType* condShapeInfo,
                       const void* vx, const LongType* xTadShapeInfo, const LongType* xTadOffsets,
                       const void* vy, const LongType* yTadShapeInfo, const LongType* yTadOffsets,
                       void* vz, const LongType* zTadShapeInfo, const LongType* zTadOffsets,
                       LongType numTads, LongType tadLen),
                      SD_COMMON_TYPES, SD_COMMON_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
