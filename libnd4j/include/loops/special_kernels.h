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
//  @author raver119@gmail.com
//

#ifndef LIBND4J_SPECIAL_KERNELS_H
#define LIBND4J_SPECIAL_KERNELS_H
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <helpers/DebugHelper.h>
#include <helpers/TAD.h>
#include <helpers/shape.h>

namespace sd {

template <typename T>
SD_HOST void fillIsMaxGeneric(dim3 &launchDims, cudaStream_t *stream, void *dx, const sd::LongType *xShapeInfo,
                              sd::LongType length, long idx);

template <typename T>
SD_HOST void fillDimensionalIsMaxGeneric(dim3 &launchDims, cudaStream_t *stream, const void *dX, void *dZ,
                                         const sd::LongType *zShapeInfo, const sd::LongType *tadOnlyShapeInfo,
                                         LongType *dimension, LongType dimensionLength, const sd::LongType *tadOffsets);

template <typename T>
SD_HOST void convertToHalfGeneric(dim3 &launchDims, cudaStream_t *stream, void *dx, sd::LongType n, half *dz);

template <typename T>
SD_HOST void tearKernelGeneric(dim3 &launchDims, cudaStream_t *stream, void *vx, sd::LongType const *xShapeInfo,
                               sd::Pointer *targets, sd::LongType const *zShapeInfo, sd::LongType const *tadShapeInfo,
                               sd::LongType const *tadOffsets);

template <typename T>
SD_HOST void shuffleKernelGeneric(dim3 &launchDims, cudaStream_t *stream, void **vdX, sd::LongType **xShapeInfo,
                                  void **vdZ, int N, int *shuffleMap, sd::LongType **tadOnlyShapeInfo,
                                  sd::LongType **tadOffsets);

template <typename T>
SD_HOST void convertHalfsToGeneric(dim3 &launchDims, cudaStream_t *stream, half *dx, sd::LongType n, void *dz);

template <typename T>
SD_HOST void concatKernelVStackGeneric(dim3 &launchDims, cudaStream_t *stream, int numArrays, sd::Pointer *data,
                                       sd::Pointer *inputShapeInfos, void *vz, sd::LongType const *zShapeInfo);

template <typename T>
SD_HOST void concatKernelScalarGeneric(dim3 &launchDims, cudaStream_t *stream, int numArrays, sd::Pointer *data,
                                       void *vresult);

template <typename T>
SD_HOST void concatKernelHStackGeneric(dim3 &launchDims, cudaStream_t *stream, int numArrays, sd::Pointer *data,
                                       sd::Pointer *inputShapeInfos, void *vresult,
                                       sd::LongType const *resultShapeInfo);

template <typename T>
SD_HOST void concatKernelGeneric(dim3 &launchDims, cudaStream_t *stream, int numArrays, sd::Pointer *data,
                                 sd::Pointer *inputShapeInfos, void *vresult, sd::LongType const *resultShapeInfo,
                                 sd::Pointer *tadPointers, sd::Pointer *offsetPointers, sd::LongType const *zTadShape,
                                 sd::LongType const *zOffsets);

template <typename T>
SD_HOST void pullRowsKernelGeneric(dim3 &launchDims, cudaStream_t *stream, void *vx, void *vz, sd::LongType n,
                                   sd::LongType *indexes, sd::LongType const *tadShapeInfo,
                                   sd::LongType const *tadOffsets, sd::LongType const *zTadShapeInfo,
                                   sd::LongType const *zTadOffsets);

template <typename T>
SD_HOST void averagingKernelGeneric(dim3 &launchDims, cudaStream_t *stream, void **vdx, void *vdz, int n,
                                    sd::LongType length, bool propagate);

template <typename T>
SD_HOST void accumulateKernelGeneric(dim3 &launchDims, cudaStream_t *stream, void **vx, void *vz, int n,
                                     const sd::LongType length);

template <typename T>
SD_HOST void flattenKernelGeneric(dim3 &launchDims, cudaStream_t *stream, sd::Pointer *extraPointers, int dOffset,
                                  char order, void *vz, sd::LongType *zShapeInfo, void *vy, sd::LongType *yShapeInfo);

template <typename T>
SD_HOST void tileKernelH(void const *inputBuffer, sd::LongType const *inputShape, void *outputBuffer,
                         sd::LongType const *outputShape, sd::LongType resultLength, cudaStream_t *stream);
template <typename X, typename Y>
SD_HOST void tileKernelHH(void const *inputBuffer, sd::LongType const *inputShape, void *outputBuffer,
                          sd::LongType const *outputShape, sd::LongType resultLength, sd::LongType ews,
                          cudaStream_t *stream);

class NDArray;
template <typename T>
SD_HOST void setDiagonalValueUpper(void *buffer, sd::LongType const *shape, NDArray const &value, int diagonal,
                                   sd::LongType rows, sd::LongType cols, cudaStream_t &stream);

template <typename T>
SD_HOST void setDiagonalValueLower(void *buffer, sd::LongType const *shape, NDArray const &value, int diagonal,
                                   sd::LongType rows, sd::LongType cols, cudaStream_t &stream);

template <typename T>
SD_HOST void templatedSwapUnsafe(void *theFirstBuffer, sd::LongType const *theFirstShape, void *theSecondBuffer,
                                 sd::LongType const *theSecondShape, cudaStream_t *theStream);

}  // namespace sd

#endif
