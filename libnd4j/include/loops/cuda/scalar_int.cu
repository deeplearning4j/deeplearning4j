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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 08.11.2018
// @author raver119@gmail.com
//
#include <system/op_boilerplate.h>
#include <types/types.h>

#include "../legacy_ops.h"
#include "../scalar_int.h"

using namespace simdOps;

////////////////////////////////////////////////////////////////////////
template <typename X, typename OpType>
SD_KERNEL void scalarAlongDimension(void const* x, sd::LongType const* xShapeInfo, void* extraParams, void* z,
                                    sd::LongType const* zShapeInfo, void const* scalars, sd::LongType* dimension,
                                    long long int dimensionLength, sd::LongType const* tadShapeInfo,
                                    sd::LongType const* tadOffsets, sd::LongType const* tadShapeInfoZ,
                                    sd::LongType const* tadOffsetsZ) {
  functions::scalar::ScalarIntTransform<X>::template transformCuda<OpType>(
      x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets,
      tadShapeInfoZ, tadOffsetsZ);
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename OpType>
SD_KERNEL void scalarSimpleShaped(void const* x, void const* y, sd::LongType const* xShapeInfo, void* params, void* z,
                                  sd::LongType const* zShapeInfo, sd::LongType * allocationBuffer) {
  functions::scalar::ScalarIntTransform<X>::template transformCuda<OpType>(y, x, xShapeInfo, params, z, zShapeInfo,
                                                                           allocationBuffer);
}

// *********************************************************************//
// *********************************************************************//
namespace functions {
namespace scalar {

////////////////////////////////////////////////////////////////////////
template <typename X>
template <typename OpType>
SD_DEVICE void ScalarIntTransform<X>::transformCuda(void const* vscalar, void const* vy, sd::LongType const* yShapeInfo,
                                                    void* vparams, void* vz, sd::LongType const* zShapeInfo,
                                                    sd::LongType* allocationBuffer) {
  auto scalar = reinterpret_cast<X const*>(vscalar)[0];
  auto y = reinterpret_cast<X const*>(vy);
  auto params = reinterpret_cast<X*>(vparams);
  auto z = reinterpret_cast<X*>(vz);

  auto yRank = shape::rank(yShapeInfo);
  auto yEWS = shape::elementWiseStride(yShapeInfo);
  auto yShape = shape::shapeOf(yShapeInfo);
  auto yStride = shape::stride(yShapeInfo);

  auto zRank = shape::rank(zShapeInfo);
  auto zEWS = shape::elementWiseStride(zShapeInfo);
  auto zShape = shape::shapeOf(zShapeInfo);
  auto zStride = shape::stride(zShapeInfo);

  int totalThreads = gridDim.x * blockDim.x;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ int len;
  if (threadIdx.x == 0) len = shape::length(yShapeInfo);
  __syncthreads();

  if (yEWS >= 1 && zEWS >= 1 && shape::order(yShapeInfo) == shape::order(zShapeInfo)) {
    transformCuda<OpType>(len, vscalar, vy, yEWS, vparams, vz, zEWS, allocationBuffer);
  } else {
    for (sd::LongType i = tid; i < len; i += totalThreads)
      z[shape::getIndexOffset(i, zShapeInfo)] = OpType::op(y[shape::getIndexOffset(i, yShapeInfo)], scalar, params);
  }
}

////////////////////////////////////////////////////////////////////////
template <typename X>
template <typename OpType>
SD_DEVICE void ScalarIntTransform<X>::transformCuda(sd::LongType len, void const* vx, void const* vy, sd::LongType yEWS,
                                                    void* vparams, void* vz, sd::LongType zEWS,
                                                    sd::LongType* allocationBuffer) {
  auto x = reinterpret_cast<X const*>(vx)[0];
  auto y = reinterpret_cast<X const*>(vy);
  auto z = reinterpret_cast<X*>(vz);
  auto params = reinterpret_cast<X*>(vparams);

  int totalThreads = gridDim.x * blockDim.x;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  sd::LongType i = tid;
  if (yEWS == 1 && zEWS == 1) {
    for (; i < len; i += totalThreads) z[i] = OpType::op(y[i], x, params);
  } else {
    for (; i < len; i += totalThreads) z[i * zEWS] = OpType::op(y[i * yEWS], x, params);
  }
}

////////////////////////////////////////////////////////////////////////
template <typename X>
template <typename OpType>
SD_DEVICE void ScalarIntTransform<X>::transformCuda(void const* vx, sd::LongType const* xShapeInfo, void* vextraParams,
                                                    void* vz, sd::LongType const* zShapeInfo, void const* vscalars,
                                                    sd::LongType* dimension, sd::LongType dimensionLength,
                                                    sd::LongType const* tadShapeInfo, sd::LongType const* tadOffsets,
                                                    sd::LongType const* tadShapeInfoZ,
                                                    sd::LongType const* tadOffsetsZ) {
  auto x = reinterpret_cast<X const*>(vx);
  auto scalars = reinterpret_cast<X const*>(vscalars);
  auto z = reinterpret_cast<X*>(vz);
  auto extraParams = reinterpret_cast<X*>(vextraParams);

  if (tadShapeInfoZ == nullptr) {
    tadShapeInfoZ = tadShapeInfo;
    tadOffsetsZ = tadOffsets;
  }

  // tad preparation
  auto tadEws = shape::elementWiseStride(tadShapeInfo);
  auto zEws = shape::elementWiseStride(tadShapeInfoZ);
  auto tadLength = shape::length(tadShapeInfo);
  auto numTads = shape::length(xShapeInfo) / tadLength;

  if (tadEws > 0 && zEws > 0 && shape::order(tadShapeInfo) == shape::order(zShapeInfo)) {
    // main loop, rolling over tads
    for (int r = blockIdx.x; r < numTads; r += gridDim.x) {
      X* oZ = z + tadOffsetsZ[r];
      auto oX = x + tadOffsets[r];

      auto s = scalars[r];

      for (int f = threadIdx.x; f < tadLength; f += blockDim.x)
        oZ[f * zEws] = OpType::op(oX[f * tadEws], s, extraParams);
    }
  } else {
    // main loop, rolling over tads
    for (int r = blockIdx.x; r < numTads; r += gridDim.x) {
      X* oZ = z + tadOffsetsZ[r];
      auto oX = x + tadOffsets[r];

      auto s = scalars[r];

      for (int f = threadIdx.x; f < tadLength; f += blockDim.x)
        oZ[shape::getIndexOffset(f, tadShapeInfoZ)] =
            OpType::op(oX[shape::getIndexOffset(f, tadShapeInfo)], s, extraParams);
    }
  }
}

////////////////////////////////////////////////////////////////////////
template <typename X>
template <typename OpType>
SD_HOST void ScalarIntTransform<X>::intermediateAlongDimension(
    dim3& launchDims, cudaStream_t* stream, void const* x, sd::LongType const* xShapeInfo, void* z,
    sd::LongType const* zShapeInfo, void const* scalars, void* extraParams, sd::LongType * dimension,
    sd::LongType dimensionLength,
    sd::LongType const* tadShapeInfo, sd::LongType const* tadOffsets, sd::LongType const* tadShapeInfoZ,
    sd::LongType const* tadOffsetsZ) {
  scalarAlongDimension<X, OpType><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
      x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets,
      tadShapeInfoZ, tadOffsetsZ);
  sd::DebugHelper::checkErrorCode(stream, "ScalarIntTransform intermediateAlongDimension(...) failed");

}

////////////////////////////////////////////////////////////////////////
template <typename X>
template <typename OpType>
void SD_HOST ScalarIntTransform<X>::intermediateShaped(dim3& launchDims, cudaStream_t* stream, void const* vx,
                                                       sd::LongType const* xShapeInfo, void* vz,
                                                       sd::LongType const* zShapeInfo, void const* vscalar,
                                                       void* vextraParams, sd::LongType * allocPointer) {
  scalarSimpleShaped<X, OpType><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
      vx, vscalar, xShapeInfo, vextraParams, vz, zShapeInfo, allocPointer);
}

////////////////////////////////////////////////////////////////////////
template <typename X>
void ScalarIntTransform<X>::executeCudaShaped(dim3& launchDims, cudaStream_t* stream, int opNum, void const* vx,
                                              sd::LongType const* xShapeInfo, void* vz, sd::LongType const* zShapeInfo,
                                              void const* vscalar, void* vextraParams) {
  if (sd::Environment::getInstance().isDebugAndVerbose()) printf("H14 opNum:[%i]\n", opNum);

  DISPATCH_BY_OPNUM_T(intermediateShaped,
                      PARAMS(launchDims, stream, vx, xShapeInfo, vz, zShapeInfo, vscalar, vextraParams, nullptr),
                      SCALAR_INT_OPS);
}

////////////////////////////////////////////////////////////////////////
template <typename X>
void ScalarIntTransform<X>::executeCudaAlongDimension(dim3& launchDims, cudaStream_t* stream, int opNum, void const* vx,
                                                      sd::LongType const* xShapeInfo, void* vz,
                                                      sd::LongType const* zShapeInfo, void const* vscalars,
                                                      void* vextraParams, sd::LongType* dimension,
                                                      sd::LongType dimensionLength,
                                                      sd::LongType const* tadShapeInfo, sd::LongType const* tadOffsets,
                                                      sd::LongType const* tadShapeInfoZ,
                                                      sd::LongType const* tadOffsetsZ) {
  DISPATCH_BY_OPNUM_T(intermediateAlongDimension,
                      PARAMS(launchDims, stream, vx, xShapeInfo, vz, zShapeInfo, vscalars, vextraParams, dimension,
                             dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ),
                      SCALAR_INT_OPS);
}

BUILD_SINGLE_TEMPLATE(template class ScalarIntTransform, , SD_INTEGER_TYPES);

}  // namespace scalar
}  // namespace functions
