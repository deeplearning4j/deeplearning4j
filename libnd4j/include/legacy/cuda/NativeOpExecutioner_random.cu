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
#include <array/ConstantDataBuffer.h>
#include <array/DataTypeUtils.h>
#include <array/ShapeDescriptor.h>
#include <exceptions/cuda_exception.h>
#include <exceptions/datatype_exception.h>
#include <execution/cuda/LaunchDims.h>
#include <helpers/DebugHelper.h>
#include <helpers/PointersManager.h>
#include <legacy/NativeOpExecutioner.h>
#include <loops/broadcasting.h>
#include <loops/broadcasting_bool.h>
#include <loops/broadcasting_int.h>
#include <loops/indexreduce.h>
#include <loops/pairwise_bool.h>
#include <loops/pairwise_int.h>
#include <loops/pairwise_transform.h>
#include <loops/random.h>
#include <loops/reduce3.h>
#include <loops/reduce_bool.h>
#include <loops/reduce_float.h>
#include <loops/reduce_long.h>
#include <loops/reduce_same.h>
#include <loops/scalar.h>
#include <loops/scalar_bool.h>
#include <loops/scalar_int.h>
#include <loops/special_kernels.h>
#include <loops/summarystatsreduce.h>
#include <loops/transform_any.h>
#include <loops/transform_bool.h>
#include <loops/transform_float.h>
#include <loops/transform_same.h>
#include <loops/transform_strict.h>
#include <system/op_boilerplate.h>
#include <helpers/ConstantTadHelper.h>
#include <system/selective_rendering.h>

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execRandom(sd::LaunchContext* lc, int opNum, sd::Pointer stateHost, void* hZ,
                                     sd::LongType const* hZShapeInfo, void* dZ, sd::LongType const* dZShapeInfo,
                                     void* extraArguments) {
  auto stream = lc->getCudaStream();
  auto sizeOf = sizeof(sd::graph::RandomGenerator);
  sd::Pointer stateDevice;

  cudaError_t res = cudaMalloc(reinterpret_cast<void**>(&stateDevice), sizeOf);
  checkCudaErrors(cudaStreamSynchronize(*stream));
  checkCudaErrors(cudaMemcpyAsync(stateDevice, stateHost, sizeOf, cudaMemcpyHostToDevice, *stream));

  dim3 launchDims = getLaunchDims("random");
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

  auto rng = reinterpret_cast<sd::graph::RandomGenerator*>(stateHost);

  BUILD_SINGLE_SELECTOR(zType, functions::random::RandomFunction,
                        ::executeCudaSingle(launchDims, stream, opNum, stateDevice, dZ, dZShapeInfo, extraArguments),
                        SD_FLOAT_TYPES);

  res = cudaStreamSynchronize(*stream);
  if (res != 0) {
    std::string errorMessage = "execRandom X failed with error code: " + std::to_string(static_cast<int>(res));
    THROW_EXCEPTION(errorMessage.c_str());
  }
  cudaFree(stateDevice);

  rng->rewindH(shape::length(hZShapeInfo));
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execRandom(sd::LaunchContext* lc, int opNum, sd::Pointer stateHost, void const* hX,
                                     sd::LongType const* hXShapeInfo, void const* dX, sd::LongType const* dXShapeInfo, void* hZ,
                                     sd::LongType const* hZShapeInfo, void* dZ, sd::LongType const* dZShapeInfo,
                                     void* extraArguments) {
  auto stream = lc->getCudaStream();

  auto sizeOf = sizeof(sd::graph::RandomGenerator);
  sd::Pointer stateDevice;

  cudaError_t res = cudaMalloc(reinterpret_cast<void**>(&stateDevice), sizeOf);
  checkCudaErrors(cudaStreamSynchronize(*stream));
  checkCudaErrors(cudaMemcpyAsync(stateDevice, stateHost, sizeOf, cudaMemcpyHostToDevice, *stream));

  auto rng = reinterpret_cast<sd::graph::RandomGenerator*>(stateHost);

  dim3 launchDims = getLaunchDims("random");
  auto xType = sd::ArrayOptions::dataType(hZShapeInfo);

  BUILD_SINGLE_SELECTOR(
      xType, functions::random::RandomFunction,
      ::executeCudaDouble(launchDims, stream, opNum, stateDevice, dX, dXShapeInfo, dZ, dZShapeInfo, extraArguments),
      SD_FLOAT_TYPES);

  res = cudaStreamSynchronize(*stream);
  if (res != 0) {
    std::string errorMessage = "execRandom XY failed with error code: " + std::to_string(static_cast<int>(res));
    THROW_EXCEPTION(errorMessage.c_str());
  }
  cudaFree(stateDevice);

  rng->rewindH(shape::length(hZShapeInfo));
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execRandom(sd::LaunchContext* lc, int opNum, sd::Pointer stateHost, void const* hX,
                                     sd::LongType const* hXShapeInfo, void const* dX, sd::LongType const* dXShapeInfo,
                                     void const* hY, sd::LongType const* hYShapeInfo, void const* dY,
                                     sd::LongType const* dYShapeInfo, void* hZ, sd::LongType const* hZShapeInfo, void* dZ,
                                     sd::LongType const* dZShapeInfo, void* extraArguments) {
  auto stream = lc->getCudaStream();
  auto sizeOf = sizeof(sd::graph::RandomGenerator);
  sd::Pointer stateDevice;

  cudaError_t res = cudaMalloc(reinterpret_cast<void**>(&stateDevice), sizeOf);
  checkCudaErrors(cudaStreamSynchronize(*stream));
  checkCudaErrors(cudaMemcpyAsync(stateDevice, stateHost, sizeOf, cudaMemcpyHostToDevice, *stream));

  auto rng = reinterpret_cast<sd::graph::RandomGenerator*>(stateHost);

  dim3 launchDims = getLaunchDims("random");
  auto xType = sd::ArrayOptions::dataType(hZShapeInfo);

  BUILD_SINGLE_SELECTOR(xType, functions::random::RandomFunction,
                        ::executeCudaTriple(launchDims, stream, opNum, stateDevice, dX, dXShapeInfo, dY, dYShapeInfo,
                                            dZ, dZShapeInfo, extraArguments),
                        SD_FLOAT_TYPES);

  res = cudaStreamSynchronize(*stream);
  if (res != 0) {
    std::string errorMessage = "execRandom XYZ failed with error code: " + std::to_string(static_cast<int>(res));
    THROW_EXCEPTION(errorMessage.c_str());
  }
  cudaFree(stateDevice);

  rng->rewindH(shape::length(hZShapeInfo));
}

////////////////////////////////////////////////////////////////////////
