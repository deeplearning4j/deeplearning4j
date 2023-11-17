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
// @author raver119@gmail.com
//
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>
#include <ops/declarable/helpers/dynamic.h>

#include "execution/cuda/LaunchDims.h"
#include "helpers/DebugHelper.h"

namespace sd {
namespace ops {
namespace helpers {

template <typename X, typename Y>
static SD_KERNEL void dynamicPartitionScalarKernel(const void *vx, const LongType *xShapeInfo, const void *vi,
                                                   const LongType *iShapeInfo, void **vz, LongType **zShapeInfos, const LongType numOutputs) {
  auto x = reinterpret_cast<const X *>(vx);
  auto i = reinterpret_cast<const Y *>(vi);
  auto xLength = shape::length(xShapeInfo);
  auto iLength = shape::length(iShapeInfo);

  extern __shared__ char shmem[];
  __shared__ Y *rawIndices;
  __shared__ Y *trueIndices;

  if (threadIdx.x == 0) {
    rawIndices = reinterpret_cast<Y *>(shmem);
    trueIndices = rawIndices + blockDim.x;
  }
  __syncthreads();

  // we run things in blocks, 1 partition per block of threads
  for (LongType o = blockIdx.x; o < numOutputs; o += gridDim.x) {
    auto z = reinterpret_cast<X *>(vz[o]);

    auto zShapeInfo = zShapeInfos[o];
    auto zLength = shape::length(zShapeInfo);

    // iLimit should be multiple of blockDim.x
    auto iLimit = iLength <= blockDim.x ? blockDim.x : (iLength + (blockDim.x - (iLength % blockDim.x)));
    int cnt = 0;

    for (LongType e = threadIdx.x; e < iLimit; e += blockDim.x) {
      // load set of indices into shared memory
      if (e < iLength) rawIndices[threadIdx.x] = i[shape::getIndexOffset(e, iShapeInfo)];
      __syncthreads();

      // now we need to find out where our actual updates will be mapped
      // TODO: this can be improved obviously, by using prefix-sum like approach
      if (threadIdx.x == 0) {
        for (int f = 0; f < blockDim.x; f++) {
          if (rawIndices[f] == static_cast<Y>(o))
            trueIndices[f] = cnt++;
          else
            trueIndices[f] = -1;
        }
      }
      __syncthreads();

      // doing actual update
      if (e < iLength)
        if (trueIndices[threadIdx.x] >= 0) {
          z[trueIndices[threadIdx.x]] = x[shape::getIndexOffset(e, xShapeInfo)];
        }

      __syncthreads();
    }
  }
}

template <typename X, typename Y>
static SD_KERNEL void dynamicPartitionTadKernel(const void *vx, const LongType *xTadShapeInfo,
                                                const LongType *xTadOffsets, LongType xLength,
                                                const void *vindices, const LongType *iShapeInfo, LongType iLength, void **vz,
                                                LongType **zTadShapeInfos, LongType **zTadOffsets,
                                                LongType numOutputs) {
  auto x = reinterpret_cast<const X *>(vx);
  auto indices = reinterpret_cast<const Y *>(vindices);

  // we run things in blocks, 1 partition per block of threads
  for (int i = blockIdx.x; i < numOutputs; i += gridDim.x) {
    auto z = reinterpret_cast<X *>(vz[i]);

    // each thread has own counter for partitions
    int outCnt = 0;

    for (LongType e = 0; e < iLength; e++) {
      if (indices[shape::getIndexOffset(e, iShapeInfo)] == i) {
        auto dx = x + xTadOffsets[e];
        auto dz = z + zTadOffsets[i][outCnt++];

        for (int f = threadIdx.x; f < xLength; f += blockDim.x) {
          dz[shape::getIndexOffset(f, zTadShapeInfos[i])] = dx[shape::getIndexOffset(f, xTadShapeInfo)];
        }
      }
    }
  }
}

template <typename X, typename Y>
static void _dynamicPartitionFunctor(LaunchContext *context, NDArray const *input, NDArray const *indices,
                                     std::vector<NDArray *> &outputList) {
  std::vector<std::pair<NDArray *, int>> outputs(outputList.size());
  int sourceDimsLen = input->rankOf() - indices->rankOf();

  unsigned int outSize = outputList.size();

  PointersManager pm(context, "dynamicPartition");

  if (sourceDimsLen) {  // non-linear case
    std::vector<LongType> sourceDims(sourceDimsLen);

    for (int i = sourceDimsLen; i > 0; i--) sourceDims[sourceDimsLen - i] = input->rankOf() - i;
    // compute tad array for given dimensions
    auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), &sourceDims);

    std::vector<void *> outBuffers(outSize);
    std::vector<const LongType *> tadShapes(outSize);
    std::vector<const LongType *> tadOffsets(outSize);
    std::vector<LongType> numTads(outSize);
    // fill up dimensions array for before kernel
    for (unsigned int i = 0; i < outSize; i++) {
      outputs[i].first = outputList[i];
      std::vector<LongType> outDims(outputs[i].first->rankOf() - 1);

      int r = outputs[i].first->rankOf();

      for (int k = 1; k < r; k++) outDims[k - 1] = k;

      auto packZ = ConstantTadHelper::getInstance().tadForDimensions(outputList.at(i)->shapeInfo(), &outDims);

      outBuffers[i] = outputList.at(i)->specialBuffer();
      tadShapes[i] = packZ->platformShapeInfo();
      tadOffsets[i] = packZ->platformOffsets();
    }

    // we copy pointers to device
    auto dOutBuffers =
        reinterpret_cast<void **>(pm.replicatePointer(outBuffers.data(), outBuffers.size() * sizeof(void *)));
    auto dOutTadShapes = reinterpret_cast<LongType **>(
        pm.replicatePointer(tadShapes.data(), tadShapes.size() * sizeof(LongType *)));
    auto dOutTadOffsets = reinterpret_cast<LongType **>(
        pm.replicatePointer(tadOffsets.data(), tadOffsets.size() * sizeof(LongType *)));
    // run kernel on device
    dim3 launchDims = getDynamicPartitionDims(256,sizeof(Y));

    dynamicPartitionTadKernel<X, Y><<<launchDims.y,launchDims.x, launchDims.z, *context->getCudaStream()>>>(
        input->specialBuffer(), packX->platformShapeInfo(), packX->platformOffsets(),
        shape::length(packX->primaryShapeInfo()), indices->specialBuffer(), indices->specialShapeInfo(),
        indices->lengthOf(), dOutBuffers, dOutTadShapes, dOutTadOffsets, outSize);
    DebugHelper::checkErrorCode(context->getCudaStream(),"dynamicPartitionTadKernel failed");


  } else {  // linear case
    dim3 launchDims = getDynamicPartitionDims(256,sizeof(Y));
    std::vector<void *> outBuffers;
    std::vector<const LongType *> outShapes;

    for (auto v : outputList) {
      outBuffers.emplace_back(v->specialBuffer());
      outShapes.emplace_back(v->specialShapeInfo());
    }

    auto dOutBuffers =
        reinterpret_cast<void **>(pm.replicatePointer(outBuffers.data(), outBuffers.size() * sizeof(void *)));
    auto dOutShapes = reinterpret_cast<LongType **>(
        pm.replicatePointer(outShapes.data(), outShapes.size() * sizeof(LongType *)));

    dynamicPartitionScalarKernel<X, Y><<<launchDims.y,launchDims.x, launchDims.z, *context->getCudaStream()>>>(
        input->specialBuffer(), input->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(),
        dOutBuffers, dOutShapes, outSize);
    DebugHelper::checkErrorCode(context->getCudaStream(),"dynamicPartitionScalarKernel failed");

  }

  pm.synchronize();
}

template <typename X, typename Y>
static SD_KERNEL void dynamicStitchScalarKernel(void **vx, LongType **xShapeInfos, void **vindices,
                                                LongType **iShapeInfos, int inputSize, void *vz,
                                                const LongType *zShapeInfo, LongType zLength) {
  auto z = reinterpret_cast<X *>(vz);

  for (int e = blockIdx.x; e < inputSize; e += gridDim.x) {
    auto x = reinterpret_cast<X *>(vx[e]);
    auto indices = reinterpret_cast<Y *>(vindices[e]);

    auto xShapeInfo = xShapeInfos[e];
    auto iShapeInfo = iShapeInfos[e];

    auto iLength = shape::length(iShapeInfo);

    for (int i = threadIdx.x; i < iLength; i += blockDim.x) {
      auto idx = indices[shape::getIndexOffset(i, iShapeInfo)];
      if (idx >= 0 && idx < zLength)
        z[shape::getIndexOffset(idx, zShapeInfo)] = x[shape::getIndexOffset(i, xShapeInfo)];
    }
  }
}

template <typename X, typename Y>
static SD_KERNEL void dynamicStitchTadKernel(void **vx, LongType **xTadShapeInfos, LongType **xTadOffsets,
                                             void **vindices, LongType **iShapeInfos, int inputSize, void *vz,
                                             const LongType *zTadShapeInfo, const LongType *zTadOffsets,
                                             LongType *numTadsPerInput, LongType numOutputsTad) {
  //note: this implementation is less than ideal but several forms of parallelization do not seem to work.
  //for now since this isn't a computationally intensive function this serial implementation that works correctly
  //will stay.
  auto bz = reinterpret_cast<X *>(vz);
  int arrIndex = threadIdx.x;
  //each input
  for (int e = arrIndex; e < inputSize; e++) {
    auto indices = reinterpret_cast<Y *>(vindices[e]);

    auto iShapeInfo = iShapeInfos[e];
    auto numTads = numTadsPerInput[e];
    if (shape::isEmpty(iShapeInfo)) continue;

    auto iLength = shape::length(iShapeInfo);
    auto zLength = shape::length(zTadShapeInfo);

    auto xTadShapeInfo = xTadShapeInfos[e];
    auto xTadLength = shape::length(xTadShapeInfo);

    // process each index setting values for this tad
    for (int i = 0; i < iLength; i++) {
      auto idx = indices[shape::getIndexOffset(i, iShapeInfo)];

      // the input at a given index starting at the offset for the current tad
      auto x = reinterpret_cast<X *>(vx[e]) + xTadOffsets[e][i];
      auto zTad = bz + zTadOffsets[idx];
      for (int j = 0; j < xTadLength; j++) {
        auto xIdx = shape::getIndexOffset(j, xTadShapeInfo);
        auto zIdx = shape::getIndexOffset(j, zTadShapeInfo);
        if (xIdx < xTadLength && xIdx >= 0 && zIdx < zLength && zIdx >= 0) zTad[zIdx] = x[xIdx];
      }
    }
  }

  __syncthreads();

}


template <typename X, typename Y>
static Status _dynamicStitchFunctor(LaunchContext *context, std::vector<NDArray *> const &inputs,
                                        std::vector<NDArray *> const &indices, NDArray *output) {
  LongType inputSize = inputs.size();

  PointersManager pm(context, "dynamicStitch");

  if (output->isVector()) {
    std::vector<const void *> inputBuffers(inputSize);
    std::vector<const LongType *> inputShapes(inputSize);
    std::vector<const void *> indicesBuffers(inputSize);
    std::vector<const LongType *> indicesShapes(inputSize);

    for (LongType e = 0; e < inputSize; e++) {
      inputBuffers[e] = inputs.at(e)->specialBuffer();
      indicesBuffers[e] = indices.at(e)->specialBuffer();

      inputShapes[e] = inputs.at(e)->specialShapeInfo();
      indicesShapes[e] = indices.at(e)->specialShapeInfo();
    }

    // copying pointers to buffers to device
    auto dInputBuffers =
        reinterpret_cast<void **>(pm.replicatePointer(inputBuffers.data(), inputSize * sizeof(void *)));
    auto dIndicesBuffers =
        reinterpret_cast<void **>(pm.replicatePointer(indicesBuffers.data(), inputSize * sizeof(void *)));
    auto dInputShapes =
        reinterpret_cast<LongType **>(pm.replicatePointer(inputShapes.data(), inputSize * sizeof(LongType *)));
    auto dIndicesShapes = reinterpret_cast<LongType **>(
        pm.replicatePointer(indicesShapes.data(), inputSize * sizeof(LongType *)));
    dim3 launchDims = getLaunchDims("dynamic_stitch_tad");

    dynamicStitchScalarKernel<X, Y><<<launchDims.y, launchDims.x, launchDims.z, *context->getCudaStream()>>>(
        dInputBuffers, dInputShapes, dIndicesBuffers, dIndicesShapes, inputSize, output->specialBuffer(),
        output->specialShapeInfo(), output->lengthOf());
    DebugHelper::checkErrorCode(context->getCudaStream(),"dynamicStitchScalarKernel failed");

  } else {
    std::vector<LongType> restDims(output->rankOf() - 1);
    for (int i = restDims.size(); i > 0; i--) restDims[restDims.size() - i] = output->rankOf() - i;
    auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), &restDims);

    std::vector<const void *> inputBuffers(inputSize);
    std::vector<const LongType *> inputTadShapes(inputSize);
    std::vector<const LongType *> inputTadOffsets(inputSize);

    std::vector<const void *> indicesBuffers(inputSize);
    std::vector<const LongType *> indicesShapes(inputSize);
    std::vector<LongType> inputsNumTads(inputSize);

    for (LongType e = 0; e < inputSize; e++) {
      std::vector<LongType> sourceDims(inputs[e]->rankOf() - indices[e]->rankOf());
      for (LongType i = sourceDims.size(); i > 0; i--) sourceDims[sourceDims.size() - i] = inputs[e]->rankOf() - i;

      auto packX = ConstantTadHelper::getInstance().tadForDimensions(inputs[e]->shapeInfo(), &sourceDims);
      indicesBuffers[e] = indices[e]->specialBuffer();
      indicesShapes[e] = indices[e]->specialShapeInfo();
      inputsNumTads[e] = packX->numberOfTads();
      inputBuffers[e] = inputs[e]->specialBuffer();
      inputTadShapes[e] = packX->platformShapeInfo();
      inputTadOffsets[e] = packX->platformOffsets();
    }

    // copying pointers to buffers to device
    auto dInputBuffers =
        reinterpret_cast<void **>(pm.replicatePointer(inputBuffers.data(), inputSize * sizeof(void *)));
    auto dInputTadShapes = reinterpret_cast<LongType **>(
        pm.replicatePointer(inputTadShapes.data(), inputSize * sizeof(LongType *)));
    auto dInputTadOffsets = reinterpret_cast<LongType **>(
        pm.replicatePointer(inputTadOffsets.data(), inputSize * sizeof(LongType *)));

    auto dIndicesBuffers =
        reinterpret_cast<void **>(pm.replicatePointer(indicesBuffers.data(), inputSize * sizeof(void *)));
    auto dIndicesShapes = reinterpret_cast<LongType **>(
        pm.replicatePointer(indicesShapes.data(), inputSize * sizeof(LongType *)));

    auto dNumTadsInputs = reinterpret_cast<LongType *>(
        pm.replicatePointer(inputsNumTads.data(), inputSize * sizeof(LongType *)));


    dim3 launchDims = getLaunchDims("dynamic_stitch_tad");
    dynamicStitchTadKernel<X, Y><<<launchDims.x, launchDims.y, launchDims.z, *context->getCudaStream()>>>(
        dInputBuffers, dInputTadShapes, dInputTadOffsets, dIndicesBuffers, dIndicesShapes, inputSize,
        output->specialBuffer(), packZ->platformShapeInfo(), packZ->platformOffsets(),dNumTadsInputs, packZ->numberOfTads());
    DebugHelper::checkErrorCode(context->getCudaStream(),"dynamicStitchTadKernel failed");

  }

  pm.synchronize();

  return Status::OK;
}

template <typename T>
static void _dynamicPartitionFunctorBP(NDArray const *input, NDArray const *indices,
                                       std::vector<NDArray *> const &inputGradientList,
                                       std::vector<NDArray *> &outputList) {}

void dynamicPartitionFunctor(LaunchContext *context, NDArray const *input, NDArray const *indices,
                             std::vector<NDArray *> &outputList) {
  auto xType = input->dataType();
  auto yType = indices->dataType();

  NDArray::prepareSpecialUse({}, {indices, input});

  BUILD_DOUBLE_SELECTOR(xType, yType, _dynamicPartitionFunctor, (context, input, indices, outputList), SD_NUMERIC_TYPES,
                        SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({}, {indices, input});

  // TODO: it would be nice to have NDArray::registerSpecialUse signature that accepts something else beyond
  // initializer_list
  for (auto v : outputList) {
    v->tickWriteDevice();
  }
}

template <typename T>
static Status _dynamicStitchFunctorBP(std::vector<NDArray *> const &inputs, std::vector<NDArray *> const &indices,
                                      NDArray const *gradInput, std::vector<NDArray *> &outputList) {
  THROW_EXCEPTION("Not implemented yet");
}

Status dynamicStitchFunctor(LaunchContext *context, std::vector<NDArray *> const &inputs,
                            std::vector<NDArray *> const &indices, NDArray *output) {
  auto xType = inputs.at(0)->dataType();
  auto yType = indices.at(0)->dataType();

  for (auto v : indices) {
    v->syncToDevice();
    v->tickReadDevice();
  }

  for (auto v : inputs) {
    v->syncToDevice();
    v->tickReadDevice();
  }

  NDArray::prepareSpecialUse({output}, {});

  BUILD_DOUBLE_SELECTOR(xType, yType, _dynamicStitchFunctor, (context, inputs, indices, output), SD_NUMERIC_TYPES,
                        SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({output}, {});

  return Status::OK;
}

Status dynamicStitchFunctorBP(LaunchContext *context, std::vector<NDArray *> const &inputs,
                                  std::vector<NDArray *> const &indices, NDArray const *gradInput,
                                  std::vector<NDArray *> &outputList) {
  auto xType = inputs.at(0)->dataType();

  BUILD_SINGLE_SELECTOR(xType, return _dynamicStitchFunctorBP, (inputs, indices, gradInput, outputList),
                        SD_NUMERIC_TYPES);
}

void dynamicPartitionFunctorBP(LaunchContext *context, NDArray const *input, NDArray const *indices,
                               std::vector<NDArray *> const &inputGradientList, std::vector<NDArray *> &outputList) {
  auto xType = input->dataType();

  BUILD_SINGLE_SELECTOR(xType, _dynamicPartitionFunctorBP, (input, indices, inputGradientList, outputList),
                        SD_NUMERIC_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
