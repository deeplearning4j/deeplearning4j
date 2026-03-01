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
// @author Adam Gibson
//

#include <array/NDArrayFactory.h>
#include <ops/declarable/OpRegistrator.h>

#include "cudnnUtils.h"

namespace sd {
namespace ops {
namespace platforms {

// GRU implementation is designed for 1 physical layer
constexpr int numLayers = 1;

// Copy weights for GRU: 3 gates (reset, update, hidden) instead of 4 for LSTM
void copyGruWeights(const cudaStream_t &stream, bool isBidirectional, uint8_t *weightsSpace, size_t weightsSize,
                    uint8_t *inputWeightsData, uint8_t *recurrentWeightsData, uint8_t *biasesData, LongType inputSize,
                    int hiddenSize, int dataTypeSize) {
  int pseudo_layer_count = isBidirectional ? 2 : 1;
  uint8_t *wptr = weightsSpace;
  auto wEnd = wptr + weightsSize;

  // GRU has 3 gates: reset (r), update (z), hidden (h)
  // copy size for 1 full pseudo layer
  auto input_pseudo_size = 3 * inputSize * hiddenSize * dataTypeSize;
  auto hidden_pseudo_size = 3 * hiddenSize * hiddenSize * dataTypeSize;

  for (LongType i = 0; i < pseudo_layer_count; i++) {
    if (wptr + input_pseudo_size + hidden_pseudo_size > wEnd) return;
    // copy input weights
    if (inputWeightsData) {
      cudaMemcpyAsync(wptr, inputWeightsData, input_pseudo_size, cudaMemcpyDeviceToDevice, stream);
      inputWeightsData += input_pseudo_size;
    }
    wptr += input_pseudo_size;
    // copy recurrent weights
    if (recurrentWeightsData) {
      cudaMemcpyAsync(wptr, recurrentWeightsData, hidden_pseudo_size, cudaMemcpyDeviceToDevice, stream);
      recurrentWeightsData += hidden_pseudo_size;
    }
    wptr += hidden_pseudo_size;
  }

  // copy biases (3 gates for GRU)
  auto bias_size = 3 * hiddenSize * dataTypeSize;
  for (int i = 0; i < pseudo_layer_count; i++) {
    // refill first 3 biases
    if (biasesData && wptr + bias_size < wEnd) {
      cudaMemcpyAsync(wptr, biasesData, bias_size, cudaMemcpyDeviceToDevice, stream);
      biasesData += bias_size;
    }
    wptr += bias_size;
    // refill next 3 with zeros (cuDNN expects double bias)
    if (wptr + bias_size < wEnd) {
      cudaMemsetAsync(wptr, 0, bias_size, stream);
      wptr += bias_size;
    }
  }
  // memset the rest
  if (wEnd - wptr) cudaMemsetAsync(wptr, 0, wEnd - wptr, stream);
}

// Old RNN API (cuDNN < 9.0) — removed APIs in cuDNN 9.0
#if CUDNN_VERSION < 9000
void cudnn_gru_old(LaunchContext *contextPtr, NDArray *input, NDArray *inputWeights,
                   NDArray *recurrentWeights, NDArray *biases, NDArray *prevAct,
                   NDArray *outputActivations, NDArray *finalTimeStepActivations,
                   LongType maxSeqLength, LongType batchSize, LongType inputSize, LongType hiddenSize,
                   bool isBidirectional) {
  sd_debug("cudnn gru api %s \n", "v6");

  bool training = false;
  cudnnHandle_t handle = *(reinterpret_cast<cudnnHandle_t *>(contextPtr->getCuDnnHandle()));

  auto stream = *(contextPtr->getCudaStream());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(handle, stream));

  CudnnTensorList xDescList(maxSeqLength);
  CudnnTensorList yDescList(maxSeqLength);

  auto cudnnType = cudnnDataType(input->dataType());
  auto dataTypeSize = input->sizeOfT();

  CudnnTensor hxDesc, hyDesc;

  constexpr int rankOf = 3;
  const int numDirections = isBidirectional ? 2 : 1;

  const int dimsX[rankOf] = {static_cast<int>(batchSize), static_cast<int>(inputSize), 1};
  const int stridesX[rankOf] = {static_cast<int>(inputSize), 1, 1};

  const int dimsY[rankOf] = {static_cast<int>(batchSize), static_cast<int>(hiddenSize * numDirections), 1};
  const int stridesY[rankOf] = {static_cast<int>(hiddenSize * numDirections), 1, 1};

  const int dimC[rankOf] = {static_cast<int>(numLayers * numDirections), static_cast<int>(batchSize), static_cast<int>(hiddenSize)};
  const int strideC[rankOf] = {static_cast<int>(batchSize * hiddenSize), static_cast<int>(hiddenSize), 1};

  for (int i = 0; i < maxSeqLength; i++) {
    xDescList.set(i, cudnnType, rankOf, dimsX, stridesX);
    yDescList.set(i, cudnnType, rankOf, dimsY, stridesY);
  }

  auto xDesc0 = xDescList.get(0);

  hxDesc.set(cudnnType, rankOf, dimC, strideC);
  hyDesc.set(cudnnType, rankOf, dimC, strideC);

  PointersManager manager(contextPtr, __func__);

  // dropout section
  DropoutDesc dropoutDesc(nullptr);
  float dropout = 0;
  size_t sizeInBytes = 0;
  void *droupoutMem = nullptr;
  uint64_t seed = 1;
  if (dropout != 0) {
    dropoutDesc.create();
    CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnDropoutGetStatesSize), cudnnDropoutGetStatesSize(handle, &sizeInBytes));
    droupoutMem = manager.allocateDevMem(sizeInBytes);
    dropoutDesc.set(handle, dropout, droupoutMem, sizeInBytes, seed);
  }

  // RNN for GRU
  RnnDesc rnnDesc;
  cudnnRNNMode_t rnnCellMode = CUDNN_GRU;  // Use GRU instead of LSTM
  cudnnRNNAlgo_t algo = CUDNN_RNN_ALGO_STANDARD;

  auto direction = isBidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL;
  auto mathPrec = cudnnType;

  constexpr auto inputMode = CUDNN_LINEAR_INPUT;
  rnnDesc.setUsingOldAPI(handle, inputMode, direction, rnnCellMode, algo, mathPrec, hiddenSize, numLayers, dropoutDesc);

  // set up parameters
  size_t weightsSize = 0;
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnGetRNNParamsSize),
                          cudnnGetRNNParamsSize(handle, rnnDesc, xDesc0, &weightsSize, cudnnType));

  FilterDesc wDesc;
  int dimW[] = {static_cast<int>(weightsSize / dataTypeSize), 1, 1};

  wDesc.set(cudnnType, CUDNN_TENSOR_NCHW, 3, dimW);
  void *weightsSpace = manager.allocateDevMem(weightsSize);

  size_t workSpaceSizeInBytes = 0;
  size_t reserveSpaceSizeInBytes = 0;

  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnGetRNNWorkspaceSize),
      cudnnGetRNNWorkspaceSize(handle, rnnDesc, maxSeqLength, xDescList.getDescriptors(), &workSpaceSizeInBytes));

  void *workSpace = manager.allocateDevMem(workSpaceSizeInBytes);
  void *reserveSpace = nullptr;

  if (training) {
    CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnGetRNNTrainingReserveSize),
                            cudnnGetRNNTrainingReserveSize(handle, rnnDesc, maxSeqLength, xDescList.getDescriptors(),
                                                           &reserveSpaceSizeInBytes));
    reserveSpace = manager.allocateDevMem(reserveSpaceSizeInBytes);
  }

  NDArray::prepareSpecialUse({outputActivations, finalTimeStepActivations},
                             {input, inputWeights, recurrentWeights, biases, prevAct});

  uint8_t *biasesData = biases ? (uint8_t *)biases->specialBuffer() : nullptr;
  auto prevActData = prevAct ? prevAct->specialBuffer() : nullptr;
  auto finalTimeStepActivationsData = finalTimeStepActivations ? finalTimeStepActivations->specialBuffer() : nullptr;

  // transpose weights
  NDArray inputWeightsT, recurrentWeightsT;
  uint8_t *inputWeightsData = nullptr;
  uint8_t *recurrentWeightsData = nullptr;
  if (inputWeights) {
    auto* tPtr = inputWeights->transpose();
    auto* dPtr = tPtr->dup('c');
    inputWeightsT = std::move(*dPtr);
    delete dPtr;
    delete tPtr;
    inputWeightsData = (uint8_t *)inputWeightsT.specialBuffer();
  }
  if (recurrentWeights) {
    auto* tPtr = recurrentWeights->transpose();
    auto* dPtr = tPtr->dup('c');
    recurrentWeightsT = std::move(*dPtr);
    delete dPtr;
    delete tPtr;
    recurrentWeightsData = (uint8_t *)recurrentWeightsT.specialBuffer();
  }

  copyGruWeights(stream, isBidirectional, (uint8_t *)weightsSpace, weightsSize, inputWeightsData, recurrentWeightsData,
                 biasesData, inputSize, hiddenSize, dataTypeSize);

  NDArray *argX = input;
  NDArray *argOutput = outputActivations;
  NDArray permutedX, outputH;

  if (outputActivations != nullptr && outputActivations->ordering() != 'c') {
    outputH = NDArray('c', std::vector<LongType>{maxSeqLength, batchSize, (numDirections * hiddenSize)},
                      outputActivations->dataType(), contextPtr);
    argOutput = &outputH;
  }

  auto xData = argX->specialBuffer();
  auto yData = argOutput ? argOutput->specialBuffer() : nullptr;

  if (training) {
    CHECK_CUDNN_FAILURE_MSG(
        STRINGIZE(cudnnRNNForwardTraining),
        cudnnRNNForwardTraining(handle, rnnDesc, (int)maxSeqLength, xDescList.getDescriptors(), xData, hxDesc,
                                prevActData, nullptr, nullptr, wDesc, weightsSpace, yDescList.getDescriptors(),
                                yData, hyDesc, finalTimeStepActivationsData, nullptr, nullptr, workSpace,
                                workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes));
  } else {
    CHECK_CUDNN_FAILURE_MSG(
        STRINGIZE(cudnnRNNForwardInference),
        cudnnRNNForwardInference(handle, rnnDesc, (int)maxSeqLength, xDescList.getDescriptors(), xData, hxDesc,
                                 prevActData, nullptr, nullptr, wDesc, weightsSpace, yDescList.getDescriptors(),
                                 yData, hyDesc, finalTimeStepActivationsData, nullptr, nullptr, workSpace,
                                 workSpaceSizeInBytes));
  }

  // remap output
  if (outputActivations != nullptr && argOutput != outputActivations) {
    outputActivations->assign(argOutput);
  }

  NDArray::registerSpecialUse({outputActivations, finalTimeStepActivations},
                              {input, inputWeights, recurrentWeights, biases, prevAct});

  return;
}
#endif  // CUDNN_VERSION < 9000

#if CUDNN_VERSION >= CUDNN_NEW_RNN_API_VER

void cudnn_gru_v8(LaunchContext *contextPtr, NDArray *input, NDArray *seqLengthArray,
                  NDArray *inputWeights, NDArray *recurrentWeights, NDArray *biases, NDArray *prevAct,
                  NDArray *outputActivations, NDArray *finalTimeStepActivations,
                  int maxSeqLength, int batchSize, int inputSize, int hiddenSize,
                  bool isBidirectional) {
  sd_debug("cudnn gru api %s \n", "v8");

  NDArray *argSeqNdArray = nullptr;
  NDArray seqArrIntData;
  if (seqLengthArray) {
    auto isCContiguous = [](NDArray& a) {
      auto r = a.rankOf(); auto s = a.shapeOf(); auto st = a.stridesOf();
      sd::LongType exp = 1;
      for (int i = r - 1; i >= 0; --i) { if (s[i] == 1) continue; if (st[i] != exp) return false; exp *= s[i]; }
      return true;
    };
    if (isCContiguous(*seqLengthArray) && seqLengthArray->dataType() == INT32) {
      argSeqNdArray = seqLengthArray;
    } else {
      if (seqLengthArray->dataType() != INT32) {
        auto* cPtr = seqLengthArray->cast(INT32);
        seqArrIntData = std::move(*cPtr);
        delete cPtr;
        if (!isCContiguous(seqArrIntData)) {
          auto* dPtr = seqArrIntData.dup('c');
          seqArrIntData = std::move(*dPtr);
          delete dPtr;
        }
      } else {
        auto* dPtr = seqLengthArray->dup('c');
        seqArrIntData = std::move(*dPtr);
        delete dPtr;
      }
      argSeqNdArray = &seqArrIntData;
    }
  } else {
    seqArrIntData = NDArray('c', std::vector<LongType>{(LongType)batchSize}, INT32, contextPtr);
    seqArrIntData.assign(maxSeqLength);
    argSeqNdArray = &seqArrIntData;
  }

  PointersManager manager(contextPtr, __func__);
  bool training = false;
  cudnnHandle_t handle = *(reinterpret_cast<cudnnHandle_t *>(contextPtr->getCuDnnHandle()));
  auto stream = *(contextPtr->getCudaStream());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(handle, stream));

  auto cudnnType = cudnnDataType(input->dataType());
  auto dataTypeSize = input->sizeOfT();

  CudnnTensor hDesc;

  constexpr int rankOf = 3;
  const int numDirections = isBidirectional ? 2 : 1;

  const int dimC[rankOf] = {numLayers * numDirections, batchSize, hiddenSize};
  const int strideC[rankOf] = {batchSize * hiddenSize, hiddenSize, 1};

  hDesc.set(cudnnType, rankOf, dimC, strideC);

  // dropout section
  DropoutDesc dropoutDesc(nullptr);
  float dropout = 0;
  size_t sizeInBytes = 0;
  void *droupoutMem = nullptr;
  uint64_t seed = 1;
  if (dropout != 0) {
    dropoutDesc.create();
    CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnDropoutGetStatesSize), cudnnDropoutGetStatesSize(handle, &sizeInBytes));
    droupoutMem = manager.allocateDevMem(sizeInBytes);
    dropoutDesc.set(handle, dropout, droupoutMem, sizeInBytes, seed);
  }

  // RNN for GRU
  RnnDesc rnnDesc;
  cudnnRNNMode_t rnnCellMode = CUDNN_GRU;  // Use GRU instead of LSTM
  cudnnRNNAlgo_t algo = CUDNN_RNN_ALGO_STANDARD;
  auto direction = isBidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL;
  auto mathPrec = cudnnType;

  constexpr auto inputMode = CUDNN_LINEAR_INPUT;
  bool use_tensor_ops = false;
#if CUDNN_VERSION >= CUDNN_NEW_RNN_API_VER
  cudnnMathType_t mathType = use_tensor_ops ? CUDNN_TENSOR_OP_MATH : CUDNN_FMA_MATH;
#else
  cudnnMathType_t mathType = use_tensor_ops ? CUDNN_TENSOR_OP_MATH : CUDNN_DEFAULT_MATH;
#endif

  int projSize = hiddenSize;
  cudnnRNNBiasMode_t bias_mode = CUDNN_RNN_DOUBLE_BIAS;
  uint32_t aux_flags = CUDNN_RNN_PADDED_IO_ENABLED;

  rnnDesc.set(algo, rnnCellMode, bias_mode, direction, inputMode, cudnnType, mathPrec, mathType, inputSize, hiddenSize,
              projSize, numLayers, dropoutDesc, aux_flags);

  // set Data desc
  RnnDataDesc xDataDesc, yDataDesc;
  float padding_fill = 0.0f;
  auto hostSeqArr = bufferInHost<int>(*argSeqNdArray);
  // Data format: 0 = [sL, bS, nIn] (seq_major)
  cudnnRNNDataLayout_t layout = CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED;
  xDataDesc.set(cudnnType, layout, maxSeqLength, batchSize, inputSize, hostSeqArr, (void *)&padding_fill);
  yDataDesc.set(cudnnType, layout, maxSeqLength, batchSize, hiddenSize * numDirections, hostSeqArr,
                (void *)&padding_fill);

  // set up parameters
  size_t weightsSize = 0;
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnGetRNNWeightSpaceSize),
                          cudnnGetRNNWeightSpaceSize(handle, rnnDesc, &weightsSize));

  void *weightsSpace = manager.allocateDevMem(weightsSize);

  // Set up work space and reserved memory
  void *workSpace = nullptr;
  void *reserveSpace = nullptr;

  size_t workSpaceSizeInBytes = 0;
  size_t reserveSpaceSizeInBytes = 0;

  cudnnForwardMode_t fwdMode = training ? CUDNN_FWD_MODE_TRAINING : CUDNN_FWD_MODE_INFERENCE;
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnGetRNNTempSpaceSizes),
      cudnnGetRNNTempSpaceSizes(handle, rnnDesc, fwdMode, xDataDesc, &workSpaceSizeInBytes, &reserveSpaceSizeInBytes));
  workSpace = manager.allocateDevMem(workSpaceSizeInBytes);

  if (training) {
    reserveSpace = manager.allocateDevMem(reserveSpaceSizeInBytes);
  }

  NDArray::prepareSpecialUse({outputActivations, finalTimeStepActivations},
                             {input, inputWeights, recurrentWeights, biases, prevAct, argSeqNdArray});

  auto xData = input->specialBuffer();
  uint8_t *biasesData = biases ? (uint8_t *)biases->specialBuffer() : nullptr;
  auto prevActData = prevAct ? prevAct->specialBuffer() : nullptr;
  auto yData = outputActivations ? outputActivations->specialBuffer() : nullptr;
  auto finalTimeStepActivationsData = finalTimeStepActivations ? finalTimeStepActivations->specialBuffer() : nullptr;

  // transpose weights
  NDArray inputWeightsT, recurrentWeightsT;
  uint8_t *inputWeightsData = nullptr;
  uint8_t *recurrentWeightsData = nullptr;
  if (inputWeights) {
    auto* tPtr = inputWeights->transpose();
    auto* dPtr = tPtr->dup('c');
    inputWeightsT = std::move(*dPtr);
    delete dPtr;
    delete tPtr;
    inputWeightsData = (uint8_t *)inputWeightsT.specialBuffer();
  }
  if (recurrentWeights) {
    auto* tPtr = recurrentWeights->transpose();
    auto* dPtr = tPtr->dup('c');
    recurrentWeightsT = std::move(*dPtr);
    delete dPtr;
    delete tPtr;
    recurrentWeightsData = (uint8_t *)recurrentWeightsT.specialBuffer();
  }

  copyGruWeights(stream, isBidirectional, (uint8_t *)weightsSpace, weightsSize, inputWeightsData, recurrentWeightsData,
                 biasesData, inputSize, hiddenSize, dataTypeSize);

  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnRNNForward),
      cudnnRNNForward(handle, rnnDesc, fwdMode, (const int32_t *)argSeqNdArray->specialBuffer(), xDataDesc, xData,
                      yDataDesc, yData, hDesc, prevActData, finalTimeStepActivationsData, nullptr, nullptr,
                      nullptr, weightsSize, weightsSpace, workSpaceSizeInBytes, workSpace,
                      reserveSpaceSizeInBytes, reserveSpace));

  NDArray::registerSpecialUse({outputActivations, finalTimeStepActivations},
                              {input, inputWeights, recurrentWeights, biases, prevAct});

  return;
}

#endif

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(gru, ENGINE_CUDA) {
  auto x = INPUT_VARIABLE(0);   // input [time, bS, nIn]
  auto hI = INPUT_VARIABLE(1);  // initial cell output (at time step = 0) [bS, nOut]
  auto Wx = INPUT_VARIABLE(2);  // input-to-hidden weights, [nIn, 3*nOut]
  auto Wh = INPUT_VARIABLE(3);  // hidden-to-hidden weights, [nOut, 3*nOut]
  auto b = INPUT_VARIABLE(4);   // biases, [3*nOut]

  auto h = OUTPUT_VARIABLE(0);  // cell outputs [time, bS, nOut]

  const LongType time = x->sizeAt(0);
  const LongType bS = x->sizeAt(1);
  const LongType nIn = x->sizeAt(2);
  const LongType nOut = hI->sizeAt(1);

  auto contextPtr = block.launchContext();
  bool isBidirectional = false;  // Standard GRU is unidirectional

#if CUDNN_VERSION < CUDNN_NEW_RNN_API_VER
  cudnn_gru_old(contextPtr, x, Wx, Wh, b, hI, h, nullptr, time, bS, nIn, nOut, isBidirectional);
#elif CUDNN_VERSION < 9000
  if (cudnnGetVersion() >= CUDNN_NEW_RNN_API_VER) {
    cudnn_gru_v8(contextPtr, x, nullptr, Wx, Wh, b, hI, h, nullptr, time, bS, nIn, nOut, isBidirectional);
  } else {
    cudnn_gru_old(contextPtr, x, Wx, Wh, b, hI, h, nullptr, time, bS, nIn, nOut, isBidirectional);
  }
#else
  // cuDNN 9.0+ — old RNN API removed, always use v8
  cudnn_gru_v8(contextPtr, x, nullptr, Wx, Wh, b, hI, h, nullptr, time, bS, nIn, nOut, isBidirectional);
#endif

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(gru, ENGINE_CUDA) {
  auto x = INPUT_VARIABLE(0);   // input [time, bS, nIn]
  auto hI = INPUT_VARIABLE(1);  // initial cell output [bS, nOut]
  auto Wx = INPUT_VARIABLE(2);  // input-to-hidden weights [nIn, 3*nOut]
  auto Wh = INPUT_VARIABLE(3);  // hidden-to-hidden weights [nOut, 3*nOut]
  auto b = INPUT_VARIABLE(4);   // biases [3*nOut]
  auto h = OUTPUT_VARIABLE(0);  // output [time, bS, nOut]

  DataType xType = x->dataType();

  Requirements req("CUDNN GRU OP");
  req.expectEq(makeInfoVariable(x->ordering(), ORDERING_MSG_INPUT0), 'c') &&
      req.expectEq(makeInfoVariable(Wx->dataType(), TYPE_MSG_INPUT2), makeInfoVariable(xType, TYPE_MSG_INPUT0)) &&
      req.expectEq(makeInfoVariable(Wh->dataType(), TYPE_MSG_INPUT3), makeInfoVariable(xType, TYPE_MSG_INPUT0)) &&
      req.expectEq(makeInfoVariable(b->dataType(), TYPE_MSG_INPUT_ "#4"), makeInfoVariable(xType, TYPE_MSG_INPUT0)) &&
      req.expectIn(makeInfoVariable(xType, TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectEq(makeInfoVariable(hI->ordering(), ORDERING_MSG_INPUT1), 'c') &&
      req.expectEq(makeInfoVariable(h->ordering(), ORDERING_MSG_OUTPUT0), 'c');

  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
