/*******************************************************************************
 *
 * Copyright (c) 2021 Konduit K.K.
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
//
// @author AbdelRauf
//
#include <array/NDArrayFactory.h>

#include <vector>

#include "cudnnUtils.h"

namespace sd {
namespace ops {
namespace platforms {

std::vector<int> getConcatTargets(NDArray&targetLabels, NDArray&targetLabelLengths) {
  // concatenate target labels
  const int32_t *tlabels = bufferInHost<int32_t>(targetLabels);
  const int32_t *tlens = bufferInHost<int32_t>(targetLabelLengths);
  int32_t nextOffset = targetLabels.strideAt(0);
  int32_t elStride = targetLabels.strideAt(1);
  int32_t batchCount = targetLabelLengths.lengthOf();
  std::vector<int> labels;
  labels.resize(targetLabels.lengthOf());
  int j = 0;
  if (targetLabels.ews()) {
    for (int i = 0; i < batchCount; i++) {
      int count = tlens[i];
      for (int k = 0; k < count; k++) {
        labels[j] = tlabels[k];
        j++;
      }
      tlabels += nextOffset;
    }
  } else {
    for (int i = 0; i < batchCount; i++) {
      int count = tlens[i];
      for (int k = 0; k < count; k++) {
        labels[j] = tlabels[k * elStride];
        j++;
      }
      tlabels += nextOffset;
    }
  }
  return labels;
}

void cudnnCtcLoss(const LaunchContext &context, NDArray&probs, const int32_t *targetLabelsPtr,
                  NDArray&probInputLengthes, NDArray&targetLabelLengths, NDArray &ctcLosses,
                  NDArray &grads) {
  const int dims[] = {(int)probs.sizeAt(0), (int)probs.sizeAt(1), (int)probs.sizeAt(2)};
  const int strides[] = {(int)probs.strideAt(0), (int)probs.strideAt(1), (int)probs.strideAt(2)};
  auto handle = reinterpret_cast<cudnnHandle_t *>(context.getCuDnnHandle());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, *context.getCudaStream()));

  CTCLossDesc ctcLossDesc;
  CudnnTensor probsDesc, gradsDesc(nullptr);
  bool calcGrads = !grads.isEmpty();
  auto cudnnType = cudnnDataType(probs.dataType());
  ctcLossDesc.set(cudnnType, CUDNN_LOSS_NORMALIZATION_SOFTMAX, CUDNN_PROPAGATE_NAN);
  probsDesc.set(cudnnType, probs.rankOf(), dims, strides);

  if (calcGrads) {
    gradsDesc.create();
    const int gradStrides[] = {(int)grads.strideAt(0), (int)grads.strideAt(1), (int)grads.strideAt(2)};
    gradsDesc.set(cudnnDataType(grads.dataType()), grads.rankOf(), dims, gradStrides);
  }

  size_t tempWorkSpaceSize = 0;
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnGetCTCLossWorkspaceSize),
      cudnnGetCTCLossWorkspaceSize(*handle, probsDesc, gradsDesc, targetLabelsPtr,
                                   bufferInHost<int32_t>(targetLabelLengths), bufferInHost<int32_t>(probInputLengthes),
                                   CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, ctcLossDesc, &tempWorkSpaceSize));

  PointersManager manager(&context, __func__);
  // Allocate temp tempWorkspace buffer
  void *tempWorkSpace = manager.allocateDevMem(tempWorkSpaceSize);

  NDArray::prepareSpecialUse({&ctcLosses, &grads}, {&probs});
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnCTCLoss),
      cudnnCTCLoss(*handle, probsDesc, probs.specialBuffer(), targetLabelsPtr,
                   bufferInHost<int32_t>(targetLabelLengths), bufferInHost<int32_t>(probInputLengthes),
                   ctcLosses.specialBuffer(), gradsDesc, calcGrads ? grads.specialBuffer() : nullptr,
                   CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, ctcLossDesc, tempWorkSpace, tempWorkSpaceSize));

  NDArray::registerSpecialUse({&ctcLosses, &grads}, {&probs});

  return;
}

PLATFORM_IMPL(ctc_loss, ENGINE_CUDA) {
  auto targetLabels = INPUT_VARIABLE(0);
  auto logitInput = INPUT_VARIABLE(1);
  auto targetLabelLengths = INPUT_VARIABLE(2);
  auto logitInputLengths = INPUT_VARIABLE(3);
  auto outputLosses = OUTPUT_VARIABLE(0);
  auto context = block.launchContext();
  // in Cudnn Batch is in the middle dimension
  logitInput->permutei({1, 0, 2});
  // in Cudnn targets are concantenated instead of batched as matrix
  auto labels = getConcatTargets(*targetLabels, *targetLabelLengths);
  const int32_t *ldata = labels.data();
  auto emptyGrads = NDArrayFactory::empty<float>();
  cudnnCtcLoss(*context, *logitInput, ldata, *logitInputLengths, *targetLabelLengths, *outputLosses, emptyGrads);
  return Status::OK;
}

template <typename T>
bool checkLabelLength(NDArray&labelLengthArr) {
  // check label lengths
  auto lenBatch = labelLengthArr.lengthOf();
  for (int i = 0; i < lenBatch; i++) {
    // The labelLengths is greater than 256.
    if (labelLengthArr.e<int32_t>(i) > 256) return false;
  }
  return true;
}

PLATFORM_CHECK(ctc_loss, ENGINE_CUDA) {
  auto targetLabels = INPUT_VARIABLE(0);
  auto logitInput = INPUT_VARIABLE(1);
  auto targetLabelLengths = INPUT_VARIABLE(2);
  auto logitInputLengths = INPUT_VARIABLE(3);
  auto outputLosses = OUTPUT_VARIABLE(0);
  int blankIndex = INT_ARG(0);

  Requirements req("CUDNN CTC_LOSS OP");
  req.expectEq(makeInfoVariable(blankIndex, "Blank Index"), 0) &&
      req.expectEq(makeInfoVariable(logitInput->dataType(), TYPE_MSG_INPUT1), FLOAT32) &&
      req.expectEq(makeInfoVariable(targetLabelLengths->dataType(), TYPE_MSG_INPUT2), INT32) &&
      req.expectEq(makeInfoVariable(targetLabels->ews(), EWS_MSG_INPUT0), 1) &&
      req.expectEq(makeInfoVariable(targetLabelLengths->ews(), EWS_MSG_INPUT2), 1) &&
      req.expectEq(makeInfoVariable(logitInputLengths->ews(), EWS_MSG_INPUT3), 1) &&
      req.expectEq(makeInfoVariable(outputLosses->ews(), EWS_MSG_OUTPUT), 1) &&
      req.expectTrue(
          makeInfoVariable(checkLabelLength<int32_t>(*targetLabelLengths), "target Label lengthes should be <= 256"),
          NO_MSG);
  req.logTheSuccess();
  return req;
}

PLATFORM_IMPL(ctc_loss_grad, ENGINE_CUDA) {
  auto targetLabels = INPUT_VARIABLE(0);
  auto logitInput = INPUT_VARIABLE(1);
  auto targetLabelLengths = INPUT_VARIABLE(2);
  auto logitInputLengths = INPUT_VARIABLE(3);
  auto outputGradients = OUTPUT_VARIABLE(0);
  auto context = block.launchContext();
  REQUIRE_TRUE(outputGradients->isSameShape(logitInput), 0,
               "CtcLoss Gradient: wrong shape of output array, expected is %s but got %s instead !",
               ShapeUtils::shapeAsString(logitInput).c_str(), ShapeUtils::shapeAsString(outputGradients).c_str());
  // in Cudnn Batch is in the middle dimension
  logitInput->permutei({1, 0, 2});
  outputGradients->permutei({1, 0, 2});
  // in Cudnn targets are concantenated instead of batched as matrix
  auto labels = getConcatTargets(*targetLabels, *targetLabelLengths);
  const int32_t *ldata = labels.data();
  auto tempLosses = NDArrayFactory::create<float>('c', {logitInputLengths->sizeAt(0)});
  cudnnCtcLoss(*context, *logitInput, ldata, *logitInputLengths, *targetLabelLengths, tempLosses, *outputGradients);
  // restore grads shape from {T, BATCH, C} -> {BATCHS, T, C}
  outputGradients->permutei({1, 0, 2});

  return Status::OK;
}

PLATFORM_CHECK(ctc_loss_grad, ENGINE_CUDA) {
  auto targetLabels = INPUT_VARIABLE(0);
  auto logitInput = INPUT_VARIABLE(1);
  auto targetLabelLengths = INPUT_VARIABLE(2);
  auto logitInputLengths = INPUT_VARIABLE(3);
  auto outputGrads = OUTPUT_VARIABLE(0);
  int blankIndex = INT_ARG(0);

  Requirements req("CUDNN CTC_LOSS_GRAD OP");
  req.expectEq(makeInfoVariable(blankIndex, "Blank Index"), 0) &&
      req.expectEq(makeInfoVariable(logitInput->dataType(), TYPE_MSG_INPUT1), FLOAT32) &&
      req.expectEq(makeInfoVariable(targetLabelLengths->dataType(), TYPE_MSG_INPUT2), INT32) &&
      req.expectEq(makeInfoVariable(targetLabels->ews(), EWS_MSG_INPUT0), 1) &&
      req.expectEq(makeInfoVariable(targetLabelLengths->ews(), EWS_MSG_INPUT2), 1) &&
      req.expectEq(makeInfoVariable(logitInputLengths->ews(), EWS_MSG_INPUT3), 1) &&
      req.expectEq(makeInfoVariable(outputGrads->ews(), EWS_MSG_OUTPUT), 1) &&
      req.expectTrue(
          makeInfoVariable(checkLabelLength<int32_t>(*targetLabelLengths), "target Label lengthes should be <= 256"),
          NO_MSG);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
