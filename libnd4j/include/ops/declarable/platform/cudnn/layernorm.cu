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
// cuDNN-based layer normalization
// cuDNN 8.1+ has native layer normalization support via cudnnNormalizationForward
//

#include <helpers/PointersManager.h>

#include "cudnnUtils.h"

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
// Layer normalization forward using cuDNN
// For older cuDNN versions, we use batch normalization with reshaped tensors
static void layerNormCUDNN(const LaunchContext* context, NDArray* input, NDArray* gamma, NDArray* beta,
                           NDArray* output, NDArray* mean, NDArray* variance,
                           const std::vector<LongType>& normAxes, double epsilon) {
  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  auto stream = cudnnCaptureAwareStream(context->getCudaStream());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, stream));

  const cudnnDataType_t dataType = cudnnDataType(input->dataType());
  const int rank = input->rankOf();

  PointersManager manager(context, __func__);

#if CUDNN_VERSION >= 8100
  // cuDNN 8.1+ has native layer normalization via cudnnNormalizationForward
  // with CUDNN_NORM_PER_ACTIVATION mode

  // For now, use the batch normalization approach which works on all versions
#endif

  // Layer normalization can be implemented using batch normalization
  // by treating the normalized dimensions as the "batch" dimension
  // and the remaining dimensions as the "channel" dimension

  // Calculate the product of normalized dimensions and remaining dimensions
  LongType normSize = 1;
  LongType batchSize = 1;

  std::set<LongType> normAxesSet(normAxes.begin(), normAxes.end());

  for (int i = 0; i < rank; i++) {
    if (normAxesSet.count(i) > 0) {
      normSize *= input->sizeAt(i);
    } else {
      batchSize *= input->sizeAt(i);
    }
  }

  // Reshape for batch normalization: [batchSize, normSize, 1, 1]
  CudnnTensor xDesc, yDesc;
  xDesc.set4D(CUDNN_TENSOR_NCHW, dataType, static_cast<int>(batchSize), static_cast<int>(normSize), 1, 1);
  yDesc.set4D(CUDNN_TENSOR_NCHW, dataType, static_cast<int>(batchSize), static_cast<int>(normSize), 1, 1);

  // Scale/bias descriptor
  CudnnTensor bnScaleBiasMeanVarDesc;
  bnScaleBiasMeanVarDesc.set4D(CUDNN_TENSOR_NCHW, dataType, 1, static_cast<int>(normSize), 1, 1);

  // Scaling parameters
  static const float alpha32 = 1.0f, beta32 = 0.0f;
  static const double alpha64 = 1.0, beta64 = 0.0;
  const void* pAlpha = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* pBeta = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  // Allocate temp buffers for mean and variance if not provided
  void* saveMean = mean ? mean->specialBuffer() : manager.allocateDevMem(normSize * input->sizeOfT());
  void* saveInvVar = variance ? variance->specialBuffer() : manager.allocateDevMem(normSize * input->sizeOfT());

  NDArray::prepareSpecialUse({output}, {input, gamma, beta});

  // Use batch normalization in training mode to compute mean/variance per instance
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnBatchNormalizationForwardTraining),
      cudnnBatchNormalizationForwardTraining(
          *handle, CUDNN_BATCHNORM_PER_ACTIVATION,
          pAlpha, pBeta,
          xDesc, input->specialBuffer(),
          yDesc, output->specialBuffer(),
          bnScaleBiasMeanVarDesc,
          gamma ? gamma->specialBuffer() : nullptr,
          beta ? beta->specialBuffer() : nullptr,
          1.0,  // exponential average factor (use current batch only)
          nullptr,  // running mean
          nullptr,  // running variance
          epsilon,
          saveMean,
          saveInvVar));

  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    auto cudaErr = cudaStreamSynchronize(stream);
    if (cudaErr != 0) throw cuda_exception::build("layerNormCUDNN: cudaStreamSynchronize failed!", cudaErr);
  }

  NDArray::registerSpecialUse({output}, {input, gamma, beta});
}

//////////////////////////////////////////////////////////////////////////
// Layer normalization backward using cuDNN
static void layerNormBpCUDNN(const LaunchContext* context, NDArray* input, NDArray* gradO,
                              NDArray* gamma, NDArray* mean, NDArray* variance,
                              NDArray* gradI, NDArray* gradGamma, NDArray* gradBeta,
                              const std::vector<LongType>& normAxes, double epsilon) {
  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  auto stream = cudnnCaptureAwareStream(context->getCudaStream());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, stream));

  const cudnnDataType_t dataType = cudnnDataType(input->dataType());
  const int rank = input->rankOf();

  PointersManager manager(context, __func__);

  // Calculate dimensions
  LongType normSize = 1;
  LongType batchSize = 1;

  std::set<LongType> normAxesSet(normAxes.begin(), normAxes.end());

  for (int i = 0; i < rank; i++) {
    if (normAxesSet.count(i) > 0) {
      normSize *= input->sizeAt(i);
    } else {
      batchSize *= input->sizeAt(i);
    }
  }

  // Tensor descriptors
  CudnnTensor xDesc, dyDesc, dxDesc;
  xDesc.set4D(CUDNN_TENSOR_NCHW, dataType, static_cast<int>(batchSize), static_cast<int>(normSize), 1, 1);
  dyDesc.set4D(CUDNN_TENSOR_NCHW, dataType, static_cast<int>(batchSize), static_cast<int>(normSize), 1, 1);
  dxDesc.set4D(CUDNN_TENSOR_NCHW, dataType, static_cast<int>(batchSize), static_cast<int>(normSize), 1, 1);

  // Scale/bias descriptor
  CudnnTensor bnScaleBiasMeanVarDesc;
  bnScaleBiasMeanVarDesc.set4D(CUDNN_TENSOR_NCHW, dataType, 1, static_cast<int>(normSize), 1, 1);

  // Scaling parameters
  static const float alpha32 = 1.0f, beta32 = 0.0f;
  static const double alpha64 = 1.0, beta64 = 0.0;
  const void* pAlphaData = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* pBetaData = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);
  const void* pAlphaParam = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* pBetaParam = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  NDArray::prepareSpecialUse({gradI, gradGamma, gradBeta}, {input, gradO, gamma, mean, variance});

  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnBatchNormalizationBackward),
      cudnnBatchNormalizationBackward(
          *handle, CUDNN_BATCHNORM_PER_ACTIVATION,
          pAlphaData, pBetaData,
          pAlphaParam, pBetaParam,
          xDesc, input->specialBuffer(),
          dyDesc, gradO->specialBuffer(),
          dxDesc, gradI->specialBuffer(),
          bnScaleBiasMeanVarDesc,
          gamma ? gamma->specialBuffer() : nullptr,
          gradGamma ? gradGamma->specialBuffer() : nullptr,
          gradBeta ? gradBeta->specialBuffer() : nullptr,
          epsilon,
          mean ? mean->specialBuffer() : nullptr,
          variance ? variance->specialBuffer() : nullptr));

  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    auto cudaErr = cudaStreamSynchronize(stream);
    if (cudaErr != 0) throw cuda_exception::build("layerNormBpCUDNN: cudaStreamSynchronize failed!", cudaErr);
  }

  NDArray::registerSpecialUse({gradI, gradGamma, gradBeta}, {input, gradO, gamma, mean, variance});
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(layer_norm, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto gamma = INPUT_VARIABLE(1);
  auto beta = INPUT_VARIABLE(2);

  auto output = OUTPUT_VARIABLE(0);
  auto mean = OUTPUT_VARIABLE(1);      // optional
  auto variance = OUTPUT_VARIABLE(2);  // optional

  // Get normalization axes (default: last axis)
  std::vector<LongType> normAxes;
  if (block.getIArguments()->size() > 0) {
    for (size_t i = 0; i < block.getIArguments()->size(); i++) {
      normAxes.push_back(I_ARG(i));
    }
  } else {
    normAxes.push_back(input->rankOf() - 1);
  }

  double epsilon = block.getTArguments()->size() > 0 ? T_ARG(0) : 1e-5;

  layerNormCUDNN(block.launchContext(), input, gamma, beta, output,
                 block.width() > 3 ? mean : nullptr,
                 block.width() > 4 ? variance : nullptr,
                 normAxes, epsilon);

  return Status::OK;
}

PLATFORM_CHECK(layer_norm, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto gamma = INPUT_VARIABLE(1);
  auto beta = INPUT_VARIABLE(2);

  Requirements req("CUDNN LAYER_NORM OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c') &&
      req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 5);

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(layer_norm_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto gamma = INPUT_VARIABLE(1);
  auto beta = INPUT_VARIABLE(2);
  auto gradO = INPUT_VARIABLE(3);

  auto gradI = OUTPUT_VARIABLE(0);
  auto gradGamma = OUTPUT_VARIABLE(1);
  auto gradBeta = OUTPUT_VARIABLE(2);

  // Get normalization axes
  std::vector<LongType> normAxes;
  if (block.getIArguments()->size() > 0) {
    for (size_t i = 0; i < block.getIArguments()->size(); i++) {
      normAxes.push_back(I_ARG(i));
    }
  } else {
    normAxes.push_back(input->rankOf() - 1);
  }

  double epsilon = block.getTArguments()->size() > 0 ? T_ARG(0) : 1e-5;

  // Compute forward pass to get mean/variance
  NDArray mean(gamma->shapeInfo(), false, block.launchContext(), true);
  NDArray variance(gamma->shapeInfo(), false, block.launchContext(), true);
  NDArray tempOutput(input->shapeInfo(), false, block.launchContext(), true);

  layerNormCUDNN(block.launchContext(), input, gamma, beta, &tempOutput, &mean, &variance, normAxes, epsilon);

  layerNormBpCUDNN(block.launchContext(), input, gradO, gamma, &mean, &variance,
                   gradI, gradGamma, gradBeta, normAxes, epsilon);

  return Status::OK;
}

PLATFORM_CHECK(layer_norm_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);

  Requirements req("CUDNN LAYER_NORM_BP OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c') &&
      req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 5);

  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
