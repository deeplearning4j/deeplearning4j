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

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>

#include "cudnnUtils.h"

namespace sd {
namespace ops {
namespace platforms {

// Activation descriptor RAII wrapper
struct ActivationDesc {
  MOVEONLY_DESC_FULL_IMPL(ActivationDesc, ActivationDescriptor)

  template <typename... Args>
  void set(Args&&... args) {
    CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetActivationDescriptor),
                            cudnnSetActivationDescriptor(desc, std::forward<Args>(args)...));
  }
};

//////////////////////////////////////////////////////////////////////////
static void activationCUDNN(const LaunchContext* context, NDArray* input, NDArray* output,
                            cudnnActivationMode_t mode, double coef = 0.0) {
  // cuDNN activation works on 4D tensors, we need to reshape

  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  auto stream = cudnnCaptureAwareStream(context->getCudaStream());
  CHECK_CUDNN_FAILURE(cudnnSetStream(*handle, stream));

  const cudnnDataType_t dataType = cudnnDataType(input->dataType());

  // Reshape to 4D: [total_elements, 1, 1, 1]
  const LongType length = input->lengthOf();

  // Set up tensor descriptors
  CudnnTensor x, z;
  x.set4D(CUDNN_TENSOR_NCHW, dataType, static_cast<int>(length), 1, 1, 1);
  z.set4D(CUDNN_TENSOR_NCHW, dataType, static_cast<int>(length), 1, 1, 1);

  // Set up activation descriptor
  ActivationDesc actDesc;
  actDesc.set(mode, CUDNN_PROPAGATE_NAN, coef);

  // Scaling factors
  static const float alpha32 = 1.0f, beta32 = 0.0f;
  static const double alpha64 = 1.0, beta64 = 0.0;
  const void* ptrAlpha = output->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* ptrBeta = output->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  NDArray::prepareSpecialUse({output}, {input});

  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnActivationForward),
      cudnnActivationForward(*handle, actDesc, ptrAlpha, x, input->specialBuffer(),
                             ptrBeta, z, output->specialBuffer()));

  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    auto cudaErr = cudaStreamSynchronize(stream);
    if (cudaErr != 0) throw cuda_exception::build("activationCUDNN: cudaStreamSynchronize failed!", cudaErr);
  }

  NDArray::registerSpecialUse({output}, {input});
}

//////////////////////////////////////////////////////////////////////////
static void activationBpCUDNN(const LaunchContext* context, NDArray* input, NDArray* gradO,
                              NDArray* output, NDArray* gradI, cudnnActivationMode_t mode, double coef = 0.0) {
  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  auto stream = cudnnCaptureAwareStream(context->getCudaStream());
  CHECK_CUDNN_FAILURE(cudnnSetStream(*handle, stream));

  const cudnnDataType_t dataType = cudnnDataType(input->dataType());

  const LongType length = input->lengthOf();

  // Set up tensor descriptors
  CudnnTensor x, y, dy, dx;
  x.set4D(CUDNN_TENSOR_NCHW, dataType, static_cast<int>(length), 1, 1, 1);
  y.set4D(CUDNN_TENSOR_NCHW, dataType, static_cast<int>(length), 1, 1, 1);
  dy.set4D(CUDNN_TENSOR_NCHW, dataType, static_cast<int>(length), 1, 1, 1);
  dx.set4D(CUDNN_TENSOR_NCHW, dataType, static_cast<int>(length), 1, 1, 1);

  // Set up activation descriptor
  ActivationDesc actDesc;
  actDesc.set(mode, CUDNN_PROPAGATE_NAN, coef);

  // Scaling factors
  static const float alpha32 = 1.0f, beta32 = 0.0f;
  static const double alpha64 = 1.0, beta64 = 0.0;
  const void* ptrAlpha = gradI->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* ptrBeta = gradI->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  NDArray::prepareSpecialUse({gradI}, {input, output, gradO});

  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnActivationBackward),
      cudnnActivationBackward(*handle, actDesc, ptrAlpha,
                              y, output->specialBuffer(),
                              dy, gradO->specialBuffer(),
                              x, input->specialBuffer(),
                              ptrBeta, dx, gradI->specialBuffer()));

  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    auto cudaErr = cudaStreamSynchronize(stream);
    if (cudaErr != 0) throw cuda_exception::build("activationBpCUDNN: cudaStreamSynchronize failed!", cudaErr);
  }

  NDArray::registerSpecialUse({gradI}, {input, output, gradO});
}

//////////////////////////////////////////////////////////////////////////
// RELU
//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(relu, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  activationCUDNN(block.launchContext(), input, output, CUDNN_ACTIVATION_RELU);

  return Status::OK;
}

PLATFORM_CHECK(relu, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("CUDNN RELU OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
  req.expectIn(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT0), {HALF, FLOAT32, DOUBLE}) &&
  req.expectLessEq(makeInfoVariable(input->lengthOf(), LENGTH_MSG_INPUT0), static_cast<LongType>(INT_MAX));

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// SIGMOID
//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(sigmoid, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  activationCUDNN(block.launchContext(), input, output, CUDNN_ACTIVATION_SIGMOID);

  return Status::OK;
}

PLATFORM_CHECK(sigmoid, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("CUDNN SIGMOID OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
  req.expectIn(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT0), {HALF, FLOAT32, DOUBLE}) &&
  req.expectLessEq(makeInfoVariable(input->lengthOf(), LENGTH_MSG_INPUT0), static_cast<LongType>(INT_MAX));

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// TANH
//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(tanh, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  activationCUDNN(block.launchContext(), input, output, CUDNN_ACTIVATION_TANH);

  return Status::OK;
}

PLATFORM_CHECK(tanh, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("CUDNN TANH OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
  req.expectIn(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT0), {HALF, FLOAT32, DOUBLE}) &&
  req.expectLessEq(makeInfoVariable(input->lengthOf(), LENGTH_MSG_INPUT0), static_cast<LongType>(INT_MAX));

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// ELU
//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(elu, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  // ELU alpha coefficient, default is 1.0
  double alpha = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0;

  activationCUDNN(block.launchContext(), input, output, CUDNN_ACTIVATION_ELU, alpha);

  return Status::OK;
}

PLATFORM_CHECK(elu, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("CUDNN ELU OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
  req.expectIn(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT0), {HALF, FLOAT32, DOUBLE}) &&
  req.expectLessEq(makeInfoVariable(input->lengthOf(), LENGTH_MSG_INPUT0), static_cast<LongType>(INT_MAX));

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// RELU6 (Clipped ReLU with ceiling of 6)
//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(relu6, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  // CUDNN_ACTIVATION_CLIPPED_RELU clips to [0, coef] where coef is the ceiling
  double ceiling = block.getTArguments()->size() > 0 ? T_ARG(0) : 6.0;

  activationCUDNN(block.launchContext(), input, output, CUDNN_ACTIVATION_CLIPPED_RELU, ceiling);

  return Status::OK;
}

PLATFORM_CHECK(relu6, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("CUDNN RELU6 OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
  req.expectIn(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT0), {HALF, FLOAT32, DOUBLE}) &&
  req.expectLessEq(makeInfoVariable(input->lengthOf(), LENGTH_MSG_INPUT0), static_cast<LongType>(INT_MAX));

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// Backward passes for activation functions
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
// RELU_BP
//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(relu_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto gradO = INPUT_VARIABLE(1);
  auto gradI = OUTPUT_VARIABLE(0);

  // For ReLU backward, we need the forward output
  // Create temporary array for forward pass result
  NDArray output(input->shapeInfo(), false, block.launchContext(), true);
  activationCUDNN(block.launchContext(), input, &output, CUDNN_ACTIVATION_RELU);

  activationBpCUDNN(block.launchContext(), input, gradO, &output, gradI, CUDNN_ACTIVATION_RELU);

  return Status::OK;
}

PLATFORM_CHECK(relu_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto gradO = INPUT_VARIABLE(1);
  auto gradI = OUTPUT_VARIABLE(0);

  Requirements req("CUDNN RELU_BP OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
  req.expectIn(makeInfoVariable(gradO->dataType(), TYPE_MSG_INPUT1), {HALF, FLOAT32, DOUBLE}) &&
  req.expectIn(makeInfoVariable(gradI->dataType(), TYPE_MSG_OUTPUT0), {HALF, FLOAT32, DOUBLE}) &&
  req.expectLessEq(makeInfoVariable(input->lengthOf(), LENGTH_MSG_INPUT0), static_cast<LongType>(INT_MAX));

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// SIGMOID_BP
//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(sigmoid_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto gradO = INPUT_VARIABLE(1);
  auto gradI = OUTPUT_VARIABLE(0);

  NDArray output(input->shapeInfo(), false, block.launchContext(), true);
  activationCUDNN(block.launchContext(), input, &output, CUDNN_ACTIVATION_SIGMOID);

  activationBpCUDNN(block.launchContext(), input, gradO, &output, gradI, CUDNN_ACTIVATION_SIGMOID);

  return Status::OK;
}

PLATFORM_CHECK(sigmoid_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto gradO = INPUT_VARIABLE(1);
  auto gradI = OUTPUT_VARIABLE(0);

  Requirements req("CUDNN SIGMOID_BP OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
  req.expectIn(makeInfoVariable(gradO->dataType(), TYPE_MSG_INPUT1), {HALF, FLOAT32, DOUBLE}) &&
  req.expectIn(makeInfoVariable(gradI->dataType(), TYPE_MSG_OUTPUT0), {HALF, FLOAT32, DOUBLE}) &&
  req.expectLessEq(makeInfoVariable(input->lengthOf(), LENGTH_MSG_INPUT0), static_cast<LongType>(INT_MAX));

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// TANH_BP
//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(tanh_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto gradO = INPUT_VARIABLE(1);
  auto gradI = OUTPUT_VARIABLE(0);

  NDArray output(input->shapeInfo(), false, block.launchContext(), true);
  activationCUDNN(block.launchContext(), input, &output, CUDNN_ACTIVATION_TANH);

  activationBpCUDNN(block.launchContext(), input, gradO, &output, gradI, CUDNN_ACTIVATION_TANH);

  return Status::OK;
}

PLATFORM_CHECK(tanh_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto gradO = INPUT_VARIABLE(1);
  auto gradI = OUTPUT_VARIABLE(0);

  Requirements req("CUDNN TANH_BP OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
  req.expectIn(makeInfoVariable(gradO->dataType(), TYPE_MSG_INPUT1), {HALF, FLOAT32, DOUBLE}) &&
  req.expectIn(makeInfoVariable(gradI->dataType(), TYPE_MSG_OUTPUT0), {HALF, FLOAT32, DOUBLE}) &&
  req.expectLessEq(makeInfoVariable(input->lengthOf(), LENGTH_MSG_INPUT0), static_cast<LongType>(INT_MAX));

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// ELU_BP
//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(elu_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto gradO = INPUT_VARIABLE(1);
  auto gradI = OUTPUT_VARIABLE(0);

  double alpha = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0;

  NDArray output(input->shapeInfo(), false, block.launchContext(), true);
  activationCUDNN(block.launchContext(), input, &output, CUDNN_ACTIVATION_ELU, alpha);

  activationBpCUDNN(block.launchContext(), input, gradO, &output, gradI, CUDNN_ACTIVATION_ELU, alpha);

  return Status::OK;
}

PLATFORM_CHECK(elu_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto gradO = INPUT_VARIABLE(1);
  auto gradI = OUTPUT_VARIABLE(0);

  Requirements req("CUDNN ELU_BP OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
  req.expectIn(makeInfoVariable(gradO->dataType(), TYPE_MSG_INPUT1), {HALF, FLOAT32, DOUBLE}) &&
  req.expectIn(makeInfoVariable(gradI->dataType(), TYPE_MSG_OUTPUT0), {HALF, FLOAT32, DOUBLE}) &&
  req.expectLessEq(makeInfoVariable(input->lengthOf(), LENGTH_MSG_INPUT0), static_cast<LongType>(INT_MAX));

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// RELU6_BP
//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(relu6_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto gradO = INPUT_VARIABLE(1);
  auto gradI = OUTPUT_VARIABLE(0);

  double ceiling = block.getTArguments()->size() > 0 ? T_ARG(0) : 6.0;

  NDArray output(input->shapeInfo(), false, block.launchContext(), true);
  activationCUDNN(block.launchContext(), input, &output, CUDNN_ACTIVATION_CLIPPED_RELU, ceiling);

  activationBpCUDNN(block.launchContext(), input, gradO, &output, gradI, CUDNN_ACTIVATION_CLIPPED_RELU, ceiling);

  return Status::OK;
}

PLATFORM_CHECK(relu6_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto gradO = INPUT_VARIABLE(1);
  auto gradI = OUTPUT_VARIABLE(0);

  Requirements req("CUDNN RELU6_BP OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
  req.expectIn(makeInfoVariable(gradO->dataType(), TYPE_MSG_INPUT1), {HALF, FLOAT32, DOUBLE}) &&
  req.expectIn(makeInfoVariable(gradI->dataType(), TYPE_MSG_OUTPUT0), {HALF, FLOAT32, DOUBLE}) &&
  req.expectLessEq(makeInfoVariable(input->lengthOf(), LENGTH_MSG_INPUT0), static_cast<LongType>(INT_MAX));

  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
