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

//////////////////////////////////////////////////////////////////////////
static void softmaxCUDNN(const LaunchContext* context, NDArray* input, NDArray* output, const int dimension) {
  // cuDNN softmax works on 4D tensors in NCHW format
  // We need to reshape our input to fit this format

  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  auto stream = cudnnCaptureAwareStream(context->getCudaStream());
  CHECK_CUDNN_FAILURE(cudnnSetStream(*handle, stream));

  const cudnnDataType_t dataType = cudnnDataType(input->dataType());

  const int rank = input->rankOf();
  const int dim = dimension < 0 ? rank + dimension : dimension;

  // For cuDNN softmax, we need to reshape the tensor to 4D
  // The dimension we want to apply softmax on should be the channel dimension (C)
  // Shape should be: [N, C, H, W] where softmax is applied along C

  // Calculate the dimensions for the 4D tensor
  LongType N = 1;  // batch dimension (everything before the softmax dim)
  LongType C = input->sizeAt(dim);  // the dimension to apply softmax on
  LongType H = 1;  // height (everything after the softmax dim)
  LongType W = 1;  // width (always 1 for this reshaping)

  for (int i = 0; i < dim; i++) {
    N *= input->sizeAt(i);
  }
  for (int i = dim + 1; i < rank; i++) {
    H *= input->sizeAt(i);
  }

  // Set up tensor descriptors
  CudnnTensor x, z;
  x.set4D(CUDNN_TENSOR_NCHW, dataType, static_cast<int>(N), static_cast<int>(C), static_cast<int>(H), static_cast<int>(W));
  z.set4D(CUDNN_TENSOR_NCHW, dataType, static_cast<int>(N), static_cast<int>(C), static_cast<int>(H), static_cast<int>(W));

  // Scaling factors — MUST be static so their addresses persist for CUDA graph replay.
  // During graph capture, cuDNN records H2D memcpy nodes from these host addresses.
  // Stack-local scalars would be dangling on replay, causing GPU DMA hang.
  static const float alpha32 = 1.0f, beta32 = 0.0f;
  static const double alpha64 = 1.0, beta64 = 0.0;
  const void* ptrAlpha = output->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* ptrBeta = output->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  NDArray::prepareSpecialUse({output}, {input});

  // Use CUDNN_SOFTMAX_ACCURATE for better numerical precision
  // CUDNN_SOFTMAX_MODE_CHANNEL applies softmax across the channel dimension
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnSoftmaxForward),
      cudnnSoftmaxForward(*handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                          ptrAlpha, x, input->specialBuffer(),
                          ptrBeta, z, output->specialBuffer()));

  // During CUDA graph capture, stream synchronization is illegal (error 900).
  // Operations are only being recorded, not executed — nothing to synchronize.
  if (!tl_graphExecutionActive) {
    auto cudaErr = cudaStreamSynchronize(stream);
    if (cudaErr != 0) throw cuda_exception::build("softmaxCUDNN: cudaStreamSynchronize failed!", cudaErr);
  }

  NDArray::registerSpecialUse({output}, {input});
}

//////////////////////////////////////////////////////////////////////////
static void softmaxBpCUDNN(const LaunchContext* context, NDArray* input, NDArray* gradO,
                           NDArray* softmaxOutput, NDArray* gradI, const int dimension) {
  // cuDNN softmax backward

  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  auto stream = cudnnCaptureAwareStream(context->getCudaStream());
  CHECK_CUDNN_FAILURE(cudnnSetStream(*handle, stream));

  const cudnnDataType_t dataType = cudnnDataType(input->dataType());

  const int rank = input->rankOf();
  const int dim = dimension < 0 ? rank + dimension : dimension;

  // Calculate the dimensions for the 4D tensor (same as forward pass)
  LongType N = 1;
  LongType C = input->sizeAt(dim);
  LongType H = 1;
  LongType W = 1;

  for (int i = 0; i < dim; i++) {
    N *= input->sizeAt(i);
  }
  for (int i = dim + 1; i < rank; i++) {
    H *= input->sizeAt(i);
  }

  // Set up tensor descriptors
  CudnnTensor y, dy, dx;
  y.set4D(CUDNN_TENSOR_NCHW, dataType, static_cast<int>(N), static_cast<int>(C), static_cast<int>(H), static_cast<int>(W));
  dy.set4D(CUDNN_TENSOR_NCHW, dataType, static_cast<int>(N), static_cast<int>(C), static_cast<int>(H), static_cast<int>(W));
  dx.set4D(CUDNN_TENSOR_NCHW, dataType, static_cast<int>(N), static_cast<int>(C), static_cast<int>(H), static_cast<int>(W));

  // Scaling factors — MUST be static for CUDA graph replay safety (see forward pass comment).
  static const float alpha32 = 1.0f, beta32 = 0.0f;
  static const double alpha64 = 1.0, beta64 = 0.0;
  const void* ptrAlpha = gradI->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* ptrBeta = gradI->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  NDArray::prepareSpecialUse({gradI}, {softmaxOutput, gradO});

  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnSoftmaxBackward),
      cudnnSoftmaxBackward(*handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                           ptrAlpha, y, softmaxOutput->specialBuffer(),
                           dy, gradO->specialBuffer(),
                           ptrBeta, dx, gradI->specialBuffer()));

  if (!tl_graphExecutionActive) {
    auto cudaErr = cudaStreamSynchronize(stream);
    if (cudaErr != 0) throw cuda_exception::build("softmaxBpCUDNN: cudaStreamSynchronize failed!", cudaErr);
  }

  NDArray::registerSpecialUse({gradI}, {softmaxOutput, gradO});
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(softmax, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  const int rank = input->rankOf();
  const int dim = block.getIArguments()->size() > 0 ? INT_ARG(0) : rank - 1;

  REQUIRE_TRUE(dim < rank && dim >= 0, 0,
               "SOFTMAX CUDNN OP: the value of input integer parameter (dimension) must be in range [0, %i), "
               "but got dimension = %i instead!",
               rank, dim);

  softmaxCUDNN(block.launchContext(), input, output, dim);

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(softmax, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  const int rank = input->rankOf();
  const int dim = block.getIArguments()->size() > 0 ? INT_ARG(0) : rank - 1;

  Requirements req("CUDNN SOFTMAX OP");
  // cuDNN softmax supports FLOAT16, FLOAT32, FLOAT64
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
  req.expectIn(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT0), {HALF, FLOAT32, DOUBLE}) &&
  // Ensure the tensor can be reshaped to 4D for cuDNN
  req.expectGreaterEq(makeInfoVariable(rank, RANK_MSG_INPUT0), 1) &&
  req.expectLess(makeInfoVariable(dim, "dimension"), rank) &&
  req.expectGreaterEq(makeInfoVariable(dim, "dimension"), 0) &&
  // cuDNN has limits on tensor dimensions
  req.expectLessEq(makeInfoVariable(input->lengthOf(), LENGTH_MSG_INPUT0), static_cast<LongType>(INT_MAX));

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(softmax_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto gradO = INPUT_VARIABLE(1);
  auto softmaxOutput = INPUT_VARIABLE(2);
  auto gradI = OUTPUT_VARIABLE(0);

  const int rank = input->rankOf();
  const int dim = block.getIArguments()->size() > 0 ? INT_ARG(0) : rank - 1;

  REQUIRE_TRUE(dim < rank && dim >= 0, 0,
               "SOFTMAX_BP CUDNN OP: the value of input integer parameter (dimension) must be in range [0, %i), "
               "but got dimension = %i instead!",
               rank, dim);

  softmaxBpCUDNN(block.launchContext(), input, gradO, softmaxOutput, gradI, dim);

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(softmax_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto gradO = INPUT_VARIABLE(1);
  auto softmaxOutput = INPUT_VARIABLE(2);
  auto gradI = OUTPUT_VARIABLE(0);

  const int rank = input->rankOf();
  const int dim = block.getIArguments()->size() > 0 ? INT_ARG(0) : rank - 1;

  Requirements req("CUDNN SOFTMAX_BP OP");
  // cuDNN softmax supports FLOAT16, FLOAT32, FLOAT64
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
  req.expectIn(makeInfoVariable(gradO->dataType(), TYPE_MSG_INPUT1), {HALF, FLOAT32, DOUBLE}) &&
  req.expectIn(makeInfoVariable(softmaxOutput->dataType(), TYPE_MSG_INPUT2), {HALF, FLOAT32, DOUBLE}) &&
  req.expectIn(makeInfoVariable(gradI->dataType(), TYPE_MSG_OUTPUT0), {HALF, FLOAT32, DOUBLE}) &&
  req.expectGreaterEq(makeInfoVariable(rank, RANK_MSG_INPUT0), 1) &&
  req.expectLess(makeInfoVariable(dim, "dimension"), rank) &&
  req.expectGreaterEq(makeInfoVariable(dim, "dimension"), 0) &&
  req.expectLessEq(makeInfoVariable(input->lengthOf(), LENGTH_MSG_INPUT0), static_cast<LongType>(INT_MAX));

  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
