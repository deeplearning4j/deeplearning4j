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
static void logSoftmaxCUDNN(const LaunchContext* context, NDArray* input, NDArray* output, const int dimension) {
  // cuDNN log softmax uses CUDNN_SOFTMAX_LOG algorithm

  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  auto stream = cudnnCaptureAwareStream(context->getCudaStream());
  CHECK_CUDNN_FAILURE(cudnnSetStream(*handle, stream));

  const cudnnDataType_t dataType = cudnnDataType(input->dataType());

  const int rank = input->rankOf();
  const int dim = dimension < 0 ? rank + dimension : dimension;

  // Calculate the dimensions for the 4D tensor
  // The dimension we want to apply softmax on should be the channel dimension (C)
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

  // Scaling factors
  static const float alpha32 = 1.0f, beta32 = 0.0f;
  static const double alpha64 = 1.0, beta64 = 0.0;
  const void* ptrAlpha = output->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* ptrBeta = output->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  NDArray::prepareSpecialUse({output}, {input});

  // Use CUDNN_SOFTMAX_LOG for log softmax
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnSoftmaxForward),
      cudnnSoftmaxForward(*handle, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_CHANNEL,
                          ptrAlpha, x, input->specialBuffer(),
                          ptrBeta, z, output->specialBuffer()));

  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    auto cudaErr = cudaStreamSynchronize(stream);
    if (cudaErr != 0) { std::string msg = "logSoftmaxCUDNN: cudaStreamSynchronize failed!; Error code: [" + std::to_string(cudaErr) + "]"; THROW_EXCEPTION(msg.c_str()); }
  }

  NDArray::registerSpecialUse({output}, {input});
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(log_softmax, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  const int rank = input->rankOf();
  const int dim = block.getIArguments()->size() > 0 ? INT_ARG(0) : rank - 1;

  REQUIRE_TRUE(dim < rank && dim >= 0, 0,
               "LOG_SOFTMAX CUDNN OP: the value of input integer parameter (dimension) must be in range [0, %i), "
               "but got dimension = %i instead!",
               rank, dim);

  logSoftmaxCUDNN(block.launchContext(), input, output, dim);

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(log_softmax, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  const int rank = input->rankOf();
  const int dim = block.getIArguments()->size() > 0 ? INT_ARG(0) : rank - 1;

  Requirements req("CUDNN LOG_SOFTMAX OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
  req.expectIn(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT0), {HALF, FLOAT32, DOUBLE}) &&
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
