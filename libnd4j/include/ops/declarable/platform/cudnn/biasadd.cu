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
// cuDNN-based biasadd operation
//

#include <helpers/PointersManager.h>

#include "cudnnUtils.h"

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
static void biasaddCUDNN(const LaunchContext* context, NDArray* input, NDArray* bias, NDArray* output, bool isNCHW) {
  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, *context->getCudaStream()));

  const cudnnDataType_t dataType = cudnnDataType(input->dataType());
  const int rank = input->rankOf();

  // For NCHW: channel is dim 1, for NHWC: channel is last dim
  const int channelDim = isNCHW ? 1 : rank - 1;
  const LongType numChannels = input->sizeAt(channelDim);

  PointersManager manager(context, __func__);

  // Create input/output tensor descriptor
  CudnnTensor xDesc;

  if (rank == 4) {
    // Standard 4D case (images)
    const LongType bS = input->sizeAt(0);
    const LongType C = isNCHW ? input->sizeAt(1) : input->sizeAt(3);
    const LongType H = isNCHW ? input->sizeAt(2) : input->sizeAt(1);
    const LongType W = isNCHW ? input->sizeAt(3) : input->sizeAt(2);

    cudnnTensorFormat_t format = isNCHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;
    xDesc.set4D(format, dataType, bS, C, H, W);
  } else if (rank == 5) {
    // 5D case (3D convolutions)
    std::vector<int> dims(rank);
    std::vector<int> strides(rank);
    for (int i = 0; i < rank; i++) {
      dims[i] = static_cast<int>(input->sizeAt(i));
      strides[i] = static_cast<int>(input->strideAt(i));
    }
    xDesc.set(dataType, rank, dims.data(), strides.data());
  } else if (rank == 3) {
    // 3D case (sequences) - reshape to 4D
    const LongType bS = input->sizeAt(0);
    const LongType C = isNCHW ? input->sizeAt(1) : input->sizeAt(2);
    const LongType W = isNCHW ? input->sizeAt(2) : input->sizeAt(1);

    cudnnTensorFormat_t format = isNCHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;
    xDesc.set4D(format, dataType, bS, C, 1, W);
  } else if (rank == 2) {
    // 2D case (fully connected) - [batch, features]
    const LongType bS = input->sizeAt(0);
    const LongType C = input->sizeAt(1);
    xDesc.set4D(CUDNN_TENSOR_NCHW, dataType, bS, C, 1, 1);
  } else {
    THROW_EXCEPTION("biasaddCUDNN: unsupported input rank");
  }

  // Create bias tensor descriptor [1, C, 1, 1]
  CudnnTensor bDesc;
  bDesc.set4D(CUDNN_TENSOR_NCHW, dataType, 1, numChannels, 1, 1);

  // Scaling parameters
  const float alpha32 = 1.0f;
  const double alpha64 = 1.0;
  const void* alpha = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);

  // First copy input to output if they're different
  if (input != output && input->specialBuffer() != output->specialBuffer()) {
    NDArray::prepareSpecialUse({output}, {input});
    cudaMemcpyAsync(output->specialBuffer(), input->specialBuffer(),
                    input->lengthOf() * input->sizeOfT(), cudaMemcpyDeviceToDevice,
                    *context->getCudaStream());
    NDArray::registerSpecialUse({output}, {input});
  }

  NDArray::prepareSpecialUse({output}, {bias});

  // Add bias using cudnnAddTensor: C = alpha * A + C
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnAddTensor),
      cudnnAddTensor(*handle, alpha, bDesc, bias->specialBuffer(), alpha, xDesc, output->specialBuffer()));

  auto cudaErr = cudaStreamSynchronize(*context->getCudaStream());
  if (cudaErr != 0) throw cuda_exception::build("biasaddCUDNN: cudaStreamSynchronize failed!", cudaErr);

  NDArray::registerSpecialUse({output}, {bias});
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(biasadd, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto bias = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);

  const bool isNCHW = !block.getBArguments()->empty() ? B_ARG(0) : false;

  biasaddCUDNN(block.launchContext(), input, bias, output, isNCHW);

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(biasadd, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto bias = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);

  const bool isNCHW = !block.getBArguments()->empty() ? B_ARG(0) : false;
  const int channelDim = isNCHW ? 1 : input->rankOf() - 1;

  Requirements req("CUDNN BIASADD OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0),
                   makeInfoVariable(bias->dataType(), TYPE_MSG_INPUT1)) &&
      req.expectEq(makeInfoVariable(bias->rankOf(), RANK_MSG_INPUT1), 1) &&
      req.expectEq(makeInfoVariable(bias->sizeAt(0), "bias size"),
                   makeInfoVariable(input->sizeAt(channelDim), "input channel size")) &&
      req.expectIn(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), {2, 3, 4, 5}) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c');

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// Backward pass for biasadd - gradients for input and bias
PLATFORM_IMPL(biasadd_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto bias = INPUT_VARIABLE(1);
  auto gradO = INPUT_VARIABLE(2);

  auto gradI = OUTPUT_VARIABLE(0);
  auto gradB = OUTPUT_VARIABLE(1);

  const bool isNCHW = !block.getBArguments()->empty() ? B_ARG(0) : false;
  const int channelDim = isNCHW ? 1 : input->rankOf() - 1;

  // Gradient w.r.t. input is just gradO (identity)
  if (gradI != gradO) {
    gradI->assign(gradO);
  }

  // Gradient w.r.t. bias is sum of gradO along all dimensions except channel
  std::vector<LongType> reduceDims;
  for (int i = 0; i < input->rankOf(); i++) {
    if (i != channelDim) {
      reduceDims.push_back(i);
    }
  }

  gradO->reduceAlongDimension(reduce::Sum, gradB, &reduceDims, false);

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(biasadd_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto gradO = INPUT_VARIABLE(2);

  Requirements req("CUDNN BIASADD_BP OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0),
                   makeInfoVariable(gradO->dataType(), TYPE_MSG_INPUT2)) &&
      req.expectIn(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), {2, 3, 4, 5});

  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
