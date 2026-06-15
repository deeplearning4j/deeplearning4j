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
// cuDNN-based reduce operations: reduce_sum, reduce_mean, reduce_max, reduce_min
//

#include <helpers/ShapeUtils.h>
#include <ops/declarable/helpers/axis.h>

#include "cudnnUtils.h"

namespace sd {
namespace ops {
namespace platforms {

// RAII wrapper for cuDNN reduce tensor descriptor
struct ReduceTensorDesc {
  MOVEONLY_DESC_FULL_IMPL(ReduceTensorDesc, ReduceTensorDescriptor)

  template <typename... Args>
  void set(Args&&... args) {
    CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetReduceTensorDescriptor),
                            cudnnSetReduceTensorDescriptor(desc, std::forward<Args>(args)...));
  }
};

//////////////////////////////////////////////////////////////////////////
static void reduceTensorCUDNN(const LaunchContext* context, NDArray* input, NDArray* output,
                               const std::vector<LongType>& dimensions, cudnnReduceTensorOp_t reduceOp,
                               bool keepDims) {
  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  auto stream = cudnnCaptureAwareStream(context->getCudaStream());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, stream));

  const cudnnDataType_t dataType = cudnnDataType(input->dataType());
  const int rank = input->rankOf();

  PointersManager manager(context, __func__);

  // Create input tensor descriptor
  CudnnTensor xDesc;
  std::vector<int> xDims(rank);
  std::vector<int> xStrides(rank);
  for (int i = 0; i < rank; i++) {
    xDims[i] = static_cast<int>(input->sizeAt(i));
    xStrides[i] = static_cast<int>(input->strideAt(i));
  }
  xDesc.set(dataType, rank, xDims.data(), xStrides.data());

  // Create output tensor descriptor
  // For reduce, output dimensions are 1 for reduced dims (if keepDims) or collapsed
  CudnnTensor yDesc;
  std::vector<int> yDims(rank);
  std::vector<int> yStrides(rank);

  if (keepDims) {
    for (int i = 0; i < rank; i++) {
      bool isReduceDim = false;
      for (const auto& d : dimensions) {
        LongType dim = d < 0 ? d + rank : d;
        if (dim == i) {
          isReduceDim = true;
          break;
        }
      }
      yDims[i] = isReduceDim ? 1 : static_cast<int>(input->sizeAt(i));
    }
    // Calculate strides for keepDims case
    int stride = 1;
    for (int i = rank - 1; i >= 0; i--) {
      yStrides[i] = stride;
      stride *= yDims[i];
    }
    yDesc.set(dataType, rank, yDims.data(), yStrides.data());
  } else {
    // When not keeping dims, we need to handle output shape differently
    int outRank = output->rankOf();
    if (outRank == 0) {
      // Scalar output - use rank 1 with size 1
      int scalarDim = 1;
      int scalarStride = 1;
      yDesc.set(dataType, 1, &scalarDim, &scalarStride);
    } else {
      std::vector<int> outDims(outRank);
      std::vector<int> outStrides(outRank);
      for (int i = 0; i < outRank; i++) {
        outDims[i] = static_cast<int>(output->sizeAt(i));
        outStrides[i] = static_cast<int>(output->strideAt(i));
      }
      yDesc.set(dataType, outRank, outDims.data(), outStrides.data());
    }
  }

  // Create reduce descriptor
  ReduceTensorDesc reduceDesc;
  cudnnDataType_t compType = (dataType == CUDNN_DATA_DOUBLE) ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
  cudnnNanPropagation_t nanOpt = CUDNN_PROPAGATE_NAN;
  cudnnReduceTensorIndices_t indices = CUDNN_REDUCE_TENSOR_NO_INDICES;
  cudnnIndicesType_t indicesType = CUDNN_32BIT_INDICES;
  reduceDesc.set(reduceOp, compType, nanOpt, indices, indicesType);

  // Get workspace size
  size_t wsSize = 0;
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnGetReductionWorkspaceSize),
                          cudnnGetReductionWorkspaceSize(*handle, reduceDesc, xDesc, yDesc, &wsSize));
  void* wsData = wsSize > 0 ? manager.allocateDevMem(wsSize) : nullptr;

  // Scaling parameters
  static const float alpha32 = 1.0f, beta32 = 0.0f;
  static const double alpha64 = 1.0, beta64 = 0.0;
  const void* alpha = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* beta = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  NDArray::prepareSpecialUse({output}, {input});

  // Execute reduction
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnReduceTensor),
      cudnnReduceTensor(*handle, reduceDesc, nullptr, 0, wsData, wsSize,
                        alpha, xDesc, input->specialBuffer(),
                        beta, yDesc, output->specialBuffer()));

  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    auto cudaErr = cudaStreamSynchronize(stream);
    if (cudaErr != 0) throw cuda_exception::build("reduceTensorCUDNN: cudaStreamSynchronize failed!", cudaErr);
  }

  NDArray::registerSpecialUse({output}, {input});
}

//////////////////////////////////////////////////////////////////////////
// Helper to check if cuDNN reduce is applicable
static bool canUseCudnnReduce(NDArray* input, NDArray* output, const std::vector<LongType>& dimensions) {
  // cuDNN reduce works best for contiguous tensors
  if (input->ordering() != 'c') return false;

  // Check supported data types
  DataType dt = input->dataType();
  if (dt != FLOAT32 && dt != DOUBLE && dt != HALF) return false;

  // cuDNN has limitations on tensor dimensions (max 8)
  if (input->rankOf() > 8 || input->rankOf() < 1) return false;

  return true;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(reduce_sum, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  std::vector<LongType> dimensions;
  if (block.width() > 1) {
    auto axesVector = INPUT_VARIABLE(1);
    helpers::adjustAxis(input->rankOf(), axesVector, dimensions);
  } else if (block.getIArguments()->size()) {
    dimensions = *block.getIArguments();
  }

  bool keepDims = false;
  if (block.getBArguments()->size())
    keepDims = B_ARG(0);
  else if (block.getTArguments()->size())
    keepDims = (bool)T_ARG(0);

  // If reducing all dimensions or no dimensions specified, reduce to scalar
  if (dimensions.empty()) {
    for (int i = 0; i < input->rankOf(); i++) {
      dimensions.push_back(i);
    }
  }

  reduceTensorCUDNN(block.launchContext(), input, output, dimensions, CUDNN_REDUCE_TENSOR_ADD, keepDims);

  return Status::OK;
}

PLATFORM_CHECK(reduce_sum, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  std::vector<LongType> dimensions;
  if (block.width() > 1) {
    auto axesVector = INPUT_VARIABLE(1);
    helpers::adjustAxis(input->rankOf(), axesVector, dimensions);
  } else if (block.getIArguments()->size()) {
    dimensions = *block.getIArguments();
  }

  Requirements req("CUDNN REDUCE_SUM OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 8) &&
      req.expectGreaterEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 1) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c');

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(reduce_mean, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  std::vector<LongType> dimensions;
  if (block.width() > 1) {
    auto axesVector = INPUT_VARIABLE(1);
    helpers::adjustAxis(input->rankOf(), axesVector, dimensions);
  } else if (block.getIArguments()->size()) {
    dimensions = *block.getIArguments();
  }

  bool keepDims = false;
  if (block.getBArguments()->size())
    keepDims = B_ARG(0);
  else if (block.getTArguments()->size())
    keepDims = (bool)T_ARG(0);

  if (dimensions.empty()) {
    for (int i = 0; i < input->rankOf(); i++) {
      dimensions.push_back(i);
    }
  }

  reduceTensorCUDNN(block.launchContext(), input, output, dimensions, CUDNN_REDUCE_TENSOR_AVG, keepDims);

  return Status::OK;
}

PLATFORM_CHECK(reduce_mean, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("CUDNN REDUCE_MEAN OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 8) &&
      req.expectGreaterEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 1) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c');

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(reduce_max, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  std::vector<LongType> dimensions;
  if (block.width() > 1) {
    auto axesVector = INPUT_VARIABLE(1);
    helpers::adjustAxis(input->rankOf(), axesVector, dimensions);
  } else if (block.getIArguments()->size()) {
    dimensions = *block.getIArguments();
  }

  bool keepDims = false;
  if (block.getBArguments()->size())
    keepDims = B_ARG(0);
  else if (block.getTArguments()->size())
    keepDims = (bool)T_ARG(0);

  if (dimensions.empty()) {
    for (int i = 0; i < input->rankOf(); i++) {
      dimensions.push_back(i);
    }
  }

  reduceTensorCUDNN(block.launchContext(), input, output, dimensions, CUDNN_REDUCE_TENSOR_MAX, keepDims);

  return Status::OK;
}

PLATFORM_CHECK(reduce_max, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("CUDNN REDUCE_MAX OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 8) &&
      req.expectGreaterEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 1) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c');

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(reduce_min, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  std::vector<LongType> dimensions;
  if (block.width() > 1) {
    auto axesVector = INPUT_VARIABLE(1);
    helpers::adjustAxis(input->rankOf(), axesVector, dimensions);
  } else if (block.getIArguments()->size()) {
    dimensions = *block.getIArguments();
  }

  bool keepDims = false;
  if (block.getBArguments()->size())
    keepDims = B_ARG(0);
  else if (block.getTArguments()->size())
    keepDims = (bool)T_ARG(0);

  if (dimensions.empty()) {
    for (int i = 0; i < input->rankOf(); i++) {
      dimensions.push_back(i);
    }
  }

  reduceTensorCUDNN(block.launchContext(), input, output, dimensions, CUDNN_REDUCE_TENSOR_MIN, keepDims);

  return Status::OK;
}

PLATFORM_CHECK(reduce_min, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("CUDNN REDUCE_MIN OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 8) &&
      req.expectGreaterEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 1) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c');

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(reduce_prod, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  std::vector<LongType> dimensions;
  if (block.width() > 1) {
    auto axesVector = INPUT_VARIABLE(1);
    helpers::adjustAxis(input->rankOf(), axesVector, dimensions);
  } else if (block.getIArguments()->size()) {
    dimensions = *block.getIArguments();
  }

  bool keepDims = false;
  if (block.getBArguments()->size())
    keepDims = B_ARG(0);
  else if (block.getTArguments()->size())
    keepDims = (bool)T_ARG(0);

  if (dimensions.empty()) {
    for (int i = 0; i < input->rankOf(); i++) {
      dimensions.push_back(i);
    }
  }

  reduceTensorCUDNN(block.launchContext(), input, output, dimensions, CUDNN_REDUCE_TENSOR_MUL, keepDims);

  return Status::OK;
}

PLATFORM_CHECK(reduce_prod, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("CUDNN REDUCE_PROD OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 8) &&
      req.expectGreaterEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 1) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c');

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(reduce_norm2, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  std::vector<LongType> dimensions;
  if (block.width() > 1) {
    auto axesVector = INPUT_VARIABLE(1);
    helpers::adjustAxis(input->rankOf(), axesVector, dimensions);
  } else if (block.getIArguments()->size()) {
    dimensions = *block.getIArguments();
  }

  bool keepDims = false;
  if (block.getBArguments()->size())
    keepDims = B_ARG(0);
  else if (block.getTArguments()->size())
    keepDims = (bool)T_ARG(0);

  if (dimensions.empty()) {
    for (int i = 0; i < input->rankOf(); i++) {
      dimensions.push_back(i);
    }
  }

  reduceTensorCUDNN(block.launchContext(), input, output, dimensions, CUDNN_REDUCE_TENSOR_NORM2, keepDims);

  return Status::OK;
}

PLATFORM_CHECK(reduce_norm2, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("CUDNN REDUCE_NORM2 OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 8) &&
      req.expectGreaterEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 1) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c');

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(reduce_norm1, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  std::vector<LongType> dimensions;
  if (block.width() > 1) {
    auto axesVector = INPUT_VARIABLE(1);
    helpers::adjustAxis(input->rankOf(), axesVector, dimensions);
  } else if (block.getIArguments()->size()) {
    dimensions = *block.getIArguments();
  }

  bool keepDims = false;
  if (block.getBArguments()->size())
    keepDims = B_ARG(0);
  else if (block.getTArguments()->size())
    keepDims = (bool)T_ARG(0);

  if (dimensions.empty()) {
    for (int i = 0; i < input->rankOf(); i++) {
      dimensions.push_back(i);
    }
  }

  reduceTensorCUDNN(block.launchContext(), input, output, dimensions, CUDNN_REDUCE_TENSOR_NORM1, keepDims);

  return Status::OK;
}

PLATFORM_CHECK(reduce_norm1, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("CUDNN REDUCE_NORM1 OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 8) &&
      req.expectGreaterEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 1) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c');

  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
