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
// cuDNN-based element-wise tensor operations using cudnnOpTensor
// Supports: add, multiply, min, max
//

#include <helpers/PointersManager.h>

#include "cudnnUtils.h"

namespace sd {
namespace ops {
namespace platforms {

// RAII wrapper for cuDNN OpTensor descriptor
struct OpTensorDesc {
  MOVEONLY_DESC_FULL_IMPL(OpTensorDesc, OpTensorDescriptor)

  template <typename... Args>
  void set(Args&&... args) {
    CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetOpTensorDescriptor),
                            cudnnSetOpTensorDescriptor(desc, std::forward<Args>(args)...));
  }
};

//////////////////////////////////////////////////////////////////////////
static void opTensorCUDNN(const LaunchContext* context, NDArray* a, NDArray* b, NDArray* c,
                          cudnnOpTensorOp_t opType, float alpha1 = 1.0f, float alpha2 = 1.0f, float beta = 0.0f) {
  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  auto stream = cudnnCaptureAwareStream(context->getCudaStream());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, stream));

  const cudnnDataType_t dataType = cudnnDataType(a->dataType());
  const int rank = a->rankOf();

  PointersManager manager(context, __func__);

  // Helper lambda to create tensor descriptor
  auto createTensorDesc = [&dataType](CudnnTensor& desc, NDArray* arr) {
    const int r = arr->rankOf();
    if (r <= 4) {
      // Use 4D descriptor
      int n = 1, c = 1, h = 1, w = 1;
      if (r >= 1) w = static_cast<int>(arr->sizeAt(r - 1));
      if (r >= 2) h = static_cast<int>(arr->sizeAt(r - 2));
      if (r >= 3) c = static_cast<int>(arr->sizeAt(r - 3));
      if (r >= 4) n = static_cast<int>(arr->sizeAt(r - 4));
      desc.set4D(CUDNN_TENSOR_NCHW, dataType, n, c, h, w);
    } else {
      // Use ND descriptor
      std::vector<int> dims(r);
      std::vector<int> strides(r);
      for (int i = 0; i < r; i++) {
        dims[i] = static_cast<int>(arr->sizeAt(i));
        strides[i] = static_cast<int>(arr->strideAt(i));
      }
      desc.set(dataType, r, dims.data(), strides.data());
    }
  };

  CudnnTensor aDesc, bDesc, cDesc;
  createTensorDesc(aDesc, a);
  createTensorDesc(bDesc, b);
  createTensorDesc(cDesc, c);

  // Create op tensor descriptor
  OpTensorDesc opDesc;
  cudnnDataType_t compType = (dataType == CUDNN_DATA_DOUBLE) ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
  opDesc.set(opType, compType, CUDNN_PROPAGATE_NAN);

  // Scaling parameters
  const float alpha1_32 = alpha1, alpha2_32 = alpha2, beta_32 = beta;
  const double alpha1_64 = alpha1, alpha2_64 = alpha2, beta_64 = beta;

  const void* pAlpha1 = a->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha1_32) : reinterpret_cast<const void*>(&alpha1_64);
  const void* pAlpha2 = a->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha2_32) : reinterpret_cast<const void*>(&alpha2_64);
  const void* pBeta = a->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta_32) : reinterpret_cast<const void*>(&beta_64);

  NDArray::prepareSpecialUse({c}, {a, b});

  // Execute op tensor: C = alpha1 * A op alpha2 * B + beta * C
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnOpTensor),
      cudnnOpTensor(*handle, opDesc,
                    pAlpha1, aDesc, a->specialBuffer(),
                    pAlpha2, bDesc, b->specialBuffer(),
                    pBeta, cDesc, c->specialBuffer()));

  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    auto cudaErr = cudaStreamSynchronize(stream);
    if (cudaErr != 0) { std::string msg = "opTensorCUDNN: cudaStreamSynchronize failed!; Error code: [" + std::to_string(cudaErr) + "]"; THROW_EXCEPTION(msg.c_str()); }
  }

  NDArray::registerSpecialUse({c}, {a, b});
}

//////////////////////////////////////////////////////////////////////////
// Check if tensors are compatible for cuDNN OpTensor
static bool canUseOpTensor(NDArray* a, NDArray* b, NDArray* c) {
  // Must have same data type
  if (a->dataType() != b->dataType() || a->dataType() != c->dataType()) return false;

  // Check supported types
  DataType dt = a->dataType();
  if (dt != FLOAT32 && dt != DOUBLE && dt != HALF) return false;

  // Must be contiguous
  if (a->ordering() != 'c' || b->ordering() != 'c' || c->ordering() != 'c') return false;

  // Rank limit
  if (a->rankOf() > 8 || b->rankOf() > 8 || c->rankOf() > 8) return false;

  return true;
}

//////////////////////////////////////////////////////////////////////////
// add operation: C = A + B
PLATFORM_IMPL(add, ENGINE_CUDA) {
  auto a = INPUT_VARIABLE(0);
  auto b = INPUT_VARIABLE(1);
  auto c = OUTPUT_VARIABLE(0);

  opTensorCUDNN(block.launchContext(), a, b, c, CUDNN_OP_TENSOR_ADD);

  return Status::OK;
}

PLATFORM_CHECK(add, ENGINE_CUDA) {
  auto a = INPUT_VARIABLE(0);
  auto b = INPUT_VARIABLE(1);
  auto c = OUTPUT_VARIABLE(0);

  // cuDNN OpTensor requires same shape or broadcastable in specific ways
  // For simplicity, we require same shape
  bool sameShape = a->isSameShape(b) && a->isSameShape(c);

  Requirements req("CUDNN ADD OP");
  req.expectIn(makeInfoVariable(a->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectEq(makeInfoVariable(a->dataType(), TYPE_MSG_INPUT0),
                   makeInfoVariable(b->dataType(), TYPE_MSG_INPUT1)) &&
      req.expectEq(makeInfoVariable(a->ordering(), ORDERING_MSG_INPUT0), 'c') &&
      req.expectEq(makeInfoVariable(b->ordering(), ORDERING_MSG_INPUT1), 'c') &&
      req.expectTrue(makeInfoVariable(sameShape, "same shape"));

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// multiply operation: C = A * B
PLATFORM_IMPL(multiply, ENGINE_CUDA) {
  auto a = INPUT_VARIABLE(0);
  auto b = INPUT_VARIABLE(1);
  auto c = OUTPUT_VARIABLE(0);

  opTensorCUDNN(block.launchContext(), a, b, c, CUDNN_OP_TENSOR_MUL);

  return Status::OK;
}

PLATFORM_CHECK(multiply, ENGINE_CUDA) {
  auto a = INPUT_VARIABLE(0);
  auto b = INPUT_VARIABLE(1);
  auto c = OUTPUT_VARIABLE(0);

  bool sameShape = a->isSameShape(b) && a->isSameShape(c);

  Requirements req("CUDNN MULTIPLY OP");
  req.expectIn(makeInfoVariable(a->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectEq(makeInfoVariable(a->dataType(), TYPE_MSG_INPUT0),
                   makeInfoVariable(b->dataType(), TYPE_MSG_INPUT1)) &&
      req.expectEq(makeInfoVariable(a->ordering(), ORDERING_MSG_INPUT0), 'c') &&
      req.expectEq(makeInfoVariable(b->ordering(), ORDERING_MSG_INPUT1), 'c') &&
      req.expectTrue(makeInfoVariable(sameShape, "same shape"));

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// maximum operation: C = max(A, B)
PLATFORM_IMPL(maximum, ENGINE_CUDA) {
  auto a = INPUT_VARIABLE(0);
  auto b = INPUT_VARIABLE(1);
  auto c = OUTPUT_VARIABLE(0);

  opTensorCUDNN(block.launchContext(), a, b, c, CUDNN_OP_TENSOR_MAX);

  return Status::OK;
}

PLATFORM_CHECK(maximum, ENGINE_CUDA) {
  auto a = INPUT_VARIABLE(0);
  auto b = INPUT_VARIABLE(1);
  auto c = OUTPUT_VARIABLE(0);

  bool sameShape = a->isSameShape(b) && a->isSameShape(c);

  Requirements req("CUDNN MAXIMUM OP");
  req.expectIn(makeInfoVariable(a->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectEq(makeInfoVariable(a->dataType(), TYPE_MSG_INPUT0),
                   makeInfoVariable(b->dataType(), TYPE_MSG_INPUT1)) &&
      req.expectEq(makeInfoVariable(a->ordering(), ORDERING_MSG_INPUT0), 'c') &&
      req.expectEq(makeInfoVariable(b->ordering(), ORDERING_MSG_INPUT1), 'c') &&
      req.expectTrue(makeInfoVariable(sameShape, "same shape"));

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// minimum operation: C = min(A, B)
PLATFORM_IMPL(minimum, ENGINE_CUDA) {
  auto a = INPUT_VARIABLE(0);
  auto b = INPUT_VARIABLE(1);
  auto c = OUTPUT_VARIABLE(0);

  opTensorCUDNN(block.launchContext(), a, b, c, CUDNN_OP_TENSOR_MIN);

  return Status::OK;
}

PLATFORM_CHECK(minimum, ENGINE_CUDA) {
  auto a = INPUT_VARIABLE(0);
  auto b = INPUT_VARIABLE(1);
  auto c = OUTPUT_VARIABLE(0);

  bool sameShape = a->isSameShape(b) && a->isSameShape(c);

  Requirements req("CUDNN MINIMUM OP");
  req.expectIn(makeInfoVariable(a->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectEq(makeInfoVariable(a->dataType(), TYPE_MSG_INPUT0),
                   makeInfoVariable(b->dataType(), TYPE_MSG_INPUT1)) &&
      req.expectEq(makeInfoVariable(a->ordering(), ORDERING_MSG_INPUT0), 'c') &&
      req.expectEq(makeInfoVariable(b->ordering(), ORDERING_MSG_INPUT1), 'c') &&
      req.expectTrue(makeInfoVariable(sameShape, "same shape"));

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// subtract operation: C = A - B (using add with alpha2 = -1)
PLATFORM_IMPL(subtract, ENGINE_CUDA) {
  auto a = INPUT_VARIABLE(0);
  auto b = INPUT_VARIABLE(1);
  auto c = OUTPUT_VARIABLE(0);

  opTensorCUDNN(block.launchContext(), a, b, c, CUDNN_OP_TENSOR_ADD, 1.0f, -1.0f, 0.0f);

  return Status::OK;
}

PLATFORM_CHECK(subtract, ENGINE_CUDA) {
  auto a = INPUT_VARIABLE(0);
  auto b = INPUT_VARIABLE(1);
  auto c = OUTPUT_VARIABLE(0);

  bool sameShape = a->isSameShape(b) && a->isSameShape(c);

  Requirements req("CUDNN SUBTRACT OP");
  req.expectIn(makeInfoVariable(a->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectEq(makeInfoVariable(a->dataType(), TYPE_MSG_INPUT0),
                   makeInfoVariable(b->dataType(), TYPE_MSG_INPUT1)) &&
      req.expectEq(makeInfoVariable(a->ordering(), ORDERING_MSG_INPUT0), 'c') &&
      req.expectEq(makeInfoVariable(b->ordering(), ORDERING_MSG_INPUT1), 'c') &&
      req.expectTrue(makeInfoVariable(sameShape, "same shape"));

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// Square root using cudnnOpTensor with SQRT op (cuDNN 7+)
// Note: CUDNN_OP_TENSOR_SQRT operates on A only, B is ignored
PLATFORM_IMPL(sqrt, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  auto handle = reinterpret_cast<cudnnHandle_t*>(block.launchContext()->getCuDnnHandle());
  auto stream = cudnnCaptureAwareStream(block.launchContext()->getCudaStream());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, stream));

  const cudnnDataType_t dataType = cudnnDataType(input->dataType());
  const int rank = input->rankOf();

  // Create tensor descriptor
  CudnnTensor xDesc;
  if (rank <= 4) {
    int n = 1, c = 1, h = 1, w = 1;
    if (rank >= 1) w = static_cast<int>(input->sizeAt(rank - 1));
    if (rank >= 2) h = static_cast<int>(input->sizeAt(rank - 2));
    if (rank >= 3) c = static_cast<int>(input->sizeAt(rank - 3));
    if (rank >= 4) n = static_cast<int>(input->sizeAt(rank - 4));
    xDesc.set4D(CUDNN_TENSOR_NCHW, dataType, n, c, h, w);
  } else {
    std::vector<int> dims(rank);
    std::vector<int> strides(rank);
    for (int i = 0; i < rank; i++) {
      dims[i] = static_cast<int>(input->sizeAt(i));
      strides[i] = static_cast<int>(input->strideAt(i));
    }
    xDesc.set(dataType, rank, dims.data(), strides.data());
  }

  // Create op tensor descriptor for SQRT
  OpTensorDesc opDesc;
  cudnnDataType_t compType = (dataType == CUDNN_DATA_DOUBLE) ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
  opDesc.set(CUDNN_OP_TENSOR_SQRT, compType, CUDNN_PROPAGATE_NAN);

  // Scaling parameters
  static const float alpha32 = 1.0f, beta32 = 0.0f;
  static const double alpha64 = 1.0, beta64 = 0.0;
  const void* alpha = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* beta = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  NDArray::prepareSpecialUse({output}, {input});

  // For SQRT, we use A = input, B = input (B is ignored), C = output
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnOpTensor),
      cudnnOpTensor(*handle, opDesc,
                    alpha, xDesc, input->specialBuffer(),
                    alpha, xDesc, input->specialBuffer(),  // B is ignored for SQRT
                    beta, xDesc, output->specialBuffer()));

  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    auto cudaErr = cudaStreamSynchronize(stream);
    if (cudaErr != 0) { std::string msg = "sqrt CUDNN: cudaStreamSynchronize failed!; Error code: [" + std::to_string(cudaErr) + "]"; THROW_EXCEPTION(msg.c_str()); }
  }

  NDArray::registerSpecialUse({output}, {input});

  return Status::OK;
}

PLATFORM_CHECK(sqrt, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("CUDNN SQRT OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c') &&
      req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 8);

  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
