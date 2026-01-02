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
// OneDNN implementation of batched_gemm operation
// Uses DNNL's native matmul for each batch element
//

#include <helpers/MKLDNNStream.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>

#include "mkldnnUtils.h"

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
static void batchedGemmONEDNN(std::vector<NDArray*>& vA, std::vector<NDArray*>& vB, std::vector<NDArray*>& vC,
                               NDArray* alphas, NDArray* betas, int transA, int transB,
                               int M, int N, int K, int ldA, int ldB, int ldC) {

  const int batchSize = vA.size();
  if (batchSize == 0) return;

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());
  dnnl::stream stream(engine);

  // Get data type
  dnnl::memory::data_type dtype;
  auto xType = vA[0]->dataType();
  if (xType == DataType::FLOAT32)
    dtype = dnnl::memory::data_type::f32;
  else if (xType == DataType::HALF)
    dtype = dnnl::memory::data_type::f16;
  else if (xType == DataType::BFLOAT16)
    dtype = dnnl::memory::data_type::bf16;
  else
    return; // Unsupported type, fall back to default implementation

  // Convert BLAS transpose flags (111=NoTrans, 112=Trans)
  bool transposeA = (transA == 112);
  bool transposeB = (transB == 112);

  // For BLAS GEMM: C = alpha * op(A) * op(B) + beta * C
  // OneDNN matmul: C = A * B (with optional post-ops for alpha/beta)
  //
  // BLAS uses column-major storage, where a MxK matrix has stride [1, lda]
  // OneDNN uses row-major by default

  // Process each batch
  for (int batch = 0; batch < batchSize; ++batch) {
    auto A = vA[batch];
    auto B = vB[batch];
    auto C = vC[batch];

    // Get alpha and beta for this batch
    float alpha = alphas->isScalar() ? alphas->e<float>(0) : alphas->e<float>(batch);
    float beta = betas->isScalar() ? betas->e<float>(0) : betas->e<float>(batch);

    // OneDNN memory dimensions (row-major interpretation)
    // For op(A) with shape [M, K], OneDNN needs source memory
    // For op(B) with shape [K, N], OneDNN needs weights memory
    // Result C has shape [M, N]
    dnnl::memory::dims aShape = {M, K};
    dnnl::memory::dims bShape = {K, N};
    dnnl::memory::dims cShape = {M, N};

    // Create memory descriptors with proper strides
    // The BLAS matrices are stored column-major, which means:
    // - For non-transposed: element (i,j) is at offset i + j*lda
    // - For transposed: we read as if it were transposed

    dnnl::memory::dims aStrides, bStrides, cStrides;

    if (transposeA) {
      // A is transposed: original storage is [K, M] column-major with stride [1, ldA]
      // After transpose, logical shape is [M, K] with strides [ldA, 1]
      aStrides = {ldA, 1};
    } else {
      // A is not transposed: [M, K] column-major with stride [1, ldA]
      aStrides = {1, ldA};
    }

    if (transposeB) {
      // B is transposed: original storage is [N, K] column-major with stride [1, ldB]
      // After transpose, logical shape is [K, N] with strides [ldB, 1]
      bStrides = {ldB, 1};
    } else {
      // B is not transposed: [K, N] column-major with stride [1, ldB]
      bStrides = {1, ldB};
    }

    // C is [M, N] column-major with stride [1, ldC]
    cStrides = {1, ldC};

    // Create memory descriptors
    dnnl::memory::desc a_md(aShape, dtype, aStrides);
    dnnl::memory::desc b_md(bShape, dtype, bStrides);
    dnnl::memory::desc c_md(cShape, dtype, cStrides);

    // Create primitive attributes for alpha scaling and beta accumulation
    dnnl::primitive_attr attr;

    // Handle beta (accumulation with existing C values)
    if (beta != 0.f) {
      dnnl::post_ops po;
      po.append_sum(beta);
      attr.set_post_ops(po);
    }

    // Create primitive descriptor (OneDNN 3.x API - no separate desc class)
    dnnl::matmul::primitive_desc op_prim_desc(engine, a_md, b_md, c_md, attr);

    // Create memory objects
    dnnl::memory a_mem(a_md, engine, A->buffer());
    dnnl::memory b_mem(b_md, engine, B->buffer());
    dnnl::memory c_mem(c_md, engine, C->buffer());

    // Execute
    std::unordered_map<int, dnnl::memory> args;
    args[DNNL_ARG_SRC] = a_mem;
    args[DNNL_ARG_WEIGHTS] = b_mem;
    args[DNNL_ARG_DST] = c_mem;

    dnnl::matmul(op_prim_desc).execute(stream, args);

    // Apply alpha scaling if needed (OneDNN 3.x matmul doesn't have built-in alpha)
    if (alpha != 1.f) {
      C->applyScalar(sd::scalar::Multiply, alpha, C);
    }
  }

  stream.wait();
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(batched_gemm, ENGINE_CPU) {
  // Get transpose flags
  int transA = INT_ARG(0);
  int transB = INT_ARG(1);

  // Get alpha and beta
  auto alpha = INPUT_VARIABLE(0);
  auto beta = INPUT_VARIABLE(1);

  // Calculate batch size
  int batchSize = (block.width() - 2) / 2;

  if (batchSize <= 0) return sd::Status::OK;

  // Get first matrices to infer dimensions
  auto firstA = INPUT_VARIABLE(2);
  auto firstB = INPUT_VARIABLE(2 + batchSize);

  // Infer dimensions based on transpose flags
  // BLAS convention: transA/transB are 0 or 1 (or 111/112)
  bool doTransA = (transA == 1 || transA == 112);
  bool doTransB = (transB == 1 || transB == 112);

  int M = doTransA ? firstA->sizeAt(1) : firstA->sizeAt(0);
  int K = doTransA ? firstA->sizeAt(0) : firstA->sizeAt(1);
  int N = doTransB ? firstB->sizeAt(0) : firstB->sizeAt(1);

  // Infer leading dimensions
  int ldA = firstA->sizeAt(0);
  int ldB = firstB->sizeAt(0);
  int ldC = M;

  // Convert transpose flags to BLAS format
  int transABlas = doTransA ? 112 : 111;  // 112 = CblasTrans, 111 = CblasNoTrans
  int transBBlas = doTransB ? 112 : 111;

  // Collect matrices
  std::vector<NDArray*> vA(batchSize);
  std::vector<NDArray*> vB(batchSize);
  std::vector<NDArray*> vC(batchSize);

  for (int e = 0; e < batchSize; e++) {
    vA[e] = INPUT_VARIABLE(e + 2);
    vB[e] = INPUT_VARIABLE(e + 2 + batchSize);
    vC[e] = OUTPUT_VARIABLE(e);
  }

  batchedGemmONEDNN(vA, vB, vC, alpha, beta, transABlas, transBBlas, M, N, K, ldA, ldB, ldC);

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(batched_gemm, ENGINE_CPU) {
  // DISABLED: OneDNN batched_gemm has significant primitive creation overhead per batch
  // that makes it slower than OpenBLAS for typical workloads. The generic implementation
  // uses optimized OpenBLAS GEMM calls which are faster.
  return Requirements("ONEDNN BATCHED_GEMM OP - DISABLED");
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
