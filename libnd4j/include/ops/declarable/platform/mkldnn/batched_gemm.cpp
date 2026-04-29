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
#if HAVE_ONEDNN
#include <dnnl.h>
#endif

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

  // OneDNN memory dimensions (row-major interpretation)
  dnnl::memory::dims aShape = {M, K};
  dnnl::memory::dims bShape = {K, N};
  dnnl::memory::dims cShape = {M, N};

  // Create memory descriptors with proper strides
  // The BLAS matrices are stored column-major, which means:
  // - For non-transposed: element (i,j) is at offset i + j*lda
  // - For transposed: we read as if it were transposed

  dnnl::memory::dims aStrides, bStrides, cStrides;

  if (transposeA) {
    aStrides = {ldA, 1};
  } else {
    aStrides = {1, ldA};
  }

  if (transposeB) {
    bStrides = {ldB, 1};
  } else {
    bStrides = {1, ldB};
  }

  cStrides = {1, ldC};

  // Create memory descriptors once — reused for all batches
  dnnl::memory::desc a_md(aShape, dtype, aStrides);
  dnnl::memory::desc b_md(bShape, dtype, bStrides);
  dnnl::memory::desc c_md(cShape, dtype, cStrides);

  // Check if all batches share the same alpha/beta (common case)
  float alpha0 = alphas->isScalar() ? alphas->e<float>(0) : alphas->e<float>(0);
  float beta0 = betas->isScalar() ? betas->e<float>(0) : betas->e<float>(0);
  bool uniformAlpha = alphas->isScalar();
  bool uniformBeta = betas->isScalar();

  // Create primitive attributes for alpha scaling and beta accumulation
  dnnl::primitive_attr attr;
  if (beta0 != 0.f) {
    dnnl::post_ops po;
    po.append_sum(beta0);
    attr.set_post_ops(po);
  }

  // Create primitive descriptor ONCE outside the loop
  dnnl::matmul::primitive_desc op_prim_desc(engine, a_md, b_md, c_md, attr);
  dnnl::matmul matmul_prim(op_prim_desc);

  // Pre-allocate args map
  std::unordered_map<int, dnnl::memory> args;
  dnnl::memory a_mem(a_md, engine, nullptr);
  dnnl::memory b_mem(b_md, engine, nullptr);
  dnnl::memory c_mem(c_md, engine, nullptr);
  args[DNNL_ARG_SRC] = a_mem;
  args[DNNL_ARG_WEIGHTS] = b_mem;
  args[DNNL_ARG_DST] = c_mem;

  // If alpha/beta vary per batch, we need per-batch primitives for different beta values
  // For the common case (uniform alpha/beta), reuse the same primitive
  for (int batch = 0; batch < batchSize; ++batch) {
    auto A = vA[batch];
    auto B = vB[batch];
    auto C = vC[batch];

    float alpha = uniformAlpha ? alpha0 : alphas->e<float>(batch);
    float beta = uniformBeta ? beta0 : betas->e<float>(batch);

    // Update memory object data handles (no reallocation)
    a_mem.set_data_handle(A->buffer());
    b_mem.set_data_handle(B->buffer());
    c_mem.set_data_handle(C->buffer());

    if (!uniformBeta && beta != beta0) {
      // Different beta — need a new primitive (rare path)
      dnnl::primitive_attr batchAttr;
      if (beta != 0.f) {
        dnnl::post_ops po;
        po.append_sum(beta);
        batchAttr.set_post_ops(po);
      }
      dnnl::matmul::primitive_desc batchPd(engine, a_md, b_md, c_md, batchAttr);
      dnnl::matmul(batchPd).execute(stream, args);
    } else {
      matmul_prim.execute(stream, args);
    }

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
  int batchSize = (block.width() - 2) / 2;
  if (batchSize <= 0) return Requirements("ONEDNN BATCHED_GEMM OP - EMPTY BATCH");

  auto firstA = INPUT_VARIABLE(2);
  auto firstB = INPUT_VARIABLE(2 + batchSize);
  auto firstC = OUTPUT_VARIABLE(0);

  auto xType = firstA->dataType();
  auto wType = firstB->dataType();
  auto zType = firstC->dataType();

  bool isFloat32 = (xType == DataType::FLOAT32 && wType == DataType::FLOAT32 && zType == DataType::FLOAT32);
  bool isBFloat16 = false;
  if (xType == DataType::BFLOAT16 && wType == DataType::BFLOAT16 &&
      (zType == DataType::BFLOAT16 || zType == DataType::FLOAT32)) {
#if HAVE_ONEDNN
    dnnl_cpu_isa_t isa = dnnl_get_effective_cpu_isa();
    isBFloat16 = (isa >= dnnl_cpu_isa_avx512_core_bf16);
#endif
  }
  bool isFloat16 = false;
  if (xType == DataType::HALF && wType == DataType::HALF &&
      (zType == DataType::HALF || zType == DataType::FLOAT32)) {
#if HAVE_ONEDNN
    dnnl_cpu_isa_t isa = dnnl_get_effective_cpu_isa();
    isFloat16 = (isa >= dnnl_cpu_isa_avx512_core_amx_fp16);
#endif
  }
  bool isSupportedType = isFloat32 || isBFloat16 || isFloat16;

  Requirements req("ONEDNN BATCHED_GEMM OP");
  req.expectTrue(makeInfoVariable(isSupportedType, TYPE_MSG_INPUT),
                 "Must be FLOAT32, BF16, or FP16") &&
      req.expectEq(makeInfoVariable(firstA->rankOf(), RANK_MSG_INPUT0), 2) &&
      req.expectEq(makeInfoVariable(firstB->rankOf(), RANK_MSG_INPUT1), 2);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
