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
//  @author raver119@gmail.com
//  @author Yurii Shyrma (iuriish@yahoo.com)
//
#include <cublas_v2.h>
#include <exceptions/cuda_exception.h>
#include <helpers/PointersManager.h>
#include <ops/declarable/helpers/batched_gemm.h>
#include <ops/specials_cuda.h>
#include <system/op_boilerplate.h>
#include <types/float16.h>

#include <indexing/NDIndexUtils.h>
#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops {
namespace helpers {


void bgemm(NDArray *a, NDArray *b, NDArray *c,  NDArray *alphas,  NDArray *betas,
                  int transA, int transB, int M, int N, int K, const int lda, const int ldb, const int ldc, NDArray *all) {
  NDArray *allIndex = nullptr;
  if(all != nullptr)
    allIndex = all;
  else {
    NDArray allLocal = NDIndexUtils::createAll();
    all = &allLocal;
  }


  int batchSize = a->sizeAt(0) / 2;
  std::vector<NDArray *>inputs;
  std::vector<NDArray *> keyInputs;
  std::vector<NDArray *> outputs;

  create_view createView;

  //add alpha and beta before the batch gemm, this just needs to be broadcasted
  inputs.push_back(alphas);
  inputs.push_back(betas);

  //divide by 2: queries and keys
  for(int i = 0; i < batchSize; i++) {
    auto point = NDIndexUtils::createPoint(i);
    auto aSlice = createView.evaluate({a,&point,all,all},{},{});
    auto bSlice = createView.evaluate({b,&point,all,all},{},{});
    auto outSlice = createView.evaluate({c,&point,all,all},{},{});
    inputs.push_back(aSlice.at(0));
    keyInputs.push_back(bSlice.at(0));
    outputs.push_back(outSlice.at(0));
  }

  bgemm(inputs,keyInputs,outputs,alphas,betas,transA,transB,M,N,K,lda,ldb,ldc);

}

//////////////////////////////////////////////////////////////////////////////
// bsxMXK x bSxKxN = bSxMxN
void bgemm( std::vector<NDArray *> &vA,  std::vector<NDArray *> &vB, std::vector<NDArray *> &vC,
           NDArray *alphas,  NDArray *betas, int transA, int transB, int M, int N, int K,  int lda,
           int ldb,  int ldc) {
  const auto bS = vA.size();  // batch size

  std::vector<NDArray*> pA(bS), pB(bS), pC(bS);

  std::vector<NDArray*> toDelete;

  for (int i = 0; i < bS; ++i) {
    if (vA[i]->ews() != 1) {
      pA[i] = new NDArray(vA[i]->dup('f'));
      toDelete.emplace_back(pA[i]);
    } else
      pA[i] = vA[i];

    if (vB[i]->ews() != 1) {
      pB[i] = new NDArray(vB[i]->dup('f'));
      toDelete.emplace_back(pB[i]);
    } else
      pB[i] = vB[i];

    if (vC[i]->ews() != 1) {
      pC[i] = new NDArray(vC[i]->dup('f'));
      toDelete.emplace_back(pC[i]);
    } else
      pC[i] = vC[i];

    if (pC[i]->ordering() != 'f') {
      auto temp = pA[i];
      pA[i] = new NDArray(pB[i]->permute({1, 0}));
      pB[i] = new NDArray(temp->permute({1, 0}));
      pC[i] = new NDArray(pC[i]->permute({1, 0}));
      toDelete.push_back(pA[i]);
      toDelete.push_back(pB[i]);
      toDelete.push_back(pC[i]);
      M = pA[i]->sizeAt(0);
      K = pA[i]->sizeAt(1);
      N = pB[i]->sizeAt(1);
    }

    NDArray::prepareSpecialUse({pC[i]}, {pA[i], pB[i]});
    NDArray::registerSpecialUse({pC[i]}, {pA[i], pB[i]});
  }

  NDArray::prepareSpecialUse({}, {alphas, betas});
  NDArray::registerSpecialUse({}, {alphas, betas});

  std::vector<void*> pAbuffs(bS), pBbuffs(bS), pCbuffs(bS);
  for (int i = 0; i < bS; ++i) {
    pAbuffs[i] = pA[i]->specialBuffer();
    pBbuffs[i] = pB[i]->specialBuffer();
    pCbuffs[i] = pC[i]->specialBuffer();
  }

  LaunchContext * context = vA[0]->getContext();
  PointersManager manager(context, "helpers::bgemm cuda");

  const void** aBuffers = reinterpret_cast<const void**>(manager.replicatePointer(pAbuffs.data(), bS * sizeof(void*)));
  const void** bBuffers = reinterpret_cast<const void**>(manager.replicatePointer(pBbuffs.data(), bS * sizeof(void*)));
  void** cBuffers = reinterpret_cast<void**>(manager.replicatePointer(pCbuffs.data(), bS * sizeof(void*)));



  const cublasOperation_t transAblas = transA == 112 ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t transBblas = transB == 112 ? CUBLAS_OP_T : CUBLAS_OP_N;

  if(M < 0) THROW_EXCEPTION("M < 0");
  if(N < 0) THROW_EXCEPTION("N < 0");
  if(K < 0) THROW_EXCEPTION("K < 0");


  const auto aType = pA[0]->dataType();
  const auto bType = pB[0]->dataType();
  const auto cType = pC[0]->dataType();

  std::lock_guard<std::mutex> lock(*LaunchContext::deviceMutex());

  auto handle = reinterpret_cast<cublasHandle_t*>(context->getCublasHandle());
  auto stream = context->getCudaStream();

  auto status = cublasSetStream_v2(*handle, *stream);

  if (status != CUBLAS_STATUS_SUCCESS) throw cuda_exception::build("MmulHelper::mmulMxM cuda set stream failed ! Please double check the passed in handle.", status);

  const bool AB(aType == bType), AC(aType == cType), ABC(AB && AC);

  // choose appropriate cuda gemm api depending on data types
  if (ABC && aType == DOUBLE) {
    double alpha = alphas->e<double>(0);
    double beta = betas->e<double>(0);
    status = cublasDgemmBatched(*handle, transAblas, transBblas, M, N, K, &alpha, (const double**)aBuffers, lda,
                                (const double**)bBuffers, ldb, &beta, (double**)cBuffers, ldc, bS);
  } else if (ABC && aType == FLOAT32) {
    float alpha = alphas->e<float>(0);
    float beta = betas->e<float>(0);
    status = cublasSgemmBatched(*handle, transAblas, transBblas, M, N, K, &alpha, (const float**)aBuffers, lda,
                                (const float**)bBuffers, ldb, &beta, (float**)cBuffers, ldc, bS);
  } else if (ABC && aType == HALF) {
    __half alpha = alphas->e<float>(0);
    __half beta = betas->e<float>(0);
    status = cublasHgemmBatched(*handle, transAblas, transBblas, M, N, K, &alpha, (const __half**)aBuffers, lda,
                                (const __half**)bBuffers, ldb, &beta, (__half**)cBuffers, ldc, bS);
  } else if (AB && aType == INT8 && cType == FLOAT32) {
    float alpha = alphas->e<float>(0);
    float beta = betas->e<float>(0);
    status = cublasGemmBatchedEx(*handle, transAblas, transBblas, M, N, K, &alpha, aBuffers, CUDA_R_8I, lda, bBuffers,
                                 CUDA_R_8I, ldb, &beta, cBuffers, CUDA_R_32F, ldc, bS, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
  } else if (AB && aType == HALF && cType == FLOAT32) {
    float alpha = alphas->e<float>(0);
    float beta = betas->e<float>(0);
    status =
        cublasGemmBatchedEx(*handle, transAblas, transBblas, M, N, K, &alpha, aBuffers, CUDA_R_16F, lda, bBuffers,
                            CUDA_R_16F, ldb, &beta, cBuffers, CUDA_R_32F, ldc, bS, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
  } else
    THROW_EXCEPTION("batched gemm cuda: this mode is not implemented yet !");

  if (status != CUBLAS_STATUS_SUCCESS) {
    sd_printf("Status was: %d\n",status);
    throw cuda_exception::build("MmulHelper::mmulMxM cuda execution failed !", status);
  }

  auto cudaResult = cudaStreamSynchronize(*stream);
  if (cudaResult != 0) {
    throw cuda_exception::build("MmulHelper::mmulMxM cuda stream synchronize failed !", cudaResult);
  }

  for (int i = 0; i < bS; ++i)
    if (vC[i]->ews() != 1) vC[i]->assign(pC[i]);

  for (int i = toDelete.size() - 1; i >= 0; --i) delete toDelete[i];
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
